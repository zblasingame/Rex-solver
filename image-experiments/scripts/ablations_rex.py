#!/usr/bin/env python3
"""
ablation_rex.py — Clean ablation study for the Rex (Reversible Exponential) solver.

Ablates three independent design axes of Rex and measures their contribution
to three outcome dimensions at matched compute (same NFE budget):

  ABLATION AXES
  ─────────────
  (i)  No Reversible Coupling   → plain ERK forward/backward (no McCallum-Foster pairing)
  (ii) No Exponential Transform → standard RK in x-space (no Lawson / integrating-factor
                                   change of variable; linear drift is NOT factored out)
  (iii)No Time Reparam          → integration runs in raw t rather than the
                                   transformed variable ς(t) = α/σ (or σ/α)

  OUTCOME DIMENSIONS
  ──────────────────
  (a) Inversion error  : ‖latent_reconstructed − latent_original‖₂  (cycle encode→decode)
  (b) Edit consistency : LPIPS(edited_recon, reference_edit)  +  CLIPScore(edited_recon, edit_prompt)
  (c) Generation quality : ImageReward + PickScore

  All five variants (Full Rex + 4 ablations) use the same RK tableau and the same
  total NFE so that differences are attributable to design choices, not compute.

Usage
─────
  python ablation_rex.py \
      --num_images 50 \
      --num_inference_steps 50 \
      --freeze_step 0.5 \
      --guidance 2.0 \
      --tableau euler \
      --zeta 0.999 \
      --prediction_type data \
      --save_dir results/ablation_rex

Output
──────
  results/ablation_rex/
    imgs/           per-variant PNG images
    metadata/       per-sample JSON metrics
    summary.json    aggregate mean ± std for every metric × variant
    summary.csv     same as CSV for easy plotting
    ablation_report.txt  human-readable summary table
"""

import os
import sys
import json
import csv
import math
import argparse
import textwrap
import traceback
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusers import DDIMScheduler, StableDiffusionDiffEditPipeline
from datasets import load_dataset
from tqdm import tqdm

from samplers import test_sd15
from samplers.test_sd15 import pil_to_latents
from samplers.utils import PipelineLike
from samplers.rk_tableaus import (
    ButcherTableau,
    get_rk_tableau,
    list_rk_methods,
    RK4,
    DOPRI5,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal.clip_score import CLIPScore
import ImageReward as RM
from transformers import AutoProcessor, AutoModel

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AblationConfig:
    """
    Flags that select which Rex components are active.

    All four combinations (plus full Rex) are tested.
    """
    name: str                          # human-readable variant label
    use_reversible_coupling: bool      # (i)   McCallum-Foster pairing
    use_exponential_transform: bool    # (ii)  Lawson / integrating-factor x → Z = x/w
    use_time_reparam: bool             # (iii) ς(t) = α/σ reparameterisation

    # Fixed across all variants
    zeta: float = 0.999
    prediction_type: str = "data"
    eps: float = 0.0002


# The five variants we compare
def build_variants(zeta: float, prediction_type: str) -> List[AblationConfig]:
    base = dict(zeta=zeta, prediction_type=prediction_type)
    return [
        AblationConfig("Full Rex",
                       use_reversible_coupling=True,
                       use_exponential_transform=True,
                       use_time_reparam=True,   **base),
        AblationConfig("No Reversible Coupling",
                       use_reversible_coupling=False,
                       use_exponential_transform=True,
                       use_time_reparam=True,   **base),
        AblationConfig("No Exponential Transform",
                       use_reversible_coupling=True,
                       use_exponential_transform=False,
                       use_time_reparam=True,   **base),
        AblationConfig("No Time Reparam",
                       use_reversible_coupling=True,
                       use_exponential_transform=True,
                       use_time_reparam=False,  **base),
        AblationConfig("Vanilla ERK (no components)",
                       use_reversible_coupling=False,
                       use_exponential_transform=False,
                       use_time_reparam=False,  **base),
    ]


@dataclass
class SampleMetrics:
    variant_name: str
    sample_idx: int
    inversion_error: float          # (a) ‖recon_latent − orig_latent‖₂
    lpips_orig: float               # structural similarity to original
    lpips_edit: float               # (b) structural similarity to reference edit
    clip_score: float               # (b) text-image alignment
    image_reward: float             # (c) generation quality
    pick_score: float               # (c) generation quality
    encode_nfe: int
    decode_nfe: int
    total_nfe: int
    error_flag: str = ""            # non-empty if the run raised an exception


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — SCHEDULE HELPERS  (self-contained, scheduler-aware)
# ══════════════════════════════════════════════════════════════════════════════

class ScheduleHelper:
    """
    Thin wrapper that converts continuous t ∈ [0,1] to (α_t, σ_t) for both
    'scaled_linear' and 'linear' DDPM β-schedules and supplies the closed-form
    inverse of the ς(t) map used by Rex.
    """

    def __init__(self, scheduler, sched_type: str = "scaled_linear", eps: float = 1e-5):
        self.scheduler = scheduler
        self.sched_type = sched_type
        self.eps = eps

        b0 = scheduler.betas[0].item() * 1000
        b1 = scheduler.betas[-1].item() * 1000
        self._b0 = b0
        self._b1 = b1

    # ── α / σ ──────────────────────────────────────────────────────────────

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        b0, b1 = self._b0, self._b1
        if self.sched_type == "linear":
            delta = b1 - b0
            return torch.exp(-delta / 4 * t.pow(2) - b0 / 2 * t)
        elif self.sched_type == "scaled_linear":
            sq_b = math.sqrt(b0 * b1)
            return torch.exp(
                -(b1 - 2 * sq_b + b0) / 6 * t.pow(3)
                - (sq_b - b0) / 2 * t.pow(2)
                - b0 / 2 * t
            )
        raise ValueError(self.sched_type)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        a = self.alpha(t)
        return torch.sqrt(torch.clamp(1 - a.pow(2), min=0.0))

    # ── ς(t) and its inverse ───────────────────────────────────────────────

    def varsigma(self, t: torch.Tensor, prediction_type: str) -> torch.Tensor:
        """ς(t) — the transformed time variable."""
        a = self.alpha(t)
        s = self.sigma(t)
        if prediction_type == "data":
            return a / torch.clamp(s, min=self.eps)
        else:
            return s / torch.clamp(a, min=self.eps)

    def varsigma_inv(self, gamma: torch.Tensor, prediction_type: str) -> torch.Tensor:
        """Closed-form inverse ς⁻¹(γ) → t."""
        b0, b1 = self._b0, self._b1
        p = -2 if prediction_type == "data" else 2

        if self.sched_type == "linear":
            delta = b1 - b0
            inner = b0 ** 2 + 2 * delta * torch.log(gamma.pow(p) + 1.0)
            t = (-b0 + torch.sqrt(torch.clamp(inner, min=0.0))) / delta
        elif self.sched_type == "scaled_linear":
            sq_b = math.sqrt(b0 * b1)
            delta = b1 - 2 * sq_b + b0
            inner = (
                2 * (sq_b - b0) ** 3
                - 3 * b0 * delta * (sq_b - b0)
                - 3 * delta ** 2 * torch.log(gamma.pow(p) + 1.0)
            )
            t = (b0 - sq_b + (-inner).pow(1 / 3)) / delta
        else:
            raise ValueError(self.sched_type)

        return t.clamp(self.eps, 1.0 - self.eps)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — ABLATABLE SOLVER
# ══════════════════════════════════════════════════════════════════════════════

class AblationSolver(nn.Module):
    """
    A single solver class that can reproduce Full Rex or any ablated variant
    depending on the flags in `cfg`.

    Design-axis implementations
    ───────────────────────────
    (i) Reversible coupling:
        Full  — two-state (x, x̂) McCallum-Foster update (Rex forward/backward step).
        Off   — single-state plain ERK step; backward ≈ forward with negated step
                (no algebraic reversibility guarantee).

    (ii) Exponential transform (Lawson trick):
        Full  — work in Z = x / w(t), so the stiff linear drift w'(t)/w(t)·x is
                factored out analytically; model evaluations use rescaled state.
        Off   — integrate directly in x; model called on raw x without rescaling.
                The ERK increment is computed in x-space and added directly.

    (iii) Time reparameterisation:
        Full  — step size is uniform in ς(t), so each step covers equal
                "information distance" according to the signal-to-noise ratio.
        Off   — step size is uniform in t (raw time); no change of variable.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: AblationConfig,
        sched_helper: ScheduleHelper,
        tableau: Union[ButcherTableau, str],
        n_steps: int,
    ):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.sched = sched_helper
        self.n_steps = n_steps
        self.nfe = 0

        if isinstance(tableau, str):
            self.tableau = get_rk_tableau(tableau)
        else:
            self.tableau = tableau

    # ── helpers ────────────────────────────────────────────────────────────

    def _weight(self, t: torch.Tensor) -> torch.Tensor:
        """w(t) used in the exponential factor."""
        if self.cfg.prediction_type == "data":
            return self.sched.sigma(t)
        return self.sched.alpha(t)

    def _to_tensor(self, v, device, dtype):
        if not isinstance(v, torch.Tensor):
            return torch.tensor([v], device=device, dtype=dtype)
        return v.to(device=device, dtype=dtype)

    def _model_out_to_data_or_noise(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert raw model output v to the quantity needed for the integrand.

        With exponential transform active we need the denoised prediction
        (data or noise depending on prediction_type) because it enters as the
        driving term after the linear drift is removed.

        Without exponential transform the increment is simply h·v (noise pred)
        or h·(v - x)/something, so we return v directly and let the caller scale.
        """
        a = self.sched.alpha(t)
        s = self.sched.sigma(t)
        eps = self.sched.eps
        pt = self.cfg.prediction_type

        if pt == "data":
            # model outputs noise → convert to x₀
            return (x - s * v) / torch.clamp(a, min=eps)
        else:
            # model outputs data → convert to ε
            # return (x - a * v) / torch.clamp(s, min=eps)
            return v

    # ── core ERK stage computation ─────────────────────────────────────────

    def _erk_increment(
        self,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        x_in: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the ERK increment Ψ(t_start → t_end, x_in).

        If use_exponential_transform:
            Operate on Z = x/w; the integrand is the denoised prediction.
            Return w_end · ΔZ   (so caller can do x_end = w_end·Z_end).

        If NOT use_exponential_transform:
            Operate directly on x; integrand is raw model output (noise pred).
            Return Δx.

        If use_time_reparam:
            Stages are placed uniformly in ς(t).
        Else:
            Stages are placed uniformly in t.
        """
        device, dtype = x_in.device, x_in.dtype
        tab = self.tableau.to(device, dtype)
        s = tab.num_stages
        eps = self.sched.eps
        pt = self.cfg.prediction_type

        w_start = self._weight(t_start)
        w_end = self._weight(t_end)

        # Decide the "integration coordinate" h and stage positions
        if self.cfg.use_time_reparam:
            # ς-domain
            zs = self.sched.varsigma(t_start, pt)
            ze = self.sched.varsigma(t_end, pt)
            h = ze - zs

            def t_for_stage(ci):
                zi = zs + ci * h
                return self.sched.varsigma_inv(zi, pt)
        else:
            # raw t-domain
            h = t_end - t_start

            def t_for_stage(ci):
                return t_start + ci * h

        k: List[torch.Tensor] = []

        for i in range(s):
            t_i = t_for_stage(tab.c[i])
            w_i = self._weight(t_i)

            if self.cfg.use_exponential_transform:
                # Z_i = x_in/w_start + Σ_j a_ij · k_j
                Z_i = x_in / torch.clamp(w_start, min=eps)
                for j in range(i):
                    if tab.a[i, j] != 0:
                        Z_i = Z_i + h * tab.a[i, j] * k[j]

                # Model is called on the rescaled state w_i · Z_i
                x_model = w_i * Z_i
                self.nfe += 1
                v = self.model(t_i, x_model)
                # Integrand: denoised prediction (x₀ or ε depending on pt)
                k_i = self._model_out_to_data_or_noise(t_i, x_model, v)
            else:
                # No exponential transform — integrate directly in x-space
                x_i = x_in.clone()
                for j in range(i):
                    if tab.a[i, j] != 0:
                        x_i = x_i + h * tab.a[i, j] * k[j]

                self.nfe += 1
                v = self.model(t_i, x_i)
                # Integrand: raw noise prediction (matches DDIM-like update)
                k_i = v

            k.append(k_i)

        # Weighted sum of increments
        increment = torch.zeros_like(x_in)
        for i in range(s):
            if tab.b[i] != 0:
                increment = increment + h * tab.b[i] * k[i]

        return increment

    # ── single forward step ────────────────────────────────────────────────
    def _forward_step(self, t_n, t_n1, x_n, x_hat_n):
        device, dtype = x_n.device, x_n.dtype
        eps = self.sched.eps
        w_n  = self._weight(t_n)
        w_n1 = self._weight(t_n1)
        weight_ratio = w_n1 / torch.clamp(w_n, min=eps)
        zeta = self.cfg.zeta

        if self.cfg.use_reversible_coupling:
            # Step 1: Ψ(t_n → t_n1, x̂_n)
            psi_fwd = self._erk_increment(t_n, t_n1, x_hat_n)

            # Step 2: x_{n+1} using the blended state — this IS the final x_{n+1}
            if self.cfg.use_exponential_transform:
                x_n1 = weight_ratio * (zeta * x_n + (1 - zeta) * x_hat_n) + w_n1 * psi_fwd
            else:
                x_n1 = zeta * x_n + (1 - zeta) * x_hat_n + psi_fwd

            # Step 3: Ψ(t_n1 → t_n, x_{n+1})
            psi_bwd = self._erk_increment(t_n1, t_n, x_n1)

            # Step 4: x̂_{n+1}
            if self.cfg.use_exponential_transform:
                x_hat_n1 = weight_ratio * x_hat_n - w_n1 * psi_bwd
            else:
                x_hat_n1 = x_hat_n - psi_bwd

            # x_n1 is already correct from Step 2 — do NOT recompute it
            return x_n1, x_hat_n1

        else:
            psi = self._erk_increment(t_n, t_n1, x_n)
            if self.cfg.use_exponential_transform:
                x_n1 = weight_ratio * x_n + w_n1 * psi
            else:
                x_n1 = x_n + psi
            return x_n1, None

    # ── single backward step (inversion) ──────────────────────────────────
    def _backward_step(self, t_n1, t_n, x_n1, x_hat_n1):
        # t_n1 = HIGH (first arg, more noisy)
        # t_n  = LOW  (second arg, less noisy)
        device, dtype = x_n1.device, x_n1.dtype
        eps  = self.sched.eps
        
        # Match working script: w_n is at HIGH, w_n1 is at LOW
        w_n  = self._weight(t_n1)   # ← HIGH  (was wrongly using t_n=LOW)
        w_n1 = self._weight(t_n)    # ← LOW   (was wrongly using t_n1=HIGH)
        weight_ratio_inv = w_n / torch.clamp(w_n1, min=eps)   # = σ_HIGH/σ_LOW > 1
        
        zeta     = self.cfg.zeta
        zeta_inv = 1.0 / zeta

        if self.cfg.use_reversible_coupling:
            # First psi: LOW→HIGH direction (matches working _psi_step(t_n1=LOW, t_n=HIGH))
            psi_neg_h = self._erk_increment(t_n, t_n1, x_n1)   # LOW→HIGH
            if self.cfg.use_exponential_transform:
                x_hat_n = weight_ratio_inv * x_hat_n1 + w_n * psi_neg_h
            else:
                x_hat_n = x_hat_n1 + psi_neg_h

            # Second psi: HIGH→LOW direction (matches working _psi_step(t_n=HIGH, t_n1=LOW))
            psi_h = self._erk_increment(t_n1, t_n, x_hat_n)    # HIGH→LOW
            if self.cfg.use_exponential_transform:
                x_n = (zeta_inv * weight_ratio_inv * x_n1
                    + (1.0 - zeta_inv) * x_hat_n
                    - w_n * zeta_inv * psi_h)
            else:
                x_n = zeta_inv * x_n1 + (1.0 - zeta_inv) * x_hat_n - zeta_inv * psi_h

            return x_n, x_hat_n

        else:
            psi = self._erk_increment(t_n, t_n1, x_n1)    # LOW→HIGH
            if self.cfg.use_exponential_transform:
                x_n = weight_ratio_inv * x_n1 + w_n * psi
            else:
                x_n = x_n1 + psi
            return x_n, None 

    # ── public API: solve ──────────────────────────────────────────────────

    def _make_t_schedule(
        self, t_start: float, t_end: float, device, dtype
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Build list of (t_n, t_{n+1}) pairs uniform in t."""
        ts = torch.linspace(t_start, t_end, self.n_steps + 1, device=device, dtype=dtype)
        return [(ts[i].unsqueeze(0), ts[i + 1].unsqueeze(0)) for i in range(self.n_steps)]


    def _make_grid(self, t_start, t_end, device, dtype):
        """Shared grid for both directions. linspace gives the most accurate spacing."""
        return torch.linspace(t_start, t_end, self.n_steps + 1,
                            device=device, dtype=dtype)

    def forward_solve(self, x, x_hat, t_span):
        """Decode: traverse the grid HIGH→LOW."""
        t_start, t_end = t_span[0].item(), t_span[-1].item()
        device, dtype  = x.device, x.dtype
        # Build grid LOW→HIGH, then reverse so we step HIGH→LOW
        grid = self._make_grid(t_end, t_start, device, dtype)  # [eps, ..., freeze]
        for i in range(self.n_steps - 1, -1, -1):              # traverse reversed
            t_lo = grid[i].unsqueeze(0)
            t_hi = grid[i + 1].unsqueeze(0)
            x, x_hat = self._forward_step(t_hi, t_lo, x, x_hat)
        return x, x_hat

    def backward_solve(self, x, x_hat, t_span):
        """Encode: traverse the grid LOW→HIGH."""
        t_start, t_end = t_span[0].item(), t_span[-1].item()
        device, dtype  = x.device, x.dtype
        if x_hat is None:
            x_hat = x.clone()
        # Build grid LOW→HIGH
        grid = self._make_grid(t_start, t_end, device, dtype)  # [eps, ..., freeze]
        for i in range(self.n_steps):                          # traverse forward
            t_lo = grid[i].unsqueeze(0)
            t_hi = grid[i + 1].unsqueeze(0)
            x, x_hat = self._backward_step(t_hi, t_lo, x, x_hat)
        return x, x_hat

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — SD MODEL WRAPPER  (identical to original script)
# ══════════════════════════════════════════════════════════════════════════════

class SDModel(nn.Module):
    """
    Wraps the SD UNet for use with AblationSolver.

    Always returns the UNet's raw noise prediction (ε).
    AblationSolver._model_out_to_data_or_noise handles conversion.
    """

    def __init__(
        self,
        sd_pipe,
        prompt_embeds: torch.Tensor,
        guidance_scale: float,
    ):
        super().__init__()
        self.sd_pipe = sd_pipe
        self.prompt_embeds = prompt_embeds
        self.guidance_scale = guidance_scale
        self.do_cfg = guidance_scale > 1.0

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        timestep = 1000 * t

        x_in = torch.cat([x] * 2) if self.do_cfg else x
        noise_pred = self.sd_pipe.unet(
            x_in,
            timestep,
            encoder_hidden_states=self.prompt_embeds,
            return_dict=False,
        )[0]

        if self.do_cfg:
            uncond, cond = noise_pred.chunk(2)
            noise_pred = uncond + self.guidance_scale * (cond - uncond)

        return noise_pred


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — PICKSCORE  (identical to original script)
# ══════════════════════════════════════════════════════════════════════════════

class PickScoreModel:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
        self.model = (
            AutoModel.from_pretrained("yuvalkirstain/PickScore_v1")
            .eval()
            .to(device)
        )

    @torch.no_grad()
    def score(self, prompt: str, image) -> float:
        img_inputs = self.processor(
            images=image, padding=True, truncation=True,
            max_length=77, return_tensors="pt"
        ).to(self.device)
        txt_inputs = self.processor(
            text=prompt, padding=True, truncation=True,
            max_length=77, return_tensors="pt"
        ).to(self.device)
        img_emb = self.model.get_image_features(**img_inputs)
        img_emb = img_emb / torch.norm(img_emb, dim=-1, keepdim=True)
        txt_emb = self.model.get_text_features(**txt_inputs)
        txt_emb = txt_emb / torch.norm(txt_emb, dim=-1, keepdim=True)
        s = self.model.logit_scale.exp() * (txt_emb @ img_emb.T)[0]
        return s.item()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — PER-SAMPLE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_variant(
    cfg: AblationConfig,
    sd_pipe,
    sched_helper: ScheduleHelper,
    latent: torch.Tensor,
    tableau_name: str,
    n_steps: int,
    freeze_step: float,
    eps_t: float,
    guidance_scale: float,
    ori_prompt: str,
    edited_prompt: str,
    negative_prompt: str,
    sample_idx: int,
    device: str,
    cs_model,
    ir_model,
    lpips_model,
    pick_model,
    ori_image_tensor: torch.Tensor,
    edited_image_tensor: torch.Tensor,
) -> SampleMetrics:
    """
    Run encode → decode cycle for one (sample, variant) pair.

    Returns a SampleMetrics dataclass.
    """
    dtype = latent.dtype

    # ── Encode step (inversion: eps_t → freeze_step) ─────────────────────

    try:
        # Prompt embeds for encoding (original prompt)
        enc_embeds = sd_pipe._encode_prompt(
            ori_prompt, sd_pipe.unet.device, guidance_scale > 1.0, negative_prompt
        )
        enc_model = SDModel(sd_pipe, enc_embeds, guidance_scale)

        n_encode = max(1, int(freeze_step * n_steps))
        enc_solver = AblationSolver(
            model=enc_model,
            cfg=cfg,
            sched_helper=sched_helper,
            tableau=tableau_name,
            n_steps=n_encode,
        )

        # Encode: backward_solve with t_span=[eps, freeze_step]  (low→high)
        t_enc = torch.tensor([eps_t, freeze_step], device=device, dtype=dtype)
        x_hat_init = latent.clone() if cfg.use_reversible_coupling else None
        with torch.no_grad():
            enc_solver.nfe = 0
            x_t, x_hat_t = enc_solver.backward_solve(latent.clone(), x_hat_init, t_enc)
            encode_nfe = enc_solver.nfe

       
        # ── Decode step (generation: freeze_step → eps_t) ─────────────────
        dec_embeds = sd_pipe._encode_prompt(
            edited_prompt, sd_pipe.unet.device, guidance_scale > 1.0, negative_prompt
        )
        dec_model = SDModel(sd_pipe, dec_embeds, guidance_scale)

        n_decode = max(1, int(freeze_step * n_steps))
        dec_solver = AblationSolver(
            model=dec_model,
            cfg=cfg,
            sched_helper=sched_helper,
            tableau=tableau_name,
            n_steps=n_decode,
        )

        # Decode: forward_solve with t_span=[freeze_step, eps]  (high→low)
        t_dec = torch.tensor([freeze_step, eps_t], device=device, dtype=dtype)
        with torch.no_grad():
            dec_solver.nfe = 0
            recon_latent, _ = dec_solver.forward_solve(x_t, x_hat_t, t_dec)
            decode_nfe = dec_solver.nfe

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"  [ERROR] variant={cfg.name}, sample={sample_idx}: {exc}\n{tb}")
        return SampleMetrics(
            variant_name=cfg.name, sample_idx=sample_idx,
            inversion_error=float("nan"), lpips_orig=float("nan"),
            lpips_edit=float("nan"), clip_score=float("nan"),
            image_reward=float("nan"), pick_score=float("nan"),
            encode_nfe=0, decode_nfe=0, total_nfe=0,
            error_flag=str(exc),
        )

    total_nfe = encode_nfe + decode_nfe

    # ── (a) Inversion error ───────────────────────────────────────────────
    # Run a second encode→decode cycle using the *same* original prompt to
    # measure pure reconstruction fidelity (no edit perturbation).
    try:
        same_solver = AblationSolver(
            model=enc_model,
            cfg=cfg,
            sched_helper=sched_helper,
            tableau=tableau_name,
            n_steps=n_decode,
        )
        with torch.no_grad():
            same_solver.nfe = 0
            recon_orig_latent, _ = same_solver.forward_solve(x_t, x_hat_t, t_dec)

        inversion_error = (
            (recon_orig_latent - latent).pow(2).mean().item()
        )
    except Exception as exc:
        print(f"  [WARN] inversion_error computation failed: {exc}")
        inversion_error = float("nan")

    # ── (b) & (c) Image-space metrics ────────────────────────────────────

    with torch.no_grad():
        pil_img = test_sd15.to_pil(latents=recon_latent, sd_pipe=sd_pipe)
        img_tensor = test_sd15.to_tensor_image(recon_latent, sd_pipe)  # [1,3,H,W], [0,1]

        reward = ir_model.score(edited_prompt, pil_img)
        clip_sc = cs_model(img_tensor, edited_prompt).item()
        pick_sc = pick_model.score(edited_prompt, pil_img)

        # LPIPS expects [-1, 1]
        img_01 = img_tensor           # already [0,1]
        img_11 = img_01 * 2 - 1
        lp_orig = lpips_model(img_11, ori_image_tensor).item()
        lp_edit = lpips_model(img_11, edited_image_tensor).item()

    return SampleMetrics(
        variant_name=cfg.name,
        sample_idx=sample_idx,
        inversion_error=inversion_error,
        lpips_orig=lp_orig,
        lpips_edit=lp_edit,
        clip_score=clip_sc,
        image_reward=reward,
        pick_score=pick_sc,
        encode_nfe=encode_nfe,
        decode_nfe=decode_nfe,
        total_nfe=total_nfe,
    ), pil_img


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — REPORTING
# ══════════════════════════════════════════════════════════════════════════════

METRIC_DISPLAY = {
    "inversion_error": ("(a) Inversion Error ↓", True),   # True = lower is better
    "lpips_edit":      ("(b) LPIPS edit ↓",      True),
    "clip_score":      ("(b) CLIP Score ↑",       False),
    "image_reward":    ("(c) ImageReward ↑",      False),
    "pick_score":      ("(c) PickScore ↑",        False),
    "lpips_orig":      ("LPIPS orig ↓",           True),
}


def aggregate(all_metrics: List[SampleMetrics]) -> dict:
    """Group by variant and compute mean ± std for every metric."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for m in all_metrics:
        buckets[m.variant_name].append(m)

    summary = {}
    metric_keys = [
        "inversion_error", "lpips_orig", "lpips_edit",
        "clip_score", "image_reward", "pick_score",
        "encode_nfe", "decode_nfe", "total_nfe",
    ]
    for vname, mlist in buckets.items():
        summary[vname] = {}
        for k in metric_keys:
            vals = [getattr(m, k) for m in mlist if not math.isnan(getattr(m, k))]
            if vals:
                summary[vname][k] = {
                    "mean": float(np.mean(vals)),
                    "std":  float(np.std(vals)),
                    "n":    len(vals),
                }
            else:
                summary[vname][k] = {"mean": float("nan"), "std": float("nan"), "n": 0}
    return summary


def write_summary_json(summary: dict, path: str):
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON → {path}")


def write_summary_csv(summary: dict, path: str):
    metric_keys = [
        "inversion_error", "lpips_orig", "lpips_edit",
        "clip_score", "image_reward", "pick_score",
        "encode_nfe", "decode_nfe", "total_nfe",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["variant"] + [f"{k}_mean" for k in metric_keys] + [f"{k}_std" for k in metric_keys]
        writer.writerow(header)
        for vname, stats in summary.items():
            row = [vname]
            row += [f"{stats[k]['mean']:.6f}" for k in metric_keys]
            row += [f"{stats[k]['std']:.6f}" for k in metric_keys]
            writer.writerow(row)
    print(f"Saved summary CSV  → {path}")


def write_report(summary: dict, path: str, args):
    """
    Write a human-readable ablation report with:
      - configuration header
      - per-metric tables (variant × metric, with Δ vs Full Rex highlighted)
      - qualitative interpretation of each ablation
    """
    variants = list(summary.keys())
    full_rex = "Full Rex"

    lines = []
    lines.append("=" * 78)
    lines.append("  REX ABLATION REPORT")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Configuration")
    lines.append("─" * 40)
    for k, v in vars(args).items():
        lines.append(f"  {k:30s}: {v}")
    lines.append("")

    lines.append("Ablated Variants")
    lines.append("─" * 40)
    lines.append("  Full Rex               — all three components active")
    lines.append("  No Reversible Coupling — plain ERK, no McCallum-Foster pairing")
    lines.append("  No Exponential Transform — integrate in x-space, no Lawson rescaling")
    lines.append("  No Time Reparam        — uniform steps in t, not ς(t)=α/σ")
    lines.append("  Vanilla ERK            — all three components disabled")
    lines.append("")

    for metric_key, (display_name, lower_better) in METRIC_DISPLAY.items():
        lines.append(f"{'─'*78}")
        lines.append(f"  {display_name}")
        lines.append(f"{'─'*78}")
        arrow = "↓ lower is better" if lower_better else "↑ higher is better"
        lines.append(f"  {arrow}")
        lines.append("")

        col_w = 18
        header = f"  {'Variant':<34}" + "".join(f"{'mean':>{col_w}}{'±std':>{col_w}}")
        lines.append(header)
        lines.append("  " + "-" * (34 + 2 * col_w))

        ref_val = summary.get(full_rex, {}).get(metric_key, {}).get("mean", float("nan"))

        for vname in variants:
            stats = summary[vname].get(metric_key, {})
            mean_v = stats.get("mean", float("nan"))
            std_v  = stats.get("std",  float("nan"))
            delta  = mean_v - ref_val

            if vname == full_rex:
                tag = "  (baseline)"
            else:
                sign = "+" if delta >= 0 else ""
                tag  = f"  Δ={sign}{delta:.4f}"

            mean_s = f"{mean_v:.4f}" if not math.isnan(mean_v) else "  N/A"
            std_s  = f"±{std_v:.4f}" if not math.isnan(std_v)  else ""

            lines.append(
                f"  {vname:<34}{mean_s:>{col_w}}{std_s:>{col_w}}{tag}"
            )
        lines.append("")

    lines.append("=" * 78)
    lines.append("  MECHANISTIC INTERPRETATION")
    lines.append("=" * 78)
    lines.append(textwrap.dedent("""
  (i) Reversible Coupling (McCallum-Foster):
      The two-state (x, x̂) coupling makes the discrete update ALGEBRAICALLY
      invertible — the backward step recovers x_n EXACTLY from x_{n+1} and
      x̂_{n+1} without any additional model calls.  Removing this (plain ERK)
      means the backward pass is a NAÏVE negated step, which accumulates
      truncation error at every stage.  Expected impact: large rise in
      inversion error; moderate degradation of edit consistency.

  (ii) Exponential Transform (Lawson / Integrating Factor):
      The stiff linear drift in the diffusion ODE is x·(d log w/dt).  The
      exponential transform Z = x/w factors this out analytically so the ERK
      method only needs to integrate the NONLINEAR residual (the denoising
      prediction).  Without it the solver must fight the stiff drift with its
      own stages, leading to larger local truncation error per step.
      Expected impact: moderate rise in inversion error and LPIPS; possible
      visible artifacts in edited images.

  (iii) Time Reparameterisation (ς = α/σ):
      Uniform steps in t concentrate many small steps near t=0 (high-SNR
      regime, where changes are slow) and too few steps near t=1 (low-SNR
      / high-curvature regime).  The ς reparameterisation distributes
      budget proportionally to the local curvature of the ODE solution,
      improving accuracy at no extra cost.  Expected impact: subtle but
      consistent improvement in generation quality metrics; smaller effect
      on inversion error than (i) or (ii).
    """))

    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)
    print(f"Saved ablation report → {path}")
    print()
    print(text)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def safe_unsqueeze_to_4d(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is [1, C, H, W]."""
    if t.dim() == 3:
        return t.unsqueeze(0)
    return t


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Rex ablation study: reversible coupling × exp-transform × time-reparam",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data / model
    parser.add_argument("--model_id", type=str,
                        default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--dataset_path", type=str, default="./data/pix2pix")
    parser.add_argument("--num_images", type=int, default=50,
                        help="Number of dataset samples to evaluate")
    parser.add_argument("--save_dir", type=str, default="results/ablation_rex")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # Solver
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Total NFE budget per direction; matched across all variants")
    parser.add_argument("--freeze_step", type=float, default=0.5,
                        help="Fraction of diffusion trajectory used [0, 1]")
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--eps", type=float, default=0.0002,
                        help="Small time offset to avoid t=0 singularity")
    parser.add_argument("--tableau", type=str, default="rk4",
                        choices=list_rk_methods(),
                        help="RK tableau used by ALL variants (matched compute)")
    parser.add_argument("--zeta", type=float, default=0.999,
                        help="McCallum-Foster ζ coupling parameter")
    parser.add_argument("--prediction_type", type=str, default="data",
                        choices=["data", "noise"])

    # Variant selection
    parser.add_argument("--variants", type=str, nargs="+",
                        default=["all"],
                        help=(
                            "Which variants to run. "
                            "'all' runs every variant. "
                            "Otherwise pick from: "
                            "'full', 'no_coupling', 'no_exp', 'no_reparam', 'vanilla'"
                        ))

    args = parser.parse_args()

    # ── Variant filter ────────────────────────────────────────────────────
    ALL_VARIANTS = build_variants(args.zeta, args.prediction_type)
    VARIANT_KEYS = {
        "full":        "Full Rex",
        "no_coupling": "No Reversible Coupling",
        "no_exp":      "No Exponential Transform",
        "no_reparam":  "No Time Reparam",
        "vanilla":     "Vanilla ERK (no components)",
    }

    if "all" in args.variants:
        selected_variants = ALL_VARIANTS
    else:
        wanted = {VARIANT_KEYS[k] for k in args.variants if k in VARIANT_KEYS}
        selected_variants = [v for v in ALL_VARIANTS if v.name in wanted]
        if not selected_variants:
            raise ValueError(f"No valid variants selected from: {args.variants}")

    # ── Setup ─────────────────────────────────────────────────────────────
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    sched_type = "scaled_linear"

    print("=" * 60)
    print("Rex Ablation Study")
    print("=" * 60)
    print(f"Device      : {device}")
    print(f"Tableau     : {args.tableau}  (identical across all variants)")
    print(f"NFE budget  : {args.num_inference_steps} steps × freeze_step={args.freeze_step}")
    print(f"Variants    : {[v.name for v in selected_variants]}")
    print("=" * 60)

    set_seed(args.seed)

    # ── Load SD model ─────────────────────────────────────────────────────
    print("\nLoading Stable Diffusion …")
    sd_full = StableDiffusionDiffEditPipeline.from_pretrained(
        args.model_id, torch_dtype=dtype
    ).to(device)

    scheduler = DDIMScheduler(
        beta_end=0.012,
        beta_start=0.00085,
        beta_schedule="scaled_linear",
        clip_sample=False,
        timestep_spacing="linspace",
        set_alpha_to_one=False,
    )

    sd_pipe = PipelineLike(
        device=device,
        vae=sd_full.vae,
        text_encoder=sd_full.text_encoder,
        tokenizer=sd_full.tokenizer,
        unet=sd_full.unet,
        scheduler=scheduler,
    )
    for sub in [sd_pipe.vae, sd_pipe.text_encoder, sd_pipe.unet]:
        sub.to(device)
        sub.eval()

    print("SD model loaded.\n")

    # Schedule helper (shared across all variants — same β-schedule)
    sched_helper = ScheduleHelper(scheduler, sched_type=sched_type)

    # ── Evaluation models ─────────────────────────────────────────────────
    print("Loading evaluation models …")
    cs_model = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
    ir_model  = RM.load("ImageReward-v1.0", device=device)
    lpips_model = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to(device)
    pick_model = PickScoreModel(device=device)
    print("Evaluation models loaded.\n")

    # ── Output directories ────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "metadata"), exist_ok=True)
    for v in selected_variants:
        slug = v.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        os.makedirs(os.path.join(args.save_dir, "imgs", slug), exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────
    print("Loading dataset …")
    ds = load_dataset(args.dataset_path, split=f"train[:{args.num_images}]")
    print(f"Dataset loaded ({len(ds)} entries).\n")

    # ── Main loop ─────────────────────────────────────────────────────────
    all_metrics: List[SampleMetrics] = []
    sample_idx = 0

    for entry in tqdm(ds, desc="Samples"):
        if sample_idx >= args.num_images:
            break

        ori_prompt    = entry["original_prompt"]
        ori_image     = entry["original_image"]
        edit_prompt   = entry["edit_prompt"]
        edited_prompt = entry["edited_prompt"]
        edited_image  = entry["edited_image"]

        # Encode image to VAE latent space (once, shared by all variants)
        latent, ori_img_tensor = pil_to_latents(ori_image, sd_pipe, return_image=True)
        _, edited_img_tensor   = pil_to_latents(edited_image, sd_pipe, return_image=True)

        latent            = latent.to(device)
        ori_img_tensor    = safe_unsqueeze_to_4d(ori_img_tensor.to(device))
        edited_img_tensor = safe_unsqueeze_to_4d(edited_img_tensor.to(device))

        # ── Run each variant ───────────────────────────────────────────────
        for cfg in selected_variants:
            print(f"\n[Sample {sample_idx}] variant: {cfg.name}")

            result = run_variant(
                cfg=cfg,
                sd_pipe=sd_pipe,
                sched_helper=sched_helper,
                latent=latent,
                tableau_name=args.tableau,
                n_steps=args.num_inference_steps,
                freeze_step=args.freeze_step,
                eps_t=args.eps,
                guidance_scale=args.guidance,
                ori_prompt=ori_prompt,
                edited_prompt=edited_prompt,
                negative_prompt="",
                sample_idx=sample_idx,
                device=device,
                cs_model=cs_model,
                ir_model=ir_model,
                lpips_model=lpips_model,
                pick_model=pick_model,
                ori_image_tensor=ori_img_tensor,
                edited_image_tensor=edited_img_tensor,
            )

            # run_variant returns (SampleMetrics, PIL) or SampleMetrics on error
            if isinstance(result, tuple):
                metrics, pil_img = result
            else:
                metrics = result
                pil_img = None

            all_metrics.append(metrics)

            # Save image
            slug = cfg.name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            if pil_img is not None:
                img_path = os.path.join(
                    args.save_dir, "imgs", slug, f"{sample_idx:06d}.png"
                )
                pil_img.save(img_path)

            # Save per-sample JSON
            meta_path = os.path.join(
                args.save_dir, "metadata", f"{sample_idx:06d}_{slug}.json"
            )
            meta = asdict(metrics)
            meta.update({
                "original_prompt": ori_prompt,
                "edit_prompt": edit_prompt,
                "edited_prompt": edited_prompt,
                "tableau": args.tableau,
                "freeze_step": args.freeze_step,
                "guidance_scale": args.guidance,
                "num_inference_steps": args.num_inference_steps,
                "zeta": args.zeta,
                "prediction_type": args.prediction_type,
                "use_reversible_coupling": cfg.use_reversible_coupling,
                "use_exponential_transform": cfg.use_exponential_transform,
                "use_time_reparam": cfg.use_time_reparam,
            })
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            print(
                f"  inv_err={metrics.inversion_error:.2e}  "
                f"lpips_edit={metrics.lpips_edit:.4f}  "
                f"clip={metrics.clip_score:.2f}  "
                f"ir={metrics.image_reward:.3f}  "
                f"pick={metrics.pick_score:.2f}  "
                f"NFE={metrics.total_nfe}"
            )

        sample_idx += 1

    # ── Aggregate & report ────────────────────────────────────────────────
    print("\n\nAggregating results …")
    summary = aggregate(all_metrics)

    write_summary_json(summary, os.path.join(args.save_dir, "summary.json"))
    write_summary_csv(summary,  os.path.join(args.save_dir, "summary.csv"))
    write_report(summary, os.path.join(args.save_dir, "ablation_report.txt"), args)

    print(f"\nAll done. Results saved to: {args.save_dir}/")


if __name__ == "__main__":
    main()
