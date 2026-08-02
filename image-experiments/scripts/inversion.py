"""
Reconstruction experiment using the Rex (Reversible Exponential) solver.

For each sample:
  1. Encode image -> VAE latent (x0)
  2. Rex backward pass: x0 (t=eps) -> xT (t=freeze_step)   [encode to noise]
  3. Rex forward  pass: xT (t=freeze_step) -> x0_recon (t=eps) [decode back]
  4. Measure MSE between x0 and x0_recon  (latent-space error)
  5. Decode both x0 and x0_recon through the VAE decoder and
     measure MSE in pixel/image space

Usage:
    python scripts/inversion.py \
        --n_samples 50 \
        --num_inference_steps 50 \
        --freeze_step 1.0 \
        --tableau euler \
        --zeta 0.999 \
        --prediction_type data \
        --save_dir results/reconstruction/rex_euler
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusers import DDIMScheduler, StableDiffusionDiffEditPipeline
from datasets import load_dataset
from tqdm import tqdm

from samplers import test_sd15, BELM, BDIA, edict, DDIM
from samplers.test_sd15 import pil_to_latents
from samplers.utils import PipelineLike
from samplers.rk_tableaus import list_rk_methods
from samplers.rex import RexTorchdynWrapper, create_rex_solver

import torch.nn as nn


# ============================================================
#  SD model wrapper
# ============================================================

class SDRexModel:
    """Adapts the SD UNet for RexTorchdynWrapper (handles CFG)."""

    def __init__(self, sd_pipe, prompt_embeds, guidance_scale=7.5,
                 scheduler=None, sched_type="scaled_linear"):
        self.sd_pipe = sd_pipe
        self.prompt_embeds = prompt_embeds
        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = guidance_scale > 1.0
        self.scheduler = scheduler
        self.sched_type = sched_type
        self.eps = 1e-5

    def __call__(self, t, x):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        timestep = 1000 * t
        x_input = torch.cat([x] * 2) if self.do_classifier_free_guidance else x
        noise_pred = self.sd_pipe.unet(
            x_input, timestep,
            encoder_hidden_states=self.prompt_embeds,
            return_dict=False,
        )[0]
        if self.do_classifier_free_guidance:
            uncond, cond = noise_pred.chunk(2)
            noise_pred = uncond + self.guidance_scale * (cond - uncond)
        return noise_pred



# ============================================================
#  Encode / Decode helpers (adapted for reconstruction)
# ============================================================

def rex_encode(
    sd_pipe, prompt_embeds, latent,
    tableau, n_steps, prediction_type, adaptive, zeta,
    eps, freeze_step, sched_type,
):
    """
    Invert latent x0 at t=eps -> intermediate xT at t=freeze_step.
    Returns (x_t, x_hat_t, nfe).
    """
    model = SDRexModel(
        sd_pipe, prompt_embeds,
        guidance_scale=1.0,          # guidance=1 for reconstruction
        scheduler=sd_pipe.scheduler,
        sched_type=sched_type,
    )
    n_encode_steps = max(1, int(freeze_step * n_steps))
    solver = create_rex_solver(
        model, tableau=tableau, n_steps=n_encode_steps,
        prediction_type=prediction_type, adaptive=adaptive,
        zeta=zeta, eps=eps, scheduler=sd_pipe.scheduler, sched_type=sched_type,
    )
    device = latent.device
    t_span = torch.tensor([eps, freeze_step], device=device)
    x, x_hat = latent.clone(), latent.clone()
    solver.nfe = 0
    with torch.no_grad():
        x_t, x_hat_t = solver.backward_solve(x, x_hat, t_span)
        nfe = solver.nfe
    return x_t, x_hat_t, nfe


def rex_decode(
    sd_pipe, prompt_embeds, x_t, x_hat_t,
    tableau, n_steps, prediction_type, adaptive, zeta,
    eps, freeze_step, sched_type,
):
    """
    Reconstruct latent from intermediate xT at t=freeze_step -> x0_recon at t=eps.
    Returns (x0_recon, nfe).
    """
    model = SDRexModel(
        sd_pipe, prompt_embeds,
        guidance_scale=1.0,
        scheduler=sd_pipe.scheduler,
        sched_type=sched_type,
    )
    n_decode_steps = max(1, int(freeze_step * n_steps))
    solver = create_rex_solver(
        model, tableau=tableau, n_steps=n_decode_steps,
        prediction_type=prediction_type, adaptive=adaptive,
        zeta=zeta, eps=eps, scheduler=sd_pipe.scheduler, sched_type=sched_type,
    )
    device = x_t.device
    t_span = torch.tensor([freeze_step, eps], device=device)
    solver.nfe = 0
    with torch.no_grad():
        x0_recon, _ = solver.forward_solve(x_t, x_hat_t, t_span)
        nfe = solver.nfe
    return x0_recon, nfe


# ============================================================
#  Utilities
# ============================================================

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def latent_to_image_tensor(latent, sd_pipe):
    """
    Decode a VAE latent to a float32 image tensor in [0, 1].
    Shape: (1, C, H, W)
    """
    latent_scaled = latent / sd_pipe.vae.config.scaling_factor
    with torch.no_grad():
        image = sd_pipe.vae.decode(latent_scaled).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image


def mse(a, b):
    """Element-wise MSE averaged over all dimensions."""
    return F.mse_loss(a, b).item()


# ============================================================
#  Main reconstruction loop
# ============================================================

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Reconstruction experiment with the Rex solver."
    )

    # Experiment
    parser.add_argument("--n_samples",          type=int,   default=50,
                        help="Number of images to reconstruct")
    parser.add_argument("--num_inference_steps", type=int,   default=50)
    parser.add_argument("--freeze_step",         type=float, default=1.0,
                        help="Encode all the way to t=freeze_step (1.0 = full noise)")
    parser.add_argument("--eps",                 type=float, default=0.0002)
    parser.add_argument("--seed",                type=int,   default=42)

    # Rex
    parser.add_argument("--tableau",          type=str,   default="euler",
                        choices=list_rk_methods())
    parser.add_argument("--zeta",             type=float, default=0.999)
    parser.add_argument("--prediction_type",  type=str,   default="data",
                        choices=["data", "noise"])
    parser.add_argument("--adaptive",         action="store_true")
    parser.add_argument("--step_domain",      type=str,   default="t",
                        choices=["t", "varsigma"])
    parser.add_argument("--atol",             type=float, default=1e-5)
    parser.add_argument("--rtol",             type=float, default=1e-5)

    # Model / data
    parser.add_argument("--model_id", type=str,
                        default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--save_dir", type=str,
                        default="results/reconstruction/rex")
    parser.add_argument("--device",   type=int, default=0)
    parser.add_argument("--dtype", choices=["fp16", "fp32", "bf16"], default="fp32")

    # Comparison with other methods
    parser.add_argument('--sampler_type', type=str, default='rex',
                        choices=['rex', 'ddim', 'bdia', 'edict', 'belm'],
                        help='Sampler to use (rex uses new wrapper)')
    parser.add_argument('--bdia_gamma', type=float, default=0.96)
    parser.add_argument('--edict_p', type=float, default=0.93)

    args = parser.parse_args()

    device     = f"cuda:{args.device}"
    dtype      = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }[args.dtype]
    sched_type = "scaled_linear"

    print(f"=== Rex Reconstruction Experiment ===")
    print(f"tableau={args.tableau}  zeta={args.zeta}  "
          f"prediction_type={args.prediction_type}  "
          f"freeze_step={args.freeze_step}  steps={args.num_inference_steps}")
    print(f"n_samples={args.n_samples}  device={device}  dtype={args.dtype}")

    # ----------------------------------------------------------
    # Load model
    # ----------------------------------------------------------
    sd = StableDiffusionDiffEditPipeline.from_pretrained(
        args.model_id, torch_dtype=dtype
    ).to(device)

    scheduler = DDIMScheduler(
        beta_end=0.012, beta_start=0.00085,
        beta_schedule="scaled_linear",
        clip_sample=False,
        timestep_spacing="linspace",
        set_alpha_to_one=False,
    )

    sd_pipe = PipelineLike(
        device=device,
        vae=sd.vae,
        text_encoder=sd.text_encoder,
        tokenizer=sd.tokenizer,
        unet=sd.unet,
        scheduler=scheduler,
    )
    sd_pipe.vae.to(device)
    sd_pipe.text_encoder.to(device)
    sd_pipe.unet.to(device)
    print("Model loaded.")

    set_seed(args.seed)

    # ----------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------
    ds = load_dataset("./data/pix2pix", split=f"train[:{args.n_samples}]")

    # ----------------------------------------------------------
    # Output directories
    # ----------------------------------------------------------
    os.makedirs(f"{args.save_dir}/imgs_original",    exist_ok=True)
    os.makedirs(f"{args.save_dir}/imgs_reconstructed", exist_ok=True)
    os.makedirs(f"{args.save_dir}/metadata",          exist_ok=True)

    # ----------------------------------------------------------
    # Per-sample metrics accumulators
    # ----------------------------------------------------------
    all_mse_latent = []   # MSE in latent space:  x0  vs  x0_recon
    all_mse_pixel  = []   # MSE in pixel  space:  img vs  img_recon
    all_encode_nfe = []
    all_decode_nfe = []

    # ----------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------
    for idx, entry in enumerate(tqdm(ds, total=args.n_samples)):
        if idx >= args.n_samples:
            break

        ori_prompt = entry["original_prompt"]
        ori_image  = entry["original_image"]

        # ---- 1. Encode image -> VAE latent (x0) ----
        latent, ori_image_tensor = pil_to_latents(
            ori_image, sd_pipe, return_image=True
        )
        latent = latent.to(device)                         # (1, 4, 64, 64)
        ori_image_tensor = ori_image_tensor.to(device)     # (C, H, W)  in [0,1]

        # Keep a clean copy of the original latent for MSE
        x0_original = latent.clone()

        # ---- 2. Encode prompt once (same for both passes) ----
        #         guidance_scale=1 -> no CFG needed
        prompt_embeds = sd_pipe._encode_prompt(
            ori_prompt, sd_pipe.unet.device,
            do_classifier_free_guidance=False,
            negative_prompt="",
        )

        enc_nfe = dec_nfe = 0
        num_inference_steps = args.num_inference_steps

        sd_params = {
            'prompt': ori_prompt,
            'negative_prompt': '',
            'seed': args.seed,
            'guidance_scale': 1.0,
            'num_inference_steps': num_inference_steps,
            'width': 512,
            'height': 512
        }

        if args.sampler_type == 'rex':
            x_t, x_hat_t, enc_nfe = rex_encode(
                sd_pipe        = sd_pipe,
                prompt_embeds  = prompt_embeds,
                latent         = latent,
                tableau        = args.tableau,
                n_steps        = args.num_inference_steps,
                prediction_type= args.prediction_type,
                adaptive       = args.adaptive,
                zeta           = args.zeta,
                eps            = args.eps,
                freeze_step    = args.freeze_step,
                sched_type     = sched_type,
            )

            # ---- 4. Rex forward pass: xT -> x0_recon ----
            x0_recon, dec_nfe = rex_decode(
                sd_pipe        = sd_pipe,
                prompt_embeds  = prompt_embeds,
                x_t            = x_t,
                x_hat_t        = x_hat_t,
                tableau        = args.tableau,
                n_steps        = args.num_inference_steps,
                prediction_type= args.prediction_type,
                adaptive       = args.adaptive,
                zeta           = args.zeta,
                eps            = args.eps,
                freeze_step    = args.freeze_step,
                sched_type     = sched_type,
            )

        elif args.sampler_type == 'ddim':
            freeze_step_idx = int((1. - args.freeze_step) * num_inference_steps)
            intermediate = DDIM.latent_to_intermediate(
                sd_pipe=sd_pipe, sd_params=sd_params,
                latent=latent, freeze_step=freeze_step_idx
            )

            x0_recon = DDIM.intermediate_to_latent(
                sd_pipe=sd_pipe, sd_params=sd_params,
                intermediate=intermediate, freeze_step=freeze_step_idx
            )
            
        elif args.sampler_type == 'edict':
            freeze_step_idx = int((1. - args.freeze_step) * num_inference_steps)
            x_intermediate, y_intermediate = edict.latent_to_intermediate(
                sd_pipe=sd_pipe, sd_params=sd_params,
                latent=latent, freeze_step=freeze_step_idx, p=args.edict_p
            )
            x0_recon, _ = edict.intermediate_to_latent(
                sd_pipe=sd_pipe, sd_params=sd_params,
                x_intermediate=x_intermediate, y_intermediate=y_intermediate,
                p=args.edict_p, freeze_step=freeze_step_idx
            ) 
            
        elif args.sampler_type == 'bdia':
            freeze_step_idx = int((1. - args.freeze_step) * num_inference_steps)
            intermediate, second_intermediate = BDIA.latent_to_intermediate(
                sd_pipe=sd_pipe, sd_params=sd_params,
                latent=latent, gamma=args.bdia_gamma, freeze_step=freeze_step_idx
            )
            x0_recon = BDIA.intermediate_to_latent(
                sd_pipe=sd_pipe, sd_params=sd_params,
                intermediate=intermediate, intermediate_second=second_intermediate,
                gamma=args.bdia_gamma, freeze_step=freeze_step_idx
            )
            
        elif args.sampler_type == 'belm':
            freeze_step_idx = int((1. - args.freeze_step) * num_inference_steps)
            intermediate, second_intermediate = BELM.latent_to_intermediate(
                sd_pipe=sd_pipe, sd_params=sd_params,
                latent=latent, freeze_step=freeze_step_idx
            )
            x0_recon = BELM.intermediate_to_latent(
                sd_pipe=sd_pipe, sd_params=sd_params,
                intermediate=intermediate, intermediate_second=second_intermediate,
                freeze_step=freeze_step_idx
            )

        # ---- 5a. MSE in latent space ----
        mse_latent = mse(x0_recon, x0_original)

        # ---- 5b. MSE in pixel space ----
        img_recon    = latent_to_image_tensor(x0_recon,    sd_pipe)
        img_original = (ori_image_tensor/2) + 0.5
        mse_pixel    = mse(img_recon, img_original)

        # ---- 6. Accumulate ----
        all_mse_latent.append(mse_latent)
        all_mse_pixel.append(mse_pixel)
        all_encode_nfe.append(enc_nfe)
        all_decode_nfe.append(dec_nfe)

        # ---- 7. Save images ----
        # Original
        from PIL import Image as PILImage
        img_pil_orig  = PILImage.fromarray(
            (img_original.permute(1,2,0).float().cpu().numpy() * 255).clip(0,255).astype(np.uint8)
        )
        img_pil_recon = PILImage.fromarray(
            (img_recon[0].permute(1,2,0).float().cpu().numpy() * 255).clip(0,255).astype(np.uint8)
        )
        img_pil_orig.save(f"{args.save_dir}/imgs_original/{idx:06d}.png")
        img_pil_recon.save(f"{args.save_dir}/imgs_reconstructed/{idx:06d}.png")

        # ---- 8. Save per-sample metadata ----
        save_dict = {
            "idx":             idx,
            "prompt":          ori_prompt,
            "sampler":         "rex",
            "tableau":         args.tableau,
            "zeta":            args.zeta,
            "prediction_type": args.prediction_type,
            "freeze_step":     args.freeze_step,
            "num_steps":       args.num_inference_steps,
            "enc_nfe":         enc_nfe,
            "dec_nfe":         dec_nfe,
            "total_nfe":       enc_nfe + dec_nfe,
            "mse_latent":      mse_latent,
            "mse_pixel":       mse_pixel,
        }
        with open(f"{args.save_dir}/metadata/{idx:06d}.json", "w") as f:
            json.dump(save_dict, f, indent=2)

        print(
            f"[{idx+1:>3}/{args.n_samples}]  "
            f"MSE_latent={mse_latent:.6f}  "
            f"MSE_pixel={mse_pixel:.6f}  "
            f"NFE={enc_nfe+dec_nfe}"
        )

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    mean_mse_latent = float(np.mean(all_mse_latent))
    mean_mse_pixel  = float(np.mean(all_mse_pixel))
    mean_enc_nfe    = float(np.mean(all_encode_nfe))
    mean_dec_nfe    = float(np.mean(all_decode_nfe))

    summary = {
        "n_samples":        args.n_samples,
        "tableau":          args.tableau,
        "zeta":             args.zeta,
        "prediction_type":  args.prediction_type,
        "freeze_step":      args.freeze_step,
        "num_steps":        args.num_inference_steps,
        # ---- KEY RESULTS ----
        "mean_mse_latent":  mean_mse_latent,
        "mean_mse_pixel":   mean_mse_pixel,
        "mean_enc_nfe":     mean_enc_nfe,
        "mean_dec_nfe":     mean_dec_nfe,
        "mean_total_nfe":   mean_enc_nfe + mean_dec_nfe,
        # raw lists for further analysis
        "per_sample_mse_latent": all_mse_latent,
        "per_sample_mse_pixel":  all_mse_pixel,
    }

    summary_path = f"{args.save_dir}/reconstruction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("="*50)
    print(" RECONSTRUCTION EXPERIMENT SUMMARY")
    print("="*50)
    print(f"  Solver type      : {args.sampler_type}")
    print(f"  Samples          : {args.n_samples}")
    print(f"  Tableau          : {args.tableau}")
    print(f"  Zeta             : {args.zeta}")
    print(f"  Prediction type  : {args.prediction_type}")
    print(f"  Freeze step      : {args.freeze_step}")
    print(f"  Inference steps  : {args.num_inference_steps}")
    print("-"*50)
    print(f"  Mean MSE (latent): {mean_mse_latent:.2e}")
    print(f"  Mean MSE (pixel) : {mean_mse_pixel:.2e}")
    print(f"  Mean total NFE   : {mean_enc_nfe + mean_dec_nfe:.1f}")
    print("="*50)
    print(f"Results saved to: {args.save_dir}")
    print(f"Summary JSON:      {summary_path}")


if __name__ == "__main__":
    main()
