"""
Image editing script using the refreshed Rex (Reversible Exponential) solver.

This script implements image editing for diffusion models using the new Rex wrapper
from rex_wrapper.py, which combines:
1. Exponential RK methods (to handle the linear drift in diffusion ODEs)
2. McCallum-Foster reversible coupling (for algebraic reversibility)

The Rex solver enables exact forward/backward inversion, making it ideal for 
image editing tasks where we need to:
1. Encode an image to a latent representation (backward pass)
2. Modify the conditioning (prompt)
3. Decode back to an edited image (forward pass)

Usage:
    python scripts/image_editing_rex.py \
        --num_inference_steps 50 \
        --freeze_step 0.5 \
        --guidance 2.0 \
        --tableau rk4 \
        --zeta 0.5 \
        --prediction_type data \
        --save_dir results/image_edits/rex_rk4
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

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

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal.clip_score import CLIPScore
import ImageReward as RM
from transformers import AutoProcessor, AutoModel


class PickScoreModel:
    """
    PickScore model for evaluating image-text alignment.
    
    Based on the PickScore model from https://github.com/yuvalkirstain/PickScore
    Uses CLIP features with a trained scoring head.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        processor_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_name = "yuvalkirstain/PickScore_v1"
        
        self.processor = AutoProcessor.from_pretrained(processor_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(device)
    
    @torch.no_grad()
    def score(self, prompt: str, image) -> float:
        """
        Compute PickScore for an image given a text prompt.
        
        Args:
            prompt: Text description/prompt
            image: PIL Image or tensor image
            
        Returns:
            PickScore value (higher = better alignment)
        """
        # Process image
        image_inputs = self.processor(
            images=image,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        
        # Process text
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        
        # Get embeddings
        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        
        # Compute score
        score = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
        return score.item()


class SDRexModel:
    """
    Wrapper to adapt Stable Diffusion UNet for use with RexTorchdynWrapper.
    
    Converts the UNet's noise prediction to data prediction when needed,
    and handles classifier-free guidance.
    """
    
    def __init__(
        self,
        sd_pipe,
        prompt_embeds: torch.Tensor,
        guidance_scale: float = 7.5,
        scheduler=None,
        sched_type: str = 'scaled_linear',
    ):
        self.sd_pipe = sd_pipe
        self.prompt_embeds = prompt_embeds
        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = guidance_scale > 1.0
        self.scheduler = scheduler
        self.sched_type = sched_type
        self.eps = 1e-5
        
    def _t_to_sigma_alpha(self, t: torch.Tensor):
        """Convert continuous time t to (alpha, sigma) using the scheduler's beta schedule."""
        beta_0 = self.scheduler.betas[0] * 1000
        beta_1 = self.scheduler.betas[-1] * 1000
        
        if self.sched_type == 'linear':
            delta = beta_1 - beta_0
            alpha_t = torch.exp(-delta/4 * t.pow(2) - beta_0/2 * t)
            sigma_t = torch.sqrt(1 - alpha_t.pow(2))
        elif self.sched_type == 'scaled_linear':
            alpha_t = torch.exp(
                -(beta_1 - 2 * torch.sqrt(beta_0 * beta_1) + beta_0) / 6 * t.pow(3)
                - (torch.sqrt(beta_0 * beta_1) - beta_0) / 2 * t.pow(2)
                - beta_0/2 * t
            )
            sigma_t = torch.sqrt(1 - alpha_t.pow(2))
        else:
            raise ValueError(f"Unknown schedule type: {self.sched_type}")
            
        return alpha_t, sigma_t
    
    def __call__(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the UNet with guidance.
        
        Args:
            t: Time in [0, 1] (will be scaled to [0, 1000] for UNet)
            x: Latent state
            
        Returns:
            Model output (noise prediction)
        """
        # Handle scalar t
        if t.dim() == 0:
            t = t.unsqueeze(0)
            
        # Scale time to UNet's expected range [0, 1000]
        timestep = 1000 * t
        
        # Prepare input for classifier-free guidance
        if self.do_classifier_free_guidance:
            x_input = torch.cat([x] * 2)
        else:
            x_input = x
            
        # Get noise prediction from UNet
        noise_pred = self.sd_pipe.unet(
            x_input,
            timestep,
            encoder_hidden_states=self.prompt_embeds,
            return_dict=False,
        )[0]
        
        # Apply classifier-free guidance
        if self.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
        return noise_pred



def sd_rex_encode(
    sd_pipe,
    sd_params: dict,
    latent: torch.Tensor,
    tableau: str = "rk4",
    n_steps: int = 50,
    prediction_type: str = "data",
    adaptive: bool = False,
    zeta: float = 0.5,
    eps: float = 0.0002,
    freeze_step: float = 0.5,
    sched_type: str = 'scaled_linear',
) -> tuple:
    """
    Encode an image latent to intermediate representation using Rex backward pass.
    
    This inverts the diffusion process to find the latent at time `freeze_step`.
    
    Args:
        sd_pipe: Stable Diffusion pipeline
        sd_params: Dictionary with prompt, guidance_scale, etc.
        latent: Image encoded to VAE latent space
        tableau: RK method name
        n_steps: Number of integration steps
        prediction_type: "data" or "noise"
        zeta: McCallum-Foster coupling parameter
        eps: Small time to avoid singularity at t=0
        freeze_step: Time to stop encoding (higher = more noise)
        sched_type: Schedule type
        
    Returns:
        (x_t, x_hat_t): Encoded states at time freeze_step
    """
    prompt = sd_params['prompt']
    negative_prompt = sd_params['negative_prompt']
    guidance_scale = sd_params['guidance_scale']
    
    # Encode prompts
    prompt_embeds = sd_pipe._encode_prompt(
        prompt,
        sd_pipe.unet.device,
        guidance_scale > 1.0,
        negative_prompt
    )
    
    # Create model wrapper
    model = SDRexModel(
        sd_pipe,
        prompt_embeds,
        guidance_scale,
        sd_pipe.scheduler,
        sched_type,
    )
    
    # Create Rex solver
    n_encode_steps = int(freeze_step * n_steps)
    solver = create_rex_solver(
        model,
        tableau=tableau,
        n_steps=n_encode_steps,
        prediction_type=prediction_type,
        adaptive=adaptive,
        zeta=zeta,
        scheduler=sd_pipe.scheduler,
        sched_type=sched_type,
    )
    
    # Time span: from eps to freeze_step (backward = inversion)
    # In Rex, backward_solve goes from t_end to t_start
    device = latent.device
    t_span = torch.tensor([eps, freeze_step], device=device)
    
    # Initialize both states with the latent
    x = latent.clone()
    x_hat = latent.clone()
    
    with torch.no_grad():
        # Reset NFE counter before solve
        solver.nfe = 0
        # Use backward_solve to invert (encode image -> noise direction)
        x_t, x_hat_t = solver.backward_solve(x, x_hat, t_span)
        encode_nfe = solver.nfe
    
    print(f"Encoded: t={eps} -> t={freeze_step}, var(x_t)={torch.var(x_t).item():.4f}, NFE={encode_nfe}")
    
    return x_t, x_hat_t, encode_nfe


def sd_rex_decode(
    sd_pipe,
    sd_params: dict,
    x_t: torch.Tensor,
    x_hat_t: torch.Tensor,
    tableau: str = "rk4",
    n_steps: int = 50,
    prediction_type: str = "data",
    adaptive: bool = False,
    zeta: float = 0.5,
    eps: float = 0.0002,
    freeze_step: float = 0.5,
    sched_type: str = 'scaled_linear',
) -> torch.Tensor:
    """
    Decode intermediate representation back to image using Rex forward pass.
    
    This runs the diffusion process forward from time `freeze_step` to `eps`.
    
    Args:
        sd_pipe: Stable Diffusion pipeline
        sd_params: Dictionary with prompt (can be different from encoding)
        x_t: Encoded state at time freeze_step
        x_hat_t: Auxiliary encoded state
        tableau: RK method name
        n_steps: Number of integration steps
        prediction_type: "data" or "noise"
        zeta: McCallum-Foster coupling parameter
        eps: Small time to avoid singularity at t=0
        freeze_step: Time to start decoding
        sched_type: Schedule type
        
    Returns:
        Decoded image latent
    """
    prompt = sd_params['prompt']
    negative_prompt = sd_params['negative_prompt']
    guidance_scale = sd_params['guidance_scale']
    
    # Encode prompts (potentially different from encoding)
    prompt_embeds = sd_pipe._encode_prompt(
        prompt,
        sd_pipe.unet.device,
        guidance_scale > 1.0,
        negative_prompt
    )
    
    # Create model wrapper
    model = SDRexModel(
        sd_pipe,
        prompt_embeds,
        guidance_scale,
        sd_pipe.scheduler,
        sched_type,
    )
    
    # Create Rex solver
    n_decode_steps = int(freeze_step * n_steps)
    solver = create_rex_solver(
        model,
        tableau=tableau,
        n_steps=n_decode_steps,
        prediction_type=prediction_type,
        adaptive=adaptive,
        zeta=zeta,
        scheduler=sd_pipe.scheduler,
        sched_type=sched_type,
    )
    
    # Time span: from freeze_step to eps (forward = generation)
    device = x_t.device
    t_span = torch.tensor([freeze_step, eps], device=device)
    
    with torch.no_grad():
        # Reset NFE counter before solve
        solver.nfe = 0
        # Use forward_solve to decode (noise -> image direction)
        x_0, x_hat_0 = solver.forward_solve(x_t, t_span)
        decode_nfe = solver.nfe
    
    print(f"Decoded: t={freeze_step} -> t={eps}, NFE={decode_nfe}")
    
    return x_0, decode_nfe


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Image editing with Rex solver")
    
    # General settings
    parser.add_argument('--num_inference_steps', type=int, default=200)
    parser.add_argument('--num_images', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--freeze_step', type=float, default=0.5,
                        help='Fraction of diffusion process to use (0-1)')
    parser.add_argument('--guidance', type=float, default=2.0)
    parser.add_argument('--eps', type=float, default=0.0002,
                        help='Small time to avoid singularity')
    
    # Rex-specific settings
    parser.add_argument('--tableau', type=str, default='rk4',
                        choices=list_rk_methods(),
                        help='RK method for integration')
    parser.add_argument('--zeta', type=float, default=0.5,
                        help='McCallum-Foster coupling parameter (0-1)')
    parser.add_argument('--prediction_type', type=str, default='data',
                        choices=['data', 'noise'],
                        help='Type of model prediction to use')
    parser.add_argument('--adaptive', action='store_true',
                        help='Use adaptive step size control')
    parser.add_argument('--step_domain', type=str, default='t',
                        choices=['t', 'varsigma'],
                        help='Domain for adaptive step size control')
    parser.add_argument('--atol', type=float, default=1e-5,
                        help='Absolute tolerance for adaptive stepping')
    parser.add_argument('--rtol', type=float, default=1e-5,
                        help='Relative tolerance for adaptive stepping')
    
    # Model and data
    parser.add_argument('--model_id', type=str, 
                        default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--save_dir', type=str, default='results/image_edits/rex')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    
    # Comparison with other methods
    parser.add_argument('--sampler_type', type=str, default='rex',
                        choices=['rex', 'ddim', 'bdia', 'edict', 'belm'],
                        help='Sampler to use (rex uses new wrapper)')
    parser.add_argument('--bdia_gamma', type=float, default=0.96)
    parser.add_argument('--edict_p', type=float, default=0.93)
    
    args = parser.parse_args()
    
    # Setup
    device = f'cuda:{args.device}'
    dtype = torch.float32
    sched_type = 'scaled_linear'
    
    print(f"Using device: {device}")
    print(f"Rex configuration: tableau={args.tableau}, zeta={args.zeta}, "
          f"prediction_type={args.prediction_type}")
    
    # Load model
    sd = StableDiffusionDiffEditPipeline.from_pretrained(
        args.model_id, torch_dtype=dtype
    ).to(device)
    
    scheduler = DDIMScheduler(
        beta_end=0.012, 
        beta_start=0.00085, 
        beta_schedule='scaled_linear',
        clip_sample=False,
        timestep_spacing='linspace', 
        set_alpha_to_one=False
    )
    
    sd_pipe = PipelineLike(
        device=device, 
        vae=sd.vae, 
        text_encoder=sd.text_encoder,
        tokenizer=sd.tokenizer,
        unet=sd.unet, 
        scheduler=scheduler
    )
    sd_pipe.vae.to(device)
    sd_pipe.text_encoder.to(device)
    sd_pipe.unet.to(device)
    print('Model loaded')
    
    # Evaluation models
    cs_model = CLIPScore(
        model_name_or_path='openai/clip-vit-base-patch16'
    ).to(device)
    ir_model = RM.load('ImageReward-v1.0', device=device)
    lpips = LearnedPerceptualImagePatchSimilarity(
        net_type='squeeze'
    ).to(device)
    pick_model = PickScoreModel(device=device)
    print('Evaluation models loaded')
    
    set_seed(args.seed)
    
    # Load dataset
    ds = load_dataset('./data/pix2pix', split='train[:1000]')
    
    # Create output directories
    os.makedirs(f'{args.save_dir}/imgs', exist_ok=True)
    os.makedirs(f'{args.save_dir}/metadata', exist_ok=True)
    
    count = 0
    
    for entry in tqdm(ds):
        ori_prompt = entry['original_prompt']
        ori_image = entry['original_image']
        edit_prompt = entry['edit_prompt']
        edited_prompt = entry['edited_prompt']
        edited_image = entry['edited_image']
        
        # Generate mask using DiffEdit
        mask_image = sd.generate_mask(
            image=ori_image, 
            source_prompt=ori_prompt, 
            target_prompt=edited_prompt
        )
        
        # Convert images to latents
        latent, ori_image_tensor = pil_to_latents(ori_image, sd_pipe, return_image=True)
        _, edited_image_tensor = pil_to_latents(edited_image, sd_pipe, return_image=True)
        
        latent = latent.to(device)
        ori_image_tensor = ori_image_tensor.to(device)
        edited_image_tensor = edited_image_tensor.to(device)
        
        # Prepare mask
        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
        mask_tensor = torch.from_numpy(mask_image).to(device)
        # mask_tensor = 0.
        
        negative_prompt = ''
        guidance_scale = args.guidance
        num_inference_steps = args.num_inference_steps
        
        sd_params = {
            'prompt': ori_prompt,
            'negative_prompt': negative_prompt,
            'seed': args.seed,
            'guidance_scale': guidance_scale,
            'num_inference_steps': num_inference_steps,
            'width': 512,
            'height': 512
        }
        
        # === ENCODING (Image -> Intermediate) ===
        encode_nfe = 0
        decode_nfe = 0
        
        if args.sampler_type == 'rex':
            # Use the new Rex wrapper
            x_t, x_hat_t, encode_nfe = sd_rex_encode(
                sd_pipe,
                sd_params,
                latent,
                tableau=args.tableau,
                n_steps=num_inference_steps,
                prediction_type=args.prediction_type,
                adaptive=args.adaptive,
                zeta=args.zeta,
                eps=args.eps,
                freeze_step=args.freeze_step,
                sched_type=sched_type,
            )
            
            # Apply mask (blend with noise in masked regions)
            noise = torch.randn_like(latent)
            # x_t = noise
            # x_hat_t = noise
            # x_t = mask_tensor * noise + (1. - mask_tensor) * x_t
            # x_hat_t = mask_tensor * noise + (1. - mask_tensor) * x_hat_t
            
        elif args.sampler_type == 'ddim':
            freeze_step_idx = int((1. - args.freeze_step) * num_inference_steps)
            intermediate = DDIM.latent_to_intermediate(
                sd_pipe=sd_pipe, sd_params=sd_params,
                latent=latent, freeze_step=freeze_step_idx
            )
            noise = torch.randn_like(intermediate)
            intermediate = mask_tensor * noise + (1. - mask_tensor) * intermediate
            
        elif args.sampler_type == 'edict':
            freeze_step_idx = int((1. - args.freeze_step) * num_inference_steps)
            x_intermediate, y_intermediate = edict.latent_to_intermediate(
                sd_pipe=sd_pipe, sd_params=sd_params,
                latent=latent, freeze_step=freeze_step_idx, p=args.edict_p
            )
            noise = torch.randn_like(x_intermediate)
            x_intermediate = mask_tensor * noise + (1. - mask_tensor) * x_intermediate
            y_intermediate = mask_tensor * noise + (1. - mask_tensor) * y_intermediate
            
        elif args.sampler_type == 'bdia':
            freeze_step_idx = int((1. - args.freeze_step) * num_inference_steps)
            intermediate, second_intermediate = BDIA.latent_to_intermediate(
                sd_pipe=sd_pipe, sd_params=sd_params,
                latent=latent, gamma=args.bdia_gamma, freeze_step=freeze_step_idx
            )
            noise = torch.randn_like(intermediate)
            intermediate = mask_tensor * noise + (1. - mask_tensor) * intermediate
            second_intermediate = mask_tensor * noise + (1. - mask_tensor) * second_intermediate
            
        elif args.sampler_type == 'belm':
            freeze_step_idx = int((1. - args.freeze_step) * num_inference_steps)
            intermediate, second_intermediate = BELM.latent_to_intermediate(
                sd_pipe=sd_pipe, sd_params=sd_params,
                latent=latent, freeze_step=freeze_step_idx
            )
            noise = torch.randn_like(intermediate)
            intermediate = mask_tensor * noise + (1. - mask_tensor) * intermediate
            second_intermediate = mask_tensor * noise + (1. - mask_tensor) * second_intermediate
        
        # === DECODING (Intermediate -> Edited Image) ===
        # Update prompt for editing
        sd_params['prompt'] = edited_prompt
        
        if args.sampler_type == 'rex':
            recon_latent, decode_nfe = sd_rex_decode(
                sd_pipe,
                sd_params,
                x_t,
                x_hat_t,
                tableau=args.tableau,
                n_steps=num_inference_steps,
                prediction_type=args.prediction_type,
                adaptive=args.adaptive,
                zeta=args.zeta,
                eps=args.eps,
                freeze_step=args.freeze_step,
                sched_type=sched_type,
            )
            total_nfe = encode_nfe + decode_nfe
            print(f"Total NFE: {total_nfe} (encode: {encode_nfe}, decode: {decode_nfe})")
            
        elif args.sampler_type == 'ddim':
            recon_latent = DDIM.intermediate_to_latent(
                sd_pipe=sd_pipe, sd_params=sd_params,
                intermediate=intermediate, freeze_step=freeze_step_idx
            )
            
        elif args.sampler_type == 'edict':
            recon_latent, _ = edict.intermediate_to_latent(
                sd_pipe=sd_pipe, sd_params=sd_params,
                x_intermediate=x_intermediate, y_intermediate=y_intermediate,
                p=args.edict_p, freeze_step=freeze_step_idx
            )
            
        elif args.sampler_type == 'bdia':
            recon_latent = BDIA.intermediate_to_latent(
                sd_pipe=sd_pipe, sd_params=sd_params,
                intermediate=intermediate, intermediate_second=second_intermediate,
                gamma=args.bdia_gamma, freeze_step=freeze_step_idx
            )
            
        elif args.sampler_type == 'belm':
            recon_latent = BELM.intermediate_to_latent(
                sd_pipe=sd_pipe, sd_params=sd_params,
                intermediate=intermediate, intermediate_second=second_intermediate,
                freeze_step=freeze_step_idx
            )
        
        # === EVALUATION ===
        # Save image
        pil = test_sd15.to_pil(latents=recon_latent, sd_pipe=sd_pipe)
        pil.save(os.path.join(args.save_dir, f'imgs/{count:06d}.png'))
        print(f'Image {count} editing finished')
        
        # Compute metrics
        new_img = test_sd15.to_tensor_image(recon_latent, sd_pipe)
        
        reward = ir_model.score(edited_prompt, pil)
        clip_score = cs_model(new_img, edited_prompt)
        pick_score = pick_model.score(edited_prompt, pil)
        
        lpips_orig = lpips(new_img * 2 - 1, ori_image_tensor.unsqueeze(0))
        lpips_edit = lpips(new_img * 2 - 1, edited_image_tensor.unsqueeze(0))
        
        # Save metadata
        save_dict = {
            'original_prompt': ori_prompt,
            'edit_prompt': edit_prompt,
            'edited_prompt': edited_prompt,
            'sampler_type': args.sampler_type,
            'tableau': args.tableau if args.sampler_type == 'rex' else None,
            'zeta': args.zeta if args.sampler_type == 'rex' else None,
            'prediction_type': (
                args.prediction_type if args.sampler_type == 'rex' else None
            ),
            'freeze_step': args.freeze_step,
            'guidance_scale': args.guidance,
            'num_inference_steps': args.num_inference_steps,
            'encode_nfe': encode_nfe,
            'decode_nfe': decode_nfe,
            'total_nfe': encode_nfe + decode_nfe,
            'IR': reward,
            'CLIPScore': clip_score.detach().item(),
            'PickScore': pick_score,
            'LPIPS_orig_vs_recon': lpips_orig.detach().item(),
            'LPIPS_edit_vs_recon': lpips_edit.detach().item(),
        }
        
        with open(f'{args.save_dir}/metadata/{count:06d}.json', 'w') as f:
            json.dump(save_dict, f, indent=2)
        
        count += 1
        
        if count >= args.num_images:
            break
    
    print(f"Completed {count} image edits. Results saved to {args.save_dir}")


if __name__ == '__main__':
    main()
