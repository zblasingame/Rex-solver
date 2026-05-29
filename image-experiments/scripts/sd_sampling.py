import sys

import torch
import random
import os
import json
from tqdm import tqdm
import argparse
sys.path.append(os.getcwd())
from samplers import test_sd15, BELM, BDIA, edict, DDIM
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import glob
from diffusers import StableDiffusionPipeline, DDIMScheduler
from samplers.test_sd15 import  center_crop, load_im_into_format_from_path, pil_to_latents
from samplers.utils import PipelineLike

from torchsde import BrownianInterval
from samplers.rex import rex_forward, SDE_SOLVERS

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 


def sd_rex_forward(sd_pipe, sd_params, solver):
    prompt = sd_params['prompt']
    negative_prompt = sd_params['negative_prompt']
    seed = sd_params['seed']
    guidance_scale = sd_params['guidance_scale']
    num_inference_steps = sd_params['num_inference_steps']
    width = sd_params['width']
    height = sd_params['height']
    dtype = torch.float32

    prompt_embeds = sd_pipe._encode_prompt(
        prompt,
        sd_pipe.unet.device,
        guidance_scale > 1.0,
        negative_prompt
    )

    torch.manual_seed(seed)

    do_classifier_free_guidance = guidance_scale > 1.0

    def model_func(t, x):
        x = torch.cat([x] * 2) if do_classifier_free_guidance else x
        noise_pred = sd_pipe.unet(
            x,
            1000 * t,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        return noise_pred

    shape = (1, 4, 64, 64)
    xt = torch.randn(shape, generator=None, device=sd_pipe.unet.device, dtype=dtype)
    xt_hat = xt.clone()

    bm = None
    if solver in SDE_SOLVERS:
        bm = BrownianInterval(t0=0., t1=1e5, size=shape, entropy=seed, tol=1e-5, device='cpu', levy_area_approximation='space-time')
        # Need for ShARK
        # seed = seed + 1 if seed is not None else seed
        # bm2 = BrownianInterval(t0=0., t1=1e5, size=image_shape, entropy=seed, tol=1e-5, device='cpu')
        # bm = (bm, bm2)

    timesteps = torch.linspace(1.0, 0.0002, num_inference_steps+1, device=xt.device, dtype=torch.float32)
    # timesteps = torch.linspace(1.0, 0.0, num_inference_steps+1, device=xt.device, dtype=torch.float32)

    with torch.no_grad():
        print(solver)
        image, _ = rex_forward(model_func, sd_pipe.scheduler, xt, xt_hat, timesteps, solver=solver, bm=bm, low_order_final_n_steps=2, sched_type='scaled_linear')

    del bm # clear up memory

    return image


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--num_inference_steps', type=int, default=200)
    parser.add_argument('--guidance', type=float, default=4.0)
    parser.add_argument('--sampler_type', type = str,default='lag', choices=['lag', 'ddim', 'bdia', 'edict', 'belm', 'rex'])
    parser.add_argument('--save_dir', type=str, default='xx')
    parser.add_argument('--model_id', type=str, default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--bdia_gamma', type=float, default=0.96)
    parser.add_argument('--edict_p', type=float, default=0.93)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--solver', type=str, default='rk4')
    parser.add_argument('--seed', type=int, default=38018716)
    args = parser.parse_args()

    sampler_type = args.sampler_type
    guidance_scale = args.guidance
    num_inference_steps = args.num_inference_steps
    model_id = args.model_id
    device = f'cuda:{args.device}'
    dtype = torch.float32

    # load model
    sd = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    sche = DDIMScheduler(beta_end=0.012, beta_start=0.00085, beta_schedule='scaled_linear', clip_sample=False,
                         timestep_spacing='linspace', set_alpha_to_one=False)

    sd_pipe = PipelineLike(device=device, vae=sd.vae, text_encoder=sd.text_encoder, tokenizer=sd.tokenizer,
                           unet=sd.unet, scheduler=sche)
    sd_pipe.vae.to(device)
    sd_pipe.text_encoder.to(device)
    sd_pipe.unet.to(device)
    print('model loaded')

    prompts = []

    set_seed(args.seed)

    with open('data/COCO_Captions.csv') as f:
        prompts = [line.rstrip() for line in f]

    for prompt in tqdm(prompts):
        # intermediate to latent
        sd_params = {'prompt': prompt, 'negative_prompt': None, 'seed': args.seed,
                     'guidance_scale': guidance_scale,
                     'num_inference_steps': num_inference_steps, 'width': 512, 'height': 512}
        if sampler_type in ['ddim']:
            result_latent = DDIM.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params, intermediate=None,freeze_step=0)
        elif sampler_type in ['edict']:
            result_latent, _ = edict.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params,x_intermediate=None,y_intermediate=None,p=args.edict_p,freeze_step=0)
        elif sampler_type in ['bdia']:
            result_latent = BDIA.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params, intermediate=None,
                                 intermediate_second=None,gamma= args.bdia_gamma,freeze_step=0)
        elif sampler_type in ['lag', 'belm']:
            result_latent = BELM.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params,
                                                                          intermediate=None,
                                                                         intermediate_second=None,freeze_step=0)
        elif sampler_type in ['rex']:
            result_latent = sd_rex_forward(sd_pipe, sd_params, args.solver)


        pil = test_sd15.to_pil(latents=result_latent, sd_pipe=sd_pipe)
        pil.save(os.path.join(args.save_dir,
            f'{prompt}.png'))
        print('sampling finished')


if __name__ == '__main__':
    main()
