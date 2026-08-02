import sys

import torch
import os
import json
import argparse
sys.path.append(os.getcwd())
from samplers import test_sd15, BELM, BDIA, edict, DDIM
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import glob
from diffusers import StableDiffusionPipeline, DDIMScheduler, StableDiffusionInstructPix2PixPipeline, StableDiffusionDiffEditPipeline
from torch.utils.data import DataLoader
from datasets import load_dataset
from samplers.test_sd15 import  center_crop, load_im_into_format_from_path, pil_to_latents
from samplers.utils import PipelineLike
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal.clip_score import CLIPScore
import ImageReward as RM

from tqdm import tqdm

from torchsde import BrownianInterval
from samplers.rex import rex_forward, rex_backward, SDE_SOLVERS

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



def sd_rex_forward(sd_pipe, sd_params, solver, p, xt, xt_hat, eps=0.0002, bm=None, coupling=0.999, pred_type='data'):
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

    print(prompt)

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

    # timesteps = torch.linspace(p, eps, int(p*num_inference_steps)+1, device=xt.device, dtype=torch.float32)
    timesteps = torch.linspace(eps, p, int(p*num_inference_steps)+1, device=xt.device, dtype=torch.float32)

    with torch.no_grad():
        print(solver)
        # image, _ = rex_forward(model_func, sd_pipe.scheduler, xt, xt_hat, timesteps, solver=solver, bm=bm, low_order_final_n_steps=0, sched_type='scaled_linear', coupling=coupling, pred_type=pred_type)
        image, _ = rex_backward(model_func, sd_pipe.scheduler, xt, xt_hat, timesteps, solver=solver, bm=bm, low_order_final_n_steps=0, sched_type='scaled_linear', coupling=coupling, pred_type=pred_type)

    return image


def sd_rex_backward(sd_pipe, sd_params, solver, p, xt, xt_hat, eps=0.0002, bm=None, coupling=0.999, pred_type='data'):
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

    print(prompt)

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

    # timesteps = torch.linspace(p, eps, int(p*num_inference_steps)+1, device=xt.device, dtype=torch.float32)
    timesteps = torch.linspace(eps, p, int(p*num_inference_steps)+1, device=xt.device, dtype=torch.float32)

    with torch.no_grad():
        print(solver)
        # xt, xt_hat = rex_backward(model_func, sd_pipe.scheduler, xt, xt_hat, timesteps, solver=solver, bm=bm, low_order_final_n_steps=0, sched_type='scaled_linear', coupling=coupling, pred_type=pred_type)
        xt, xt_hat = rex_forward(model_func, sd_pipe.scheduler, xt, xt_hat, timesteps, solver=solver, bm=bm, low_order_final_n_steps=0, sched_type='scaled_linear', coupling=coupling, pred_type=pred_type)

    return xt, xt_hat

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_inference_steps', type=int, default=200)
    parser.add_argument('--num_images', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--freeze_step', type=float, default=0.5)
    parser.add_argument('--guidance', type=float, default=2.0)
    parser.add_argument('--sampler_type', type = str,default='lag', choices=['lag', 'ddim', 'bdia', 'edict', 'belm', 'rex'])
    parser.add_argument('--save_dir', type=str, default='xx')
    parser.add_argument('--model_id', type=str, default='stable-diffusion-v1-5/stable-diffusion-v1-5')
    parser.add_argument('--bdia_gamma', type=float, default=0.96)
    parser.add_argument('--edict_p', type=float, default=0.93)
    parser.add_argument('--eps', type=float, default=0.0002)
    parser.add_argument('--coupling', type=float, default=0.999)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--solver', type=str, default='rk4')
    parser.add_argument('--pred_type', type=str, default='data')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    freeze_step = args.freeze_step
    sampler_type = args.sampler_type
    guidance_scale = args.guidance
    num_inference_steps = args.num_inference_steps
    model_id = args.model_id
    device = f'cuda:{args.device}'
    dtype = torch.float32

    # load model
    # model_id = "timbrooks/instruct-pix2pix"
    # sd = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float32, safety_checker=None)
    # sd = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    sd = StableDiffusionDiffEditPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)

    sche = DDIMScheduler(beta_end=0.012, beta_start=0.00085, beta_schedule='scaled_linear', clip_sample=False,
                         timestep_spacing='linspace', set_alpha_to_one=False)

    sd_pipe = PipelineLike(device=device, vae=sd.vae, text_encoder=sd.text_encoder, tokenizer=sd.tokenizer,
                           unet=sd.unet, scheduler=sche)
    sd_pipe.vae.to(device)
    sd_pipe.text_encoder.to(device)
    sd_pipe.unet.to(device)
    print('model loaded')

    # eval models
    cs_model = CLIPScore(model_name_or_path='openai/clip-vit-large-patch14').to(device)
    ir_model = RM.load('ImageReward-v1.0', device=device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
    print('eval models loaded')

    bm = None
    if args.solver in SDE_SOLVERS:
        bm = BrownianInterval(t0=0., t1=1e5, size=(1, 4, 64, 64), entropy=args.seed, tol=1e-5, device='cpu', levy_area_approximation='space-time')


    set_seed(args.seed)

    # ds = load_dataset("timbrooks/instructpix2pix-clip-filtered", )
    ds = load_dataset('./data/pix2pix', split='train[:1000]')
    # dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    os.makedirs(f'{args.save_dir}/imgs', exist_ok=True)
    os.makedirs(f'{args.save_dir}/metadata', exist_ok=True)

    count = 0

    # for ori_prompt, ori_image, edit_prompt, edited_prompt, edited_image in tqdm(ds):
    for entry in tqdm(ds):
        # ori_prompt, ori_image, edit_prompt, edited_prompt, edited_image = entry
        ori_prompt = entry['original_prompt']
        ori_image = entry['original_image']
        edit_prompt = entry['edit_prompt']
        edited_prompt = entry['edited_prompt']
        edited_image = entry['edited_image']

        mask_image = sd.generate_mask(image=ori_image, source_prompt=ori_prompt, target_prompt=edited_prompt)

        latent, ori_image = pil_to_latents(ori_image, sd_pipe, return_image=True)
        _, edited_image = pil_to_latents(edited_image, sd_pipe, return_image=True)

        # print(ori_prompt, ori_image, edit_prompt, edit_prompt, edited_image)
        latent.to(device)
        ori_image.to(device)
        edited_image.to(device)

        # print(f'Original prompt: "{ori_prompt}"')
        # print(f'Edited prompt: "{edited_prompt}"')

        negative_prompt = ''
        # ori_prompt = ''
        sd_params = {'prompt': ori_prompt, 'negative_prompt':negative_prompt, 'seed': args.seed, 'guidance_scale': guidance_scale, 'num_inference_steps':num_inference_steps , 'width':512, 'height':512}

        freeze_step = int((1. - args.freeze_step) * args.num_inference_steps)

        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
        mask_image = torch.from_numpy(mask_image).to(device)
        # mask_image = 0.

        # latent to intermediate
        if sampler_type in ['ddim']:
            intermediate = DDIM.latent_to_intermediate(sd_pipe=sd_pipe, sd_params=sd_params, latent=latent,freeze_step=freeze_step)

            noise = torch.randn_like(intermediate)
            intermediate = mask_image * noise + (1. - mask_image) * intermediate
        elif sampler_type in ['edict']:
            x_intermediate, y_intermediate = edict.latent_to_intermediate(sd_pipe=sd_pipe, sd_params=sd_params,
                                                                          latent=latent,freeze_step=freeze_step,p=args.edict_p)
            noise = torch.randn_like(x_intermediate)
            x_intermediate = mask_image * noise + (1. - mask_image) * x_intermediate
            y_intermediate = mask_image * noise + (1. - mask_image) * y_intermediate

        elif sampler_type in ['bdia']:
            intermediate, second_intermediate = BDIA.latent_to_intermediate(sd_pipe=sd_pipe, sd_params=sd_params,
                                                                            latent=latent,gamma= args.bdia_gamma ,freeze_step=freeze_step)

            noise = torch.randn_like(intermediate)
            intermediate = mask_image * noise + (1. - mask_image) * intermediate
            second_intermediate = mask_image * noise + (1. - mask_image) * second_intermediate

        elif sampler_type in ['lag', 'belm']:
            intermediate, second_intermediate = BELM.latent_to_intermediate(sd_pipe=sd_pipe,
                                                                                           sd_params=sd_params,
                                                                                           latent=latent,
                                                                                           freeze_step=freeze_step)
            noise = torch.randn_like(intermediate)
            intermediate = mask_image * noise + (1. - mask_image) * intermediate
            second_intermediate = mask_image * noise + (1. - mask_image) * second_intermediate


        elif sampler_type in ['rex']:
            xt, xt_hat = sd_rex_backward(sd_pipe, sd_params, args.solver, args.freeze_step, latent, latent, args.eps, bm, args.coupling, args.pred_type)

            print(torch.var(xt))
            noise = torch.randn_like(xt)
            xt = mask_image * noise + (1. - mask_image) * xt
            xt_hat = mask_image * noise + (1. - mask_image) * xt_hat

        # intermediate to latent
        sd_params = {'prompt': edit_prompt, 'negative_prompt': negative_prompt, 'seed': args.seed,
                     'guidance_scale': guidance_scale,
                     'num_inference_steps': num_inference_steps, 'width': 512, 'height': 512}

        if sampler_type in ['ddim']:
            recon_latent = DDIM.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params, intermediate=intermediate,freeze_step=freeze_step)
        elif sampler_type in ['edict']:
            recon_latent, _ = edict.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params,x_intermediate=x_intermediate,y_intermediate=y_intermediate,p=args.edict_p,freeze_step=freeze_step)
        elif sampler_type in ['bdia']:
            recon_latent = BDIA.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params, intermediate=intermediate,
                                 intermediate_second=second_intermediate,gamma= args.bdia_gamma,freeze_step=freeze_step)
        elif sampler_type in ['lag', 'belm']:
            recon_latent = BELM.intermediate_to_latent(sd_pipe=sd_pipe, sd_params=sd_params,
                                                                          intermediate=intermediate,
                                                                         intermediate_second=second_intermediate,freeze_step=freeze_step)
        elif sampler_type in ['rex']:
            recon_latent = sd_rex_forward(sd_pipe, sd_params, args.solver, args.freeze_step, xt, xt_hat, args.eps, bm, args.coupling, args.pred_type)
        
        # Eval and save image

        pil = test_sd15.to_pil(latents=recon_latent, sd_pipe=sd_pipe)
        pil.save(os.path.join(args.save_dir, f'imgs/{count:06d}.png'))
        print('editing finished')

        new_img = test_sd15.to_tensor_image(recon_latent, sd_pipe)

        
        reward = ir_model.score(edited_prompt, pil)
        clip_score = cs_model(new_img, edited_prompt)

        # print(torch.max(ori_image), torch.min(ori_image))

        lpips_score1 = lpips(new_img * 2 - 1, ori_image.unsqueeze(0))
        lpips_score2 = lpips(new_img * 2 - 1, edited_image.unsqueeze(0))

        save_dict = {
            'orginal_prompt': ori_prompt,
            'edit_prompt': edit_prompt,
            'edited_prompt': edited_prompt,
            'IR': reward,
            'CLIPScore': clip_score.detach().item(),
            'LPIPS_orig_vs_recon': lpips_score1.detach().item(),
            'LPIPS_edit_vs_recon': lpips_score2.detach().item(),
        }

        with open(f'{args.save_dir}/metadata/{count:06d}.json', 'w') as f:
            json.dump(save_dict, f)

        count += 1

if __name__ == '__main__':
    main()
