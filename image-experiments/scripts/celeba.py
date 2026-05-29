import sys

from tqdm import tqdm

import torch
import random
import numpy as np
import os
import json
import argparse
from torchsde import BrownianInterval
sys.path.append(os.getcwd())
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline
from samplers.rex import rex_forward, SDE_SOLVERS, psi

def ddim_forward(ddpm_pipe, seed, num_inference_steps, states=None):

    dtype = torch.float32
    device = ddpm_pipe.unet.device
    # torch.manual_seed(seed)
    ddpm_pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = ddpm_pipe.scheduler.timesteps

    xis = []
    # Sample gaussian noise to begin loop
    if isinstance(ddpm_pipe.unet.config.sample_size, int):
        image_shape = (
            1,
            ddpm_pipe.unet.config.in_channels,
            ddpm_pipe.unet.config.sample_size,
            ddpm_pipe.unet.config.sample_size,
        )
    else:
        image_shape = (1, ddpm_pipe.unet.config.in_channels, *ddpm_pipe.unet.config.sample_size)
    states = torch.randn(image_shape, generator=None, device=device, dtype=dtype)

    xis.append(states)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # print('###', i)
            noise_pred = ddpm_pipe.unet(
                states,
                t,
                return_dict=False,
            )[0]

            if i < num_inference_steps - 1:
                alpha_s = ddpm_pipe.scheduler.alphas_cumprod[timesteps[i + 1]].to(torch.float32)
                alpha_t = ddpm_pipe.scheduler.alphas_cumprod[t].to(torch.float32)
            else:
                alpha_s = 1
                alpha_t = ddpm_pipe.scheduler.alphas_cumprod[t].to(torch.float32)

            sigma_s = (1 - alpha_s)**0.5
            sigma_t = (1 - alpha_t)**0.5
            alpha_s = alpha_s**0.5
            alpha_t = alpha_t**0.5

            coef_xt = alpha_s / alpha_t
            coef_eps = sigma_s - sigma_t * coef_xt
            states = coef_xt * states + coef_eps * noise_pred
            xis.append(states)
    image = xis[-1]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = ddpm_pipe.numpy_to_pil(image)
    return image

def belm_forward(ddpm_pipe, batch_size, num_inference_steps, states=None):
    dtype = torch.float32
    device = ddpm_pipe.unet.device
    ddpm_pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = ddpm_pipe.scheduler.timesteps

    xis = []
    # Sample gaussian noise to begin loop
    if isinstance(ddpm_pipe.unet.config.sample_size, int):
        image_shape = (
            batch_size,
            ddpm_pipe.unet.config.in_channels,
            ddpm_pipe.unet.config.sample_size,
            ddpm_pipe.unet.config.sample_size,
        )
    else:
        image_shape = (batch_size, ddpm_pipe.unet.config.in_channels, *ddpm_pipe.unet.config.sample_size)
    states = torch.randn(image_shape, generator=None, device=device, dtype=dtype)

    xis.append(states)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            noise_pred = ddpm_pipe.unet(
                states,
                t,
                return_dict=False,
            )[0]

            if i < num_inference_steps - 1:
                alpha_s = ddpm_pipe.scheduler.alphas_cumprod[timesteps[i + 1]].to(torch.float32)
                alpha_t = ddpm_pipe.scheduler.alphas_cumprod[t].to(torch.float32)
            else:
                alpha_s = 1
                alpha_t = ddpm_pipe.scheduler.alphas_cumprod[t].to(torch.float32)

            sigma_s = (1 - alpha_s)**0.5
            sigma_t = (1 - alpha_t)**0.5
            alpha_s = alpha_s**0.5
            alpha_t = alpha_t**0.5

            coef_xt = alpha_s / alpha_t
            coef_eps = sigma_s - sigma_t * coef_xt
            if i == 0:
                states = coef_xt * states + coef_eps * noise_pred
            else:
                # calculate i-1
                alpha_p = ddpm_pipe.scheduler.alphas_cumprod[timesteps[i - 1]].to(torch.float32)
                sigma_p = (1 - alpha_p) ** 0.5
                alpha_p = alpha_p ** 0.5

                # calculate t
                t_p, t_t, t_s = sigma_p / alpha_p, sigma_t / alpha_t, sigma_s / alpha_s

                # calculate delta
                delta_1 = t_t - t_p
                delta_2 = t_s - t_t
                delta_3 = t_s - t_p

                # calculate coef
                coef_1 = delta_2 * delta_3 * alpha_s / delta_1
                coef_2 = (delta_2 / delta_1) ** 2 * (alpha_s / alpha_p)
                coef_3 = (delta_1 - delta_2) * delta_3 / (delta_1 ** 2) * (alpha_s / alpha_t)

                # iterate
                states = coef_1 * noise_pred + coef_2 * xis[-2] + coef_3 * xis[-1]

            xis.append(states)
    image = xis[-1]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = ddpm_pipe.numpy_to_pil(image)
    return image

def bdia_forward(ddpm_pipe, batch_size, num_inference_steps, states=None, gamma = 1.0):
    dtype = torch.float32
    device = ddpm_pipe.unet.device
    # torch.manual_seed(seed)
    ddpm_pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = ddpm_pipe.scheduler.timesteps

    xis = []
    # Sample gaussian noise to begin loop
    if isinstance(ddpm_pipe.unet.config.sample_size, int):
        image_shape = (
            batch_size,
            ddpm_pipe.unet.config.in_channels,
            ddpm_pipe.unet.config.sample_size,
            ddpm_pipe.unet.config.sample_size,
        )
    else:
        image_shape = (batch_size, ddpm_pipe.unet.config.in_channels, *ddpm_pipe.unet.config.sample_size)
    states = torch.randn(image_shape, generator=None, device=device, dtype=dtype)

    xis.append(states)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # print('###', i)
            noise_pred = ddpm_pipe.unet(
                states,
                t,
                return_dict=False,
            )[0]

            if i < num_inference_steps - 1:
                alpha_s = ddpm_pipe.scheduler.alphas_cumprod[timesteps[i + 1]].to(torch.float32)
                alpha_t = ddpm_pipe.scheduler.alphas_cumprod[t].to(torch.float32)
            else:
                alpha_s = 1
                alpha_t = ddpm_pipe.scheduler.alphas_cumprod[t].to(torch.float32)

            sigma_s = (1 - alpha_s)**0.5
            sigma_t = (1 - alpha_t)**0.5
            alpha_s = alpha_s**0.5
            alpha_t = alpha_t**0.5

            coef_xt = alpha_s / alpha_t
            coef_eps = sigma_s - sigma_t * coef_xt
            if i == 0:
                states = coef_xt * states + coef_eps * noise_pred
            else:
                alpha_p = ddpm_pipe.scheduler.alphas_cumprod[timesteps[i - 1]].to(torch.float32)
                sigma_p = (1 - alpha_p) ** 0.5
                alpha_p = alpha_p ** 0.5
                coef_xt = coef_xt - gamma * alpha_p / alpha_t
                coef_eps_2 = sigma_p - sigma_t * alpha_p / alpha_t
                coef_eps = coef_eps - gamma * coef_eps_2
                states = gamma * xis[-2] + coef_xt * xis[-1] + coef_eps * noise_pred

            xis.append(states)
    image = xis[-1]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = ddpm_pipe.numpy_to_pil(image)
    return image

def edict_forward(ddpm_pipe, batch_size, num_inference_steps, states=None, p = 0.93):
    dtype = torch.float32
    device = ddpm_pipe.unet.device
    # torch.manual_seed(seed)
    ddpm_pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = ddpm_pipe.scheduler.timesteps

    xis = []
    yis = []
    # Sample gaussian noise to begin loop
    if isinstance(ddpm_pipe.unet.config.sample_size, int):
        image_shape = (
            batch_size,
            ddpm_pipe.unet.config.in_channels,
            ddpm_pipe.unet.config.sample_size,
            ddpm_pipe.unet.config.sample_size,
        )
    else:
        image_shape = (batch_size, ddpm_pipe.unet.config.in_channels, *ddpm_pipe.unet.config.sample_size)
    x_states = torch.randn(image_shape, generator=None, device=device, dtype=dtype)
    y_states = x_states.clone()
    xis.append(x_states)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # print('###', i)
            noise_pred = ddpm_pipe.unet(
                x_states,
                t,
                return_dict=False,
            )[0]

            if i < num_inference_steps - 1:
                alpha_s = ddpm_pipe.scheduler.alphas_cumprod[timesteps[i + 1]].to(torch.float32)
                alpha_t = ddpm_pipe.scheduler.alphas_cumprod[t].to(torch.float32)
            else:
                alpha_s = 1
                alpha_t = ddpm_pipe.scheduler.alphas_cumprod[t].to(torch.float32)

            sigma_s = (1 - alpha_s)**0.5
            sigma_t = (1 - alpha_t)**0.5
            alpha_s = alpha_s**0.5
            alpha_t = alpha_t**0.5

            coef_xt = alpha_s / alpha_t
            coef_eps = sigma_s - sigma_t * coef_xt
            x_inter = coef_xt * x_states + coef_eps * noise_pred
            noise_pred = ddpm_pipe.unet(
                x_inter,
                t,
                return_dict=False,
            )[0]
            y_inter = coef_xt * y_states + coef_eps * noise_pred
            x_states = p * x_inter + (1.0 - p) * y_inter
            y_states = p * y_inter + (1.0 - p) * x_states

            xis.append(x_states)
            yis.append(y_states)

    image = xis[-1]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = ddpm_pipe.numpy_to_pil(image)
    return image


def ddpm_rex_forward(ddpm_pipe, batch_size, num_inference_steps, solver, eps=0.0002, bm=None, coupling=0.999, pred_type='data'):
    dtype = torch.float32

    # Sample gaussian noise to begin loop
    if isinstance(ddpm_pipe.unet.config.sample_size, int):
        image_shape = (
            batch_size,
            ddpm_pipe.unet.config.in_channels,
            ddpm_pipe.unet.config.sample_size,
            ddpm_pipe.unet.config.sample_size,
        )
    else:
        image_shape = (batch_size, ddpm_pipe.unet.config.in_channels, *ddpm_pipe.unet.config.sample_size)

    xt = torch.randn(image_shape, generator=None, device=ddpm_pipe.unet.device, dtype=dtype)
    xt_hat = xt.clone()

    model_func = lambda t, x: ddpm_pipe.unet(x, t * 1000, return_dict=False)[0]

    timesteps = torch.linspace(1., eps, num_inference_steps+1, device=xt.device, dtype=torch.float32)

    with torch.no_grad():
        print(solver)
        image, _ = rex_forward(model_func, ddpm_pipe.scheduler, xt, xt_hat, timesteps, solver=solver, bm=bm, low_order_final_n_steps=2, coupling=coupling, pred_type=pred_type)


    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = ddpm_pipe.numpy_to_pil(image)

    return image


def ddpm_psi_forward(ddpm_pipe, batch_size, num_inference_steps, solver, eps=0.0002, bm=None, coupling=0.999, pred_type='data'):
    device = ddpm_pipe.unet.device
    dtype = torch.float32

    # Sample gaussian noise to begin loop
    if isinstance(ddpm_pipe.unet.config.sample_size, int):
        image_shape = (
            batch_size,
            ddpm_pipe.unet.config.in_channels,
            ddpm_pipe.unet.config.sample_size,
            ddpm_pipe.unet.config.sample_size,
        )
    else:
        image_shape = (batch_size, ddpm_pipe.unet.config.in_channels, *ddpm_pipe.unet.config.sample_size)

    xt = torch.randn(image_shape, generator=None, device=ddpm_pipe.unet.device, dtype=dtype)
    xt_hat = xt.clone()

    model_func = lambda t, x: ddpm_pipe.unet(x, t * 1000, return_dict=False)[0]

    timesteps = torch.linspace(1.0, eps, num_inference_steps+1, device=xt.device, dtype=torch.float32)

    with torch.no_grad():
        print('psi')
        print(solver)
        image = psi(model_func, ddpm_pipe.scheduler, xt, timesteps, solver=solver, bm=bm, low_order_final_n_steps=2, pred_type=pred_type)


    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = ddpm_pipe.numpy_to_pil(image)

    return image


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_num', type=int, default=1000)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_inference_steps', type=int, default=20)
    parser.add_argument('--sampler_type', type = str,default='lag', choices=['lag', 'ddim', 'bdia', 'edict', 'belm', 'rex', 'psi'])
    parser.add_argument('--save_dir', type=str, default='xxxx')
    parser.add_argument('--model_id', type=str, default='google/ddpm-celebahq-256')
    parser.add_argument('--bdia_gamma', type=float, default=0.5)
    parser.add_argument('--edict_p', type=float, default=0.5)
    parser.add_argument('--solver', type=str, default='rk4')
    parser.add_argument('--eps', type=float, default=0.0002)
    parser.add_argument('--coupling', type=float, default=0.999)
    parser.add_argument('--pred_type', type=str, default='data')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    start_index = args.start_index
    batch_size = args.batch_size
    sampler_type = args.sampler_type
    test_num = args.test_num
    num_inference_steps = args.num_inference_steps
    gamma = args.bdia_gamma
    p = args.edict_p
    model_id = args.model_id

    ddpm = DDIMPipeline.from_pretrained(model_id,torch_dtype=torch.float32)
    ddpm.unet.to(f'cuda:{args.device}')

    bm = None
    if args.solver in SDE_SOLVERS:
        bm = BrownianInterval(t0=0., t1=1e5, size=(1, 3, 256, 256), entropy=args.seed, tol=1e-5, device='cpu', levy_area_approximation='space-time')


    # print(ddpm.scheduler.config)
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with torch.no_grad():
        for seed in tqdm(range(start_index,start_index+test_num)):
            set_seed(seed)
            print('prepare to sample')
            if sampler_type in ['lag','belm']:
                images = belm_forward(ddpm_pipe=ddpm,batch_size=batch_size,num_inference_steps=num_inference_steps)
                for i,image in enumerate(images):
                    image.save(os.path.join(save_dir, f"belm_celebhq_inference{num_inference_steps}_seed{seed}_{i}.png"))
                print(f"belm batch##{seed},done")
            elif sampler_type in ['ddim']:
                images = ddpm(num_inference_steps = num_inference_steps, batch_size = batch_size).images
                for i,image in enumerate(images):
                    image.save(os.path.join(save_dir, f"ddim_celebhq_inference_pipe{num_inference_steps}_seed{seed}_{i}.png"))
                print(f"ddim batch##{seed},done")
            elif sampler_type in ['bdia']:
                images = bdia_forward(ddpm_pipe=ddpm,batch_size=batch_size,num_inference_steps=num_inference_steps,gamma=gamma)
                for i,image in enumerate(images):
                    image.save(os.path.join(save_dir, f"bdia_celebhq_inference_pipe{num_inference_steps}_seed{seed}_{i}.png"))
                print(f"bdia##{seed},done")
            elif sampler_type in ['edict']:
                print(f"edict##{seed},ready")
                images = edict_forward(ddpm_pipe=ddpm, batch_size=batch_size, num_inference_steps=num_inference_steps, p=p)
                for i, image in enumerate(images):
                    image.save(
                        os.path.join(save_dir, f"edict_celebhq_inference_pipe{num_inference_steps}_seed{seed}_{i}.png"))
                print(f"edict##{seed},done")
            elif sampler_type in ['rex']:
                print(f"reversible_dpm##{seed},ready")
                images = ddpm_rex_forward(ddpm_pipe=ddpm, batch_size=batch_size, num_inference_steps=num_inference_steps, solver=args.solver, bm=bm, eps=args.eps, coupling=args.coupling, pred_type=args.pred_type)
                for i, image in enumerate(images):
                    image.save(
                        os.path.join(save_dir, f"rex_celebhq_inference_pipe{num_inference_steps}_seed{seed}_{i}.png"))
                print(f"reversible_dpm##{seed},done")
            elif sampler_type in ['psi']:
                print(f"psi##{seed},ready")
                images = ddpm_psi_forward(ddpm_pipe=ddpm, batch_size=batch_size, num_inference_steps=num_inference_steps, solver=args.solver, bm=bm, eps=args.eps, coupling=args.coupling, pred_type=args.pred_type)
                for i, image in enumerate(images):
                    image.save(
                        os.path.join(save_dir, f"psi_celebhq_inference_pipe{num_inference_steps}_seed{seed}_{i}.png"))
                print(f"psi##{seed},done")


if __name__ == '__main__':
    main()

