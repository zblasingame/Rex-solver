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
from diffusers import StableDiffusionPipeline, DDIMScheduler, StableDiffusionInstructPix2PixPipeline, StableDiffusionDiffEditPipeline, DDIMPipeline
from torch.utils.data import DataLoader
from datasets import load_dataset
from samplers.test_sd15 import  center_crop, load_im_into_format_from_path, pil_to_latents
from samplers.utils import PipelineLike
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal.clip_score import CLIPScore
import ImageReward as RM

import torchvision

from tqdm import tqdm

from torchsde import BrownianInterval
from samplers.rex import rex_forward, rex_backward, SDE_SOLVERS


class MorphDataset(torch.utils.data.Dataset):
    def __init__(self, ref_folder, folder, image_size):
        self.ref_folder = ref_folder
        self.folder = folder
        self.image_size = image_size
        self.files = os.listdir(self.ref_folder)

        transform = [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((.5,.5,.5),(.5,.5,.5))
        ]
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.files)

    def getitem_frll(self, index):
        ida, idb = self.files[index].split('_')
        idb, _ = idb.split('.p')

        path = os.path.join(self.folder, f'{ida}_03.png')
        img = Image.open(path)
        img = img.convert('RGB')
        img_a = self.transform(img)

        path = os.path.join(self.folder, f'{idb}_03.png')
        img = Image.open(path)
        img = img.convert('RGB')
        img_b = self.transform(img)

        return img_a, ida, img_b, idb

    def getitem_syn_mad22(self, index):
        ida, idb = self.files[index].split('_')
        idb, _ = idb.split('.p')

        path = os.path.join(self.folder, f'{ida}_03.png')
        img = Image.open(path)
        img = img.convert('RGB')
        img_a = self.transform(img)

        path = os.path.join(self.folder, f'{idb}_03.png')
        img = Image.open(path)
        img = img.convert('RGB')
        img_b = self.transform(img)

        return img_a, ida, img_b, idb

    def getitem_feret(self, index):
        parts = self.files[index].split('_')
        ida = []
        idb = []
        is_a = True
        n_digits = 0

        for part in parts:
            if part.isnumeric():
                n_digits += 1

            if n_digits >= 3:
                is_a = False

            if is_a:
                ida.append(part)
            else:
                idb.append(part)


        ida = '_'.join(ida)
        idb = '_'.join(idb)

        ida = ida.replace('.jpg', '').replace('.png', '')
        idb = idb.replace('.jpg', '').replace('.png', '')

        path = os.path.join(self.folder, f'{ida}.png')
        img = Image.open(path)
        img = img.convert('RGB')
        img_a = self.transform(img)

        path = os.path.join(self.folder, f'{idb}.png')
        img = Image.open(path)
        img = img.convert('RGB')
        img_b = self.transform(img)

        return img_a, ida, img_b, idb

    def getitem_frgc(self, index):
        ida, idb = self.files[index].split('_')

        ida = ida.replace('.jpg', '').replace('.png', '')
        idb = idb.replace('.jpg', '').replace('.png', '')

        path = os.path.join(self.folder, f'{ida}.png')
        img = Image.open(path)
        img = img.convert('RGB')
        img_a = self.transform(img)

        path = os.path.join(self.folder, f'{idb}.png')
        img = Image.open(path)
        img = img.convert('RGB')
        img_b = self.transform(img)

        return img_a, ida, img_b, idb

    def __getitem__(self, index):
        if 'frll' in self.folder or 'syn_mad22' in self.folder:
            img_a, ida, img_b, idb = self.getitem_frll(index)
        elif 'feret' in self.folder:
            img_a, ida, img_b, idb = self.getitem_feret(index)
        elif 'frgc' in self.folder:
            img_a, ida, img_b, idb = self.getitem_frgc(index)
        else:
            raise Exception('No morph parser found!')

        return img_a, ida, img_b, idb


def ddim_forward(ddpm_pipe, num_inference_steps, batch_size, intermediate=None):
    dtype = torch.float32
    # torch.manual_seed(seed)
    ddpm_pipe.scheduler.set_timesteps(num_inference_steps, device='cuda')
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
    if intermediate is None:
        intermediate = torch.randn(image_shape, generator=None, device='cuda', dtype=dtype)

    xis.append(intermediate)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # print('###', i)
            noise_pred = ddpm_pipe.unet(
                intermediate,
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
            intermediate = coef_xt * intermediate + coef_eps * noise_pred
            xis.append(intermediate)
    images = xis[-1]
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    images = ddpm_pipe.numpy_to_pil(images)
    return images


def ddim_inversion(ddpm_pipe, num_inference_steps, latent):
    dtype = torch.float32
    ddpm_pipe.scheduler.set_timesteps(num_inference_steps, device='cuda')

    xis=[]
    timesteps = ddpm_pipe.scheduler.timesteps
    xis.append(latent)
    prev_noise = None

    # print(num_inference_steps)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            index = num_inference_steps - i - 1

            time = timesteps[index + 1] if index < num_inference_steps - 1 else 1
            noise_pred = ddpm_pipe.unet(
                latent,
                time,
                return_dict=False,
            )[0]

            if index < num_inference_steps - 1:
                alpha_s = ddpm_pipe.scheduler.alphas_cumprod[timesteps[index]].to(torch.float32)
                alpha_t = ddpm_pipe.scheduler.alphas_cumprod[timesteps[index + 1]].to(torch.float32)
            else:
                alpha_s = ddpm_pipe.scheduler.alphas_cumprod[timesteps[index]].to(torch.float32)
                alpha_t = 1

            sigma_s = (1 - alpha_s) ** 0.5
            sigma_t = (1 - alpha_t) ** 0.5
            alpha_s = alpha_s ** 0.5
            alpha_t = alpha_t ** 0.5

            coef_xt = alpha_s / alpha_t
            coef_eps = sigma_s - sigma_t * coef_xt
            latent = coef_xt * latent + coef_eps * noise_pred

            xis.append(latent)
    return xis[-1]

def belm_inversion(ddpm_pipe, num_inference_steps, latent):
    dtype = torch.float32
    ddpm_pipe.scheduler.set_timesteps(num_inference_steps, device='cuda')

    xis=[]
    timesteps = ddpm_pipe.scheduler.timesteps
    xis.append(latent)
    prev_noise = None

    # print(num_inference_steps)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            index = num_inference_steps - i - 1

            time = timesteps[index + 1] if index < num_inference_steps - 1 else 1
            noise_pred = ddpm_pipe.unet(
                latent,
                time,
                return_dict=False,
            )[0]

            if index < num_inference_steps - 1:
                alpha_i = ddpm_pipe.scheduler.alphas_cumprod[timesteps[index]].to(torch.float32)
                alpha_i_minus_1 = ddpm_pipe.scheduler.alphas_cumprod[timesteps[index + 1]].to(torch.float32)
            else:
                alpha_i = ddpm_pipe.scheduler.alphas_cumprod[timesteps[index]].to(torch.float32)
                alpha_i_minus_1 = 1

            sigma_i = (1 - alpha_i) ** 0.5
            sigma_i_minus_1 = (1 - alpha_i_minus_1) ** 0.5
            alpha_i = alpha_i ** 0.5
            alpha_i_minus_1 = alpha_i_minus_1 ** 0.5

            if i == 0:
                latent = (alpha_i / alpha_i_minus_1) * latent + (sigma_i - (alpha_i / alpha_i_minus_1) * sigma_i_minus_1)
            else:
                alpha_i_minus_2 = 1 if i == 1 else ddpm_pipe.scheduler.alphas_cumprod[timesteps[index + 2]].to(torch.float32)
                sigma_i_minus_2 = (1 - alpha_i_minus_2) ** 0.5
                alpha_i_minus_2 = alpha_i_minus_2 ** 0.5

                h_i = sigma_i / alpha_i - sigma_i_minus_1 / alpha_i_minus_1
                h_i_minus_1 = sigma_i_minus_1 / alpha_i_minus_1 - sigma_i_minus_2 / alpha_i_minus_2

                coef_x_i_minus_2 = (alpha_i / alpha_i_minus_2) * (h_i ** 2) / (h_i_minus_1 ** 2)
                coef_x_i_minus_1 = (alpha_i / alpha_i_minus_1) * (h_i_minus_1 ** 2 - h_i ** 2) / (h_i_minus_1 ** 2)
                coef_eps = alpha_i * (h_i_minus_1 + h_i) * h_i / h_i_minus_1
                latent = coef_x_i_minus_2 * xis[-2] + coef_x_i_minus_1 * xis[-1] + coef_eps * noise_pred
            xis.append(latent)
        return xis[-1], xis[-2]

def belm_forward(ddpm_pipe, num_inference_steps, batch_size, intermediate, intermediate_second):
    dtype = torch.float32
    # torch.manual_seed(seed)
    ddpm_pipe.scheduler.set_timesteps(num_inference_steps, device='cuda')
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
    if intermediate is None:
        intermediate = torch.randn(image_shape, generator=None, device='cuda', dtype=dtype)

    xis.append(intermediate)
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # print('###', i)
            noise_pred = ddpm_pipe.unet(
                intermediate,
                t,
                return_dict=False,
            )[0]

            if i < num_inference_steps - 1:
                alpha_s = ddpm_pipe.scheduler.alphas_cumprod[timesteps[i + 1]].to(torch.float32)
                alpha_t = ddpm_pipe.scheduler.alphas_cumprod[t].to(torch.float32)
            else:
                alpha_s = 1
                alpha_t = ddpm_pipe.scheduler.alphas_cumprod[t].to(torch.float32)

            sigma_s = (1 - alpha_s) ** 0.5
            sigma_t = (1 - alpha_t) ** 0.5
            alpha_s = alpha_s ** 0.5
            alpha_t = alpha_t ** 0.5

            coef_xt = alpha_s / alpha_t
            coef_eps = sigma_s - sigma_t * coef_xt
            if i == 0:
                if intermediate_second is not None:
                    # print('have intermediate_second')
                    intermediate = intermediate_second.clone()
                else:
                    # print('dont have intermediate_second')
                    intermediate = coef_xt * intermediate + coef_eps * noise_pred
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
                intermediate = coef_1 * noise_pred + coef_2 * xis[-2] + coef_3 * xis[-1]
            xis.append(intermediate)
    images = xis[-1]
    # images = (images / 2 + 0.5).clamp(0, 1)
    # images = images.cpu().permute(0, 2, 3, 1).numpy()
    # images = ddpm_pipe.numpy_to_pil(images)
    return images

def slerp(val, low, high):
    low_norm = low / low.norm(dim=(-1, -2), keepdim=True)
    high_norm = high / high.norm(dim=(-1, -2), keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(dim=(-1, -2), keepdim=True))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high
    return res


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def ddpm_rex_forward(ddpm_pipe, num_inference_steps, solver, xt, xt_hat, eps=0.0002, bm=None, coupling=0.999, pred_type='data'):
    dtype = torch.float32

    model_func = lambda t, x: ddpm_pipe.unet(x, t * 1000, return_dict=False)[0]

    timesteps = torch.linspace(1., eps, num_inference_steps+1, device=xt.device, dtype=torch.float32)

    with torch.no_grad():
        print(solver)
        image, _ = rex_forward(model_func, ddpm_pipe.scheduler, xt, xt_hat, timesteps, solver=solver, bm=bm, low_order_final_n_steps=0, sched_type='scaled_linear', coupling=coupling, pred_type=pred_type)

    return image

def ddpm_rex_backward(ddpm_pipe, num_inference_steps, solver, xt, xt_hat, eps=0.0002, bm=None, coupling=0.999, pred_type='data'):
    dtype = torch.float32

    model_func = lambda t, x: ddpm_pipe.unet(x, t * 1000, return_dict=False)[0]

    timesteps = torch.linspace(1., eps, num_inference_steps+1, device=xt.device, dtype=torch.float32)
    # timesteps = torch.linspace(eps, p, int(p*num_inference_steps)+1, device=xt.device, dtype=torch.float32)

    with torch.no_grad():
        print(solver)
        xt, xt_hat = rex_backward(model_func, ddpm_pipe.scheduler, xt, xt_hat, timesteps, solver=solver, bm=bm, low_order_final_n_steps=0, sched_type='scaled_linear', coupling=coupling, pred_type=pred_type)

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
    parser.add_argument('--model_id', type=str, default='google/ddpm-celebahq-256')
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

    ddpm = DDIMPipeline.from_pretrained(model_id,torch_dtype=torch.float32)
    ddpm.unet.to(f'cuda:{args.device}')

    # eval models
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
    print('eval models loaded')

    bm = None
    if args.solver in SDE_SOLVERS:
        bm = BrownianInterval(t0=0., t1=1e5, size=(1, 3, 256, 256), entropy=args.seed, tol=1e-5, device='cpu', levy_area_approximation='space-time')

    rescale_img = lambda x: (x + 1.) / 2.

    set_seed(args.seed)

    ref_dir = 'data/ref_dir'
    raw_dir = 'data/frll'

    dataset = MorphDataset(ref_dir, raw_dir, 256)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True)

    os.makedirs(f'{args.save_dir}', exist_ok=True)

    for img_a, id_a, img_b, id_b in tqdm(dataloader):
        img_a = img_a.to(device)
        img_b = img_b.to(device)
    
        batch_size = img_a.shape[0]

        x_cat = torch.cat((img_a, img_b), dim=0)

        if sampler_type in ['lag','belm']:
            xt, xt_hat = belm_inversion(ddpm, num_inference_steps, x_cat)
        elif sampler_type in ['ddim']:
            pass
        elif sampler_type in ['rex']:
            xt, xt_hat = ddpm_rex_backward(ddpm, num_inference_steps, args.solver, x_cat, x_cat, eps=args.eps, bm=bm, coupling=args.coupling, pred_type=args.pred_type)

        xt_a, xt_b = xt.chunk(2, dim=0)
        xth_a, xth_b = xt_hat.chunk(2, dim=0)

        xs = []

        for blend in [0, 0.15, 0.35, 0.5, 0.65, 0.85, 1]:
            xt_i = slerp(blend, xt_a, xt_b)
            xth_i = slerp(blend, xth_a, xth_b)

            if sampler_type in ['lag','belm']:
                xt = belm_forward(ddpm, num_inference_steps, batch_size, xt_i, xth_i)
            elif sampler_type in ['ddim']:
                pass
            elif sampler_type in ['rex']:
                xt = ddpm_rex_forward(ddpm, num_inference_steps, args.solver, xt_i, xth_i, eps=args.eps, bm=bm, coupling=args.coupling, pred_type=args.pred_type)

            xs.append(xt)

        torchvision.utils.save_image(rescale_img(torch.cat(xs, dim=0)), f'{args.save_dir}/{id_a}_{id_b}.png', nrow=7)


if __name__ == '__main__':
    main()
