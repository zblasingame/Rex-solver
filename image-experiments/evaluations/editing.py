import os
import torch
import numpy as np
from tqdm import tqdm

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import json

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', type=str, help='Path to generated images')
    args = parser.parse_args()

    img_list = [os.path.join(args.path, image) for image in os.listdir(args.path)]

    clip_scores = []
    pick_scores = []
    ir = []
    lpips = []
    lpips_e = []

    for i, path in enumerate(tqdm(img_list)):
        if i >= 100:
            break

        with open(path, 'r') as f:
            data = json.load(f)

        clip_scores.append(data['CLIPScore'])
        pick_scores.append(data['PickScore'])
        ir.append(data['IR'])
        lpips.append(data['LPIPS_orig_vs_recon'])
        lpips_e.append(data['LPIPS_edit_vs_recon'])

    print(f'IR: {np.mean(ir):.3f} || CLIPScore: {np.mean(clip_scores):.2f} || PickScore: {np.mean(pick_scores):.3f} || LPIPS: {np.mean(lpips):.3f} || LPIPS Edit: {np.mean(lpips_e)}')
