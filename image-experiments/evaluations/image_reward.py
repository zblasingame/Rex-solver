import os
import torch
import numpy as np
import ImageReward as RM
from tqdm import tqdm

from torchmetrics.multimodal.clip_score import CLIPScore
from functools import partial

from PIL import Image
import torchvision.transforms as transforms

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files
        self.transform = transforms.PILToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = transform(Image.open(self.files[idx]))
        prompt = self.files[idx].split('/')[-1].split('.png')[0]

        return prompt, img, self.files[idx]

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', type=str, help='Path to generated images')
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()

    img_list = [os.path.join(args.path, image) for image in os.listdir(args.path)]

    metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")

    model = RM.load('ImageReward-v1.0')

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    rewards = []
    scores = []

    dataset = ImageDataset(img_list)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    with torch.no_grad():
        # for img in tqdm(img_list):
            # prompt = img.split('/')[-1].split('.png')[0]#.replace("['", '').replace("']", '')

        for prompt, img, paths in tqdm(dataloader):
            reward = model.score(list(prompt), list(paths))
            rewards.append(np.mean(reward))
            
            # img = transform(Image.open(img))
            score = metric(img, list(prompt))
            scores.append(np.mean(score.numpy()))


        print(f'IR: {np.mean(rewards):.3f} || CLIP: {np.mean(scores):.2f}')
