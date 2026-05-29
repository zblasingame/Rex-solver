import os
import torch
import numpy as np
from tqdm import tqdm

from functools import partial

from transformers import AutoProcessor, AutoModel

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

    # load model
    device = "cuda"
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)


    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    rewards = []
    scores = []

    dataset = ImageDataset(img_list)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)

    def calc_probs(prompt, images):
        # preprocess
        image_inputs = processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        text_inputs = processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)


        with torch.no_grad():
            # embed
            image_embs = model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            # score
            scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

            # get probabilities if you have multiple images to choose from
            # probs = torch.softmax(scores, dim=-1)

        return scores.cpu().tolist()

    with torch.no_grad():
        # for img in tqdm(img_list):
            # prompt = img.split('/')[-1].split('.png')[0]#.replace("['", '').replace("']", '')

        for prompt, img, paths in tqdm(dataloader):
            score = calc_probs(prompt, img)[0]
            scores.append(np.mean(score))


        print(f'Pick: {np.mean(scores):.2f}')

