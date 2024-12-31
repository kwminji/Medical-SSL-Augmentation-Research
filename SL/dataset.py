import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class LinearEvaluationDataset(Dataset):
    def __init__(self, data, image_size):
        self.data = data
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]['image_path']
        label = self.data[idx]['label']
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label
