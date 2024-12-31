from pathlib import Path
import torch.nn as nn
import torch
import torch.distributed as dist
from torch.nn import Module
from torch.nn import SyncBatchNorm

from torch.optim import Optimizer, Adam
from torch.utils.data import Dataset, DataLoader

from beartype import beartype
from beartype.typing import Optional

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torchvision import transforms
from PIL import Image
import pandas as pd
import kornia
import wandb
from torchvision.models import resnet50
import os
import json
from sklearn.model_selection import train_test_split
from NTXentLoss import NT_Xent

from augmentation4 import DataAugmentation

# Constants
DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(find_unused_parameters=True)

# Utility Functions
def exists(v):
    return v is not None

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

# Dataset Class
class CombinedDataset(Dataset):
    def __init__(self, data, image_size, augmentation=None):
        self.data = data
        self.image_size = image_size
        self.augmentation = augmentation
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # 기본 리사이즈
            transforms.ToTensor(),                       # 텐서 변환
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 이미지와 레이블 로드
        img_path = self.data[idx]['image_path']
        label = self.data[idx]['label']

        # 이미지를 불러오고 RGB로 변환
        image = Image.open(img_path).convert("RGB")

        # 데이터 증강 적용
        if self.augmentation:
            image = self.augmentation(image)
        else:
            image = self.transform(image)

        return image

# Data Loading Functions
def load_ahub_data(base_dir):
    all_data = []
    label_dir = os.path.join(base_dir, "02.라벨링데이터")
    print(f"{label_dir}에서 AIHub 데이터를 로드합니다...")

    for category in os.listdir(label_dir):
        category_path = os.path.join(label_dir, category)
        if os.path.isdir(category_path):
            print(f"카테고리 처리 중: {category}")
            for file_name in os.listdir(category_path):
                if file_name.endswith(".json"):
                    json_path = os.path.join(category_path, file_name)
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        image_path = os.path.join(data["filepath"], data["filename"])
                        label = 1 if data["label"] == "Normal" else 0
                        all_data.append({"image_path": image_path, "label": label})
    print(f"AIHub 데이터 총 샘플 수: {len(all_data)}개")
    return all_data

def load_mura_data(mura_base_dir):
    all_data = []
    print(f"{mura_base_dir}에서 MURA 데이터를 로드합니다...")
    for root, _, files in os.walk(mura_base_dir):
        for file in files:
            if file.endswith(".png"):
                label = 1 if "positive" in root else 0
                image_path = os.path.join(root, file)
                all_data.append({"image_path": image_path, "label": label})
    print(f"MURA 데이터 총 샘플 수: {len(all_data)}개")
    return all_data


# Projection Head
class ProjectionHead(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# SimCLR Model
class SimCLRModel(Module):
    def __init__(self, encoder, projection_head):
        super().__init__()
        self.encoder = encoder
        self.projection_head = projection_head

    def forward(self, x):
        z = self.encoder(x)  # High-dimensional representation
        h = self.projection_head(z)  # Low-dimensional projection
        return h

# SimCLR Trainer
class SimCLRTrainer(Module):
    def __init__(
        self,
        net: Module,
        *,
        image_size: int,
        learning_rate: float,
        dataset: Dataset,
        num_epochs: int,
        batch_size: int = 32,
        temperature: float = 0.5,
        optimizer_klass=Adam,
        checkpoint_folder: str = './checkpoints_combined',
        optimizer_kwargs: dict = dict(),
        accelerator_kwargs: dict = dict(),
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__()
        self.device = device
        self.accelerator = Accelerator(**accelerator_kwargs)
        self.net = self.accelerator.prepare(net).to(self.device)
        self.criterion = NT_Xent(batch_size, temperature).to(self.device)
        self.optimizer = optimizer_klass(self.net.parameters(), lr=learning_rate, **optimizer_kwargs)
        self.dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, drop_last=True)
        self.num_epochs = num_epochs
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok=True, parents=True)

    def forward(self):
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            for step, images in enumerate(self.dataloader):
                images = images.to(self.device)
                images = torch.cat([images, torch.flip(images, dims=[-1])], dim=0)
                z = self.net(images)
                z_i, z_j = torch.chunk(z, 2, dim=0)
                loss = self.criterion(z_i, z_j)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                wandb.log({"step_loss": loss.item(), "epoch": epoch + 1, "step": step + 1})
                epoch_loss += loss.item()

            checkpoint_path = self.checkpoint_folder / f'R+B+C.epoch{epoch + 1}.pt'
            torch.save({'encoder': simclr_model.encoder.state_dict(),
            'projection_head': simclr_model.projection_head.state_dict(),}, str(checkpoint_path))

            wandb.log({"epoch_loss": epoch_loss / len(self.dataloader), "epoch": epoch + 1})
            print(f"Epoch {epoch + 1}/{self.num_epochs} completed. Loss: {epoch_loss / len(self.dataloader)}")

# Main Execution
if __name__ == "__main__":
    ahub_base_dir = "/home/work/VisionAI/CLAHE_train/aihub"
    mura_base_dir = "/home/work/VisionAI/CLAHE_train/MURA"
    image_size = 256
    batch_size = 32
    learning_rate = 3e-4
    num_epochs = 5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project="SimCLR-Augmentation", config={
        "image_size": image_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
    })

    all_data = load_ahub_data(ahub_base_dir) + load_mura_data(mura_base_dir)


    data_augmentation = DataAugmentation(image_size=image_size)
    combined_dataset = CombinedDataset(data=all_data, image_size=image_size, augmentation=data_augmentation)

    resnet = resnet50(pretrained=False)
    resnet.fc = nn.Identity()  # Remove final FC layer
    projection_head = ProjectionHead(input_dim=2048, hidden_dim=512, output_dim=128)
    simclr_model = SimCLRModel(encoder=resnet, projection_head=projection_head)

    trainer = SimCLRTrainer(
        net=simclr_model,
        image_size=image_size,
        learning_rate=learning_rate,
        dataset=combined_dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,
        checkpoint_folder="./checkpoints_combined",
        device=device
    )
    trainer()