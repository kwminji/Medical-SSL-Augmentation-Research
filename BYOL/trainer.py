from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import Module
from torch.nn import SyncBatchNorm

from torch.optim import Optimizer, Adam
from torch.utils.data import Dataset, DataLoader

from byol_pytorch import BYOL

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


# constants

DEFAULT_DDP_KWARGS = DistributedDataParallelKwargs(
    find_unused_parameters=True
)

# functions

def exists(v):
    return v is not None

def cycle(dl):
    while True:
        for batch in dl:
            yield batch


# Dataset class for AIHUB and MURA

class CombinedDataset(Dataset):
    def __init__(self, data, image_size):
        self.data = data
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]['image_path']
        label = self.data[idx]['label']
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image


# Data loading functions

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


# main trainer

class BYOLTrainer(Module):
    @beartype
    def __init__(
        self,
        net: Module,
        *,
        image_size: int,
        hidden_layer: str,
        learning_rate: float,
        dataset: Dataset,
        num_epochs : int,
        batch_size: int = 32,
        optimizer_klass=Adam,
        checkpoint_every: int = 1000,
        checkpoint_folder: str = './checkpoints_rotation_flip_erasing_256',
        byol_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
        accelerator_kwargs: dict = dict(),
    ):
        super().__init__()

        if 'kwargs_handlers' not in accelerator_kwargs:
            accelerator_kwargs['kwargs_handlers'] = [DEFAULT_DDP_KWARGS]

        self.accelerator = Accelerator(**accelerator_kwargs)

        if dist.is_initialized() and dist.get_world_size() > 1:
            net = SyncBatchNorm.convert_sync_batchnorm(net)

        self.net = net

        self.byol = BYOL(net, image_size=image_size, hidden_layer=hidden_layer, **byol_kwargs)

        self.optimizer = optimizer_klass(self.byol.parameters(), lr=learning_rate, **optimizer_kwargs)

        self.dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

        self.num_epochs = num_epochs
        self.steps_per_epoch = len(self.dataloader)

        self.checkpoint_every = checkpoint_every
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok=True, parents=True)
        assert self.checkpoint_folder.is_dir()

        # prepare with accelerate

        (
            self.byol,
            self.optimizer,
            self.dataloader
        ) = self.accelerator.prepare(
            self.byol,
            self.optimizer,
            self.dataloader
        )

        self.register_buffer('step', torch.tensor(0))

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def print(self, msg):
        return self.accelerator.print(msg)

    def forward(self):
            step = self.step.item()
            data_it = cycle(self.dataloader)

            for epoch in range(self.num_epochs):  
                epoch_loss = 0  
                for _ in range(self.steps_per_epoch):  
                    images = next(data_it)

                    with self.accelerator.autocast():
                        loss = self.byol(images)
                        self.accelerator.backward(loss)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.byol.update_moving_average()

                    epoch_loss += loss.item()  
                    step += 1

                    wandb.log({"step_loss": loss.item(), "step": step})

                    if not (step % self.checkpoint_every) and self.accelerator.is_main_process:
                        checkpoint_num = step // self.checkpoint_every
                        checkpoint_path = self.checkpoint_folder / f'checkpoint.{checkpoint_num}.pt'
                        torch.save(self.net.state_dict(), str(checkpoint_path))

                epoch_loss /= self.steps_per_epoch
                self.print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss:.4f}")

                wandb.log({"epoch_loss": epoch_loss, "epoch": epoch + 1})

            self.print('training complete')


# Main execution

if __name__ == "__main__":
    ahub_base_dir = "/home/work/VisionAI/CLAHE_train/aihub"
    mura_base_dir = "/home/work/VisionAI/CLAHE_train/MURA"

    # Hyperparameters
    image_size = 256
    batch_size = 32
    learning_rate = 3e-4
    num_epochs = 5

    # Initialize wandb
    wandb.init(
        project="BYOL-Combined", 
        config={
            "image_size": image_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
        }
    )

    # Load and data
    all_data = load_ahub_data(ahub_base_dir) + load_mura_data(mura_base_dir)
    print(f"전체 데이터 샘플 수 : {len(all_data)}개")

    # Create a dataset with all the data
    combined_dataset = CombinedDataset(data=all_data, image_size=image_size)

    # Load ResNet backbone
    resnet = resnet50(pretrained=True)

    # Initialize trainer
    trainer = BYOLTrainer(
        net=resnet,
        image_size=image_size,
        hidden_layer="avgpool",
        learning_rate=learning_rate,
        dataset=combined_dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,
        checkpoint_every=1000,
        checkpoint_folder='./checkpoints_rotation_gaussian_chessboard'
    )

    # Train the model
    trainer()

