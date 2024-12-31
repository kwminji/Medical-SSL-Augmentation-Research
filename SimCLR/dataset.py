import os
import cv2
import torch
from torch.utils.data import Dataset

class XRayDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]["image_path"]
        label = self.data[idx]["label"]



        # 파일 경로 확인
        if not os.path.exists(img_path):
            print(f"경고: 파일을 찾을 수 없습니다: {img_path}")
            raise FileNotFoundError(f"이미지 파일이 존재하지 않습니다: {img_path}")

        # 이미지 읽기
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"경고: 이미지를 로드할 수 없습니다: {img_path}")
            raise ValueError(f"이미지를 로드하지 못했습니다: {img_path}")

        # 이미지 크기 조정 및 정규화
        img = cv2.resize(img, (224, 224)) / 255.0
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # 채널 추가
        return img, torch.tensor(label, dtype=torch.float32) ####
