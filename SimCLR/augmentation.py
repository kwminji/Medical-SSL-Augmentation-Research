from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import random
import torch.nn as nn
import torch

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)
# SimCLR 데이터 증강 클래스
class DataAugmentation:
    def __init__(self, image_size: int):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(10), 
            transforms.RandomHorizontalFlip(p=0.5),
            #RandomApply(transforms.GaussianBlur(kernel_size=(3, 3), sigma=(1.0, 2.0)), p=0.2), 
            self.random_grid_distortion(),
            transforms.ToTensor(),  # 텐서로 변환
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),  # 정규화
        ])

    def random_grid_distortion(self):
        """체스보드 패턴을 활용한 그리드 왜곡 기법 (ElasticTransform과 유사)"""
        def apply_distortion(img):
            img = np.array(img)
            h, w = img.shape[:2]
            grid_size = 20  
            num_x = w // grid_size
            num_y = h // grid_size

            # 그리드 변형을 위한 임의의 이동
            grid_distortions = np.random.uniform(-5, 5, size=(num_y, num_x, 2))  # X, Y 방향으로 임의 변형

            for i in range(num_y):
                for j in range(num_x):
                    x_start = j * grid_size
                    y_start = i * grid_size
                    x_end = min((j + 1) * grid_size, w)
                    y_end = min((i + 1) * grid_size, h)

                    distortion = grid_distortions[i, j]
                    img[y_start:y_end, x_start:x_end] = np.roll(img[y_start:y_end, x_start:x_end], int(distortion[0]), axis=1)
                    img[y_start:y_end, x_start:x_end] = np.roll(img[y_start:y_end, x_start:x_end], int(distortion[1]), axis=0)

            return Image.fromarray(img)
        return apply_distortion

    def random_erasing(self, img):
        """이미지에서 Random Erasing을 적용하는 함수"""
        if random.random() > 0.3:
            return img  # 30% 확률로 적용 안 함

        # 이미지 크기 가져오기
        w, h = img.size
        area = w * h

        # Erasing 범위 설정
        min_area = max(1, int(area * 0.01))  # 최소 면적은 1 이상
        max_area = max(1, int(area * 0.1))
        erase_area = random.randint(min_area, max_area)

        aspect_ratio = random.uniform(0.5, 2.0)
        erase_w = min(w, int((erase_area * aspect_ratio) ** 0.5))
        erase_h = min(h, max(1, erase_area // erase_w))  # 최소 높이는 1 이상

        if erase_w > w or erase_h > h:  # Erasing 크기가 이미지보다 크지 않도록 방지
            return img

        # 랜덤 위치 선택
        top = random.randint(0, h - erase_h)
        left = random.randint(0, w - erase_w)

        # 검정색 사각형을 이미지에 덮어 씌우기
        draw = ImageDraw.Draw(img)
        draw.rectangle([left, top, left + erase_w, top + erase_h], fill=(0, 0, 0))
        return img

    def __call__(self, img):
        #img = self.random_erasing(img)
        return self.transform(img)
