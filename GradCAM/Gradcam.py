import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet50
from torch import nn

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class):
        # Forward pass
        output = self.model(input_tensor)
        self.model.zero_grad()

        # Target 클래스에 대한 backward pass
        target = output[:, target_class]
        target.backward()

        # gradcam 계산하기 
        gradients = self.gradients.detach().cpu().numpy()
        activations = self.activations.detach().cpu().numpy()

        weights = np.mean(gradients, axis=(2, 3))  # Global average pooling
        cam = np.zeros(activations.shape[2:], dtype=np.float32)

        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]

        # ReLU
        cam = np.maximum(cam, 0)

        # 0으로 나누지 않도록 함 
        if np.max(cam) != 0:
            cam = cv2.resize(cam, input_tensor.shape[2:][::-1])
            cam -= np.min(cam)
            cam /= np.max(cam)

        return cam

def apply_colormap_on_image(image, cam, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    heatmap = np.float32(heatmap) / 255
    overlayed = heatmap + np.float32(image)
    overlayed = overlayed / np.max(overlayed)
    return np.uint8(255 * overlayed)

def preprocess_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def visualize_gradcam(image_path, model, target_layer, target_class, image_size, device):
    # 이미지를 전처리
    input_tensor = preprocess_image(image_path, image_size).to(device)

    # Grad-CAM 초기화
    gradcam = GradCAM(model, target_layer)

    # CAM 생성
    cam = gradcam.generate(input_tensor, target_class)

    # 원본 이미지 로드
    image = Image.open(image_path).convert("RGB")
    image = np.array(image.resize((image_size, image_size)))
    image = image / 255.0  # Normalize to [0, 1]

    # CAM을 이미지에 오버레이하기
    gradcam_image = apply_colormap_on_image(image, cam)
    return gradcam_image

if __name__ == "__main__":
    # 설정
    image_path = "/home/work/VisionAI/aihub/Train/01.원천데이터/Foot_Abnormal/Foot_Flat_Foot_00000017.png"  # 테스트할 이미지 경로
    target_class = 0  
    image_size = 226  # ResNet의 입력 크기
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드 (Pre-trained ResNet50)
    model = resnet50(weights=None)  # Pretrained weights are not used
    model.fc = nn.Linear(model.fc.in_features, 2)  # Adjust the number of output classes (2 for binary classification)
    
    # 체크포인트에서 선형 레이어 가중치 로드
    checkpoint_path = "/home/work/VisionAI/SL/SL_valid/checkpoints/logreg_epoch_5.pth"
    checkpoint = torch.load(checkpoint_path)
    
    # 모델의 fully connected 레이어(fc)에 가중치 로드
    model.fc.weight.data = checkpoint['linear.weight']  # Adjust this key if necessary
    model.fc.bias.data = checkpoint['linear.bias']  # Adjust this key if necessary

    # 모델을 디바이스로 이동
    model.to(device)
    model.eval()

    # Grad-CAM 실행
    target_layer = model.layer4[-1]  # ResNet50의 마지막 Conv 레이어
    gradcam_image = visualize_gradcam(image_path, model, target_layer, target_class, image_size, device)

    # 결과 저장
    output_path = "sl.jpg"
    cv2.imwrite(output_path, gradcam_image)
    print(f"Grad-CAM 결과가 {output_path}에 저장되었습니다.")
