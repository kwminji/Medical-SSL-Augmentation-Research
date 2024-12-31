import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from model import resnet50 as resnet  # ResNet50 모델을 사용하도록 명시적으로 변경

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        # Register hooks
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

        # Backward pass for the target class
        target = output[:, target_class]
        target.backward()

        # Compute Grad-CAM
        gradients = self.gradients.detach().cpu().numpy()
        activations = self.activations.detach().cpu().numpy()

        weights = np.mean(gradients, axis=(2, 3))  # Global average pooling
        cam = np.zeros(activations.shape[2:], dtype=np.float32)

        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]

        cam = np.maximum(cam, 0)  # ReLU
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
    # Preprocess the image
    input_tensor = preprocess_image(image_path, image_size).to(device)

    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer)

    # Generate CAM
    cam = gradcam.generate(input_tensor, target_class)

    # Load the original image
    image = Image.open(image_path).convert("RGB")
    image = np.array(image.resize((image_size, image_size)))
    image = image / 255.0  # Normalize to [0, 1]

    # Overlay CAM on the image
    gradcam_image = apply_colormap_on_image(image, cam)
    return gradcam_image

if __name__ == "__main__":
    # 설정
    image_path = "sample_image.jpg"  # 테스트할 이미지 경로
    target_class = 1  # 관심 클래스 (예: Abnormal = 1)
    image_size = 224  # ResNet의 입력 크기
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    model = resnet(weights="IMAGENET1K_V1").to(device)  # Pretrained weights 로드
    model.eval()

    # Grad-CAM 실행
    target_layer = model.layer4[-1]  # ResNet50의 마지막 Conv 레이어
    gradcam_image = visualize_gradcam(image_path, model, target_layer, target_class, image_size, device)

    # 결과 저장
    output_path = "gradcam_result.jpg"
    cv2.imwrite(output_path, gradcam_image)
    print(f"Grad-CAM 결과가 {output_path}에 저장되었습니다.")
