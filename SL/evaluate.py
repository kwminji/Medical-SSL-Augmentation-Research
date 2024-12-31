import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
from dataset import LinearEvaluationDataset
from data_loader import load_ahub_data, load_mura_data
from sklearn.metrics import classification_report, accuracy_score
import wandb
import os

# Logistic Regression 정의
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

# 모델 평가 함수
def evaluate_model(checkpoint_path, valid_loader, num_classes, device):
    """
    Evaluate the model using validation data and calculate various metrics.
    """
    print(f"Evaluating checkpoint: {checkpoint_path}")
    # ResNet50 Backbone 설정
    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    resnet = nn.Sequential(*list(resnet.children())[:-1])  # 마지막 FC 제거
    resnet = resnet.to(device)
    resnet.eval()

    # Logistic Regression Layer
    logreg = LogisticRegression(input_dim=2048, output_dim=num_classes).to(device)
    logreg.load_state_dict(torch.load(checkpoint_path, map_location=device))
    logreg.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)

            # ResNet50 Backbone에서 Feature 추출
            features = resnet(images).squeeze(-1).squeeze(-1)

            # Logistic Regression Prediction
            outputs = logreg(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # 예측 값 저장
            predictions = torch.argmax(outputs, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 평균 손실 계산
    avg_loss = total_loss / len(valid_loader)

    # 평가 지표 계산
    accuracy = accuracy_score(all_labels, all_preds) * 100
    report = classification_report(all_labels, all_preds, target_names=["Abnormal", "Normal"], output_dict=True)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1_score": report["weighted avg"]["f1-score"]
    }

if __name__ == "__main__":
    # 검증 데이터 디렉토리
    valid_ahub_dir = "/home/work/VisionAI/CLAHE_valid/aihub/02.라벨링데이터"
    valid_mura_dir = "/home/work/VisionAI/CLAHE_valid/MURA"

    # Device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    image_size = 256
    batch_size = 32
    num_classes = 2  # Binary classification
    checkpoint_dir = "./checkpoints/"  # 학습된 모델 체크포인트 경로

    # wandb 초기화
    wandb.init(
        project="LogisticRegression-Eval",
        config={
            "batch_size": batch_size,
            "learning_rate": 3e-4,
        }
    )

    # 검증 데이터 로드
    print("Loading validation data...")
    valid_ahub_data = load_ahub_data(valid_ahub_dir)
    valid_mura_data = load_mura_data(valid_mura_dir)
    valid_data = valid_ahub_data + valid_mura_data

    # 검증 데이터셋 생성
    valid_dataset = LinearEvaluationDataset(data=valid_data, image_size=image_size)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # 체크포인트별 결과 저장
    for epoch in range(1, 6):  # 체크포인트 1~5까지 순차적으로 로드
        checkpoint_path = os.path.join(checkpoint_dir, f"logreg_epoch_{epoch}.pth")
        if os.path.exists(checkpoint_path):
            metrics = evaluate_model(checkpoint_path, valid_loader, num_classes, device)

            # 결과를 wandb에 로그
            wandb.log({
                "checkpoint_epoch": epoch,
                "validation_loss": metrics["loss"],
                "validation_accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
            })

    # wandb 종료
    wandb.finish()
