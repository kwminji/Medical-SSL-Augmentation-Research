import torch
import wandb
import numpy as np
from torchvision.models import resnet50
from PIL import Image
from torchvision import transforms
from sklearn import preprocessing
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler
from train_epoch import load_ahub_data, load_mura_data
from sklearn.metrics import precision_score, recall_score, f1_score
import os

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


# Define Logistic Regression model
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

def load_encoder(checkpoint_path, device):
    resnet = resnet50(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    resnet.load_state_dict(checkpoint['encoder'], strict=False)  # 'encoder' 키로 모델 파라미터 로드
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove FC layer

    resnet = resnet.to(device)
    resnet.eval()
    
    return resnet


# Feature extraction from the encoder
def get_features_from_encoder(encoder, loader, device):
    x_features, y_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            features = encoder(x)
        x_features.extend(features.squeeze(-1).squeeze(-1).cpu())
        y_labels.extend(y.cpu())

    return torch.stack(x_features), torch.tensor(y_labels)


# Data preparation for training and testing
def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

# Main evaluation function
def evaluate_model(encoder_checkpoint, train_loader, test_loader, num_classes, device, num_epochs=100, eval_every=1):

    # Initialize wandb
    wandb.init(
        project="SimCLR-DataAugmentation-Eval",
        config={
            "epochs": num_epochs,
            "batch_size": 32,
            "learning_rate": 3e-4,
            "eval_every": eval_every,
        }
    )

    encoder = load_encoder(encoder_checkpoint, device)

    print("Extracting features from training data...")
    x_train, y_train = get_features_from_encoder(encoder, train_loader, device)
    print("Extracting features from testing data...")
    x_test, y_test = get_features_from_encoder(encoder, test_loader, device)

    print(f"Feature shapes - Train: {x_train.shape}, Test: {x_test.shape}")

    if len(x_train.shape) > 2:  # Global Average Pooling
        x_train = torch.mean(x_train, dim=[2, 3])
        x_test = torch.mean(x_test, dim=[2, 3])

    # Normalize features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train.numpy())
    x_test = scaler.transform(x_test.numpy())

    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()

    # Create data loaders for logistic regression
    train_loader, test_loader = create_data_loaders_from_arrays(x_train, y_train, x_test, y_test)

    # Initialize Logistic Regression
    input_dim = x_train.shape[1]
    logreg = LogisticRegression(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(logreg.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Training and Evaluation
    for epoch in range(num_epochs):
        logreg.train()
        total_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = logreg(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Log training loss
        wandb.log({"epoch": epoch, "train_loss": total_loss / len(train_loader)})

        if epoch % eval_every == 0:
            logreg.eval()
            total, correct = 0, 0
            val_loss = 0.0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    outputs = logreg(x_batch)
                    predictions = torch.argmax(outputs, dim=1)

                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()

                    all_preds.extend(predictions.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())

                    total += y_batch.size(0)
                    correct += (predictions == y_batch).sum().item()

            accuracy = 100 * correct / total
            precision = precision_score(all_labels, all_preds, average="binary")
            recall = recall_score(all_labels, all_preds, average="binary")
            f1 = f1_score(all_labels, all_preds, average="binary")

            print(f"Epoch {epoch}, Validation Loss: {val_loss / len(test_loader):.4f}, Accuracy: {accuracy:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch,
                "val_loss": val_loss / len(test_loader),
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1_score": f1
            })

    wandb.finish()


if __name__ == "__main__":
    # Paths
    checkpoint_path = "/home/work/VisionAI/SimCLR/checkpoints_combined/R+B+C.epoch5.pt"
    train_ahub_base_dir = "/home/work/VisionAI/CLAHE_train/aihub"
    train_mura_base_dir = "/home/work/VisionAI/CLAHE_train/MURA"
    valid_ahub_base_dir = "/home/work/VisionAI/CLAHE_valid/aihub"
    valid_mura_base_dir = "/home/work/VisionAI/CLAHE_valid/MURA"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parameters
    image_size = 256
    batch_size = 32
    num_classes = 2  # Binary classification
    num_epochs = 100
    eval_every = 1

    # Load train and valid datasets
    train_data = load_ahub_data(train_ahub_base_dir) + load_mura_data(train_mura_base_dir)
    valid_data = load_ahub_data(valid_ahub_base_dir) + load_mura_data(valid_mura_base_dir)

    train_dataset = LinearEvaluationDataset(data=train_data, image_size=image_size)
    valid_dataset = LinearEvaluationDataset(data=valid_data, image_size=image_size)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model
    evaluate_model(checkpoint_path, train_loader, test_loader, num_classes, device, num_epochs, eval_every)
