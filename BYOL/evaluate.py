import torch
import wandb
import numpy as np
from torchvision.models import resnet50
from PIL import Image
from torchvision import transforms
from sklearn import preprocessing
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler
from trainer import load_ahub_data, load_mura_data
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


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)


def load_encoder(checkpoint_path, device):
    resnet = resnet50(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    resnet.load_state_dict(checkpoint)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove projection head
    resnet = resnet.to(device)
    resnet.eval()
    return resnet


def get_features_from_encoder(encoder, loader, device):
    x_features, y_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            features = encoder(x)
        x_features.extend(features.squeeze(-1).squeeze(-1).cpu())
        y_labels.extend(y.cpu())

    return torch.stack(x_features), torch.tensor(y_labels)


def create_data_loaders_from_arrays(X_train, y_train, X_valid, y_valid):
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    return train_loader, valid_loader


def evaluate_model(encoder_checkpoint, train_loader, valid_loader, num_classes, device, num_epochs=100, eval_every=1):

    wandb.init(
        project="BYOL-LogisticRegression-Eval",
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
    print("Extracting features from validation data...")
    x_valid, y_valid = get_features_from_encoder(encoder, valid_loader, device)

    print(f"Feature shapes - Train: {x_train.shape}, Valid: {x_valid.shape}")

    if len(x_train.shape) > 2:  # Global Average Pooling
        x_train = torch.mean(x_train, dim=[2, 3])
        x_valid = torch.mean(x_valid, dim=[2, 3])

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train.numpy())
    x_valid = scaler.transform(x_valid.numpy())

    x_train = torch.from_numpy(x_train).float()
    x_valid = torch.from_numpy(x_valid).float()

    train_loader, valid_loader = create_data_loaders_from_arrays(x_train, y_train, x_valid, y_valid)

    input_dim = x_train.shape[1]
    logreg = LogisticRegression(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(logreg.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

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

        wandb.log({"epoch": epoch, "train_loss": total_loss / len(train_loader)})

        if epoch % eval_every == 0:
            logreg.eval()
            total, correct = 0, 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for x_batch, y_batch in valid_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    outputs = logreg(x_batch)
                    predictions = torch.argmax(outputs, dim=1)

                    all_preds.extend(predictions.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())
                    total += y_batch.size(0)
                    correct += (predictions == y_batch).sum().item()

            # Calculate metrics
            accuracy = 100 * correct / total
            precision = precision_score(all_labels, all_preds, average="weighted")
            recall = recall_score(all_labels, all_preds, average="weighted")
            f1 = f1_score(all_labels, all_preds, average="weighted")

            print(f"Epoch {epoch}, Validation Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

            wandb.log({
                "epoch": epoch,
                "validation_accuracy": accuracy,
                "validation_precision": precision,
                "validation_recall": recall,
                "validation_f1": f1
            })

    wandb.finish()


if __name__ == "__main__":
    checkpoint_path = "./checkpoints_rotation_gaussian_chessboard/checkpoint.18.pt"
    train_ahub_base_dir = "/home/work/VisionAI/CLAHE_train/aihub"
    train_mura_base_dir = "/home/work/VisionAI/CLAHE_train/MURA"
    valid_ahub_base_dir = "/home/work/VisionAI/CLAHE_valid/aihub"
    valid_mura_base_dir = "/home/work/VisionAI/CLAHE_valid/MURA"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = 256
    batch_size = 32
    num_classes = 2
    num_epochs = 100
    eval_every = 1

    train_data = load_ahub_data(train_ahub_base_dir) + load_mura_data(train_mura_base_dir)
    valid_data = load_ahub_data(valid_ahub_base_dir) + load_mura_data(valid_mura_base_dir)

    train_dataset = LinearEvaluationDataset(data=train_data, image_size=image_size)
    valid_dataset = LinearEvaluationDataset(data=valid_data, image_size=image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    evaluate_model(checkpoint_path, train_loader, valid_loader, num_classes, device, num_epochs, eval_every)
