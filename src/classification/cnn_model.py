"""
Convolutional Neural Network Architecture for Apple Quality Classification
PyTorch-based multi-class classifier with batch normalization and max-pooling.
"""

from typing import Tuple, List, Optional
import os
import pathlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class ConvNet(nn.Module):
    """
    3-Stage Convolutional Neural Network for Fruit Quality Grading.
    Input size: (batch_size, 3, 150, 150)
    """
    def __init__(self, num_classes: int = 5):
        super(ConvNet, self).__init__()

        # Layer 1: 3 -> 12 channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)  # 150x150 -> 75x75

        # Layer 2: 12 -> 20 channels
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # Layer 3: 20 -> 32 channels
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()

        # Fully connected layer
        self.fc = nn.Linear(in_features=32 * 75 * 75, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.pool(out)
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.bn3(self.conv3(out)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def get_default_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Returns training and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return train_transform, val_transform


def train_cnn(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 0.001,
    device: Optional[torch.device] = None
) -> List[float]:
    """Trains the ConvNet model over specified epochs and tracks loss."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    loss_history = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        loss_history.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    return loss_history


def evaluate_cnn(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[torch.device] = None
) -> dict:
    """Evaluates model performance and computes confusion matrix."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())

    acc = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, output_dict=True)

    return {
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
