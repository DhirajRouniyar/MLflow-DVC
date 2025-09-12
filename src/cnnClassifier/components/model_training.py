import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision import models
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig
# from prepare_base_model_pytorch import PrepareBaseModel, PrepareBaseModelConfig
import os

class Training:
    def __init__(self, config: TrainingConfig, device=None):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def get_base_model(self):
        """Load the updated base model weights into VGG16 architecture"""
        # 1. Create same architecture
        model = models.vgg16(weights=None)
        
        # 2. Replace classifier with correct number of classes
        in_features = 512 * 7 * 7
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, self.config.params_classes)  # <- use params_classes
        )
        
        # 3. Load saved weights
        model.load_state_dict(torch.load(self.config.updated_base_model_path, map_location=self.device))
        
        # 4. Move to device
        model.to(self.device)
        self.model = model


    def train_valid_generator(self):
        """Create PyTorch datasets and dataloaders."""
        if self.config.params_is_augmentation:
            train_transform = transforms.Compose([
                transforms.RandomRotation(40),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(self.config.params_image_size[0]),
                transforms.ColorJitter(),
                transforms.ToTensor(),
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(self.config.params_image_size[:2]),
                transforms.ToTensor()
            ])

        valid_transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:2]),
            transforms.ToTensor()
        ])

        train_dataset = datasets.ImageFolder(
            root=self.config.training_data,
            transform=train_transform
        )

        valid_dataset = datasets.ImageFolder(
            root=self.config.training_data,
            transform=valid_transform
        )

        train_size = int(0.8 * len(train_dataset))
        valid_size = len(train_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

        self.train_loader = DataLoader(train_dataset, batch_size=self.config.params_batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.config.params_batch_size, shuffle=False)

    @staticmethod
    def save_model(path: Path, model: nn.Module):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)

    def train(self, learning_rate=0.001):
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate)

        for epoch in range(self.config.params_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = correct / total
            train_loss = running_loss / total

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in self.valid_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_acc = val_correct / val_total
            val_loss /= val_total

            print(f"Epoch [{epoch+1}/{self.config.params_epochs}] "
                  f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
                  f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

        self.save_model(self.config.trained_model_path, self.model)
        self.save_model(self.config.trained_model_2nd_path, self.model)