import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import mlflow
import mlflow.pytorch
from urllib.parse import urlparse
from pathlib import Path
import json
import dagshub
from cnnClassifier.entity.config_entity import EvaluationConfig


class Evaluation:
    def __init__(self, config: EvaluationConfig, device=None):
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.valid_loader = None
        self.score = None
        self.avg_loss = None
        self.accuracy = None

    def _valid_loader(self):
        """Create validation DataLoader with split=0.3"""
        transform = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:2]),
            transforms.ToTensor(),
        ])

        full_dataset = datasets.ImageFolder(
            root=self.config.training_data,
            transform=transform
        )

        valid_size = int(0.3 * len(full_dataset))
        train_size = len(full_dataset) - valid_size
        _, valid_dataset = random_split(full_dataset, [train_size, valid_size])

        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config.params_batch_size,
            shuffle=False
        )

    @staticmethod
    def load_model(path: Path, num_classes: int, device="cpu"):
        """Recreate architecture and load trained weights"""
        model = models.vgg16(weights=None)
        in_features = 512 * 7 * 7
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, num_classes)
        )
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return model

    def evaluation(self):
        """Run evaluation and save score"""
        self.model = self.load_model(
            path=self.config.path_to_model,
            num_classes=self.config.params_classes,
            device=self.device
        )
        self._valid_loader()

        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in self.valid_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.avg_loss = total_loss / total
        self.accuracy = correct / total
        self.score = (self.avg_loss, self.accuracy)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        with open("scores.json", "w") as f:
            json.dump(scores, f, indent=4)

    def log_into_mlflow(self):
       # Connect to DagsHub MLflow
        dagshub.init(repo_owner="DhirajRouniyar", repo_name="MLflow-DVC", mlflow=True)

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({
                "loss": self.avg_loss,
                "accuracy": self.accuracy
            })

            # Log model to DagsHub
            mlflow.pytorch.log_model(self.model, "model", registered_model_name="TorchModel")