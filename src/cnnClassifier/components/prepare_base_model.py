import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = None
        self.full_model = None

    def get_base_model(self):
        # Load pretrained VGG16
        self.model = models.vgg16(
            weights=models.VGG16_Weights.DEFAULT if self.config.params_weights == "imagenet" else None
        )

        # Remove classifier if include_top = False
        if not self.config.params_include_top:
            self.model.classifier = nn.Identity()

        # Save the base model
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        # Freeze layers
        if freeze_all:
            for param in model.parameters():
                param.requires_grad = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for param in list(model.parameters())[:-freeze_till]:
                param.requires_grad = False

        # Replace classifier head
        in_features = model.classifier[-1].in_features if isinstance(model.classifier, nn.Sequential) else 25088
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, classes),
            nn.Softmax(dim=1)
        )

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        print(model)
        return model, criterion, optimizer
    
    def update_base_model(self):
        self.full_model, self.criterion, self.optimizer = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: nn.Module):
        torch.save(model.state_dict(), path)