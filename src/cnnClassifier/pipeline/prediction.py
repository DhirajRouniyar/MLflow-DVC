import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
from cnnClassifier.entity.config_entity import EvaluationConfig

class PredictionPipeline:
    def __init__(self, num_class, filename):
        
        self.num_classes = num_class
        self.filename = filename
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def predict(self):
        # Load model
        model_path = os.path.join("model", "model.pt")
        model = models.vgg16(weights=None)

        in_features = 512 * 7 * 7
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, self.num_classes)
        )

        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        # Image preprocessing
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Load and preprocess image
        img = Image.open(self.filename).convert('RGB')
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            output = model(input_batch)
            _, predicted = torch.max(output, 1)
            result = predicted.item()

        # Interpret result
        class_labels = ['Normal', 'Tumor']  # Adjust if label ordering is different
        prediction = class_labels[result]

        return [{"image": prediction}]
