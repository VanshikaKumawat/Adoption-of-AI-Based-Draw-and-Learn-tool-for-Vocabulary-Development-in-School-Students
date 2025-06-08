import torch
import torch.nn as nn
from torchvision import models

# ----------------------------
# Model Architecture
# ----------------------------

class SketchClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SketchClassifier, self).__init__()
        self.base_model = models.vgg16(weights='IMAGENET1K_V1')  # Load pretrained
        for param in self.base_model.parameters():
            param.requires_grad = False  # Freeze all layers

        # Replace the classifier to suit sketch classification
        self.base_model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.base_model(x)

# ----------------------------
# Utility Functions
# ----------------------------

def get_model(num_classes, device='cpu'):
    """Return a new model with the given number of output classes."""
    model = SketchClassifier(num_classes)
    return model.to(device)

def load_model(model_path, num_classes, device='cpu'):
    """Load model weights from file."""
    model = get_model(num_classes, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def save_model(model, path):
    """Save the model to disk."""
    torch.save(model.state_dict(), path)
