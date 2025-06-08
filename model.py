import torch
import torch.nn as nn
from torchvision import models

class SketchClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(SketchClassifier, self).__init__()
        # Load pretrained VGG16 model weights from torchvision
        if pretrained:
            weights = models.VGG16_Weights.IMAGENET1K_V1
            self.base_model = models.vgg16(weights=weights)
        else:
            self.base_model = models.vgg16(weights=None)

        # Freeze feature extractor layers
        for param in self.base_model.features.parameters():
            param.requires_grad = False

        # Optionally freeze first few classifier layers except the last
        # (adjust based on your training setup)
        for param in self.base_model.classifier[:-1].parameters():
            param.requires_grad = False

        # Replace the final fully connected layer for your number of classes
        in_features = self.base_model.classifier[6].in_features
        self.base_model.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

def get_model(num_classes, device='cpu', pretrained=True):
    """
    Initialize and return the SketchClassifier model on the specified device.
    """
    model = SketchClassifier(num_classes, pretrained=pretrained)
    return model.to(device)

def load_model(model_path, num_classes, device='cpu', pretrained=False):
    """
    Load model weights from a checkpoint file.
    The checkpoint is expected to contain the model's state_dict.
    """
    model = get_model(num_classes, device=device, pretrained=pretrained)
    checkpoint = torch.load(model_path, map_location=device)

    # If checkpoint is a dict with 'model_state_dict' key (from training script)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If raw state_dict saved
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def save_model(model, path):
    """
    Save only the model's state_dict to disk.
    """
    torch.save(model.state_dict(), path)
