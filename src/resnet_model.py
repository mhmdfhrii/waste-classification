import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


def create_model(num_classes):

    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    return model