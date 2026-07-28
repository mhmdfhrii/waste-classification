import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from resnet_model import create_model


# =====================================================
# CONFIG
# =====================================================

SEED = 42

DATASET_PATH = "../dataset/TrashNet"
MODEL_PATH = "../models/resnet_best.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16


# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE")
print("=" * 60)
print(device)


# =====================================================
# TRANSFORM
# =====================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# =====================================================
# LOAD DATASET
# =====================================================

dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=transform
)

classes = dataset.classes

targets = np.array(
    dataset.targets
)

indices = np.arange(
    len(dataset)
)


# =====================================================
# SAME VALIDATION SPLIT
# =====================================================

_, val_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=SEED,
    stratify=targets
)


val_dataset = Subset(
    dataset,
    val_indices
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)


# =====================================================
# DATASET INFO
# =====================================================

print("\n" + "=" * 60)
print("DATASET")
print("=" * 60)

print("Classes:")
print(classes)

print("\nValidation Images:")
print(len(val_dataset))


# =====================================================
# LOAD MODEL
# =====================================================

model = create_model(
    num_classes=len(classes)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)

model.eval()


print("\n" + "=" * 60)
print("MODEL")
print("=" * 60)

print("Loaded:")
print(MODEL_PATH)


# =====================================================
# PREDICTION
# =====================================================

all_labels = []
all_predictions = []


with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# =====================================================
# ACCURACY
# =====================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)


print("\n" + "=" * 60)
print("ACCURACY")
print("=" * 60)

print(
    f"Validation Accuracy : "
    f"{accuracy * 100:.2f}%"
)


# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=classes,
    digits=4
)

print(report)


# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# =====================================================
# PLOT CONFUSION MATRIX
# =====================================================

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=classes,
    yticklabels=classes,
    cmap="Blues"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.title(
    "ResNet18 Confusion Matrix"
)

plt.tight_layout()


# =====================================================
# SAVE CONFUSION MATRIX
# =====================================================

output_path = "../models/resnet_confusion_matrix.png"

plt.savefig(
    output_path,
    dpi=150
)

plt.show()


# =====================================================
# FINAL
# =====================================================

print("\n" + "=" * 60)
print("EVALUATION FINISHED")
print("=" * 60)

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Confusion Matrix : {output_path}"
)

print("=" * 60)