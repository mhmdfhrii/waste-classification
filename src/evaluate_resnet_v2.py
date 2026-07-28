import os

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = "../dataset/TrashNet"

MODEL_PATH = "../models/resnet_v2_best.pth"

IMAGE_SIZE = 224

BATCH_SIZE = 16

SEED = 42

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 60)
print("RESNET18 V2 EVALUATION")
print("=" * 60)

print("Device :", device)


# ============================================================
# TRANSFORM
# ============================================================

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


# ============================================================
# DATASET
# ============================================================

dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=transform
)


class_names = dataset.classes

num_classes = len(
    class_names
)

targets = np.array(
    dataset.targets
)

indices = np.arange(
    len(dataset)
)


print("\nClasses:")

for i, name in enumerate(
    class_names
):

    print(
        f"{i} : {name}"
    )


# ============================================================
# SAME VALIDATION SPLIT
# ============================================================

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

    num_workers=NUM_WORKERS,

    pin_memory=False
)


print("\nValidation images :", len(val_dataset))


# ============================================================
# MODEL
# ============================================================

model = models.resnet18(
    weights=None
)


# ============================================================
# FREEZE / UNFREEZE
# ============================================================

for param in model.parameters():

    param.requires_grad = False


for param in model.layer3.parameters():

    param.requires_grad = True


for param in model.layer4.parameters():

    param.requires_grad = True


# ============================================================
# SAME CLASSIFIER AS TRAINING V2
# ============================================================

num_features = (
    model.fc.in_features
)


model.fc = nn.Sequential(

    nn.Dropout(
        p=0.4
    ),

    nn.Linear(
        num_features,
        128
    ),

    nn.ReLU(),

    nn.Dropout(
        p=0.3
    ),

    nn.Linear(
        128,
        num_classes
    )
)


# ============================================================
# LOAD V2 MODEL
# ============================================================

model.load_state_dict(

    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

)


model = model.to(
    device
)


model.eval()


print("\nModel loaded successfully!")

print(
    "Model path :",
    MODEL_PATH
)


# ============================================================
# PREDICTION
# ============================================================

all_predictions = []

all_labels = []


with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(
            device
        )

        outputs = model(
            images
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(

    all_labels,

    all_predictions

)


print("\n")

print("=" * 60)

print(
    f"Validation Accuracy : "
    f"{accuracy * 100:.2f}%"
)

print("=" * 60)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")

print("=" * 60)


print(

    classification_report(

        all_labels,

        all_predictions,

        target_names=class_names,

        digits=4

    )

)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    all_labels,

    all_predictions

)


print("\nCONFUSION MATRIX")

print("=" * 60)

print(cm)


# ============================================================
# PLOT
# ============================================================

plt.figure(

    figsize=(10, 8)

)


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=class_names,

    yticklabels=class_names

)


plt.title(
    "ResNet18 V2 Confusion Matrix"
)


plt.xlabel(
    "Predicted Label"
)


plt.ylabel(
    "True Label"
)


plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

os.makedirs(

    "../models",

    exist_ok=True

)


output_path = (

    "../models/"
    "resnet_v2_confusion_matrix.png"

)


plt.savefig(

    output_path,

    dpi=300,

    bbox_inches="tight"

)


plt.show()


print("\nConfusion matrix saved:")

print(output_path)