import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

SEED = 42

DATASET_PATH = "dataset/TrashNet"

MODEL_PATH = "models/resnet_v2_best.pth"

IMAGE_SIZE = 224

BATCH_SIZE = 16

NUM_EPOCHS = 30

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

PATIENCE = 7

NUM_WORKERS = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("RESNET18 V2 TRAINING")
print("=" * 60)

print("Device :", device)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        10
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
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


val_transform = transforms.Compose([

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

train_dataset_full = datasets.ImageFolder(
    DATASET_PATH,
    transform=train_transform
)

val_dataset_full = datasets.ImageFolder(
    DATASET_PATH,
    transform=val_transform
)


class_names = train_dataset_full.classes

num_classes = len(
    class_names
)

targets = np.array(
    train_dataset_full.targets
)

indices = np.arange(
    len(train_dataset_full)
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_indices, val_indices = train_test_split(

    indices,

    test_size=0.2,

    random_state=SEED,

    stratify=targets
)


train_dataset = Subset(
    train_dataset_full,
    train_indices
)


val_dataset = Subset(
    val_dataset_full,
    val_indices
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=False
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=False
)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\nClasses:")

for i, name in enumerate(class_names):

    print(
        f"{i} : {name}"
    )


print("\nTotal images :", len(indices))

print(
    "Training     :",
    len(train_dataset)
)

print(
    "Validation   :",
    len(val_dataset)
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

train_targets = targets[
    train_indices
]


class_counts = np.bincount(
    train_targets,
    minlength=num_classes
)


print("\nClass distribution:")

for i, count in enumerate(
    class_counts
):

    print(
        f"{class_names[i]:<12} : {count}"
    )


# ------------------------------------------------------------
# Weight formula
# ------------------------------------------------------------

total_samples = len(
    train_targets
)


class_weights = (
    total_samples
    /
    (
        num_classes
        *
        class_counts
    )
)


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
)


class_weights = class_weights.to(
    device
)


print("\nClass weights:")

for name, weight in zip(
    class_names,
    class_weights
):

    print(
        f"{name:<12} : "
        f"{weight.item():.4f}"
    )


# ============================================================
# MODEL
# ============================================================

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)


# ------------------------------------------------------------
# Freeze early layers
# ------------------------------------------------------------

for param in model.parameters():

    param.requires_grad = False


# ------------------------------------------------------------
# Unfreeze layer3 + layer4
# ------------------------------------------------------------

for param in model.layer3.parameters():

    param.requires_grad = True


for param in model.layer4.parameters():

    param.requires_grad = True


# ------------------------------------------------------------
# Replace classifier
# ------------------------------------------------------------

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


model = model.to(
    device
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.AdamW(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=2
)


# ============================================================
# TRAINING VARIABLES
# ============================================================

best_val_accuracy = 0.0

patience_counter = 0


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(
    NUM_EPOCHS
):

    print(
        f"\nEpoch "
        f"{epoch + 1}/{NUM_EPOCHS}"
    )

    print("-" * 60)


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    train_loss = 0.0

    train_correct = 0

    train_total = 0


    train_bar = tqdm(
        train_loader,
        desc="Training"
    )


    for images, labels in train_bar:

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        train_loss += (
            loss.item()
            *
            images.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        train_correct += (
            predictions == labels
        ).sum().item()


        train_total += (
            labels.size(0)
        )


        train_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    train_loss /= train_total

    train_accuracy = (
        train_correct
        /
        train_total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        val_bar = tqdm(
            val_loader,
            desc="Validation"
        )


        for images, labels in val_bar:

            images = images.to(
                device
            )

            labels = labels.to(
                device
            )


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            val_loss += (
                loss.item()
                *
                images.size(0)
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            val_correct += (
                predictions == labels
            ).sum().item()


            val_total += (
                labels.size(0)
            )


    val_loss /= val_total

    val_accuracy = (
        val_correct
        /
        val_total
    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler.step(
        val_accuracy
    )


    current_lr = optimizer.param_groups[
        0
    ]["lr"]


    # ========================================================
    # PRINT METRICS
    # ========================================================

    print()

    print(
        f"Train Loss : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Acc  : "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss   : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Acc    : "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Learning Rate : "
        f"{current_lr:.8f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )

        patience_counter = 0


        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


        print(
            f"\n🔥 New Best Model!"
        )

        print(
            f"Best Validation Accuracy : "
            f"{best_val_accuracy * 100:.2f}%"
        )


    else:

        patience_counter += 1


        print(
            f"Early stopping: "
            f"{patience_counter}/{PATIENCE}"
        )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if patience_counter >= PATIENCE:

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# FINISHED
# ============================================================

print("\n")

print("=" * 60)

print(
    "RESNET18 V2 TRAINING FINISHED"
)

print("=" * 60)

print(
    f"Best Validation Accuracy : "
    f"{best_val_accuracy * 100:.2f}%"
)

print()

print(
    "Best Model :"
)

print(
    MODEL_PATH
)

print("=" * 60)