import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

from resnet_model import create_model


# =====================================================
# CONFIG
# =====================================================

SEED = 42

DATASET_PATH = "../dataset/TrashNet"
MODEL_PATH = "../models"

IMAGE_SIZE = 224
BATCH_SIZE = 16

EPOCHS = 30

LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.0001

PATIENCE = 7


# =====================================================
# CREATE MODEL DIRECTORY
# =====================================================

os.makedirs(
    MODEL_PATH,
    exist_ok=True
)


# =====================================================
# RANDOM SEED
# =====================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("DEVICE")
print("=" * 60)

print(device)


# =====================================================
# TRAIN TRANSFORM
# =====================================================

train_transform = transforms.Compose([

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.8, 1.0)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        15
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


# =====================================================
# VALIDATION TRANSFORM
# =====================================================

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


# =====================================================
# LOAD DATASET
# =====================================================

base_dataset = datasets.ImageFolder(
    DATASET_PATH
)


classes = base_dataset.classes

targets = np.array(
    base_dataset.targets
)

indices = np.arange(
    len(base_dataset)
)


# =====================================================
# PRINT DATASET
# =====================================================

print("\n" + "=" * 60)
print("DATASET")
print("=" * 60)

print("Classes:")
print(classes)

print("\nClass mapping:")
print(base_dataset.class_to_idx)

print("\nTotal images:")
print(len(base_dataset))


# =====================================================
# TRAIN / VALIDATION SPLIT
# =====================================================

train_indices, val_indices = train_test_split(

    indices,

    test_size=0.2,

    random_state=SEED,

    stratify=targets
)


print("\nTraining images:")
print(len(train_indices))

print("\nValidation images:")
print(len(val_indices))


# =====================================================
# CREATE DATASETS
# =====================================================

train_full = datasets.ImageFolder(
    DATASET_PATH,
    transform=train_transform
)


val_full = datasets.ImageFolder(
    DATASET_PATH,
    transform=val_transform
)


train_dataset = Subset(
    train_full,
    train_indices
)


val_dataset = Subset(
    val_full,
    val_indices
)


# =====================================================
# DATALOADER
# =====================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,

    pin_memory=False
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=False
)


# =====================================================
# CREATE MODEL
# =====================================================

model = create_model(
    num_classes=len(classes)
)


model = model.to(device)


print("\n" + "=" * 60)
print("MODEL")
print("=" * 60)

print("ResNet18")

print(
    f"Number of classes: {len(classes)}"
)


# =====================================================
# LOSS
# =====================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)


# =====================================================
# OPTIMIZER
# =====================================================

optimizer = AdamW(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# =====================================================
# SCHEDULER
# =====================================================

scheduler = ReduceLROnPlateau(

    optimizer,

    mode="max",

    factor=0.5,

    patience=2
)


# =====================================================
# HISTORY
# =====================================================

train_losses = []
val_losses = []

train_accs = []
val_accs = []


# =====================================================
# BEST MODEL
# =====================================================

best_accuracy = 0.0

patience_counter = 0

best_model_path = os.path.join(
    MODEL_PATH,
    "resnet_best.pth"
)


# =====================================================
# TRAINING LOOP
# =====================================================

print("\n" + "=" * 60)
print("START TRAINING")
print("=" * 60)


for epoch in range(EPOCHS):

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )


    # =================================================
    # TRAIN
    # =================================================

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0


    train_bar = tqdm(
        train_loader,
        desc="Training"
    )


    for images, labels in train_bar:

        images = images.to(device)

        labels = labels.to(device)


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


        running_loss += loss.item()


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()


        train_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    train_loss = (
        running_loss /
        len(train_loader)
    )


    train_accuracy = (
        correct /
        total *
        100
    )


    # =================================================
    # VALIDATION
    # =================================================

    model.eval()

    running_val_loss = 0.0

    val_correct = 0
    val_total = 0


    with torch.no_grad():

        val_bar = tqdm(
            val_loader,
            desc="Validation"
        )


        for images, labels in val_bar:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            running_val_loss += (
                loss.item()
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            val_total += labels.size(0)

            val_correct += (
                predictions == labels
            ).sum().item()


    val_loss = (
        running_val_loss /
        len(val_loader)
    )


    val_accuracy = (
        val_correct /
        val_total *
        100
    )


    # =================================================
    # HISTORY
    # =================================================

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    train_accs.append(
        train_accuracy
    )

    val_accs.append(
        val_accuracy
    )


    # =================================================
    # SCHEDULER
    # =================================================

    scheduler.step(
        val_accuracy
    )


    # =================================================
    # PRINT RESULT
    # =================================================

    print()

    print(
        f"Train Loss : {train_loss:.4f}"
    )

    print(
        f"Train Acc  : {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss   : {val_loss:.4f}"
    )

    print(
        f"Val Acc    : {val_accuracy:.2f}%"
    )

    print(
        "Learning Rate : "
        f"{optimizer.param_groups[0]['lr']:.8f}"
    )


    # =================================================
    # SAVE BEST MODEL
    # =================================================

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        patience_counter = 0


        torch.save(
            model.state_dict(),
            best_model_path
        )


        print(
            "✅ Best model saved!"
        )


    else:

        patience_counter += 1


        print(
            f"Early stopping: "
            f"{patience_counter}/{PATIENCE}"
        )


    # =================================================
    # EARLY STOPPING
    # =================================================

    if patience_counter >= PATIENCE:

        print(
            "\nEarly stopping triggered."
        )

        break


# =====================================================
# SAVE LAST MODEL
# =====================================================

last_model_path = os.path.join(
    MODEL_PATH,
    "resnet_last.pth"
)


torch.save(
    model.state_dict(),
    last_model_path
)


# =====================================================
# FINAL RESULT
# =====================================================

print("\n" + "=" * 60)
print("TRAINING FINISHED")
print("=" * 60)

print(
    f"Best Validation Accuracy : "
    f"{best_accuracy:.2f}%"
)

print()

print(
    f"Best Model : "
    f"{best_model_path}"
)

print(
    f"Last Model : "
    f"{last_model_path}"
)

print("=" * 60)