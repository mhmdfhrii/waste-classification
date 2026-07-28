import os
import copy
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from model import WasteCNN


# =====================================================
# CONFIG
# =====================================================

SEED = 42
BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

DATASET_PATH = "../dataset/TrashNet"
MODEL_PATH = "../models"

os.makedirs(MODEL_PATH, exist_ok=True)


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
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Device :", device)
print("=" * 60)


# =====================================================
# TRANSFORM
# =====================================================

train_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])


val_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])


# =====================================================
# DATASET
# =====================================================

full_dataset = datasets.ImageFolder(DATASET_PATH)

classes = full_dataset.classes

print(classes)

targets = full_dataset.targets

indices = np.arange(len(full_dataset))


train_idx, val_idx = train_test_split(

    indices,

    test_size=0.2,

    random_state=SEED,

    stratify=targets

)


train_dataset = copy.deepcopy(full_dataset)
val_dataset = copy.deepcopy(full_dataset)

train_dataset.transform = train_transform
val_dataset.transform = val_transform

train_dataset = Subset(
    train_dataset,
    train_idx
)

val_dataset = Subset(
    val_dataset,
    val_idx
)


print("Train :", len(train_dataset))
print("Validation :", len(val_dataset))


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
# MODEL
# =====================================================

model = WasteCNN(
    num_classes=len(classes)
).to(device)

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
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# =====================================================
# SCHEDULER
# =====================================================

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6
)

# =====================================================
# EARLY STOPPING
# =====================================================

best_acc = 0.0
patience = 7
counter = 0

# =====================================================
# HISTORY
# =====================================================

train_losses = []
val_losses = []

train_accs = []
val_accs = []

best_model_path = os.path.join(
    MODEL_PATH,
    "waste_cnn_best.pth"
)

last_model_path = os.path.join(
    MODEL_PATH,
    "waste_cnn_last.pth"
)

print("=" * 60)
print("Model Ready")
print(model)
print("=" * 60)

# =====================================================
# TRAINING
# =====================================================

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # -----------------------------
    # TRAIN
    # -----------------------------

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    train_bar = tqdm(
        train_loader,
        desc="Training",
        leave=False
    )

    for images, labels in train_bar:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (preds == labels).sum().item()

        train_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss = running_loss / len(train_loader)

    train_acc = 100 * correct / total

    train_losses.append(train_loss)

    train_accs.append(train_acc)

    # -----------------------------
    # VALIDATION
    # -----------------------------

    model.eval()

    running_val_loss = 0.0

    val_correct = 0
    val_total = 0

    val_bar = tqdm(
        val_loader,
        desc="Validation",
        leave=False
    )

    with torch.no_grad():

        for images, labels in val_bar:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_val_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (preds == labels).sum().item()

    val_loss = running_val_loss / len(val_loader)

    val_acc = 100 * val_correct / val_total

    val_losses.append(val_loss)

    val_accs.append(val_acc)

    scheduler.step()

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_acc:.2f}%")

    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Val Acc    : {val_acc:.2f}%")

    print(f"Learning Rate : {scheduler.get_last_lr()[0]:.8f}")

# =====================================================
# SAVE BEST MODEL
# =====================================================
    if val_acc > best_acc:

        best_acc = val_acc
        counter = 0

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print("✅ Best model saved!")

    else:

        counter += 1

        print(f"Early Stopping Counter : {counter}/{patience}")

# =====================================================
# EARLY STOPPING
# =====================================================

    if counter >= patience:

        print("\nEarly Stopping Triggered!")

        break


# =====================================================
# SAVE LAST MODEL
# =====================================================

torch.save(
    model.state_dict(),
    last_model_path
)

print("\nTraining Finished!")

print(f"Best Validation Accuracy : {best_acc:.2f}%")

# =====================================================
# PLOT ACCURACY
# =====================================================

plt.figure(figsize=(10,5))

plt.plot(
    train_accs,
    label="Train Accuracy",
    linewidth=2
)

plt.plot(
    val_accs,
    label="Validation Accuracy",
    linewidth=2
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy")
plt.grid(True)
plt.legend()

plt.savefig(
    os.path.join(
        MODEL_PATH,
        "accuracy.png"
    )
)

plt.show()

# =====================================================
# PLOT LOSS
# =====================================================

plt.figure(figsize=(10,5))

plt.plot(
    train_losses,
    label="Train Loss",
    linewidth=2
)

plt.plot(
    val_losses,
    label="Validation Loss",
    linewidth=2
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.legend()

plt.savefig(
    os.path.join(
        MODEL_PATH,
        "loss.png"
    )
)

plt.show()

print("=" * 60)
print("Files Saved")
print("=" * 60)
print(best_model_path)
print(last_model_path)
print(os.path.join(MODEL_PATH, "accuracy.png"))
print(os.path.join(MODEL_PATH, "loss.png"))
print("=" * 60)