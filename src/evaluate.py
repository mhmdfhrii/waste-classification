import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from model import WasteCNN


# =========================
# Device
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Transform
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# Dataset
# =========================

dataset = datasets.ImageFolder(
    "../dataset/TrashNet",
    transform=transform
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

_, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)


# =========================
# Model
# =========================

model = WasteCNN(
    num_classes=len(dataset.classes)
)

model.load_state_dict(
    torch.load("../models/waste_cnn.pth", map_location=device)
)

model.to(device)
model.eval()


# =========================
# Prediction
# =========================

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())


# =========================
# Classification Report
# =========================

print(classification_report(
    all_labels,
    all_preds,
    target_names=dataset.classes
))


# =========================
# Confusion Matrix
# =========================

cm = confusion_matrix(
    all_labels,
    all_preds
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=dataset.classes,
    yticklabels=dataset.classes
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()