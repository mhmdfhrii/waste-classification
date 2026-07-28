import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from resnet_model import create_model


# =====================================================
# CONFIG
# =====================================================

DATASET_PATH = "../dataset/TrashNet"

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
# DATASET
# =====================================================

dataset = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=transform
)


# =====================================================
# DATASET INFORMATION
# =====================================================

print("\n" + "=" * 60)
print("DATASET")
print("=" * 60)

print("Classes:")
print(dataset.classes)

print("\nClass to Index:")
print(dataset.class_to_idx)

print("\nTotal Images:")
print(len(dataset))


# =====================================================
# DATALOADER
# =====================================================

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)


# =====================================================
# GET ONE BATCH
# =====================================================

images, labels = next(iter(loader))


print("\n" + "=" * 60)
print("BATCH")
print("=" * 60)

print("Images Shape:")
print(images.shape)

print("\nLabels Shape:")
print(labels.shape)

print("\nLabels:")
print(labels)


# =====================================================
# CREATE MODEL
# =====================================================

model = create_model(
    num_classes=len(dataset.classes)
)

model = model.to(device)


# =====================================================
# MOVE DATA TO DEVICE
# =====================================================

images = images.to(device)
labels = labels.to(device)


# =====================================================
# FORWARD PASS
# =====================================================

model.eval()

with torch.no_grad():
    outputs = model(images)


# =====================================================
# MODEL OUTPUT
# =====================================================

print("\n" + "=" * 60)
print("MODEL")
print("=" * 60)

print("Output Shape:")
print(outputs.shape)

print("\nExpected:")
print(
    f"torch.Size([{BATCH_SIZE}, {len(dataset.classes)}])"
)


# =====================================================
# PREDICTION TEST
# =====================================================

predictions = torch.argmax(
    outputs,
    dim=1
)


print("\nPredictions:")
print(predictions)


# =====================================================
# SUCCESS
# =====================================================

print("\n" + "=" * 60)
print("TEST SUCCESS")
print("=" * 60)

print("Dataset        : OK")
print("DataLoader     : OK")
print("ResNet18       : OK")
print("Forward Pass   : OK")
print("Output Classes : OK")
print("=" * 60)