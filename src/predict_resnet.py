import os

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from resnet_model import create_model


# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = "../models/resnet_best.pth"

IMAGE_PATH = "../test_images/glass.png"

IMAGE_SIZE = 224

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic"
]


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
# LOAD MODEL
# =====================================================

model = create_model(
    num_classes=len(CLASS_NAMES)
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


# =====================================================
# CHECK IMAGE
# =====================================================

if not os.path.exists(IMAGE_PATH):

    raise FileNotFoundError(
        f"Gambar tidak ditemukan: {IMAGE_PATH}"
    )


# =====================================================
# LOAD IMAGE
# =====================================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")


# =====================================================
# PREPROCESS
# =====================================================

image_tensor = transform(
    image
)


image_tensor = image_tensor.unsqueeze(
    0
)


image_tensor = image_tensor.to(
    device
)


# =====================================================
# PREDICTION
# =====================================================

with torch.no_grad():

    outputs = model(
        image_tensor
    )

    probabilities = F.softmax(
        outputs,
        dim=1
    )


# =====================================================
# GET PREDICTION
# =====================================================

confidence, predicted_index = torch.max(
    probabilities,
    dim=1
)


predicted_class = CLASS_NAMES[
    predicted_index.item()
]


confidence_percentage = (
    confidence.item() * 100
)


# =====================================================
# ALL PROBABILITIES
# =====================================================

all_probabilities = probabilities[
    0
].cpu().numpy()


# =====================================================
# RESULT
# =====================================================

print("\n" + "=" * 60)
print("PREDICTION RESULT")
print("=" * 60)

print(
    f"Image      : {IMAGE_PATH}"
)

print(
    f"Prediction : {predicted_class}"
)

print(
    f"Confidence : {confidence_percentage:.2f}%"
)


print("\n" + "-" * 60)
print("ALL CLASS PROBABILITIES")
print("-" * 60)


for class_name, probability in zip(
    CLASS_NAMES,
    all_probabilities
):

    print(
        f"{class_name:<12} : "
        f"{probability * 100:.2f}%"
    )


print("=" * 60)