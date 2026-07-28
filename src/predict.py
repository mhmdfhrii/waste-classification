import torch
import torch.nn.functional as F

from torchvision import transforms
from torchvision.datasets import ImageFolder

from PIL import Image
import matplotlib.pyplot as plt

from model import WasteCNN


# ==========================================
# Device
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device : {device}")


# ==========================================
# Dataset (ambil nama class otomatis)
# ==========================================

dataset = ImageFolder("../dataset/TrashNet")

classes = dataset.classes

print("Classes :", classes)


# ==========================================
# Transform
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# Load Model
# ==========================================

model = WasteCNN(
    num_classes=len(classes)
)

model.load_state_dict(
    torch.load(
        "../models/waste_cnn.pth",
        map_location=device
    )
)

model.to(device)
model.eval()


# ==========================================
# Image Path
# ==========================================

image_path = "../test_images/paper.png"

# Ganti sesuai gambar yang ingin diprediksi
# contoh:
# image_path = "../test_images/paper.jpg"
# image_path = "../test_images/glass.jpg"


# ==========================================
# Load Image
# ==========================================

image = Image.open(image_path).convert("RGB")

input_tensor = transform(image)

input_tensor = input_tensor.unsqueeze(0)

input_tensor = input_tensor.to(device)


# ==========================================
# Prediction
# ==========================================

with torch.no_grad():

    outputs = model(input_tensor)

    probabilities = F.softmax(outputs, dim=1)

    confidence, prediction = torch.max(probabilities, dim=1)


predicted_class = classes[prediction.item()]

confidence = confidence.item() * 100


# ==========================================
# Display Result
# ==========================================

plt.figure(figsize=(6,6))
plt.imshow(image)
plt.axis("off")

plt.title(
    f"Prediction : {predicted_class}\nConfidence : {confidence:.2f}%"
)

plt.show()


print("="*50)
print("Prediction :", predicted_class)
print(f"Confidence : {confidence:.2f}%")
print("="*50)