"""
Backend API untuk Waste Classifier.

Menjalankan:

    pip install fastapi uvicorn python-multipart torch torchvision pillow

    uvicorn backend_main:app --reload --port 8000

Endpoint:

    GET  /health
    POST /predict
"""

import io
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
from torchvision import transforms, models


# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = "models/resnet_v2_best.pth"

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
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =====================================================
# APP
# =====================================================

app = FastAPI(
    title="Waste Classifier API",
    description="API klasifikasi sampah menggunakan ResNet18",
    version="1.0.0"
)


# =====================================================
# CORS
# =====================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# =====================================================
# LOAD MODEL
# =====================================================

def load_model():

    # Create ResNet18
    model = models.resnet18(
        weights=None
    )


    # Number of input features
    num_features = model.fc.in_features


    # =================================================
    # RESNET V2 CLASSIFIER
    # HARUS SAMA DENGAN SAAT TRAINING
    # =================================================

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
            len(CLASS_NAMES)
        )
    )


    # =================================================
    # CHECK MODEL
    # =================================================

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}"
        )


    # =================================================
    # LOAD WEIGHTS
    # =================================================

    checkpoint = torch.load(

        MODEL_PATH,

        map_location=device,

        weights_only=True

    )


    model.load_state_dict(
        checkpoint
    )


    # Move to device
    model = model.to(
        device
    )


    # Evaluation mode
    model.eval()


    return model


# Load model once when server starts
model = load_model()


# =====================================================
# IMAGE TRANSFORM
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
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health():

    return {

        "status": "ok",

        "model": "ResNet18 V2",

        "device": str(device),

        "classes": CLASS_NAMES

    }


# =====================================================
# PREDICT
# =====================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # =================================================
    # CHECK FILE TYPE
    # =================================================

    if not file.content_type:

        raise HTTPException(

            status_code=400,

            detail="Content type tidak ditemukan."

        )


    if not file.content_type.startswith(
        "image/"
    ):

        raise HTTPException(

            status_code=400,

            detail="File harus berupa gambar (jpg/png)."

        )


    # =================================================
    # READ IMAGE
    # =================================================

    try:

        raw_bytes = await file.read()


        image = Image.open(
            io.BytesIO(raw_bytes)
        ).convert("RGB")


    except Exception:

        raise HTTPException(

            status_code=400,

            detail=(
                "Gagal membaca gambar. "
                "Pastikan file JPG/JPEG/PNG."
            )

        )


    # =================================================
    # PREPROCESS
    # =================================================

    image_tensor = transform(
        image
    )


    image_tensor = image_tensor.unsqueeze(
        0
    )


    image_tensor = image_tensor.to(
        device
    )


    # =================================================
    # PREDICTION
    # =================================================

    with torch.no_grad():

        outputs = model(
            image_tensor
        )


        probabilities = F.softmax(
            outputs,
            dim=1
        )


    # =================================================
    # GET PREDICTION
    # =================================================

    confidence, predicted_index = torch.max(

        probabilities,

        dim=1

    )


    predicted_class = CLASS_NAMES[
        predicted_index.item()
    ]


    confidence_percentage = (

        confidence.item()
        * 100

    )


    # =================================================
    # PROBABILITIES
    # =================================================

    probability_values = (

        probabilities[0]
        .cpu()
        .numpy()

    )


    probability_dict = {

        class_name: round(
            float(probability) * 100,
            2
        )

        for class_name, probability

        in zip(
            CLASS_NAMES,
            probability_values
        )

    }


    # =================================================
    # RESPONSE
    # =================================================

    return {

        "predicted_class":
            predicted_class,

        "confidence":
            round(
                confidence_percentage,
                2
            ),

        "probabilities":
            probability_dict

    }