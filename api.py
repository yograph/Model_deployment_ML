#!/usr/bin/env python3
"""
ROI Classifier API
==================

This FastAPI service provides an ROI (Region of Interest) extraction + ConvNeXt-small classification pipeline.

Features:
  - Strict image-only validation (JPEG, PNG, BMP, GIF, TIFF)
  - Three ROI extraction methods: Otsu, Adaptive, Canny (fallback to full image)
  - Automatic resizing, normalization, and inference with three ConvNeXt-small models
  - Built-in HTML upload form at `/`
  - Swagger UI at `/docs` & ReDoc at `/redoc`

Run:
    pip install -r requirements.txt
    uvicorn api:app --reload --port 8000

"""
import cv2
import numpy as np
import torch
import logging
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from pathlib import Path
from torchvision import transforms
from torchvision.models import convnext_small

# ROI extraction functions (local module)
from roi_extract import (
    extract_roi_otsu,
    extract_roi_adaptive,
    extract_roi_canny
)

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Allowed MIME types
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/gif",
    "image/tiff",
}

# Pydantic response schemas
class ClassifierPrediction(BaseModel):
    classifier: int = Field(..., description="Index of the classifier model (0–2)")
    probs: List[float] = Field(..., description="Probability vector over classes")

class PredictResponse(BaseModel):
    roi_method: str = Field(..., description="ROI extraction method used (otsu, adaptive, canny) or 'full_image'")
    bbox: Optional[List[int]] = Field(None, description="[x0, y0, x1, y1] of ROI, or null if full image used")
    predictions: List[ClassifierPrediction] = Field(..., description="List of outputs from each classifier")

# Initialize FastAPI
app = FastAPI(
    title="ROI Classifier API",
    version="1.0.0",
    description="Extract ROI from an image and classify via ConvNeXt-small ensemble",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Load ConvNeXt-small models once
def load_models(num_classes: int, device: torch.device) -> List[torch.nn.Module]:
    """
    Load three ConvNeXt-small models and patch their heads for `num_classes`.

    Args:
        num_classes: Number of output classes for the linear layer.
        device: torch.device ('cpu' or 'cuda').
    Returns:
        List of `torch.nn.Module` ready for inference.
    """
    base = Path(__file__).parent / "model_weights"
    paths = [base / f"model_{i}_best.pth" for i in range(3)]
    models = []
    for i, wpath in enumerate(paths):
        try:
            model = convnext_small(pretrained=False)
            # patch classifier head
            if hasattr(model, 'classifier'):
                in_f = model.classifier[2].in_features
                model.classifier[2] = torch.nn.Linear(in_f, num_classes)
            else:
                in_f = model.head.in_features
                model.head = torch.nn.Linear(in_f, num_classes)
            # load weights (filter mismatches)
            sd = torch.load(wpath, map_location=device)
            st = model.state_dict()
            st.update({k: v for k, v in sd.items() if k in st and v.shape == st[k].shape})
            model.load_state_dict(st)
            model.to(device).eval()
            models.append(model)
        except Exception as e:
            logger.error(f"Error loading model_{i}: {e}")
            raise
    return models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLS_MODELS = load_models(num_classes=1, device=DEVICE)

# Preprocessing pipeline (resize → to tensor → normalize)
TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/", response_class=HTMLResponse, summary="Upload form")
async def main_form() -> str:
    """
    Displays a simple HTML form for uploading an image.

    Example:
        > Open in browser:
            http://127.0.0.1:8000/
    """
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>ROI Classifier Upload</title>
    </head>
    <body>
      <h1>ROI Classifier</h1>
      <p>Select an image to classify:</p>
      <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required />
        <button type="submit">Upload & Classify</button>
      </form>
      <hr />
      <p>Or try the <a href="/docs">Swagger UI</a> for detailed docs and examples.</p>
    </body>
    </html>
    '''

@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        200: {"description": "Successful classification"},
        400: {"description": "Bad request (invalid image)"},
        415: {"description": "Unsupported media type"},
        500: {"description": "Internal server error"}
    },
    summary="Upload image and get ROI-based classification"
)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    """
    Receives an image file, extracts an ROI, and returns classification scores.

    Args:
        file: Binary image upload. Supported formats: JPEG, PNG, BMP, GIF, TIFF.

    Returns:
        PredictResponse:
            - roi_method: method used ('extract_roi_otsu', 'extract_roi_adaptive', 'extract_roi_canny', or 'full_image')
            - bbox: [x0, y0, x1, y1] if ROI used, else null
            - predictions: list of classifier outputs (index + softmax probabilities)

    Raises:
        HTTPException(415): if uploaded file isn’t a supported image
        HTTPException(400): if decoding fails
        HTTPException(500): on processing/classification errors

    Example (curl):
        curl -X POST http://127.0.0.1:8000/predict \
             -F "file=@/path/to/image.jpg"
    """
    # Validate MIME
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning(f"Unsupported media type: {file.content_type}")
        raise HTTPException(status_code=415, detail="Unsupported media type. Upload an image.")

    # Read & decode
    try:
        data = await file.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Error decoding upload: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file data.")
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # Grayscale
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        logger.error(f"Grayscale conversion failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to preprocess image.")

    # ROI extraction
    roi_bbox = None
    used = None
    for fn in (extract_roi_otsu, extract_roi_adaptive, extract_roi_canny):
        try:
            bbox, pct = fn(gray)
            if bbox and pct and pct > 0.01:
                x0, y0, x1, y1 = bbox
                h, w = gray.shape
                if 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h:
                    roi_bbox, used = bbox, fn.__name__
                    break
        except Exception as e:
            logger.error(f"{fn.__name__} failed: {e}")

    # Crop or full image
    if roi_bbox:
        x0, y0, x1, y1 = roi_bbox
        crop = img[y0:y1, x0:x1]
    else:
        crop = img
        used = "full_image"

    # Classify
    try:
        t = TRANSFORM(crop).unsqueeze(0).to(DEVICE)
        preds: List[ClassifierPrediction] = []
        with torch.no_grad():
            for idx, model in enumerate(CLS_MODELS):
                logits = model(t)
                probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy().tolist()
                preds.append(ClassifierPrediction(classifier=idx, probs=probs))
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail="Failed to classify image.")

    return PredictResponse(roi_method=used, bbox=roi_bbox, predictions=preds)
