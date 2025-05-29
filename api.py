#!/usr/bin/env python3
"""
FastAPI service for ROI extraction + ConvNeXt-small classification
with strict image-only input validation, robust error handling,
and a simple HTML upload form at '/'.

Save this file as `api.py` in your project root.
Run with:
    uvicorn api:app --reload --port 8000
"""
import io
import cv2
import numpy as np
import torch
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pathlib import Path
from torchvision import transforms
from torchvision.models import convnext_small

# import ROI extraction functions
from roi_extract import (
    extract_roi_otsu,
    extract_roi_adaptive,
    extract_roi_canny
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ROI Classifier API",
    version="1.0.0",
    description="Extracts an ROI then classifies via ConvNeXt-small",
)

# Allowed MIME types for upload
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/gif",
    "image/tiff",
}


def load_models(num_classes: int, device: torch.device):
    """Load three convnext_small models with pretrained weights."""
    base = Path(__file__).parent / "model_weights"
    weights = [base / f"model_{i}_best.pth" for i in range(3)]
    models = []
    for w in weights:
        try:
            m = convnext_small(pretrained=False)
            # adjust final layer
            if hasattr(m, "classifier"):
                in_f = m.classifier[2].in_features
                m.classifier[2] = torch.nn.Linear(in_f, num_classes)
            else:
                in_f = m.head.in_features
                m.head = torch.nn.Linear(in_f, num_classes)
            # load & filter state dict
            sd = torch.load(w, map_location=device)
            st = m.state_dict()
            matched = {k: v for k, v in sd.items() if k in st and v.shape == st[k].shape}
            st.update(matched)
            m.load_state_dict(st)
            m.to(device).eval()
            models.append(m)
        except Exception as e:
            logger.error(f"Failed to load model {w}: {e}")
            raise
    return models


# Initialize device, models & transforms
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLS_MODELS = load_models(num_classes=1, device=DEVICE)
TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.get("/", response_class=HTMLResponse, summary="Upload form")
async def main_form():
    """Render a simple HTML page for users to upload an image."""
    return """
    <html>
      <head>
        <title>Upload an Image</title>
      </head>
      <body>
        <h1>Upload an Image for ROI Classification</h1>
        <form action=\"/predict\" enctype=\"multipart/form-data\" method=\"post\">
          <input name=\"file\" type=\"file\" accept=\"image/*\" required>
          <button type=\"submit\">Submit</button>
        </form>
        <p>Or view the <a href=\"/docs\">Swagger UI</a>.</p>
      </body>
    </html>
    """


@app.post("/predict", summary="Upload an image and get ROI-based classification")
async def predict(file: UploadFile = File(...)):
    # 1) Enforce image-only uploads
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning(f"Unsupported media type: {file.content_type}")
        raise HTTPException(415, "Unsupported media type. Please upload an image file.")

    # 2) Read and decode image
    try:
        data = await file.read()
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Error reading uploaded file: {e}")
        raise HTTPException(400, "Invalid image file data.")

    if img is None:
        raise HTTPException(400, "Could not decode image. Ensure it's a valid image file.")

    # 3) Convert to grayscale
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        logger.error(f"Error converting to grayscale: {e}")
        raise HTTPException(500, "Failed processing image.")

    # 4) ROI extraction methods
    methods = [extract_roi_otsu, extract_roi_adaptive, extract_roi_canny]
    roi_bbox = None
    used = None
    for fn in methods:
        try:
            bbox, pct = fn(gray)
            if bbox and pct and pct > 0.01:
                x0, y0, x1, y1 = bbox
                h, w = gray.shape
                if 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h:
                    roi_bbox = bbox
                    used = fn.__name__
                    break
                else:
                    logger.warning(f"Invalid bbox from {fn.__name__}: {bbox}")
        except Exception as e:
            logger.error(f"ROI extraction failed ({fn.__name__}): {e}")
            continue

    crop = img[y0:y1, x0:x1] if roi_bbox else img
    used = used or "full_image"

    # 5) Classification
    try:
        tensor = TRANSFORM(crop).unsqueeze(0).to(DEVICE)
        preds = []
        with torch.no_grad():
            for idx, model in enumerate(CLS_MODELS):
                logits = model(tensor)
                probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy().tolist()
                preds.append({"classifier": idx, "probs": probs})
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(500, "Failed to classify image.")

    return JSONResponse({
        "roi_method": used,
        "bbox": roi_bbox,
        "predictions": preds
    })
