import base64
import io
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageDraw, ImageFont, ImageOps

from rfdetr import RFDETRNano
from src.decision import generate_inspection_summary


# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

app = FastAPI(
    title="DefectSense API",
    description="PCB defect detection using RF-DETR Nano",
    version="2.0.0"
)

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "rf_detr_best_ema.pth"

CLASS_NAMES = {
    0: "open",
    1: "short",
    2: "mousebite",
    3: "spur",
    4: "copper",
    5: "pin-hole"
}


# --------------------------------------------------
# 2. Load RF-DETR Nano model
# --------------------------------------------------

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"RF-DETR checkpoint not found: {MODEL_PATH}"
    )

model = RFDETRNano.from_checkpoint(str(MODEL_PATH))

print("RF-DETR Nano model loaded successfully!")


# --------------------------------------------------
# 3. Annotate PCB image with predictions
# --------------------------------------------------

def annotate_image(image, boxes, scores, class_ids):

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    for box, score, class_id in zip(boxes, scores, class_ids):

        class_id = int(class_id)

        if class_id not in CLASS_NAMES:
            continue

        x1, y1, x2, y2 = [int(value) for value in box]

        defect_name = CLASS_NAMES[class_id]

        label = f"{defect_name}: {float(score):.2f}"

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=3
        )

        text_box = draw.textbbox(
            (x1, y1),
            label,
            font=font
        )

        draw.rectangle(
            text_box,
            fill="red"
        )

        draw.text(
            (x1, y1),
            label,
            fill="white",
            font=font
        )

    return annotated


# --------------------------------------------------
# 4. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "DefectSense API is running",
        "model": "RF-DETR Nano"
    }


# --------------------------------------------------
# 5. Health check endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model is not None and MODEL_PATH.is_file(),
        "model": "RF-DETR Nano"
    }


# --------------------------------------------------
# 6. PCB defect prediction endpoint
# --------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence: float = 0.25
):

    allowed_types = [
        "image/jpeg",
        "image/png"
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported."
        )

    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="Confidence must be between 0 and 1."
        )

    try:

        # Read and normalize uploaded image
        image = ImageOps.exif_transpose(
            Image.open(file.file)
        ).convert("RGB")

        # Run RF-DETR Nano prediction
        results = model.predict(
            image,
            threshold=confidence,
            include_source_image=False
        )

        # Extract predictions
        boxes = np.asarray(results.xyxy).reshape(-1, 4)
        scores = np.asarray(results.confidence).reshape(-1)
        class_ids = np.asarray(results.class_id).reshape(-1).astype(int)

        detections = []

        for box, score, class_id in zip(boxes, scores, class_ids):

            if class_id not in CLASS_NAMES:
                continue

            detections.append({
                "defect": CLASS_NAMES[class_id],
                "confidence": float(score)
            })

        # Generate inspection summary
        summary = generate_inspection_summary(detections)

        # Create annotated image
        annotated_image = annotate_image(
            image,
            boxes,
            scores,
            class_ids
        )

        # Convert annotated image into Base64 PNG
        image_buffer = io.BytesIO()

        annotated_image.save(
            image_buffer,
            format="PNG"
        )

        encoded_image = base64.b64encode(
            image_buffer.getvalue()
        ).decode("utf-8")

        summary["annotated_image"] = encoded_image

        return summary

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )

    finally:
        await file.close()