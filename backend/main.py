
from pathlib import Path
from io import BytesIO, StringIO
from typing import Annotated
import base64
import csv
import json
from fastapi.responses import Response

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, ImageEnhance, UnidentifiedImageError

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware


# ==================================================
# 1. FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="TrustCheck API",
    description="Research prototype for chest X-ray robustness and explanation stability.",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# 2. LOAD MODEL
# ==================================================

CHECKPOINT = Path(
    r"D:\TrustCheck\models\checkpoints\binary\densenet121.pt"
)

device = torch.device("cpu")

if not CHECKPOINT.is_file():
    raise FileNotFoundError(
        f"Model checkpoint not found: {CHECKPOINT}"
    )

model = models.densenet121(weights=None)
model.classifier = nn.Linear(
    model.classifier.in_features, 1
)

state = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(
    state.get("model_state_dict", state)
)
model.to(device)
model.eval()


# ==================================================
# 3. IMAGE PREPROCESSING
# ==================================================

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ==================================================
# 4. HELPER FUNCTIONS
# ==================================================

def read_uploaded_image(contents: bytes) -> Image.Image:
    """Validate image bytes and return a grayscale image."""

    try:
        with Image.open(BytesIO(contents)) as uploaded:
            uploaded.verify()

        with Image.open(BytesIO(contents)) as uploaded:
            return uploaded.convert("L")

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )


def validate_file(file: UploadFile) -> None:
    """Validate JPEG and PNG uploads."""

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/jpg",
    }

    content_type = (file.content_type or "").lower()

    if content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image.",
        )


def predict_image(pil_image: Image.Image):
    """Return predicted class and pneumonia score."""

    input_tensor = preprocess(
        pil_image.convert("RGB")
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(input_tensor)
        score = torch.sigmoid(logit).item()

    prediction = (
        "PNEUMONIA" if score >= 0.5 else "NORMAL"
    )

    return prediction, score


def create_perturbations(original: Image.Image):
    """Generate controlled brightness, contrast and noise variants."""

    brightness = ImageEnhance.Brightness(
        original
    ).enhance(1.2)

    contrast = ImageEnhance.Contrast(
        original
    ).enhance(1.5)

    image_array = np.asarray(
        original, dtype=np.float32
    )

    rng = np.random.default_rng(42)
    noise = rng.normal(
        0, 8, image_array.shape
    )

    noisy_array = np.clip(
        image_array + noise, 0, 255
    ).astype(np.uint8)

    noisy = Image.fromarray(noisy_array)

    return {
        "original": original,
        "brightness_increased": brightness,
        "contrast_increased": contrast,
        "gaussian_noise": noisy,
    }


# ==================================================
# 5. GRAD-CAM GENERATION
# ==================================================


def generate_gradcam(pil_image: Image.Image, target_class=None):
    """
    Generate a normalized 224x224 Grad-CAM heatmap.

    target_class:
        "PNEUMONIA" -> explain the positive logit
        "NORMAL"    -> explain the negative logit
        None        -> explain the predicted class
    """
    model.eval()

    activations = []
    gradients = []
    target_layer = model.features.denseblock4

    def forward_hook(module, inputs, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )
    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    try:
        input_tensor = preprocess(
            pil_image.convert("RGB")
        ).unsqueeze(0).to(device)

        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logit = model(input_tensor).reshape(-1)[0]

            if target_class is None:
                sign = 1.0 if logit.item() >= 0 else -1.0
            elif target_class == "PNEUMONIA":
                sign = 1.0
            elif target_class == "NORMAL":
                sign = -1.0
            else:
                raise ValueError(
                    "target_class must be NORMAL, PNEUMONIA, or None"
                )

            (sign * logit).backward()

        feature_maps = activations[0].detach()
        grads = gradients[0].detach()

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu(
            (weights * feature_maps).sum(dim=1, keepdim=True)
        )

        cam = torch.nn.functional.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].cpu().numpy()

        cam_min = float(cam.min())
        cam_max = float(cam.max())

        if cam_max - cam_min < 1e-8:
            return np.zeros((224, 224), dtype=np.float32)

        cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.astype(np.float32)

    finally:
        forward_handle.remove()
        backward_handle.remove()



# ==================================================
# 6. HEATMAP SIMILARITY
# ==================================================

def heatmap_similarity(heatmap_a, heatmap_b):
    """Calculate cosine similarity between two heatmaps safely."""

    a = heatmap_a.flatten().astype(np.float64)
    b = heatmap_b.flatten().astype(np.float64)

    # Reject invalid values
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return None

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if not np.isfinite(denominator) or denominator < 1e-12:
        return None

    similarity = float(np.dot(a, b) / denominator)

    if not np.isfinite(similarity):
        return None

    return similarity



# ==================================================
# 7. CREATE GRAD-CAM OVERLAY
# ==================================================

def create_gradcam_overlay(pil_image, heatmap):
    """Return a Base64 PNG data URL."""

    original = pil_image.convert("RGB").resize(
        (224, 224)
    )

    original_array = np.asarray(
        original, dtype=np.uint8
    )

    heatmap_uint8 = np.uint8(
        np.clip(heatmap, 0, 1) * 255
    )

    colored_heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB,
    )

    overlay_array = cv2.addWeighted(
        original_array,
        0.6,
        colored_heatmap,
        0.4,
        0,
    )

    buffer = BytesIO()

    Image.fromarray(overlay_array).save(
        buffer, format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return "data:image/png;base64," + encoded


# ==================================================
# 8. HOME AND HEALTH ENDPOINTS
# ==================================================

@app.get("/")
def home():
    return {
        "message": "TrustCheck API is running!",
        "status": "success",
        "version": app.version,
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device),
        "checkpoint_exists": CHECKPOINT.is_file(),
    }


# ==================================================
# 9. SINGLE X-RAY PREDICTION
# ==================================================

@app.post("/upload-xray")
async def upload_xray(
    file: Annotated[
        UploadFile,
        File(description="Upload one JPG or PNG X-ray image"),
    ],
):
    validate_file(file)

    contents = await file.read()
    image = read_uploaded_image(contents)

    prediction, score = predict_image(image)

    return {
        "message": "Image analyzed successfully",
        "filename": file.filename,
        "width": image.width,
        "height": image.height,
        "prediction": prediction,
        "pneumonia_score": round(score, 4),
        "research_notice": (
            "Research prototype only; not for medical diagnosis."
        ),
    }


# ==================================================
# 10. ROBUSTNESS TEST
# ==================================================

@app.post("/robustness-test")
async def robustness_test(
    file: Annotated[
        UploadFile,
        File(description="Upload one JPG or PNG X-ray image"),
    ],
):
    validate_file(file)

    contents = await file.read()
    original = read_uploaded_image(contents)

    test_images = create_perturbations(original)
    results = {}

    for name, test_image in test_images.items():
        prediction, score = predict_image(test_image)

        results[name] = {
            "prediction": prediction,
            "pneumonia_score": round(score, 4),
        }

    original_prediction = results["original"]["prediction"]

    for result in results.values():
        result["prediction_matches_original"] = (
            result["prediction"] == original_prediction
        )

    return {
        "message": "Robustness test completed",
        "filename": file.filename,
        "results": results,
        "research_notice": (
            "Research prototype only; not for medical diagnosis."
        ),
    }


# ==================================================
# 11. GRAD-CAM EXPLANATION STABILITY
# ==================================================

@app.post("/gradcam-test")
async def gradcam_test(
    file: Annotated[
        UploadFile,
        File(description="Upload one JPG or PNG X-ray image"),
    ],
):
    validate_file(file)

    contents = await file.read()
    original = read_uploaded_image(contents)

    test_images = create_perturbations(original)

    # Step 1: Get the original image's predicted class.
    original_prediction, original_score = predict_image(
        test_images["original"]
    )

    # Step 2: Use this same class for every Grad-CAM map.
    target_class = original_prediction

    heatmaps = {}
    predictions = {}
    overlays = {}

    for name, test_image in test_images.items():

        # Generate every heatmap for the SAME target class.
        heatmap = generate_gradcam(
            test_image,
            target_class=target_class,
        )
        heatmaps[name] = heatmap

        # Keep the prediction results unchanged.
        if name == "original":
            prediction, score = original_prediction, original_score
        else:
            prediction, score = predict_image(test_image)

        predictions[name] = {
            "prediction": prediction,
            "pneumonia_score": round(score, 4),
        }

        overlays[name] = create_gradcam_overlay(
            test_image, heatmap
        )

    original_heatmap = heatmaps["original"]
    similarities = {}

    for name, heatmap in heatmaps.items():
        similarity = heatmap_similarity(
            original_heatmap, heatmap
        )

        similarities[name] = (
            round(similarity, 4)
            if similarity is not None
            else None
        )

    return {
        "message": (
            "Grad-CAM explanation stability test completed"
        ),
        "filename": file.filename,
        "predictions": predictions,
        "gradcam_target_class": target_class,
        "heatmap_cosine_similarity_to_original": similarities,
        "heatmap_overlays": overlays,
        "heatmap_note": (
            "All Grad-CAM maps explain the original image's "
            "predicted class. Cosine similarity compares normalized "
            "Grad-CAM maps and is a research metric, not a clinical measure."
        ),
        "research_notice": (
            "Research prototype only; not for medical diagnosis."
        ),
    }


# ==================================================
# 12. BATCH EVALUATION AND CSV REPORT
# ==================================================

@app.post(
    "/batch-evaluate",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["files"],
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "format": "binary",
                                },
                                "description": (
                                    "Select up to 10 X-ray images "
                                    "(JPG or PNG)"
                                ),
                            },
                            "labels_json": {
                                "type": "string",
                                "default": "{}",
                                "description": (
                                    "Optional JSON labels"
                                ),
                            },
                        },
                    }
                }
            },
        }
    },
)
async def batch_evaluate(
    files: Annotated[
        list[UploadFile],
        File(
            description="Select up to 10 X-ray images (JPG or PNG)"
        ),
    ],
    labels_json: Annotated[
        str,
        Form(
            description=(
                'Optional JSON labels, e.g. '
                '{"image1.jpeg":"NORMAL"}'
            )
        ),
    ] = "{}",
):
    """
    Evaluate up to 10 X-ray images.

    Returns per-image predictions, perturbation metrics,
    aggregate metrics and CSV content.
    """

    if not files:
        raise HTTPException(
            status_code=400,
            detail="Upload at least one X-ray image.",
        )

    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Please upload at most 10 images per batch.",
        )

    try:
        actual_labels = json.loads(labels_json)

        if not isinstance(actual_labels, dict):
            raise ValueError(
                "Labels must be a JSON object."
            )

    except (json.JSONDecodeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail=(
                'labels_json must be a JSON object, e.g. '
                '{"image1.jpeg":"NORMAL"}.'
            ),
        )

    for filename, label in actual_labels.items():
        if str(label).upper() not in {
            "NORMAL",
            "PNEUMONIA",
        }:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid label for {filename}. "
                    "Use NORMAL or PNEUMONIA."
                ),
            )

    variation_names = [
        "original",
        "brightness_increased",
        "contrast_increased",
        "gaussian_noise",
    ]

    image_results = []
    csv_rows = []
    similarity_values = []

    consistent_count = 0
    labelled_count = 0
    correct_count = 0

    for uploaded_file in files:
        filename = (
            uploaded_file.filename or "unnamed_image"
        )

        try:
            validate_file(uploaded_file)

            contents = await uploaded_file.read()
            original_image = read_uploaded_image(contents)

            variants = create_perturbations(
                original_image
            )

            
            predictions = {}
            heatmaps = {}

            # First, predict the original image.
            baseline_prediction, baseline_score = predict_image(
                variants["original"]
            )

            # Keep the Grad-CAM target class fixed for every variation.
            for variation_name, variant_image in variants.items():

                if variation_name == "original":
                    prediction, score = baseline_prediction, baseline_score
                else:
                    prediction, score = predict_image(variant_image)

                predictions[variation_name] = {
                    "prediction": prediction,
                    "pneumonia_score": round(score, 4),
                }

                heatmaps[variation_name] = generate_gradcam(
                    variant_image,
                    target_class=baseline_prediction,
                )


            baseline_prediction = predictions[
                "original"
            ]["prediction"]

            baseline_score = predictions[
                "original"
            ]["pneumonia_score"]

            prediction_consistent = all(
                item["prediction"] == baseline_prediction
                for item in predictions.values()
            )

            if prediction_consistent:
                consistent_count += 1

            similarities = {}

            for variation_name, heatmap in heatmaps.items():
                similarity = heatmap_similarity(
                    heatmaps["original"],
                    heatmap,
                )

                similarities[variation_name] = (
                    round(similarity, 4)
                    if similarity is not None
                    else None
                )

                if (
                    variation_name != "original"
                    and similarity is not None
                ):
                    similarity_values.append(similarity)

            actual_label = actual_labels.get(filename)

            if actual_label is not None:
                actual_label = str(actual_label).upper()
                labelled_count += 1

                if baseline_prediction == actual_label:
                    correct_count += 1

            image_results.append({
                "filename": filename,
                "actual_label": actual_label,
                "original_prediction": baseline_prediction,
                "original_pneumonia_score": baseline_score,
                "prediction_consistent": prediction_consistent,
                "predictions": predictions,
                "heatmap_cosine_similarity_to_original": (
                    similarities
                ),
            })

            for variation_name in variation_names:
                item = predictions[variation_name]
                score = item["pneumonia_score"]
                similarity = similarities[variation_name]

                csv_rows.append({
                    "filename": filename,
                    "actual_label": actual_label or "",
                    "variation": variation_name,
                    "prediction": item["prediction"],
                    "pneumonia_score": score,
                    "score_change_from_original": round(
                        score - baseline_score, 4
                    ),
                    "prediction_matches_original": (
                        item["prediction"]
                        == baseline_prediction
                    ),
                    "heatmap_similarity_to_original": (
                        similarity
                        if similarity is not None
                        else ""
                    ),
                })

        except HTTPException as exc:
            image_results.append({
                "filename": filename,
                "error": exc.detail,
            })

    successful_results = [
        result for result in image_results
        if "error" not in result
    ]

    if not successful_results:
        raise HTTPException(
            status_code=400,
            detail="No valid images could be evaluated.",
        )

    # Generate CSV in memory.
    text_buffer = StringIO(newline="")

    fieldnames = [
        "filename",
        "actual_label",
        "variation",
        "prediction",
        "pneumonia_score",
        "score_change_from_original",
        "prediction_matches_original",
        "heatmap_similarity_to_original",
    ]

    writer = csv.DictWriter(
        text_buffer,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(csv_rows)

    csv_content = text_buffer.getvalue()

    successful_count = len(successful_results)

    summary = {
        "images_requested": len(files),
        "images_evaluated": successful_count,
        "images_with_errors": (
            len(files) - successful_count
        ),
        "prediction_consistency_count": consistent_count,
        "prediction_consistency_rate": round(
            consistent_count / successful_count, 4
        ),
        "mean_perturbed_heatmap_similarity": (
            round(
                sum(similarity_values)
                / len(similarity_values),
                4,
            )
            if similarity_values
            else None
        ),
        "labelled_images": labelled_count,
        "labelled_accuracy": (
            round(correct_count / labelled_count, 4)
            if labelled_count
            else None
        ),
        "accuracy_note": (
            "Accuracy uses only images with supplied reference labels."
            if labelled_count
            else "No reference labels supplied; accuracy was not calculated."
        ),
    }

    return {
        "message": "Batch evaluation completed",
        "summary": summary,
        "results": image_results,
        "csv_content": csv_content,
        "research_notice": (
            "Research prototype only. Scores are not calibrated "
            "probabilities and results are not medical diagnoses."
        ),
    }

# ==================================================
# 13. SAVED DATASET EVALUATION REPORT
# ==================================================

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


@app.get("/evaluation-report")
def get_evaluation_report():
    """Return the saved 200-image evaluation summary."""

    summary_path = REPORTS_DIR / "trustcheck_evaluation_summary.json"
    per_image_path = REPORTS_DIR / "trustcheck_per_image_results.json"

    if not summary_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Evaluation summary not found. Run evaluate_dataset.py first.",
        )

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        per_image = None

        if per_image_path.is_file():
            with per_image_path.open("r", encoding="utf-8") as f:
                per_image = json.load(f)

        return {
            "message": "Saved evaluation report loaded",
            "summary": summary,
            "per_image_results": per_image,
            "research_notice": (
                "Research prototype only. Results depend on this sample "
                "and do not establish clinical reliability."
            ),
        }

    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read evaluation report: {exc}",
        )


@app.get("/evaluation-report/csv")
def download_evaluation_csv():
    """Download the saved dataset evaluation results as CSV."""

    csv_path = REPORTS_DIR / "trustcheck_robustness_results.csv"

    if not csv_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Evaluation CSV not found. Run evaluate_dataset.py first.",
        )

    try:
        csv_content = csv_path.read_text(encoding="utf-8-sig")

        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": (
                    'attachment; filename="trustcheck_robustness_results.csv"'
                )
            },
        )

    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read evaluation CSV: {exc}",
        )

# ==================================================
# 14. CHEXPERT DATASET EVALUATION REPORT (EXPERIMENT B)
# ==================================================

CHEXPERT_REPORTS_DIR = REPORTS_DIR / "chexpert"


@app.get("/evaluation-report/chexpert")
def get_chexpert_evaluation_report():
    """Return the saved 100-image CheXpert evaluation summary."""

    summary_path = (
        CHEXPERT_REPORTS_DIR / "chexpert_evaluation_summary.json"
    )
    per_image_path = (
        CHEXPERT_REPORTS_DIR / "chexpert_per_image_results.json"
    )

    if not summary_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="CheXpert evaluation summary not found.",
        )

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        per_image = None
        if per_image_path.is_file():
            with per_image_path.open("r", encoding="utf-8") as f:
                per_image = json.load(f)

        return {
            "message": "CheXpert evaluation report loaded",
            "summary": summary,
            "per_image_results": per_image,
            "research_notice": summary.get(
                "research_notice",
                "Research prototype only; not for clinical use.",
            ),
        }

    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read CheXpert report: {exc}",
        )


@app.get("/evaluation-report/chexpert/csv")
def download_chexpert_evaluation_csv():
    """Download the saved CheXpert robustness results as CSV."""

    csv_path = (
        CHEXPERT_REPORTS_DIR / "chexpert_robustness_results.csv"
    )

    if not csv_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="CheXpert evaluation CSV not found.",
        )

    try:
        csv_content = csv_path.read_text(encoding="utf-8-sig")

        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": (
                    'attachment; filename="chexpert_robustness_results.csv"'
                )
            },
        )

    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read CheXpert CSV: {exc}",
        )
