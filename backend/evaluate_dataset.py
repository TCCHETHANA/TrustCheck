
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

import main as trustcheck


# ==========================================
# CONFIGURATION
# ==========================================

DATASET = Path(r"D:\TrustCheck\data\trustcheck_eval_200")
REPORT_DIR = Path(r"D:\TrustCheck\reports")

CLASSES = ["NORMAL", "PNEUMONIA"]

VARIATIONS = [
    "original",
    "brightness_increased",
    "contrast_increased",
    "gaussian_noise",
]

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_metrics(rows, class_names):
    """Calculate accuracy and confusion-matrix metrics."""

    valid = [
        row for row in rows
        if (
            row["actual_label"] in class_names
            and row["prediction"] in class_names
        )
    ]

    total = len(valid)

    correct = sum(
        row["prediction"] == row["actual_label"]
        for row in valid
    )

    matrix = {
        actual: {pred: 0 for pred in class_names}
        for actual in class_names
    }

    for row in valid:
        matrix[row["actual_label"]][row["prediction"]] += 1

    per_class = {}

    for label in class_names:
        tp = matrix[label][label]

        fp = sum(
            matrix[actual][label]
            for actual in class_names
            if actual != label
        )

        fn = sum(
            matrix[label][pred]
            for pred in class_names
            if pred != label
        )

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": sum(matrix[label].values()),
        }

    return {
        "sample_count": total,
        "correct_predictions": correct,
        "accuracy": round(correct / total, 4) if total else None,
        "confusion_matrix": matrix,
        "per_class_metrics": per_class,
    }


def main():
    start_time = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Validate dataset folders.
    for label in CLASSES:
        folder = DATASET / label

        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing dataset folder: {folder}"
            )

    # Collect image paths and their ground-truth labels.
    files = []

    for label in CLASSES:
        class_files = sorted(
            f for f in (DATASET / label).iterdir()
            if f.is_file() and f.suffix.lower() in EXTENSIONS
        )

        for path in class_files:
            files.append((path, label))

    if not files:
        raise RuntimeError("No JPG or PNG images found.")

    print(f"Dataset: {DATASET}")
    print(f"Images found: {len(files)}")
    print("Starting evaluation. Grad-CAM on CPU may take a while.")

    csv_rows = []
    image_results = []
    errors = []

    original_rows = []
    variation_rows = {
        name: [] for name in VARIATIONS
    }

    consistent_count = 0
    similarity_values = []
    processed = 0

    for index, (path, actual_label) in enumerate(files, start=1):
        try:
            # Validate that the image can be opened.
            with Image.open(path) as img:
                img.verify()

            # Load the image in grayscale.
            with Image.open(path) as img:
                original = img.convert("L")

            variants = trustcheck.create_perturbations(original)

            predictions = {}
            heatmaps = {}

            # ------------------------------------------
            # 1. Predict the original image FIRST.
            # ------------------------------------------

            baseline_prediction, baseline_score = (
                trustcheck.predict_image(variants["original"])
            )

            # ------------------------------------------
            # 2. Predict every variation independently.
            #    Keep the model's original class labels.
            # ------------------------------------------

            for variation in VARIATIONS:
                variant_image = variants[variation]

                if variation == "original":
                    prediction = baseline_prediction
                    score = baseline_score
                else:
                    prediction, score = trustcheck.predict_image(
                        variant_image
                    )

                predictions[variation] = {
                    "prediction": prediction,
                    "pneumonia_score": float(score),
                }

            # ------------------------------------------
            # 3. Fix the Grad-CAM target to the ORIGINAL
            #    image's predicted class for every variant.
            # ------------------------------------------

            gradcam_target = baseline_prediction

            for variation in VARIATIONS:
                heatmaps[variation] = trustcheck.generate_gradcam(
                    variants[variation],
                    target_class=gradcam_target,
                )

            # ------------------------------------------
            # 4. Calculate prediction consistency.
            # ------------------------------------------

            baseline = predictions["original"]
            baseline_prediction = baseline["prediction"]

            is_consistent = all(
                predictions[name]["prediction"] == baseline_prediction
                for name in VARIATIONS
            )

            consistent_count += int(is_consistent)

            # ------------------------------------------
            # 5. Compare Grad-CAM maps against original.
            # ------------------------------------------

            image_similarities = {}

            for variation in VARIATIONS:
                similarity = trustcheck.heatmap_similarity(
                    heatmaps["original"],
                    heatmaps[variation],
                )

                if similarity is not None:
                    similarity = float(similarity)

                image_similarities[variation] = (
                    round(similarity, 4)
                    if similarity is not None
                    else None
                )

                if (
                    variation != "original"
                    and similarity is not None
                    and np.isfinite(similarity)
                ):
                    similarity_values.append(similarity)

                # --------------------------------------
                # 6. Save one CSV row per image/variation.
                # --------------------------------------

                item = predictions[variation]
                score = item["pneumonia_score"]

                row = {
                    "filename": path.name,
                    "actual_label": actual_label,
                    "variation": variation,
                    "prediction": item["prediction"],
                    "pneumonia_score": round(score, 6),
                    "score_change_from_original": round(
                        score - baseline["pneumonia_score"], 6
                    ),
                    "prediction_matches_original": (
                        item["prediction"] == baseline_prediction
                    ),
                    "correct": (
                        item["prediction"] == actual_label
                    ),
                    "heatmap_similarity_to_original": (
                        image_similarities[variation]
                        if image_similarities[variation] is not None
                        else ""
                    ),
                }

                csv_rows.append(row)
                variation_rows[variation].append(row)

                if variation == "original":
                    original_rows.append(row)

            # ------------------------------------------
            # 7. Save per-image results.
            # ------------------------------------------

            image_results.append({
                "filename": path.name,
                "actual_label": actual_label,
                "original_prediction": baseline_prediction,
                "original_pneumonia_score": round(
                    baseline["pneumonia_score"], 6
                ),
                "prediction_consistent": is_consistent,
                "gradcam_target_class": gradcam_target,
                "predictions": predictions,
                "heatmap_similarity_to_original": image_similarities,
            })

            processed += 1

        except (
            OSError,
            ValueError,
            UnidentifiedImageError,
            RuntimeError,
        ) as exc:
            errors.append({
                "filename": path.name,
                "actual_label": actual_label,
                "error": str(exc),
            })

        if index % 10 == 0 or index == len(files):
            print(f"Processed {index}/{len(files)} images")

    # ==========================================
    # WRITE DETAILED CSV
    # ==========================================

    csv_path = REPORT_DIR / "trustcheck_robustness_results.csv"

    fieldnames = [
        "filename",
        "actual_label",
        "variation",
        "prediction",
        "pneumonia_score",
        "score_change_from_original",
        "prediction_matches_original",
        "correct",
        "heatmap_similarity_to_original",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # ==========================================
    # BUILD SUMMARY
    # ==========================================

    summary = {
        "dataset": str(DATASET),
        "images_requested": len(files),
        "images_evaluated": processed,
        "images_with_errors": len(errors),

        "class_counts": {
            label: sum(actual == label for _, actual in files)
            for label in CLASSES
        },

        "original_model_metrics": get_metrics(
            original_rows, CLASSES
        ),

        "metrics_by_variation": {
            variation: get_metrics(rows, CLASSES)
            for variation, rows in variation_rows.items()
        },

        "prediction_consistency": {
            "consistent_images": consistent_count,
            "evaluated_images": processed,
            "rate": round(consistent_count / processed, 4)
            if processed
            else None,
        },

        "mean_perturbed_heatmap_similarity": (
            round(float(np.mean(similarity_values)), 4)
            if similarity_values
            else None
        ),

        "gradcam_method": (
            "The original image's predicted class is held fixed "
            "as the Grad-CAM target across all perturbations."
        ),

        "errors": errors,
        "elapsed_seconds": round(time.time() - start_time, 2),

        "research_notice": (
            "Research prototype only. Results depend on this sample "
            "and do not establish clinical reliability. Check possible "
            "training/test dataset overlap before making general claims."
        ),
    }

    # ==========================================
    # WRITE JSON REPORTS
    # ==========================================

    summary_path = REPORT_DIR / "trustcheck_evaluation_summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    details_path = REPORT_DIR / "trustcheck_per_image_results.json"

    details_path.write_text(
        json.dumps(image_results, indent=2),
        encoding="utf-8",
    )

    # ==========================================
    # PRINT FINAL SUMMARY
    # ==========================================

    print("\n========== EVALUATION SUMMARY ==========")

    print(json.dumps({
        "images_evaluated": processed,
        "images_with_errors": len(errors),
        "original_accuracy": (
            summary["original_model_metrics"]["accuracy"]
        ),
        "prediction_consistency": (
            summary["prediction_consistency"]["rate"]
        ),
        "mean_perturbed_heatmap_similarity": (
            summary["mean_perturbed_heatmap_similarity"]
        ),
        "elapsed_seconds": summary["elapsed_seconds"],
    }, indent=2))

    print("\nReports saved to:", REPORT_DIR)
    print("CSV:", csv_path)
    print("Summary:", summary_path)
    print("Per-image results:", details_path)


if __name__ == "__main__":
    main()
