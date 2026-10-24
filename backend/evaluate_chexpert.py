
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

import main as trustcheck


# ==========================================
# CONFIGURATION
# ==========================================

DATA_DIR = Path(r"D:\TrustCheck\data\chexpert_eval_100")
MANIFEST = DATA_DIR / "sample_manifest.csv"
REPORT_DIR = Path(r"D:\TrustCheck\reports\chexpert")

LABELS = ["NORMAL_LIKE", "PNEUMONIA"]

VARIATIONS = [
    "original",
    "brightness_increased",
    "contrast_increased",
    "gaussian_noise",
]

EXPECTED_SAMPLE_SIZE = 100
EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ==========================================
# METRICS
# ==========================================

def calculate_metrics(rows):
    """Calculate classification metrics using successful image rows."""

    matrix = {
        actual: {pred: 0 for pred in LABELS}
        for actual in LABELS
    }

    for row in rows:
        actual = row["actual_label"]
        predicted = row["prediction"]

        if actual not in LABELS or predicted not in LABELS:
            raise ValueError(
                f"Unexpected label: actual={actual}, prediction={predicted}"
            )

        matrix[actual][predicted] += 1

    total = len(rows)

    correct = sum(
        row["actual_label"] == row["prediction"]
        for row in rows
    )

    per_class = {}

    for label in LABELS:
        tp = matrix[label][label]

        fp = sum(
            matrix[actual][label]
            for actual in LABELS
            if actual != label
        )

        fn = sum(
            matrix[label][pred]
            for pred in LABELS
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


# ==========================================
# MAIN EVALUATION
# ==========================================

def main():
    start_time = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Validate the manifest.
    if not MANIFEST.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST}"
        )

    df = pd.read_csv(MANIFEST)

    required_columns = {"Path", "trustcheck_label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Manifest must contain columns: {required_columns}"
        )

    if len(df) != EXPECTED_SAMPLE_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLE_SIZE} rows, found {len(df)}"
        )

    if df["trustcheck_label"].isna().any():
        raise ValueError("Manifest contains missing labels.")

    if set(df["trustcheck_label"].unique()) != set(LABELS):
        raise ValueError("Unexpected labels found in manifest.")

    print("Selected class counts:")
    print(df["trustcheck_label"].value_counts().to_string())

    # 2. Validate image paths and files before inference.
    files = []
    missing_files = []

    for _, row in df.iterrows():
        csv_path = str(row["Path"]).replace("\\", "/")
        prefix = "CheXpert-v1.0-small/"

        if csv_path.startswith(prefix):
            relative_path = csv_path[len(prefix):]
        else:
            relative_path = csv_path

        relative = Path(relative_path)

        # Prevent absolute paths and parent-directory traversal.
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"Unsafe image path in manifest: {csv_path}"
            )

        image_path = DATA_DIR / relative

        if image_path.suffix.lower() not in EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension: {image_path}"
            )

        if not image_path.is_file():
            missing_files.append(str(image_path))
        else:
            files.append((
                image_path,
                str(row["trustcheck_label"]),
            ))

    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} image files are missing. "
            f"Examples:\n" + "\n".join(missing_files[:5])
        )

    print(f"\nImages verified: {len(files)}")
    print("Starting evaluation. CPU Grad-CAM may take several minutes.")

    # 3. Evaluate images. An image counts as successful only when
    #    all four variations and their Grad-CAM maps complete.
    all_rows = []
    original_rows = []
    image_results = []
    errors = []

    consistent_count = 0
    similarity_values = []
    successful_images = 0

    for index, (image_path, actual_label) in enumerate(files, start=1):
        try:
            with Image.open(image_path) as img:
                img.verify()

            with Image.open(image_path) as img:
                original_image = img.convert("L")

            variants = trustcheck.create_perturbations(original_image)

            # Ensure all expected variants are present.
            for variation in VARIATIONS:
                if variation not in variants:
                    raise ValueError(
                        f"Missing perturbation: {variation}"
                    )

            predictions = {}
            heatmaps = {}

            # ------------------------------------------
            # 1. Predict the original image first.
            # ------------------------------------------

            original_prediction, original_score = (
                trustcheck.predict_image(variants["original"])
            )

            # ------------------------------------------
            # 2. Predict every variation independently.
            #    Preserve CheXpert's NORMAL_LIKE convention.
            # ------------------------------------------

            for variation in VARIATIONS:
                variant_image = variants[variation]

                if variation == "original":
                    prediction = original_prediction
                    score = original_score
                else:
                    prediction, score = trustcheck.predict_image(
                        variant_image
                    )

                if prediction == "NORMAL":
                    prediction = "NORMAL_LIKE"

                if prediction not in LABELS:
                    raise ValueError(
                        f"Unexpected model prediction: {prediction}"
                    )

                predictions[variation] = {
                    "prediction": prediction,
                    "pneumonia_score": float(score),
                }

            # ------------------------------------------
            # 3. Fix the Grad-CAM target class to the
            #    original image's predicted class.
            # ------------------------------------------

            baseline = predictions["original"]
            baseline_prediction = baseline["prediction"]

            gradcam_target = (
                "NORMAL"
                if baseline_prediction == "NORMAL_LIKE"
                else "PNEUMONIA"
            )

            for variation in VARIATIONS:
                heatmaps[variation] = trustcheck.generate_gradcam(
                    variants[variation],
                    target_class=gradcam_target,
                )

            # ------------------------------------------
            # 4. Calculate prediction consistency.
            # ------------------------------------------

            is_consistent = all(
                predictions[v]["prediction"] == baseline_prediction
                for v in VARIATIONS
            )

            # Build results locally so a failed image doesn't
            # partially contribute to the metrics.
            image_similarities = {}
            current_rows = []
            current_similarity_values = []

            # ------------------------------------------
            # 5. Compare each heatmap with the original map.
            # ------------------------------------------

            for variation in VARIATIONS:
                similarity = trustcheck.heatmap_similarity(
                    heatmaps["original"],
                    heatmaps[variation],
                )

                if similarity is not None:
                    similarity = float(similarity)

                valid_similarity = (
                    similarity is not None
                    and np.isfinite(similarity)
                )

                image_similarities[variation] = (
                    round(similarity, 4)
                    if valid_similarity
                    else None
                )

                if variation != "original" and valid_similarity:
                    current_similarity_values.append(similarity)

                item = predictions[variation]

                result = {
                    "relative_path": image_path.relative_to(
                        DATA_DIR
                    ).as_posix(),
                    "actual_label": actual_label,
                    "variation": variation,
                    "prediction": item["prediction"],
                    "pneumonia_score": round(
                        item["pneumonia_score"], 6
                    ),
                    "score_change_from_original": round(
                        item["pneumonia_score"]
                        - baseline["pneumonia_score"],
                        6,
                    ),
                    "prediction_matches_original": (
                        item["prediction"] == baseline_prediction
                    ),
                    "correct": item["prediction"] == actual_label,
                    "heatmap_similarity_to_original": (
                        round(similarity, 4)
                        if valid_similarity
                        else ""
                    ),
                }

                current_rows.append(result)

            # ------------------------------------------
            # 6. Commit results after all image processing
            #    has completed successfully.
            # ------------------------------------------

            all_rows.extend(current_rows)
            original_rows.append(current_rows[0])
            similarity_values.extend(current_similarity_values)

            successful_images += 1
            consistent_count += int(is_consistent)

            image_results.append({
                "relative_path": image_path.relative_to(
                    DATA_DIR
                ).as_posix(),
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

        except (
            OSError,
            ValueError,
            UnidentifiedImageError,
            RuntimeError,
        ) as exc:
            errors.append({
                "relative_path": image_path.relative_to(
                    DATA_DIR
                ).as_posix(),
                "actual_label": actual_label,
                "error": str(exc),
            })

        if index % 10 == 0 or index == len(files):
            print(
                f"Processed {index}/{len(files)} images "
                f"| successful: {successful_images} "
                f"| errors: {len(errors)}"
            )

    # ==========================================
    # 4. SAVE DETAILED CSV
    # ==========================================

    csv_path = REPORT_DIR / "chexpert_robustness_results.csv"

    fieldnames = [
        "relative_path",
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
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # ==========================================
    # 5. BUILD SUMMARY
    # ==========================================

    summary = {
        "dataset": "CheXpert sample from Kaggle",
        "images_requested": len(files),
        "images_evaluated": successful_images,
        "images_with_errors": len(errors),

        "class_counts_evaluated": {
            label: sum(
                row["actual_label"] == label
                for row in original_rows
            )
            for label in LABELS
        },

        "original_model_metrics": calculate_metrics(original_rows),

        "metrics_by_variation": {
            variation: calculate_metrics([
                row for row in all_rows
                if row["variation"] == variation
            ])
            for variation in VARIATIONS
        },

        "prediction_consistency": {
            "consistent_images": consistent_count,
            "evaluated_images": successful_images,
            "rate": round(
                consistent_count / successful_images, 4
            ) if successful_images else None,
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
            "Research prototype only; not for clinical use. "
            "NORMAL_LIKE refers to the CheXpert No Finding=1 label "
            "and does not guarantee a healthy patient. The selected "
            "sample is small and balanced, so its accuracy is not an "
            "estimate of real-world prevalence or clinical performance. "
            "Verify checkpoint training provenance and dataset overlap."
        ),
    }

    # ==========================================
    # 6. SAVE JSON REPORTS
    # ==========================================

    summary_path = REPORT_DIR / "chexpert_evaluation_summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    details_path = REPORT_DIR / "chexpert_per_image_results.json"

    details_path.write_text(
        json.dumps(image_results, indent=2),
        encoding="utf-8",
    )

    # ==========================================
    # 7. PRINT FINAL SUMMARY
    # ==========================================

    print("\n========== CHEXPERT EVALUATION ==========")

    print(json.dumps({
        "images_requested": len(files),
        "images_evaluated": successful_images,
        "images_with_errors": len(errors),
        "original_model_metrics": summary["original_model_metrics"],
        "prediction_consistency": summary["prediction_consistency"],
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
