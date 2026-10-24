
from pathlib import Path
import subprocess
import sys
import pandas as pd

# Project paths
DATA_DIR = Path(r"D:\TrustCheck\data")
CSV_PATH = DATA_DIR / "CheXpert-v1.0-small" / "train.csv"
OUTPUT_DIR = DATA_DIR / "chexpert_eval_100"

# Sample size per group
SAMPLE_SIZE = 50
RANDOM_SEED = 42

def main():
    if not CSV_PATH.is_file():
        print(f"ERROR: CSV not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)

    # Keep frontal X-rays with definite labels only
    frontal = df[df["Frontal/Lateral"] == "Frontal"].copy()

    positive = frontal[frontal["Pneumonia"] == 1].copy()
    negative = frontal[frontal["No Finding"] == 1].copy()

    if len(positive) < SAMPLE_SIZE or len(negative) < SAMPLE_SIZE:
        print("ERROR: Not enough eligible images.")
        return

    positive = positive.sample(
        n=SAMPLE_SIZE, random_state=RANDOM_SEED
    )
    negative = negative.sample(
        n=SAMPLE_SIZE, random_state=RANDOM_SEED
    )

    positive["trustcheck_label"] = "PNEUMONIA"
    negative["trustcheck_label"] = "NORMAL_LIKE"

    sample = pd.concat([positive, negative], ignore_index=True)
    sample = sample.sample(frac=1, random_state=RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the exact sample and its labels for reproducibility
    manifest_path = OUTPUT_DIR / "sample_manifest.csv"
    sample.to_csv(manifest_path, index=False)

    print(f"Selected {len(sample)} images.")
    print(f"Manifest saved to: {manifest_path}")
    print("Starting individual image downloads...")

    downloaded = 0
    failed = []

    for index, row in sample.iterrows():
        # CSV paths start with CheXpert-v1.0-small/
        relative_path = str(row["Path"]).replace(
            "CheXpert-v1.0-small/", "", 1
        )

        destination = OUTPUT_DIR / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.is_file() and destination.stat().st_size > 0:
            downloaded += 1
            continue

        command = [
            "kaggle", "datasets", "download",
            "-d", "ashery/chexpert",
            "-f", relative_path,
            "-p", str(destination.parent),
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=120,
            )

            # Kaggle may save the requested image by basename
            if result.returncode == 0 and destination.is_file():
                downloaded += 1
            elif result.returncode == 0:
                # Check whether a file with the expected basename exists
                alternate = destination.parent / Path(relative_path).name
                if alternate.is_file():
                    if alternate != destination:
                        alternate.rename(destination)
                    downloaded += 1
                else:
                    failed.append((row["Path"], "Downloaded file not found"))
            else:
                failed.append((row["Path"], result.stderr.strip()))

        except subprocess.TimeoutExpired:
            failed.append((row["Path"], "Download timed out"))
        except Exception as exc:
            failed.append((row["Path"], str(exc)))

        if (index + 1) % 10 == 0:
            print(
                f"Processed {index + 1}/{len(sample)} "
                f"| downloaded: {downloaded} | failed: {len(failed)}"
            )

    if failed:
        pd.DataFrame(
            failed, columns=["Path", "Error"]
        ).to_csv(OUTPUT_DIR / "download_failures.csv", index=False)

    print("\nDownload summary")
    print(f"Selected: {len(sample)}")
    print(f"Downloaded/already present: {downloaded}")
    print(f"Failed: {len(failed)}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")

    if downloaded == len(sample):
        print("\nAll sample images are ready for verification.")
    else:
        print("\nSome images need troubleshooting before evaluation.")

if __name__ == "__main__":
    main()
