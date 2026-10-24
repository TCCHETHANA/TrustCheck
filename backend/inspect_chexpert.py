
import pandas as pd

df = pd.read_csv(r"C:\Users\cheth\Downloads\train.csv.zip")

print("Total rows:", len(df))

print("\nImage views:")
print(df["Frontal/Lateral"].value_counts(dropna=False))

print("\nPneumonia labels:")
print(df["Pneumonia"].value_counts(dropna=False))

print("\nNo Finding labels:")
print(df["No Finding"].value_counts(dropna=False))

print("\nPneumonia vs No Finding:")
print(pd.crosstab(
    df["Pneumonia"],
    df["No Finding"],
    dropna=False
))

frontal = df[df["Frontal/Lateral"] == "Frontal"]

positive = frontal[frontal["Pneumonia"] == 1]
negative = frontal[frontal["No Finding"] == 1]

print("\n--- Frontal-view candidate groups ---")
print("Pneumonia-positive:", len(positive))
print("No Finding:", len(negative))

print("\nFirst pneumonia image path:")
print(positive["Path"].iloc[0] if len(positive) else "None")

print("\nFirst No Finding image path:")
print(negative["Path"].iloc[0] if len(negative) else "None")


from pathlib import Path

# Change this to the folder containing the extracted CheXpert dataset
DATASET_ROOT = Path(r"D:\TrustCheck\data\CheXpert-v1.0-small")

def check_image(path):
    # CSV paths start with CheXpert-v1.0-small/
    relative_path = str(path).replace(
        "CheXpert-v1.0-small/", "", 1
    )
    return (DATASET_ROOT / relative_path).is_file()

print("\n--- Checking image files ---")

positive_paths = positive["Path"].head(10)
negative_paths = negative["Path"].head(10)

print("Pneumonia images found:",
      sum(check_image(p) for p in positive_paths), "/ 10")

print("No Finding images found:",
      sum(check_image(p) for p in negative_paths), "/ 10")
