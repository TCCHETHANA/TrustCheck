
from pathlib import Path

DATA_DIR = Path(r"D:\TrustCheck\data\chest_xray")

for split in ["train", "test", "val"]:
    for label in ["NORMAL", "PNEUMONIA"]:
        folder = DATA_DIR / split / label

        if folder.exists():
            images = list(folder.glob("*.jpeg"))
            print(f"{split}/{label}: {len(images)} images")
        else:
            print(f"Missing folder: {folder}")
