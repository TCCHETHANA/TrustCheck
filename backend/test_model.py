
import torch
import torchxrayvision as xrv
from pathlib import Path
from PIL import Image
import numpy as np

DATA_DIR = Path(r"D:\TrustCheck\data\chest_xray\test")
SAMPLES_PER_CLASS = 20

model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.eval()

pneumonia_index = model.pathologies.index("Pneumonia")
threshold = float(model.op_threshs[pneumonia_index])

print(f"Model threshold: {threshold:.4f}\n")

for label in ["NORMAL", "PNEUMONIA"]:
    folder = DATA_DIR / label
    images = sorted(folder.glob("*.jpeg"))[:SAMPLES_PER_CLASS]

    scores = []
    flagged = 0

    for image_path in images:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((224, 224))
            array = np.asarray(img, dtype=np.float32)
        
        array = (array / 255.0) * 2048 - 1024
        array = array[np.newaxis, np.newaxis, :, :]

        with torch.no_grad():
            output = model(torch.from_numpy(array))

        score = output[0, pneumonia_index].item()
        scores.append(score)

        if score >= threshold:
            flagged += 1

    print(f"Dataset group: {label}")
    print(f"Images tested: {len(scores)}")
    print(f"Average Pneumonia score: {np.mean(scores):.4f}")
    print(f"Images above threshold: {flagged}/{len(scores)}")
    print()
