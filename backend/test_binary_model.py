
import torch
from torchvision import models, transforms
from pathlib import Path
from PIL import Image

DATA_DIR = Path(r"D:\TrustCheck\data\chest_xray\test")
CHECKPOINT = Path(
    r"D:\TrustCheck\models\checkpoints\binary\densenet121.pt"
)

model = models.densenet121(weights=None)
model.classifier = torch.nn.Linear(model.classifier.in_features, 1)

state = torch.load(
    CHECKPOINT, map_location="cpu", weights_only=True
)
model.load_state_dict(state.get("model_state_dict", state))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

for label in ["NORMAL", "PNEUMONIA"]:
    folder = DATA_DIR / label
    images = sorted(folder.glob("*.jpeg"))[:20]
    scores = []

    for path in images:
        with Image.open(path) as img:
            image = preprocess(img.convert("RGB")).unsqueeze(0)

        with torch.no_grad():
            logit = model(image)
            score = torch.sigmoid(logit).item()

        scores.append(score)

    if scores:
        avg = sum(scores) / len(scores)
        flagged = sum(score >= 0.5 for score in scores)
        print(f"\nDataset group: {label}")
        print(f"Images tested: {len(scores)}")
        print(f"Average pneumonia score: {avg:.4f}")
        print(f"Scores >= 0.5: {flagged}/{len(scores)}")
    else:
        print(f"No JPEG images found in {folder}")
