
import torch
from torchvision import models
from pathlib import Path

checkpoint = Path(
    r"D:\TrustCheck\models\checkpoints\binary\densenet121.pt"
)

model = models.densenet121(weights=None)
model.classifier = torch.nn.Linear(
    model.classifier.in_features, 1
)

state = torch.load(
    checkpoint,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state.get("model_state_dict", state))
model.eval()

print("Model loaded successfully!")
print("Architecture: DenseNet121")
print("Output classes: NORMAL / PNEUMONIA")
print("Device: CPU")
print("Parameters loaded:", len(model.state_dict()))
