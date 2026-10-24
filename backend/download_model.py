
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="luanacarolina/pneumonia-chest-xray-classifier",
    filename="checkpoints/binary/densenet121.pt",
    local_dir=r"D:\TrustCheck\models",
)

print("Download complete!")
print(path)
