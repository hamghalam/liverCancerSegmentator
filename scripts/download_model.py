import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

MODEL_REPO = "Hamghalam/liver-cancer-segmentation-nnunet"

token = os.getenv("HF_TOKEN")

snapshot_download(
    repo_id=MODEL_REPO,
    local_dir="./models",
    token=token
)

print("Model downloaded successfully!")