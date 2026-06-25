from huggingface_hub import snapshot_download

MODEL_REPO = "Hamghalam/liver-cancer-segmentation-nnunet"
LOCAL_DIR = "./models"

snapshot_download(
    repo_id=MODEL_REPO,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False
)

print("Model downloaded successfully!")