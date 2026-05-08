"""Utility script to prefetch the Donut OCR model."""

from huggingface_hub import snapshot_download

MODEL_ID = "chinmays18/medical-prescription-ocr"


def download_model(local_dir: str = "model-cache") -> None:
    """Download Donut OCR model files into local cache directory."""
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Model downloaded to: {local_dir}")


if __name__ == "__main__":
    download_model()
