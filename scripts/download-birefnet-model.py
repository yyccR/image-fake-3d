from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
REPO = "ZhengPeng7/BiRefNet_dynamic"
REVISION = "280306042f57b7a33854319da62fd86aaa89ec4c"
CACHE_DIR = ROOT / ".cache" / "birefnet"


def main() -> None:
    print("BiRefNet Dynamic is MIT licensed; loading reviewed code at a pinned revision.")
    snapshot = snapshot_download(
        repo_id=REPO,
        revision=REVISION,
        cache_dir=CACHE_DIR,
        allow_patterns=[
            "BiRefNet_config.py",
            "README.md",
            "birefnet.py",
            "config.json",
            "model.safetensors",
            "requirements.txt",
        ],
    )
    print(f"BiRefNet Dynamic snapshot is ready: {snapshot}")


if __name__ == "__main__":
    main()
