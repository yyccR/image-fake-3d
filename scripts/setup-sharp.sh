#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
sharp_commit="1eaa046834b81852261262b41b0919f5c1efdd2e"

cd "$project_dir"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/"
  exit 1
fi

uv venv --python 3.13 .venv
UV_CACHE_DIR="${TMPDIR:-/tmp}/image3d-uv-cache" uv pip install \
  --python .venv/bin/python \
  "git+https://github.com/apple/ml-sharp.git@$sharp_commit" \
  "transformers==5.14.1" \
  "einops==0.8.1" \
  "kornia==0.8.1" \
  "opencv-python-headless==5.0.0.93"

npm install
"$project_dir/scripts/download-sharp-model.sh"
"$project_dir/.venv/bin/python" "$project_dir/scripts/download-birefnet-model.py"

echo "SHARP, BiRefNet Dynamic, Depth Anything V2 Small, and OpenCV repair runtimes are ready."
