#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
sharp_commit="1eaa046834b81852261262b41b0919f5c1efdd2e"
model_url="https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
model_path="$project_dir/.cache/sharp/sharp_2572gikvuh.pt"
model_size="2809738232"

cd "$project_dir"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/"
  exit 1
fi

uv venv --python 3.13 .venv
UV_CACHE_DIR="${TMPDIR:-/tmp}/image3d-uv-cache" uv pip install \
  --python .venv/bin/python \
  "git+https://github.com/apple/ml-sharp.git@$sharp_commit"

npm install
mkdir -p "$(dirname "$model_path")"
current_size="0"
if [[ -f "$model_path" ]]; then
  current_size="$(wc -c < "$model_path" | tr -d ' ')"
fi
if [[ "$current_size" != "$model_size" ]]; then
  curl -L --fail --retry 3 -C - -o "$model_path" "$model_url"
fi

actual_size="$(wc -c < "$model_path" | tr -d ' ')"
if [[ "$actual_size" != "$model_size" ]]; then
  echo "Checkpoint size mismatch: expected $model_size bytes, got $actual_size bytes."
  exit 1
fi

echo "SHARP runtime is ready."
