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
  "git+https://github.com/apple/ml-sharp.git@$sharp_commit"

npm install
"$project_dir/scripts/download-sharp-model.sh"

echo "SHARP runtime is ready."
