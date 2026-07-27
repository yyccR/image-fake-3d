#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "$0")/.." && pwd)"
model_tag="sharp-model-2572gikvuh"
model_name="sharp_2572gikvuh.pt"
model_size="2809738232"
model_sha256="94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24f073d"
release_base_url="${SHARP_MODEL_RELEASE_BASE_URL:-https://github.com/yyccR/image-fake-3d/releases/download/$model_tag}"
model_dir="${SHARP_MODEL_DIR:-$project_dir/.cache/sharp}"
model_path="$model_dir/$model_name"
download_dir="$model_dir/.downloads/$model_tag"

parts=(
  "$model_name.part-00"
  "$model_name.part-01"
)
part_sizes=(
  "1572864000"
  "1236874232"
)
part_sha256s=(
  "482f222f7088efa93dc0b462ab9694b61e84873127b83819ca4b130fb180348a"
  "6718de3534413555de3ac724e899924b12b503bc08cfc35c3a24f4a27ef10ebe"
)

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    echo "A SHA-256 tool (shasum or sha256sum) is required." >&2
    return 1
  fi
}

file_is_valid() {
  local path="$1"
  local expected_size="$2"
  local expected_sha256="$3"
  [[ -f "$path" ]] || return 1
  [[ "$(wc -c < "$path" | tr -d ' ')" == "$expected_size" ]] || return 1
  [[ "$(sha256_file "$path")" == "$expected_sha256" ]]
}

if file_is_valid "$model_path" "$model_size" "$model_sha256"; then
  echo "SHARP checkpoint is already verified: $model_path"
  exit 0
fi

echo "Apple's SHARP checkpoint is licensed for non-commercial research only."
echo "Review $project_dir/third_party/apple-sharp/LICENSE_MODEL before use."
mkdir -p "$download_dir"

for index in "${!parts[@]}"; do
  part_name="${parts[$index]}"
  part_path="$download_dir/$part_name"
  if file_is_valid "$part_path" "${part_sizes[$index]}" "${part_sha256s[$index]}"; then
    echo "Using verified part: $part_name"
    continue
  fi

  current_size=0
  if [[ -f "$part_path" ]]; then
    current_size="$(wc -c < "$part_path" | tr -d ' ')"
  fi
  if (( current_size >= part_sizes[index] )); then
    rm -f "$part_path"
  fi

  echo "Downloading $part_name"
  curl --location --fail --retry 3 --continue-at - \
    --output "$part_path" "$release_base_url/$part_name"

  if ! file_is_valid "$part_path" "${part_sizes[$index]}" "${part_sha256s[$index]}"; then
    echo "Resume verification failed for $part_name; retrying from byte zero." >&2
    rm -f "$part_path"
    curl --location --fail --retry 3 \
      --output "$part_path" "$release_base_url/$part_name"
    if ! file_is_valid "$part_path" "${part_sizes[$index]}" "${part_sha256s[$index]}"; then
      echo "Checksum or size mismatch for $part_name." >&2
      exit 1
    fi
  fi
done

mkdir -p "$model_dir"
assembled_path="$(mktemp "$model_dir/.${model_name}.assembling.XXXXXX")"
cleanup() {
  rm -f "$assembled_path"
}
trap cleanup EXIT

for part_name in "${parts[@]}"; do
  command cat "$download_dir/$part_name" >> "$assembled_path"
done

if ! file_is_valid "$assembled_path" "$model_size" "$model_sha256"; then
  echo "Reassembled checkpoint failed verification." >&2
  exit 1
fi

mv -f "$assembled_path" "$model_path"
trap - EXIT
rm -f "${parts[@]/#/$download_dir/}"
rmdir "$download_dir" 2>/dev/null || true
echo "SHARP checkpoint verified: $model_path"
