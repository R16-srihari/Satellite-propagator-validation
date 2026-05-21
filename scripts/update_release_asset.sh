#!/usr/bin/env bash
set -euo pipefail
# Usage: edit OWNER/REPO/TAG below or export env vars OWNER,REPO,TAG before running
OWNER=${OWNER:-R16-srihari}
REPO=${REPO:-Satellite-propagator-validation}
TAG=${TAG:-stk-latest}
FILES=(STK_input/Satellite1_Results.csv STK_input/Satellite1.opm)
TEMP_DIRS=()

version_suffix() {
  date -u +%Y%m%dT%H%M%SZ-$$-$RANDOM
}

cleanup() {
  for dir in "${TEMP_DIRS[@]}"; do
    rm -rf "$dir"
  done
}

trap cleanup EXIT

# Require at least one existing file
found_any=0
for f in "${FILES[@]}"; do
  if [[ -f "$f" ]]; then
    found_any=1
  fi
done
if [[ $found_any -eq 0 ]]; then
  echo "No release asset files found: ${FILES[*]}" >&2
  exit 1
fi

for FILE in "${FILES[@]}"; do
  if [[ ! -f "$FILE" ]]; then
    echo "Skipping missing file: $FILE"
    continue
  fi

  asset_base=$(basename "$FILE")
  asset_stem=${asset_base%.*}
  asset_ext=${asset_base##*.}
  asset_name="${asset_stem}-$(version_suffix).${asset_ext}"
  temp_dir=$(mktemp -d)
  TEMP_DIRS+=("$temp_dir")
  temp_file="$temp_dir/$asset_name"

  cp "$FILE" "$temp_file"

  echo "Uploading $FILE as $asset_name to release $TAG in $OWNER/$REPO"
  gh release upload $TAG "$temp_file" --repo $OWNER/$REPO
  echo "Upload complete for $FILE"
done
