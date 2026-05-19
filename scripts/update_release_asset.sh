#!/usr/bin/env bash
set -euo pipefail
# Usage: edit OWNER/REPO/TAG below or export env vars OWNER,REPO,TAG before running
OWNER=${OWNER:-R16-srihari}
REPO=${REPO:-RK-7-8-integrator-validation}
TAG=${TAG:-stk-latest}
FILE=STK_input/Satellite1_Results.csv

if [[ ! -f "$FILE" ]]; then
  echo "File not found: $FILE" >&2
  exit 1
fi

# Try to get existing asset id and delete it if present
asset_id=$(gh api repos/$OWNER/$REPO/releases/tags/$TAG --jq ".assets[] | select(.name==\"$(basename $FILE)\") | .id" 2>/dev/null || true)
if [[ -n "$asset_id" ]]; then
  echo "Deleting existing asset id $asset_id"
  gh api repos/$OWNER/$REPO/releases/assets/$asset_id -X DELETE
fi

echo "Uploading $FILE to release $TAG in $OWNER/$REPO"
gh release upload $TAG $FILE --repo $OWNER/$REPO --clobber
echo "Upload complete"
