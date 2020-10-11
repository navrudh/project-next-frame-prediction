#!/usr/bin/env bash

UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
DATASET_PATH=$(cat "$UTILS_DIR/../resources/config/user-$USER.json" | python -c "import json,sys;obj=json.load(sys.stdin);print(obj['ucf101']['path']);")

if ! [ -x "$(command -v kaggle)" ]; then
  echo "kaggle-cli not found. Installing it."
  pip install kaggle
fi



FILE=ucf101.zip

if [[ -f "$FILE" ]]; then
   echo "$FILE exists"
else
   kaggle datasets download -d pevogam/ucf101
fi

mkdir -p "$DATASET_PATH/"
mv "$FILE" "$DATASET_PATH/"
cd "$DATASET_PATH/" || exit 1
7z x "$FILE"
