#!/usr/bin/env bash

if ! [ -x "$(command -v kaggle)" ]; then
  echo "kaggle-cli not found. Installing it."
  pip install kaggle
fi

kaggle datasets download -d pevogam/ucf101

UTILS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATASET_PATH=$(cat "$UTILS_DIR/../resources/config/user-$USER.json" | python -c "import json,sys;obj=json.load(sys.stdin);print(obj['ucf101']['path']);")

FILE=ucf101.zip
[[ -f "$FILE" ]] && echo "$FILE exists." || {echo "$FILE does not exist." && exit 1}

mkdir -p "$DATASET_PATH/"
mv "$FILE" "$DATASET_PATH/"
cd "$DATASET_PATH/"
7z x "$FILE"
