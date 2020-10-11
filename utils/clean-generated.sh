#!/usr/bin/env bash

UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
WORKDIR=$(cat "$UTILS_DIR/../resources/config/user-$USER.json" | python -c "import json,sys;obj=json.load(sys.stdin);print(obj['workdir']);")

rm -rf "$WORKDIR/generated"
