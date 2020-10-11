#!/usr/bin/env bash

UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
WORKDIR=$(cat "$UTILS_DIR/../resources/config/user-ganesh.json" | python -c "import json,sys;obj=json.load(sys.stdin);print(obj['workdir']);")
REMOTE=$(cat "$UTILS_DIR/../resources/local-remote-sync.json" | python -c "import json,sys;obj=json.load(sys.stdin);print(obj['remote']);")

# generated outputs
rsync -r -t -v --progress --delete -z -s --exclude-from="$UTILS_DIR/rsync-exclude.txt" "$REMOTE:$WORKDIR/generated" "$UTILS_DIR/../local_remote_out"
