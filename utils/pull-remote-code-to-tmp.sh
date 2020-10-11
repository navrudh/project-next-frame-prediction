#!/usr/bin/env bash

UTILS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REMOTE=$(cat "$UTILS_DIR/../resources/local-remote-sync.json" | python -c "import json,sys;obj=json.load(sys.stdin);print(obj['remote']);")
REMOTE_HOME=$(cat "$UTILS_DIR/../resources/local-remote-sync.json" | python -c "import json,sys;obj=json.load(sys.stdin);print(obj['remote-home']);")

rsync -r -t -v --progress --delete -z -s --exclude-from="$UTILS_DIR/rsync-exclude.txt" "$REMOTE:$REMOTE_HOME"/cudavision /tmp
