#!/usr/bin/env bash

UTILS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
WORKDIR=$(cat "$UTILS_DIR/../resources/config/user-ganesh.json" | python -c "import json,sys;obj=json.load(sys.stdin);print(obj['workdir']);")

# generated outputs
rsync -r -t -v --progress --delete -z -s --exclude-from="$UTILS_DIR/rsync-exclude.txt" ganesh@cuda7.informatik.uni-bonn.de:"$WORKDIR/generated" "$UTILS_DIR/../local_remote_out"

# tensorboard logs
#rsync -r -t -v --progress -z -s --exclude-from=/home/navrudh/Projects/Uni/cudavision/rsync-exclude-from-remote.txt ganesh@cuda4.informatik.uni-bonn.de:/home/user/ganesh/projects/cudavision/project/lightning_logs /home/navrudh/Projects/Uni/cudavision/project/local_remote_out

# ray tune
#rsync -r -t -v --progress --delete -z -s --exclude-from=/home/navrudh/Projects/Uni/cudavision/rsync-exclude-from-remote.txt ganesh@cuda4.informatik.uni-bonn.de:/home/user/ganesh/ray_results /home/navrudh/Projects/Uni/cudavision/project/local_remote_out
