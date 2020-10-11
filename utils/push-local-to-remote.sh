#!/usr/bin/env bash

UTILS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

rsync -r -t -v --progress -z -s --exclude-from="$UTILS_DIR/rsync-exclude.txt" "$UTILS_DIR/../../project" ganesh@cuda7.informatik.uni-bonn.de:/home/user/ganesh/cudavision
