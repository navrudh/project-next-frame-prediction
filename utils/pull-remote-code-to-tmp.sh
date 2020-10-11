#!/usr/bin/env bash

UTILS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

rsync -r -t -v --progress --delete -z -s --exclude-from="$UTILS_DIR/rsync-exclude.txt" ganesh@cuda7.informatik.uni-bonn.de:/home/user/ganesh/projects /tmp
