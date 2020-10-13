#!/usr/bin/env bash

while getopts p:f: flag
do
	case "${flag}" in
		p) PID=${OPTARG};;
		f) FILE=${OPTARG};;
		*) echo "Unknown Option (supported: p,f)" && exit
	esac
done

PID_RUNNING=1

function is_pid_running {
	if ps -p "$PID" > /dev/null; then
		PID_RUNNING=1
	else
		PID_RUNNING=0
	fi
}

while [ $PID_RUNNING -eq 1 ]; do
  echo "$PID is running"
	is_pid_running
	sleep 5m
done

python "$FILE"
