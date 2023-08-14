#!/bin/sh
venv/bin/uvicorn gateway:app --host=0.0.0.0 --port=8000 &
pid1=$!
echo "started gateway: [${pid1}]"

venv/bin/uvicorn inference:app --host=0.0.0.0 --port=8080 &
pid2=$!
echo "started inference: [${pid2}]"

trap "kill -INT -2 $pid1 $pid2" INT
wait $pid1 $pid2
echo "Exited"
