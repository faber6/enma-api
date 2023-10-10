#!/bin/sh
cd $(dirname "$0")

venv/bin/uvicorn inference:app --host=0.0.0.0 --port=8080 &
pid1=$!
#sudo cpulimit --limit 100 -b -z -p ${pid1}
echo "started inference: [${pid1}]"

venv/bin/uvicorn gateway:app --host=0.0.0.0 --port=8000 &
pid2=$!
echo "started gateway: [${pid2}]"

trap "kill -INT -2 $pid1 $pid2" INT
wait $pid1 $pid2
echo "Exited"
