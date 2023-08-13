#!/bin/sh
venv/bin/python gateway.py &
pid1=$!
echo "started gateway: [${pid1}]"

venv/bin/python inference.py &
pid2=$!
echo "started inference: [${pid2}]"

trap "kill -INT -2 $pid1 $pid2" INT
wait $pid1 $pid2
echo "Exited"
