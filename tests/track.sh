#!/bin/bash
# Debug tracker for the dev sim: the target book's and the bin's true poses once a second (gz ground truth).
B=$1; prev=""
while true; do
  p=$(timeout 3 gz model -m $B -p 2>/dev/null | grep -A1 XYZ | tail -1 | tr -s " ")
  [ "$p" != "$prev" ] && echo "$(date +%H:%M:%S) $B $p"
  prev=$p; sleep 1
done
