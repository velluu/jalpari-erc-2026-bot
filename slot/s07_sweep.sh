#!/bin/bash
# Slot step 7: one head sweep at the current base position, recorded for the detector fine-tune.
#   bash slot/s07_sweep.sh pos1 [head_sweep.py args, e.g. --tilts=-0.3,-0.6 for the bin]
# Starts a cam bag, waits 3 s, runs the kit's head sweep (inside the head caps and ranges), stops the bag, checks it.
POS=${1:?position name}; shift
D=$(cd "$(dirname "$0")" && pwd); OUT=${SLOT_OUT:-$PWD/slot_out}
SWEEP=$D/head_sweep.py; [ -f "$SWEEP" ] || SWEEP=$D/../tools/head_sweep.py
NAME=sweep_${POS}_$(date +%H%M%S)
bash $D/rec.sh cam $NAME > $OUT/$NAME.rec.log 2>&1 &
REC=$!
sleep ${REC_WARMUP:-3}
kill -0 $REC 2>/dev/null || { echo "FAIL s07/$POS: recorder died -> $(tail -2 $OUT/$NAME.rec.log)"; exit 1; }
python3 $SWEEP "$@" 2>&1 | tee $OUT/$NAME.sweep.log
RC=${PIPESTATUS[0]}
sleep 1; kill -INT $REC; wait $REC 2>/dev/null
INFO=$(ros2 bag info $OUT/bags/$NAME 2>/dev/null)
NCOL=$(echo "$INFO" | grep "image_raw/compressed " | grep -o "Count: [0-9]*" | grep -o "[0-9]*")
NDEP=$(echo "$INFO" | grep "compressedDepth" | grep -o "Count: [0-9]*" | grep -o "[0-9]*")
DUR=$(echo "$INFO" | grep "^Duration" | awk '{print $2}')
if [ "$RC" = "0" ] && [ "${NCOL:-0}" -gt 10 ]; then
  echo "PASS s07/$POS: sweep error_code 0, bag $NAME ${DUR} with ${NCOL} colour and ${NDEP:-0} depth frames"
else
  echo "FAIL s07/$POS: sweep rc $RC, ${NCOL:-0} colour frames -> rejection text is in $OUT/$NAME.sweep.log; smaller --pan or --speed 0.15"
fi
echo "{\"script\": \"s07_sweep\", \"check\": \"$POS\", \"ok\": $([ "$RC" = "0" ] && [ "${NCOL:-0}" -gt 10 ] && echo true || echo false), \"colour\": ${NCOL:-0}, \"depth\": ${NDEP:-0}, \"bag\": \"$NAME\"}" >> $OUT/verdicts.jsonl
