#!/bin/bash
# Inside a dev sim (dev_sim.sh exec): scorer + traced mission on the current layout. Prints the score.
#   bash tests/dev_mission.sh <digit> <label> <round> <dir under /opt/results> [extra bot args]
D=$1; L=$2; R=$3; DIR=/opt/results/$4; shift 4
cd /opt/bot
python3 /opt/gym/gym/scorer.py --layout $DIR/layout.json --digit $D --label $L --round $R --timeout 400 --out $DIR/score.json > $DIR/scorer.log 2>&1 &
SC=$!
sleep 3
OUT=$DIR/trace.json ERC_BOT_ORT_THREADS=4 python3 tests/t_mission.py $D $L ${MODEL:-/opt/model/det_v4_736x1280.onnx} $DIR/layout.json "$@" > $DIR/mission_trace.log 2>&1
sleep 5; kill -TERM $SC; wait $SC
grep -E "TRUTH" $DIR/mission_trace.log | python3 -c "
import sys, json
for l in sys.stdin:
    d = json.loads(l[6:]); print(d['t'], d['stage'], {k: v for k, v in d.items() if k in ('bin_err_xy', 'bin_est_world', 'tool_minus_bin', 'book_world', 'odom_drift_xy')})"
grep FAILED $DIR/mission_trace.log
python3 -c "import json; s=json.load(open('$DIR/score.json')); print('SCORE', s['score'], s['end_reason'], s['place_kind'], list(s['stages']), 'collisions', s['collisions'])"
