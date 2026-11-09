#!/bin/bash
# Inside a dev sim: scorer in the background (collisions), then any test script, then the scorer's collisions.
#   bash tests/dev_run.sh <results dir> <digit> <label> <round> <python args...>
DIR=/opt/results/$1; D=$2; L=$3; R=$4; shift 4
cd /opt/bot
python3 /opt/gym/gym/scorer.py --layout $DIR/layout.json --digit $D --label $L --round $R --timeout 900 --out $DIR/score_run.json > $DIR/scorer_run.log 2>&1 &
SC=$!; sleep 3
ERC_BOT_ORT_THREADS=4 timeout 900 python3 "$@" 2>&1 | grep -v INFO
sleep 2; kill -TERM $SC; wait $SC
python3 -c "import json; s=json.load(open('$DIR/score_run.json')); print('COLLISIONS', s['collisions'], 'placing contacts', s['placing_contacts'])"
