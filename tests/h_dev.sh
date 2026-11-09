#!/bin/bash
# Lane H, inside a dev sim (dev_sim.sh exec): scorer + truth logger + the mission through lane F's perturbation proxy.
#   ERC_PERT_ODOM_SCALE=0.03 ERC_PERT_ODOM_YAW=0.03 bash tests/h_dev.sh <digit> <label> <round> <dir under /opt/results> [bot args]
D=$1; L=$2; R=$3; DIR=/opt/results/$4; shift 4
mkdir -p $DIR; rm -f $DIR/mission.json $DIR/score.json $DIR/truth.jsonl
cd /opt/bot
TGT=$(python3 -c "
import json; L=json.load(open('$DIR/layout.json')); col=L['markers'].index($D)+1; lab='$L'.lower()
print([b['name'] for b in L['books'] if b['column']==col and (b['name'].endswith('_'+lab) or b['name'].endswith('_white_'+'$L'.upper()))][0])")
python3 /opt/gym/gym/scorer.py --layout $DIR/layout.json --digit $D --label $L --round $R --timeout 400 --out $DIR/score.json > $DIR/scorer.log 2>&1 &
SC=$!
python3 tests/h_truth.py $TGT $DIR/truth.jsonl > $DIR/truth.log 2>&1 &
TR=$!
sleep 3
ERC_BOT_ORT_THREADS=4 bash /opt/bot/pert2/perturb_run.sh $DIR/pert python3 -m erc_bot.mission --digit $D --label $L --sim-time \
  --model ${MODEL:-/opt/model/det_v4_736x1280.onnx} --out $DIR "$@" > $DIR/solution.log 2>&1
sleep 5; kill -TERM $SC; wait $SC; kill $TR
python3 -c "import json; s=json.load(open('$DIR/score.json')); m=json.load(open('$DIR/mission.json')); print('SCORE', s['score'], s['end_reason'], s['place_kind'], list(s['stages']), 'collisions', s['collisions'], m.get('result'), m.get('reason'), m.get('last_stage'))"
