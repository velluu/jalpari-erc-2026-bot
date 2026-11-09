#!/bin/bash
# The slot, one command per step:   bash slot/step.sh 1   ...   bash slot/step.sh 9      (or: bash slot/step.sh fallback)
# Every step ends with STEP n PASSED / STEP n FAILED in plain words and a NEXT line saying exactly what to run.
# Recorders run in the background (started by steps 1, 7, 8 and fallback, stopped by step 9).
S=$(cd "$(dirname "$0")" && pwd)
OUT=${SLOT_OUT:-$PWD/slot_out}; export SLOT_OUT=$OUT; mkdir -p $OUT/logs
STEP=${1:?usage: bash slot/step.sh 1..9 or fallback}
BOOK_PROMPT=${SLOT_YES:+--yes}
say(){ echo -e "\033[1m$*\033[0m"; }
mark(){ [ -f $OUT/verdicts.jsonl ] && wc -l < $OUT/verdicts.jsonl || echo 0; }
late(){ # late HH:MM -> true if the clock is past it (SLOT_TIMECHECK=0 turns this off, e.g. in the simulator)
  [ "${SLOT_TIMECHECK:-1}" = 0 ] && return 1; [ "$(date +%H%M)" \> "${1/:/}" ]; }
rec_start(){ # rec_start light|cam name
  local pidf=$OUT/rec_$1.pid
  if [ -f $pidf ] && kill -0 $(cat $pidf) 2>/dev/null; then return; fi
  nohup bash $S/rec.sh $1 $2 > $OUT/logs/rec_$2.log 2>&1 & echo $! > $pidf; sleep ${REC_WARMUP:-3}
  kill -0 $(cat $pidf) 2>/dev/null && echo "recorder $1 running ($2)" || echo "WARNING: recorder $1 did not start: $(tail -1 $OUT/logs/rec_$2.log)"; }
rec_stop(){ local pidf=$OUT/rec_$1.pid; [ -f $pidf ] || return 0
  local p=$(cat $pidf); kill -INT $p 2>/dev/null; for i in $(seq 1 20); do kill -0 $p 2>/dev/null || break; sleep 0.5; done; rm -f $pidf; }
run(){ # run logname cmd...   (output to screen and logs/). A script that ends without its summary line crashed:
  # that becomes a FAIL verdict, so a crash can never read as PASSED.
  echo "--- $*"; local l0=0; [ -f $OUT/logs/$1.log ] && l0=$(wc -l < $OUT/logs/$1.log)
  "${@:2}" 2>&1 | tee -a $OUT/logs/$1.log | grep -E "^(PASS|FAIL|==|     |\[slot\]|\[auto\]|Book|Open|sweep)"; local rc=${PIPESTATUS[0]}
  if [ "$2" = python3 ] && ! tail -n +$((l0+1)) $OUT/logs/$1.log | grep -qE "^== |^GO|^NO-GO"; then
    echo "FAIL $1: crashed (exit $rc), last line: $(tail -1 $OUT/logs/$1.log)"
    python3 -c "import json,sys; print(json.dumps({'script': sys.argv[1], 'check': 'crashed', 'ok': False, 'detail': sys.argv[2][-300:], 'on_fail': 'the script stopped with an error (saved in logs/' + sys.argv[1] + '.log). Carry on with NEXT'}))" "$1" "$(tail -3 $OUT/logs/$1.log | tr '\n' ' ')" >> $OUT/verdicts.jsonl
  fi
  return $rc; }
report(){ # report n m0 next_ok next_fail [critical-regex]
  local n=$1 m0=$2 nok=$3 nfail=$4 crit=${5:-}
  local fails=$(tail -n +$((m0+1)) $OUT/verdicts.jsonl 2>/dev/null | python3 -c "
import json,sys
for l in sys.stdin:
    r=json.loads(l)
    if not r['ok']: print(r['script']+'/'+r['check']+'|'+(r.get('on_fail') or 'carry on'))")
  echo
  if [ -z "$fails" ]; then say "STEP $n PASSED."; say "NEXT: $nok"; return 0; fi
  say "STEP $n FAILED these checks:"
  echo "$fails" | while IFS='|' read -r c f; do echo "  - $c"; echo "      what to do: $f"; done
  if [ -n "$crit" ] && echo "$fails" | cut -d'|' -f1 | grep -qE "$crit"; then say "NEXT: $nfail"; else say "NEXT (these failures are recorded, they do not stop the plan): $nok"; fi
  return 1; }
m0=$(mark)
case $STEP in
1|connect)
  say "STEP 1: connection check (1 to 2 min)"
  rec_start light slot_light_$(date +%H%M%S)
  run s00 python3 $S/s00_preflight.py
  report 1 $m0 "bash slot/step.sh 2" \
    "fix the item above, then run step 1 again. Twice failed on topics or actions: ask the organisers. If it still fails at 13:35, run: bash slot/step.sh fallback" \
    "rate /joint_states|rate /mobile_base_controller/odom|rate /scan|compressed|action servers|robot_description|rmem_max|cyclone profile|domain|rmw" ;;
2|robot)
  say "STEP 2: robot description, joint limits, camera, margin probe (2 to 3 min; torso and head move a little)"
  run s01 python3 $S/s01_urdf.py
  report 2 $m0 "bash slot/step.sh 3" "bash slot/step.sh 3" ;;
3|gripper)
  say "STEP 3: gripper on air, then the right arm reaches forward with the gripper open (2 to 3 min)"
  run s02air python3 $S/s02_gripper.py air
  run s02present python3 $S/s02_gripper.py present
  if tail -n +$((m0+1)) $OUT/verdicts.jsonl 2>/dev/null | grep -q '"check": "present pose", "ok": true'; then
    report 3 $m0 "the robot is still now. One person slides a competition book (coloured spine or word book, 3 cm) between the open fingers (spine facing out), steps back 1.5 m. Then: bash slot/step.sh 4 (have a tape measure ready)" "bash slot/step.sh 5 (book test skipped)"
  else
    report 3 $m0 "bash slot/step.sh 5 (book test skipped: no present pose)" "bash slot/step.sh 5 (book test skipped: no present pose)" "present pose"
  fi ;;
4|book)
  say "STEP 4: close the gripper on the book, read it, look at it, tape the held book's bottom edge from the floor (cm), open (the book drops) (2 min)"
  run s02book python3 $S/s02_gripper.py book $BOOK_PROMPT
  run calib python3 $S/calib.py
  report 4 $m0 "bash slot/step.sh 5" "bash slot/step.sh 5" ;;
5|motion)
  if late 13:45; then say "STEP 5 SKIPPED: it is after 13:45."; say "NEXT: bash slot/step.sh 8"; exit 0; fi
  say "STEP 5: command latency and the checking layer (6 to 8 min; head, torso, right arm and gripper make small moves)"
  run s04 python3 $S/s04_latency.py
  run s05 python3 $S/s05_reject.py
  run s02park python3 $S/s02_gripper.py park
  run calib python3 $S/calib.py
  if late 13:35; then N="bash slot/step.sh 8 (it is after 13:35: steps 6 and 7 skipped)"; else N="bash slot/step.sh 6"; fi
  report 5 $m0 "$N" "$N" ;;
6|camera)
  if late 13:35; then say "STEP 6 SKIPPED: it is after 13:35."; say "NEXT: bash slot/step.sh 8"; exit 0; fi
  say "STEP 6: camera sample, arms tucked, then odometry drift (3 to 4 min). Keep 1 m clear around the base, it drives 30 cm and turns"
  run s03 python3 $S/s03_camera.py --tag shelf
  run s02park python3 $S/s02_gripper.py park      # arms tucked before the base moves (the laser cannot see an arm, sim dry run 21:00: an unparked left arm hit the table)
  run s06 python3 $S/s06_odom.py
  report 6 $m0 "bash slot/step.sh 7" "bash slot/step.sh 7" ;;
7|sweep)
  if late 13:40; then say "STEP 7 SKIPPED: it is after 13:40."; say "NEXT: bash slot/step.sh 8"; exit 0; fi
  say "STEP 7: head sweeps recorded for the detector, from where the robot stands (1 to 2 min, only the head moves)"
  # no base rotation here: turning 180 deg in place put the parked right elbow on the table in the sim dry run (22:30),
  # the laser cannot see an arm. The mission (step 8) turns to the bin itself and step 8 records the camera throughout.
  run s07a bash $S/s07_sweep.sh shelf
  run s07b bash $S/s07_sweep.sh low --tilts=-0.3,-0.6
  report 7 $m0 "bash slot/step.sh 8" "bash slot/step.sh 8" ;;
8|bot)
  say "STEP 8: the bot's full mission, slowly (6 to 10 min). It chooses scripted or learned stages itself and says why."
  say "Digit and label: ${SLOT_DIGIT:=2} ${SLOT_LABEL:=red} (set SLOT_DIGIT / SLOT_LABEL before this command if the organisers set another book)"
  rec_start light slot_light_$(date +%H%M%S)
  rec_start cam slot_bot_$(date +%H%M%S)
  run s08 python3 $S/s08_bot.py --auto --digit $SLOT_DIGIT --label $SLOT_LABEL --speed ${SLOT_SPEED:-0.5} --tag $(date +%H%M%S)
  rec_stop cam
  if late 13:47; then N="bash slot/step.sh 9"; else N="bash slot/step.sh 8 once more if it failed (put the book back first), otherwise bash slot/step.sh 9"; fi
  report 8 $m0 "bash slot/step.sh 9" "$N" "mission" ;;
9|collect)
  say "STEP 9: stop the recorders and pack everything into one file (1 min)"
  rec_stop cam; rec_stop light
  bash $S/collect.sh ;;
fallback)
  say "FALLBACK (10 min): recording plus one scripted mission, then collect"
  rec_start light slot_light_fb_$(date +%H%M%S)
  rec_start cam slot_fb_$(date +%H%M%S)
  run s00 python3 $S/s00_preflight.py --no-buftest
  run s08 python3 $S/s08_bot.py --auto --digit ${SLOT_DIGIT:-2} --label ${SLOT_LABEL:-red} --speed ${SLOT_SPEED:-0.5} --tag fallback
  rec_stop cam; rec_stop light
  bash $S/collect.sh ;;
*) echo "usage: bash slot/step.sh 1..9 or fallback"; exit 2 ;;
esac
