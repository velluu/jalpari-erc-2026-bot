#!/bin/bash
# Pack the session into ONE file to send home:   bash slot/collect.sh
# Inside: every result .json, verdicts, timings, logs, the robot's URDF, the calibration, camera samples, the light bags
# (joints, TF, odometry, commands, rejections) in full, and an excerpt of every camera bag (a frame every 2 s).
# The full camera bags stay in $SLOT_OUT/bags (list with sizes and md5 inside the file): copy them to a USB stick if asked.
S=$(cd "$(dirname "$0")" && pwd); OUT=${SLOT_OUT:-$PWD/slot_out}
STAMP=$(date +%Y%m%d_%H%M); NAME=jalpari_slot_$STAMP; T=$(mktemp -d); P=$T/$NAME; mkdir -p $P/bag_excerpts $P/light_bags
for f in $OUT/*.json $OUT/*.jsonl $OUT/*.urdf $OUT/*.log; do [ -f "$f" ] && cp "$f" $P/; done
[ -d $OUT/logs ] && cp -r $OUT/logs $P/
for d in $OUT/camera_* $OUT/bot_*; do [ -d "$d" ] && cp -r "$d" $P/; done
for b in $OUT/bags/*/; do b=${b%/}; n=$(basename $b)
  case $n in
    *light*|dry_light*) cp -r $b $P/light_bags/ ;;
    *) python3 $S/bag_excerpt.py $b $P/bag_excerpts/$n > /dev/null 2>&1 || echo "excerpt of $n failed" ;;
  esac
done
(cd $OUT/bags 2>/dev/null && for b in */; do echo "$(du -sh $b | cut -f1) ${b%/} $(cat $b/*.db3 2>/dev/null | md5sum | cut -c1-32)"; done) > $P/full_bags.txt
cp $(dirname $S)/VERSION $P/ 2>/dev/null
python3 $S/gate.py > $P/gate.txt 2>&1
python3 - $OUT > $P/SUMMARY.txt <<'PY'
import json, os, sys
out = sys.argv[1]
rows = [json.loads(l) for l in open(os.path.join(out, 'verdicts.jsonl'))] if os.path.exists(os.path.join(out, 'verdicts.jsonl')) else []
bad = [r for r in rows if not r['ok']]
print(f'{len(rows) - len(bad)} of {len(rows)} checks passed')
for r in rows:
    print(('PASS ' if r['ok'] else 'FAIL ') + r['script'] + '/' + r['check'] + ': ' + str(r.get('detail', ''))[:300])
PY
mkdir -p ${COLLECT_DIR:-$HOME}
tar czf ${COLLECT_DIR:-$HOME}/$NAME.tar.gz -C $T $NAME; rm -rf $T
F=${COLLECT_DIR:-$HOME}/$NAME.tar.gz
md5sum $F > $F.md5
echo
sync; echo -e "\033[1mCOLLECTED: $F ($(( $(stat -c %s $F) / 1048576 )) MB), $(head -1 $OUT/verdicts.jsonl >/dev/null 2>&1 && python3 -c "import json;r=[json.loads(l) for l in open('$OUT/verdicts.jsonl')];print(sum(x['ok'] for x in r),'of',len(r),'checks passed')")\033[0m"
echo -e "\033[1mNEXT: send that one file to Karth today (WhatsApp / Drive / email from your phone or laptop). Full camera bags: $(du -sh $OUT/bags 2>/dev/null | cut -f1) in $OUT/bags, copy to a USB stick if there is one.\033[0m"
