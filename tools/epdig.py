"""Per-episode digest of a fleet run: python3 epdig.py RUN_DIR [filter: col=2 row=4 round=2]."""
import json, os, sys, glob
run = sys.argv[1]
flt = dict(a.split('=') for a in sys.argv[2:])
rows = []
for d in sorted(glob.glob(os.path.join(run, 'ep', '*'))):
    try:
        s = json.load(open(os.path.join(d, 'score.json')))
    except Exception:
        continue
    m = s.get('mission') or {}
    try:
        mj = json.load(open(os.path.join(d, 'mission.json')))
    except Exception:
        mj = {}
    ev = mj.get('events', [])
    brow = next((e.get('row') for e in ev if e.get('stage') == 'BOOK'), None)
    pc = [e.get('where') for e in ev if e.get('stage') == 'PLACE_CHECK']
    links = sorted({c['link'].replace('arm_right_', 'a').replace('_link', '') + ':' + c['object'].replace('erc_collection_', '')[:6] for c in s.get('collisions', [])})
    r = dict(ep=os.path.basename(d), rnd=s.get('round'), col=s.get('physical_column'), tgt=s.get('target', '').replace('book_col_', ''),
             brow=brow, score=s.get('score'), place=s.get('place_kind'), stages=''.join(k[0] for k in s.get('stages', {})),
             last=m.get('last_stage') or mj.get('last_stage'), res=m.get('result') or mj.get('result'),
             reason=(m.get('reason') or mj.get('reason') or '')[:60], lift=s.get('target_max_lift_held'),
             t_place=(s.get('stages', {}).get('place') or {}).get('t'), pc=','.join(map(str, pc)), coll=' '.join(links))
    ok = all(str(r.get({'row': 'brow'}.get(k, k))) == v for k, v in flt.items())
    if ok:
        rows.append(r)
for r in rows:
    print('\t'.join(f'{v}' for v in r.values()))
print(len(rows), 'episodes, mean', round(sum(r['score'] or 0 for r in rows) / max(1, len(rows)), 2))
