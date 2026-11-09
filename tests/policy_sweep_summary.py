"""Summarise an erc-gym pool sweep of learned vs scripted stages: per variant (from the solution command), round,
target row and physical column: episodes, scorer stage rates (column, grasp, carry, place, gentle), mean score,
collisions, and for learned places the release height and in-bin flag from the runner's records. Stdlib only.

  python3 tests/policy_sweep_summary.py <sweep run dir> [--json out.json]
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict


def variant(sol):
    pick = 'learned-pick' if ('LearnedPick' in sol or ('policy_probe' in sol and '--pick-impl scripted' not in sol)) else 'scripted-pick'
    place = 'learned-place' if ('LearnedPlace' in sol or '--place-checkpoint' in sol) else 'scripted-place'
    ck = re.findall(r'(?:checkpoint=|--place-checkpoint |--checkpoint )(\S+\.npz)', sol)
    names = sorted({os.path.basename(c)[:-4] for c in ck})
    harness = 'probe(truth task)' if 'policy_probe' in sol else 'mission'
    extra = ' expert-only' if 'expert_only=1' in sol else ''
    return f'{harness} {pick}+{place}{extra}' + (f' [{",".join(names)}]' if place == 'learned-place' or pick == 'learned-pick' else '')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('--json', default='')
    a = ap.parse_args()
    eps = {json.loads(l)['gid']: json.loads(l) for l in open(os.path.join(a.run, 'episodes.jsonl'))}
    rows = []
    for d in sorted(glob.glob(os.path.join(a.run, 'ep', '*'))):
        if '_notready' in d or '_lost' in d or not os.path.exists(os.path.join(d, 'score.json')):
            continue
        sc = json.load(open(os.path.join(d, 'score.json')))
        e = eps.get(sc.get('episode'))
        if e is None:
            continue
        lay = json.load(open(os.path.join(d, 'layout.json'))) if os.path.exists(os.path.join(d, 'layout.json')) else {}
        col = lay['markers'].index(e['digit']) + 1 if lay.get('markers') and e['digit'] in lay['markers'] else None
        lab = e['label'].lower()
        tb = [b for b in lay.get('books', []) if b['column'] == col and (b['name'].endswith('_' + lab) or b['name'].endswith('_white_' + lab.upper()))]
        row = tb[0]['row'] - 1 if tb else None
        st = sc.get('stages') or {}
        rec = {'dir': os.path.basename(d), 'variant': variant(e['solution']), 'round': e['round'], 'seed': e['seed'], 'row': row,
               'col': col, 'score': sc.get('score'), 'end': sc.get('end_reason'), 'n_coll': len(sc.get('collisions') or []),
               **{f'st_{k}': k in st for k in ('column', 'grasp', 'carry', 'place')},
               'gentle': bool(sc.get('gentle')) or (st.get('place') or {}).get('pts', 0) >= 4 * (1.5 if e['round'] == 2 and e['digit'] in (2, 4) else 1.0) - 1e-6}
        pf = os.path.join(d, 'policy_place.jsonl')
        if os.path.exists(pf):
            R = [json.loads(l) for l in open(pf)]
            rel = [r for r in R if r.get('released')]
            if rel:
                rec['h_release'] = rel[0].get('h'); rec['inside_release'] = rel[0].get('inside')
            ev = [r for r in R if r.get('event') == 'fallback']
            rec['place_fallback'] = bool(ev)
        mf = os.path.join(d, 'mission.json')
        if os.path.exists(mf):
            m = json.load(open(mf)); rec['result'] = m.get('result'); rec['reason'] = (m.get('reason') or '')[:90]
            rec['fallbacks'] = [x.get('why', '')[:60] for x in m.get('events', []) if x.get('stage') == 'POLICY_FALLBACK']
        rows.append(rec)
    groups = defaultdict(list)
    for r in rows:
        groups[(r['variant'],)].append(r)
        groups[(r['variant'], f"round {r['round']}")].append(r)
        groups[(r['variant'], f"row {r['row']}")].append(r)
        groups[(r['variant'], f"col {r['col']}")].append(r)
    print(f"{a.run}: {len(rows)} scored episodes")
    print(f"{'group':86s} {'n':>3s} {'grasp':>6s} {'carry':>6s} {'place':>6s} {'gentle':>6s} {'score':>6s} {'coll':>5s}")
    out = {}
    for k in sorted(groups, key=lambda k: (k[0], len(k), k[1:] if len(k) > 1 else '')):
        g = groups[k]; n = len(g)
        f = lambda key: sum(1 for r in g if r.get(key)) / n
        s = sum((r['score'] or 0) for r in g) / n
        out[' | '.join(k)] = {'n': n, 'grasp': f('st_grasp'), 'carry': f('st_carry'), 'place': f('st_place'), 'gentle': f('gentle'),
                              'mean_score': s, 'collisions': sum(r['n_coll'] for r in g)}
        print(f"{' | '.join(k)[:86]:86s} {n:3d} {f('st_grasp'):6.2f} {f('st_carry'):6.2f} {f('st_place'):6.2f} {f('gentle'):6.2f} {s:6.2f} {sum(r['n_coll'] for r in g):5d}")
    if a.json:
        json.dump({'groups': out, 'episodes': rows}, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    sys.exit(main())
