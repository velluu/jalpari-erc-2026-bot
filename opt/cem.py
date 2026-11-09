#!/usr/bin/env python3
"""Lane G optimiser: cross-entropy method over the wall-book pick parameters, one population per shelf row, trials
served by opt/trials.py servers running in erc-gym pool episodes. Stdlib only (runs on the HPC login node).

  cem.py init  --q Q --space SPACE.json [--rows 1,2,3,4] [--pop 12] [--reps 2] [--elite 3]
  cem.py run   --q Q [--gens 8]           sample, queue, wait, update, repeat (log in Q/log.jsonl)
  cem.py queue --q Q --name N --params FILE --rows 1,2,3,4 --reps K      one fixed candidate (baselines, checks)
  cem.py table --q Q                      per generation and row: best, mean, elite mean of J
  cem.py requeue --q Q [--stale-min 25]   running trials older than that go back to pending

SPACE.json: {"fixed": {section: {knob: v}}, "vars": {"wall.g_off": [lo, hi, mean0, std0], ...}, "bern": {"wall.push": p0}}
"""
import argparse, glob, json, math, os, random, sys, time


def jload(f):
    return json.load(open(f))


def jdump(o, f):
    tmp = f + '.tmp'
    json.dump(o, open(tmp, 'w'), indent=1); os.replace(tmp, f)


def setp(d, key, v):
    sec, k = key.split('.', 1)
    d.setdefault(sec, {})[k] = v


def build(space, x, b):
    p = json.loads(json.dumps(space.get('fixed', {})))
    for k, v in x.items():
        setp(p, k, round(v, 4))
    for k, v in b.items():
        setp(p, k, int(v))
    return p


def queue_trials(q, cid, params, rows, reps, tag=''):
    pf = f'{q}/params/{cid}.json'
    jdump(params, pf)
    ids = []
    for row in rows:
        for k in range(reps):
            tid = f'{cid}_r{row}_k{k}'
            if os.path.exists(f'{q}/results/{tid}.json'):
                ids.append(tid); continue
            jdump({'id': tid, 'row': row, 'params': pf, 'cand': cid, 'tag': tag}, f'{q}/pending/{tid}.json')
            ids.append(tid)
    return ids


def requeue(q, stale_min):
    n = 0
    for f in glob.glob(f'{q}/running/*'):
        if time.time() - os.path.getmtime(f) > stale_min * 60:
            tid = os.path.basename(f).split('.', 1)[0]
            if not os.path.exists(f'{q}/results/{tid}.json'):
                try:
                    os.rename(f, f'{q}/pending/{tid}.json'); n += 1
                except OSError:
                    pass
            else:
                os.remove(f)
    return n


def wait_for(q, ids, poll=30, stale_min=25):
    while True:
        left = [i for i in ids if not os.path.exists(f'{q}/results/{i}.json')]
        if not left:
            return {i: jload(f'{q}/results/{i}.json') for i in ids}
        requeue(q, stale_min)
        if os.path.exists(f'{q}/STOP_CEM'):
            sys.exit('STOP_CEM')
        time.sleep(poll)


def cmd_init(a):
    q = a.q
    for d in ('pending', 'running', 'results', 'params', 'servers'):
        os.makedirs(f'{q}/{d}', exist_ok=True)
    space = jload(a.space)
    rows = [int(r) for r in a.rows.split(',')]
    st = {'space': space, 'rows': rows, 'pop': a.pop, 'reps': a.reps, 'elite': a.elite, 'gen': 0,
          'dist': {str(r): {'mean': {k: v[2] for k, v in space['vars'].items()},
                            'std': {k: v[3] for k, v in space['vars'].items()},
                            'bern': dict(space.get('bern', {}))} for r in rows}}
    if a.seed_params:
        st['seed_params'] = jload(a.seed_params)
    jdump(st, f'{q}/state{a.tag}.json')
    print(f'init {q}: rows {rows}, pop {a.pop} x reps {a.reps}, {len(space["vars"])} vars')


def sample(dist, space, rng, use_mean=False):
    x = {}
    for k, (lo, hi, _, _) in space['vars'].items():
        m, s = dist['mean'][k], dist['std'][k]
        v = m if use_mean else rng.gauss(m, s)
        x[k] = min(hi, max(lo, v))
    b = {k: (p >= 0.5 if use_mean else rng.random() < p) for k, p in dist['bern'].items()}
    return x, b


def cmd_run(a):
    q = a.q
    rng = random.Random(1234)
    for _ in range(a.gens):
        st = jload(f'{q}/state{a.tag}.json')
        g, space = st['gen'], st['space']
        cands = {}
        ids = []
        for r in st['rows']:
            dist = st['dist'][str(r)]
            for i in range(st['pop']):
                x, b = sample(dist, space, rng, use_mean=(i == 0))
                cid = f'{a.tag}g{g:02d}_r{r}_c{i:02d}'
                p = build(space, x, b)
                cands[cid] = {'row': r, 'x': x, 'b': b}
                ids += queue_trials(q, cid, p, [r], st['reps'], tag=f'{a.tag}gen{g}')
        t0 = time.time()
        print(f'[cem {time.strftime("%H:%M:%S")}] gen {g}: {len(cands)} candidates, {len(ids)} trials queued', flush=True)
        res = wait_for(q, ids)
        log = {'gen': g, 'wall_s': round(time.time() - t0), 'rows': {}}
        for r in st['rows']:
            mine = [(cid, c) for cid, c in cands.items() if c['row'] == r]
            scored = []
            for cid, c in mine:
                tr = [res[i] for i in ids if i.startswith(cid + '_')]
                J = [t.get('J', 0.0) for t in tr]
                c['J'] = sum(J) / len(J)
                c['trials'] = [{k: t.get(k) for k in ('id', 'J', 'pick', 'held', 'stages', 'found', 'max_lift', 'bilateral')}
                               | {'ncoll': len(t.get('collisions') or [])} for t in tr]
                scored.append((c['J'], cid, c))
            scored.sort(key=lambda z: -z[0])
            el = [c for _, _, c in scored[:st['elite']]]
            dist = st['dist'][str(r)]
            for k, (lo, hi, _, s0) in space['vars'].items():
                vals = [c['x'][k] for c in el]
                m = sum(vals) / len(vals)
                sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
                dist['mean'][k] = 0.3 * dist['mean'][k] + 0.7 * m
                dist['std'][k] = max(0.08 * s0, 0.3 * dist['std'][k] + 0.7 * sd)
            for k in dist['bern']:
                pm = sum(1.0 for c in el if c['b'][k]) / len(el)
                dist['bern'][k] = min(0.9, max(0.1, 0.3 * dist['bern'][k] + 0.7 * pm))
            log['rows'][str(r)] = {'best': [scored[0][1], round(scored[0][0], 3)],
                                   'mean_J': round(sum(z[0] for z in scored) / len(scored), 3),
                                   'elite_J': round(sum(c['J'] for c in el) / len(el), 3),
                                   'cands': {cid: {'J': round(c['J'], 3), 'x': {k: round(v, 4) for k, v in c['x'].items()},
                                                   'b': c['b'], 'trials': c['trials']} for _, cid, c in scored}}
            print(f'  row {r}: best {scored[0][1]} J {scored[0][0]:.2f}, mean {log["rows"][str(r)]["mean_J"]}, '
                  f'elite {log["rows"][str(r)]["elite_J"]}', flush=True)
        with open(f'{q}/log{a.tag}.jsonl', 'a') as f:
            f.write(json.dumps(log) + '\n')
        st['gen'] = g + 1
        jdump(st, f'{q}/state{a.tag}.json')


def cmd_queue(a):
    p = jload(a.params)
    ids = queue_trials(a.q, a.name, p, [int(r) for r in a.rows.split(',')], a.reps, tag='fixed')
    print(f'queued {len(ids)} trials for {a.name}')


def cmd_table(a):
    for line in open(f'{a.q}/log{a.tag}.jsonl'):
        L = json.loads(line)
        for r, v in L['rows'].items():
            print(f"gen {L['gen']} row {r}: best {v['best'][0]} {v['best'][1]}, elite {v['elite_J']}, mean {v['mean_J']} ({L['wall_s']} s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('init'); s.add_argument('--q', required=True); s.add_argument('--space', required=True)
    s.add_argument('--rows', default='1,2,3,4'); s.add_argument('--pop', type=int, default=12)
    s.add_argument('--reps', type=int, default=2); s.add_argument('--elite', type=int, default=3)
    s.add_argument('--seed-params', default=''); s.add_argument('--tag', default='')
    s = sub.add_parser('run'); s.add_argument('--q', required=True); s.add_argument('--gens', type=int, default=8)
    s.add_argument('--tag', default='', help='second optimiser on the same queue: state, log and trial ids carry it')
    s = sub.add_parser('queue'); s.add_argument('--q', required=True); s.add_argument('--name', required=True)
    s.add_argument('--params', required=True); s.add_argument('--rows', default='1,2,3,4'); s.add_argument('--reps', type=int, default=1)
    s = sub.add_parser('table'); s.add_argument('--q', required=True); s.add_argument('--tag', default='')
    s = sub.add_parser('requeue'); s.add_argument('--q', required=True); s.add_argument('--stale-min', type=float, default=25)
    a = ap.parse_args()
    if a.cmd == 'requeue':
        print(requeue(a.q, a.stale_min))
    else:
        {'init': cmd_init, 'run': cmd_run, 'queue': cmd_queue, 'table': cmd_table}[a.cmd](a)


if __name__ == '__main__':
    main()
