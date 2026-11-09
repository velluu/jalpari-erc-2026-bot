"""Offline check of erc_bot/supervisor.py with a fake mission child (no ROS): python3 tests/t_supervisor.py"""
import json, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from erc_bot.supervisor import supervise
FAKE = r'''
import json, os, signal, sys
out = sys.argv[sys.argv.index('--out') + 1]; mode = sys.argv[1]
os.makedirs(out, exist_ok=True)
cnt = os.path.join(out, 'count'); n = int(open(cnt).read()) + 1 if os.path.exists(cnt) else 1; open(cnt, 'w').write(str(n))
def rec(**k): json.dump(dict(events=[], **k), open(os.path.join(out, 'mission.json'), 'w'))
if mode == 'ok': rec(result='done', last_stage='PARKED'); sys.exit(0)
if mode == 'segv_early': os.kill(os.getpid(), signal.SIGSEGV)
if mode == 'segv_once':
    if n == 1: os.kill(os.getpid(), signal.SIGSEGV)
    rec(result='done', last_stage='PARKED'); sys.exit(0)
if mode == 'kill_late': rec(result=None, last_stage='CARRY'); os.kill(os.getpid(), signal.SIGKILL)
if mode == 'exit_silent': sys.exit(0)
'''
d = tempfile.mkdtemp(); fake = os.path.join(d, 'fake.py'); open(fake, 'w').write(FAKE)
def run(mode):
    out = tempfile.mkdtemp()
    rc = supervise([mode, '--out', out], cmd=[sys.executable, fake])
    m = json.load(open(os.path.join(out, 'mission.json'))); n = int(open(os.path.join(out, 'count')).read())
    return rc, m, n
ok = True
def check(name, cond, info):
    global ok; ok &= bool(cond); print(('PASS' if cond else 'FAIL'), name, info)
rc, m, n = run('ok'); check('ok', rc == 0 and m['result'] == 'done' and n == 1 and 'exit' not in m, (rc, m.get('result'), n))
rc, m, n = run('segv_early'); check('segv before START: one restart, recorded', rc == 128 + 11 and m['result'] == 'killed' and n == 2 and m['restarted'] and 'SIGSEGV' in m['first_death'], (rc, m['reason'], n))
rc, m, n = run('segv_once'); check('segv once then done', rc == 0 and m['result'] == 'done' and n == 2 and 'SIGSEGV' in m['first_death'], (rc, m.get('first_death'), n))
rc, m, n = run('kill_late'); check('kill after a stage: no restart, stage kept', rc == 128 + 9 and m['result'] == 'killed' and m['last_stage'] == 'CARRY' and n == 1, (rc, m['reason'], n))
rc, m, n = run('exit_silent'); check('exit 0 without a record: one restart, then killed', m['result'] == 'killed' and n == 2, (rc, m['reason'], n))
sys.exit(0 if ok else 1)
