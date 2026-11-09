"""Mission supervisor (stdlib only, no ROS, no numpy). python3 -m erc_bot.mission runs here first; the mission itself runs
in a child process with ERC_BOT_CHILD=1."""
import json, os, signal, subprocess, sys, time

TERMINAL = ('done', 'aborted', 'stopped', 'crashed')


def _out_dir(argv):
    for i, x in enumerate(argv):
        if x == '--out' and i + 1 < len(argv):
            return argv[i + 1]
        if x.startswith('--out='):
            return x[6:]
    return ''


def supervise(argv=None, cmd=None):
    """Run the mission in a child process and record a hard death. A native crash (SIGSEGV in a library), an OOM kill or
    any exit that skips main()'s finally leaves no mission.json and nothing on screen: the fleet saw 0.5 % of episodes
    exit about 14 s in, right after 'interface ready', with no traceback. The parent prints the death and writes it into
    mission.json (result 'killed', the signal, the last stage). A death before the first stage (the robot has not moved)
    restarts the mission once. ERC_BOT_NO_SUPERVISOR=1 runs main() in this process."""
    argv = list(sys.argv[1:] if argv is None else argv)
    out = _out_dir(argv)
    if not out:                                   # the record decides restart or not: give the child a private one
        out = f'/tmp/erc_bot_mission_{os.getpid()}'
        argv += ['--out', out]
    cmd = cmd or [sys.executable, '-X', 'faulthandler', '-m', 'erc_bot.mission']
    path = os.path.join(out, 'mission.json')
    env = dict(os.environ, ERC_BOT_CHILD='1')
    stopping, child = [False], [None]

    def forward(sig, _frame):                     # a stop sent to this pid only (ros2 launch): pass it on once
        if stopping[0]:
            return
        stopping[0] = True
        if child[0] is not None and child[0].poll() is None:
            child[0].send_signal(sig)
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    restarted, first_death, t0 = False, None, time.monotonic()
    while True:
        if os.path.exists(path):
            os.remove(path)                       # a restart must not read the dead attempt's record as its own
        child[0] = subprocess.Popen(cmd + argv, env=env)
        while True:
            try:
                rc = child[0].wait()
                break
            except KeyboardInterrupt:
                continue
        try:
            rec = json.load(open(path))
        except Exception:
            rec = {}
        if rec.get('result') in TERMINAL:                 # main() recorded its own end: keep it
            if rc < 0 or restarted:
                rec.update(exit=rc, restarted=restarted, first_death=first_death)
                json.dump(rec, open(path, 'w'), indent=1)
            return rc if rc >= 0 else 128 - rc
        how = (f'signal {signal.Signals(-rc).name} (native crash or kill, no Python exception)' if rc < 0
               else f'exit code {rc} without a mission record (see the log above)')
        stage = rec.get('last_stage')
        msg = f'MISSION PROCESS DIED at stage {stage or "before START"} after {time.monotonic() - t0:.0f} s wall: {how}'
        print(f'[erc_bot supervisor] {msg}', file=sys.stderr, flush=True)
        rec.update(result='killed', reason=msg, exit=rc, restarted=restarted, first_death=first_death)
        rec.setdefault('last_stage', None); rec.setdefault('events', [])
        os.makedirs(out, exist_ok=True)
        json.dump(rec, open(path, 'w'), indent=1)
        if stopping[0] or restarted or stage is not None or time.monotonic() - t0 > 120:
            return 128 - rc if rc < 0 else (rc or 3)
        restarted, first_death = True, msg
        print('[erc_bot supervisor] nothing moved yet: restarting the mission once', file=sys.stderr, flush=True)


