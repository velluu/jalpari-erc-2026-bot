#!/bin/bash
# One command that proves the laptop is ready for the slot:   bash check.sh   (prints READY or NOT READY)
# Checks: ROS 2 Humble, the CycloneDDS RMW, every Python import the bot and slot scripts use, numpy 1.24.4, the md5
# manifest of this kit, det-v4 loads and runs one frame in onnxruntime, the bot's modules import, every slot script
# compiles, kernel receive buffer. Does not need the robot.
K=$(cd "$(dirname "$0")" && pwd); cd "$K"
bad=0
ok(){ echo "PASS check/$1: $2"; }
no(){ echo "FAIL check/$1: $2  -> $3"; bad=$((bad+1)); }
[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash
[ "${ROS_DISTRO:-}" = humble ] && ok ros "Humble" || no ros "ROS_DISTRO=${ROS_DISTRO:-unset}" "install ROS 2 Humble"
[ -d /opt/ros/humble/share/rmw_cyclonedds_cpp ] && ok rmw "rmw_cyclonedds_cpp installed" || no rmw "no rmw_cyclonedds_cpp" "bash install.sh (or sudo apt install ros-humble-rmw-cyclonedds-cpp)"
if md5sum --quiet -c MANIFEST.md5 >/tmp/kit_md5.log 2>&1; then ok manifest "$(wc -l < MANIFEST.md5) files match MANIFEST.md5 ($(head -1 VERSION))"; else no manifest "$(grep -c FAILED /tmp/kit_md5.log) files differ" "copy the kit again (cat /tmp/kit_md5.log)"; fi
out=$(PYTHONPATH="$K:$K/slot:${PYTHONPATH:-}" python3 - <<'PY' 2>&1
import importlib, sys, time, glob, py_compile
fails = []
for m in ('rclpy', 'control_msgs.action', 'trajectory_msgs.msg', 'sensor_msgs.msg', 'nav_msgs.msg', 'tf2_ros', 'rosbag2_py',
          'cv2', 'numpy', 'scipy.optimize', 'scipy.spatial', 'onnxruntime'):
    try:
        importlib.import_module(m)
    except Exception as e:
        fails.append(f'{m}: {e}')
import numpy
if numpy.__version__ != '1.24.4':
    fails.append(f'numpy {numpy.__version__} (need 1.24.4: Humble cv_bridge and OpenCV break under numpy 2)')
for m in ('erc_bot.live', 'erc_bot.kinematics', 'erc_bot.arm', 'erc_bot.perception', 'erc_bot.mission', 'erc_bot.stages', 'erc_bot.policy_runner'):
    try:
        importlib.import_module(m)
    except Exception as e:
        fails.append(f'{m}: {type(e).__name__}: {e}')
for f in sorted(glob.glob('slot/*.py')):
    try:
        py_compile.compile(f, doraise=True, cfile='/tmp/kit_pyc')
    except Exception as e:
        fails.append(f'{f}: {e}')
for mp in sorted(glob.glob('models/*.onnx')):
  try:
    import onnxruntime as ort, numpy as np
    s = ort.InferenceSession(mp, providers=['CPUExecutionProvider'])
    i = s.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in i.shape]
    x = np.zeros(shape, np.float32); s.run(None, {i.name: x}); t = time.time(); s.run(None, {i.name: x})
    print(f'MODEL {mp} input {i.shape} one frame {time.time() - t:.3f} s')
  except Exception as e:
    fails.append(f'model {mp}: {e}')
pol, pfails = [], []
for pp in sorted(glob.glob('policies/*.npz')):
  try:
    import json, numpy as np
    from erc_bot.policy_obs import NumpyPolicy
    n = NumpyPolicy(pp); m = n.meta; nm = pp.split('/')[-1][:-4]
    if not m.get('student') or m.get('task') not in ('pick', 'place'):
        raise ValueError(f"meta student={m.get('student')} task={m.get('task')}")
    if int(m.get('obs_size', n.obs_size)) != n.obs_size or int(m.get('act_size', n.act_size)) != n.act_size:
        raise ValueError(f"sizes npz {n.obs_size}/{n.act_size} vs meta {m.get('obs_size')}/{m.get('act_size')}")
    a = np.asarray(n.act(np.zeros(n.obs_size, np.float32)))
    if a.size != n.act_size or not np.all(np.isfinite(a)):
        raise ValueError(f'forward pass gave {a.size} values, finite {bool(np.all(np.isfinite(a)))}')
    pol.append((m['task'], nm))
  except Exception as e:
    pfails.append(f'{pp}: {type(e).__name__}: {e}')
if pol:
    src = open('erc_bot/policy_runner.py').read()
    if 'sp.prepare(' not in src or 'sp.finish(' not in src:
        pfails.append('erc_bot/policy_runner.py has no ScriptedPick.prepare/finish fallback (stale runner)')
for t in ('pick', 'place'):
    if sum(1 for x in pol if x[0] == t) > 1:
        pfails.append(f'more than one {t} policy')
print('POLICYFAILS ' + ' | '.join(pfails) if pfails else 'POLICIES ' + (', '.join(f'{t} {n}' for t, n in pol) or 'none (scripted stages)'))
print('IMPORTFAILS ' + ' | '.join(fails) if fails else 'IMPORTS OK')
PY
)
echo "$out" | grep -q "IMPORTS OK" && ok python "imports, numpy 1.24.4, erc_bot modules, slot scripts compile" || no python "$(echo "$out" | grep IMPORTFAILS | cut -c13-)" "bash install.sh; for a module not listed there, send this line to the lead"
echo "$out" | grep -q "^MODEL" && ok model "$(echo "$out" | grep ^MODEL | cut -c7- | tr '\n' ' '), default $(cat models/DEFAULT 2>/dev/null)" || no model "no detector ran" "check models/ and onnxruntime"
echo "$out" | grep -q "^POLICIES " && ok policies "$(echo "$out" | grep '^POLICIES ' | cut -c10-)" || no policies "$(echo "$out" | grep POLICYFAILS | cut -c13-)" "send this line to the lead; s08 runs scripted stages if the policies/ folder is moved away"
R=$(cat /proc/sys/net/core/rmem_max 2>/dev/null || echo 0)
[ "$R" -ge 4194304 ] && ok rmem "net.core.rmem_max $R" || no rmem "net.core.rmem_max $R, CRITICAL: camera frames over CycloneDDS are lost (0 Hz for 3.7 MB frames at 212992)" "sudo cp dds/60-erc-dds.conf /etc/sysctl.d/ && sudo sysctl --system"
df -Pk . | awk 'NR==2{exit !($4 > 20*1024*1024)}' && ok disk "$(df -Ph . | awk 'NR==2{print $4}') free" || no disk "$(df -Ph . | awk 'NR==2{print $4}') free" "free 20 GB for the bags"
[ $bad = 0 ] && echo "READY ($(head -1 VERSION))" || echo "NOT READY: $bad check(s) failed"
exit $bad
