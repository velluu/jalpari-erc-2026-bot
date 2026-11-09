#!/usr/bin/env python3
"""Slot step 2: the real gripper's scale. Closing on air, then on a 3 cm book held in the stopped gripper.

  python3 slot/s02_gripper.py air            close and open both grippers on air (right twice), no arm motion
  python3 slot/s02_gripper.py present        park both arms (the bot's own park), then the right gripper forward at
                                             about 0.62 m ahead, 0.24 m right, open: a book can be slid between the fingers
  (robot still, no code running: one person slides a 3 cm book between the open fingers, spine facing out, steps back 1.5 m)
  python3 slot/s02_gripper.py book           close on the book, hold 3 s, read position and effort, then open (book drops)
  python3 slot/s02_gripper.py park           both arms back to the bot's parked posture

Every arm motion is the bot's own code (erc_bot.arm), at 35 % of the speed caps. Writes gripper_<mode>.json.
--yes skips the prompts (dry run).
"""
import argparse, math, sys, time
import numpy as np
from common import start, finish, save, SIM, ask

GR = 'gripper_right_finger_joint'


def close_trace(n, ctrl, joint, target, secs=1.5, timeout=6.0):
    """One gripper goal; joint trace until it stops. Returns the numbers the bot's grasp check relies on."""
    n.js_log = []
    q0 = n.joints[joint]
    r = n.send_goal(ctrl, [joint], [(secs, [target])], wait_result=False)
    gh = r.pop('handle', None)
    res = gh.get_result_async() if gh is not None and r['accepted'] else None
    t0 = r['t_send']; still = None; last = None
    while n.now() - t0 < timeout:
        n.spin_for(0.02)
        q = n.joints[joint]
        if n.now() - t0 > secs * 0.5 and last is not None and abs(q - last) < 0.002:
            still = still or n.now()
            if n.now() - still > 0.6 and (res is None or res.done() or n.now() - t0 > secs + 2.5):
                break
        else:
            still = None
        last = q
    tr = [(t - t0, p[joint]) for t, _, p in n.js_log if joint in p]
    n.js_log = None
    qf = n.joints[joint]
    moved = [t for t, q in tr if abs(q - q0) > 0.005]
    span = qf - q0
    t95 = next((t for t, q in tr if abs(q - q0) >= 0.95 * abs(span)), None) if abs(span) > 0.01 else None
    out = {'target': target, 'q0': round(q0, 4), 'final': round(qf, 4), 'effort': n.effort.get(joint),
           'onset_s': round(moved[0], 3) if moved else None, 't95_s': round(t95, 3) if t95 is not None else None,
           'accepted': r['accepted'], 'rejections': r.get('rejections', [])}
    if res is not None and res.done():
        rr = res.result()
        out.update(error_code=int(rr.result.error_code), error_string=rr.result.error_string, status=int(rr.status))
    else:
        out.update(error_code=None, error_string='no result while the fingers were still (goal kept running)')
        if gh is not None:
            gh.cancel_goal_async(); n.spin_for(0.3)
    out['trace'] = [(round(t, 3), round(q, 4)) for t, q in tr]
    return out


def air(n, v):
    res = {}
    for side, reps in (('right', 2), ('left', 1)):
        j, c = f'gripper_{side}_finger_joint', f'gripper_{side}_controller'
        if n.joints[j] > 0.05:
            res[f'{side}_initial_open'] = close_trace(n, c, j, 0.0)      # start from open, so the first close is a full stroke
        for k in range(reps):
            cl = close_trace(n, c, j, 0.79)
            op = close_trace(n, c, j, 0.0)
            res[f'{side}_{k}'] = {'close': cl, 'open': op}
            v.check(f'{side} close on air #{k + 1}', 0.70 <= cl['final'] <= 0.803,
                    f"final {cl['final']:.3f} (bot's air value 0.79, SURE_HOLD 0.75), effort {cl['effort']}, onset {cl['onset_s']} s, "
                    f"95 % at {cl['t95_s']} s, result {cl['error_code']} {cl['error_string']!r}",
                    'the air value is off the bot\'s 0.79: report it, the bot\'s SURE_HOLD (0.75) has to sit between the air '
                    'and book values', **{k2: cl[k2] for k2 in ('final', 'effort', 'onset_s', 't95_s', 'error_code')})
            v.check(f'{side} open #{k + 1}', op['final'] <= 0.05, f"final {op['final']:.3f}, 95 % at {op['t95_s']} s, result {op['error_code']}",
                    'gripper did not open: stop, ask the organisers', final=op['final'])
    return res


def bot_arm(n):
    """The bot's Live node and Arm on the same process (slow speed)."""
    import erc_bot.live as live
    from erc_bot.arm import Arm
    live.SPEED_USE = 0.35
    L = live.Live('slot_arm', sim_time=SIM)
    w0 = time.monotonic()
    while time.monotonic() - w0 < 60 and (L.urdf is None or len(L.joints) < 20):
        L.tick(0.05)
    if L.urdf is None:
        raise RuntimeError('no /robot_description')
    w0 = time.monotonic()
    while time.monotonic() - w0 < 30 and not all(a.server_is_ready() for a in L.actions.values()):
        L.tick(0.05)
    L.wait(0.5)
    return L, Arm(L)


def present(n, v, a):
    from erc_bot.arm import transform
    L, arm = bot_arm(n)
    out = {}
    try:
        arm.park()
        v.check('park', True, 'both arms parked (bot park sequence)')
        z0 = L.joints['torso_lift_joint']
        L.gripper(False)
        last = None
        for x, z in ((0.62, 0.92), (0.60, 1.00), (0.66, 0.85), (0.58, 1.10)):
            try:
                goal = transform([x, -0.24, z + z0])
                arm.pose(goal)
                tool = arm.tool()[:3, 3]
                out = {'tool': tool.round(3).tolist(), 'torso': z0, 'q': {k: L.joints[k] for k in arm.k.names}}
                break
            except Exception as e:
                last = str(e); L.wait(0.2)
        ok = bool(out)
        v.check('present pose', ok, f"right gripper open at {out.get('tool')} in base_footprint" if ok else last,
                'no reachable present pose: run `s02_gripper.py park`, skip the book test (the bot\'s pick logs the book '
                'reading as stage GRIPPER in mission.json)')
    finally:
        L.stop(); L.destroy_node()
    return out


def book(n, v, a):
    if not ask('Book between the open fingers and everyone 1.5 m away?', a.yes):
        v.check('book', False, 'operator said no', 'place the book, step back, rerun `s02_gripper.py book`')
        return {}
    cl = close_trace(n, 'gripper_right_controller', GR, 0.79, timeout=8.0)
    n.spin_for(3.0)
    held = n.joints[GR]
    slip = held - cl['final']
    v.check('close on book', SIM or cl['final'] < 0.75, f"stopped at {cl['final']:.3f} (air about 0.79, bot SURE_HOLD 0.75), after 3 s "
            f"{held:.3f} (drift {slip:+.4f}), effort {n.effort.get(GR)}, result {cl['error_code']} {cl['error_string']!r}",
            'reads like air: the book was not between the fingers, or the real fingers squeeze a 3 cm book to near 0.79; '
            'report the number, the bot then confirms holds by sight only', final=cl['final'], held_3s=held,
            effort=n.effort.get(GR), error_code=cl['error_code'])
    try:
        hold = hold_check(v, a)
    except Exception as e:                      # never lets the hold check break the gripper step
        v.info(f'hold check not run: {type(e).__name__}: {e}')
        hold = {'error': str(e)}
    v.info('opening the gripper, the book drops')
    op = close_trace(n, 'gripper_right_controller', GR, 0.0)
    v.check('open after book', op['final'] <= 0.05, f"final {op['final']:.3f}", 'gripper did not open: stop, ask the organisers')
    return {'close': cl, 'held_3s': held, 'open': op, 'hold': hold}


BOOK_HALF_H, PRESS, HANG_MARGIN = 0.125, 0.04, 0.03   # lane B's stages.BOOK_HALF_H, place 'press' default; safety margin


def tape(prompt, auto):
    if auto:
        return None
    try:
        s = input(f'{prompt} (cm, Enter to skip): ').strip().replace(',', '.')
        return float(s) / 100.0 if s else None
    except (EOFError, ValueError):
        return None


def hold_check(v, a):
    """The bot's in-hand measurement (lane B's ScriptedPlace.hold_offset, logged as HOLD_MEASURED dz): the held book's
    spine centre above the grasping link, by sight with the kit's default detector, box centre, median depth. The place's
    release height comes from it (book_below_tool = 0.125 - dz), so the estimated hang is checked against a tape here."""
    import os
    model = os.environ.get('ERC_BOT_MODEL', '')
    if not model or not os.path.exists(model):
        v.info('hold check skipped: ERC_BOT_MODEL not set (source kit_env.sh)')
        return {'skipped': 'no model'}
    from erc_bot.perception import Eyes
    L, arm = bot_arm(None)
    try:
        eyes = Eyes(L, model, conf=0.3)
        tool = arm.tool()[:3, 3]
        cam = L.matrix('base_footprint', 'head_front_camera_color_optical_frame')[:3, 3]
        d = tool + np.array([0.0, 0.0, 0.05]) - cam              # the bot's look_at(tool + 5 cm)
        L.head(math.atan2(d[1], d[0]), math.atan2(d[2], math.hypot(d[0], d[1])))
        L.wait(0.5)
        zs, cls = [], set()
        for _ in range(8):
            dets, _ = eyes.look(None, frame='base_footprint', conf=0.3, kp=False)
            for q in dets or []:
                if q['class'].startswith('book') and np.linalg.norm(q['p'][:2] - tool[:2]) < 0.20:
                    zs.append(float(q['p'][2])); cls.add(q['class'])
            L.wait(0.1)
    finally:
        L.destroy_node()
    out = {'tool_z': round(float(tool[2]), 4), 'seen': len(zs), 'classes': sorted(cls)}
    if len(zs) < 2:
        v.check('hold seen', False, f'held book seen in {len(zs)} of 8 frames near the gripper (tool z {tool[2]:.3f} m)',
                'use a competition book (red, green, blue or yellow spine, or a word book): the detector knows only those. '
                'Not blocking', **out)
        return out
    centre = float(np.median(zs))
    out.update(spine_z=round(centre, 4), dz_sight=round(centre - float(tool[2]), 4))
    v.info(f"HOLD by sight: spine centre {centre:.3f} m above the floor, grasping link {tool[2]:.3f} m, dz {out['dz_sight']:+.3f} m "
           f"(the bot's HOLD_MEASURED), {len(zs)} detections {sorted(cls)}")
    # The place releases at floor + gap + est_hang - press (press 0.04) and the lowering ends on the floor stall, so only an
    # estimated hang more than press above the true one drops the book (lane B, 03:41). Safety rule: est <= true + 3 cm.
    est_hang = BOOK_HALF_H - out['dz_sight']                    # the bot's book_below_tool from HOLD_MEASURED dz
    lo = tape('Tape from the floor to the BOTTOM edge of the held book', a.yes)
    out['est_hang'] = round(est_hang, 4)
    if lo is None or not (0.2 < lo < float(tool[2])):
        v.info(f'no usable tape reading: est hang {est_hang:.3f} m recorded only')
        return out
    true_hang = float(tool[2]) - lo
    excess = est_hang - true_hang - HANG_MARGIN
    out.update(tape_bottom=round(lo, 4), true_hang=round(true_hang, 4), hang_excess=round(excess, 4),
               place_press=round(PRESS + excess, 3) if excess > 0 else None)
    v.check('hold hang margin', excess <= 0,
            f"estimated hang below the grasping link {est_hang:.3f} m (dz {out['dz_sight']:+.3f}) vs tape {true_hang:.3f} m: "
            f"{100 * (est_hang - true_hang):+.1f} cm, limit +{100 * HANG_MARGIN:.0f} cm",
            f"the release would aim {100 * excess:.1f} cm too high: calibration.json sets place press {PRESS + excess:.3f} "
            f"(lane B's aim-under margin, default {PRESS}), step 8 passes it as --place-opt press=. Not blocking", **out)
    return out


def park(n, v, a):
    L, arm = bot_arm(n)
    try:
        arm.park()
        v.check('park', True, 'both arms parked')
    finally:
        L.stop(); L.destroy_node()
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('air', 'present', 'book', 'park'))
    ap.add_argument('--yes', action='store_true')
    a = ap.parse_args()
    n, v = start(f's02_gripper_{a.mode}')
    if not v.check('joints', n.wait_joints([GR, 'gripper_left_finger_joint'], 10.0), 'gripper joints on /joint_states',
                   'no joint states: rerun s00_preflight'):
        return finish(n, v)
    try:
        res = {'air': air, 'present': present, 'book': book, 'park': park}[a.mode](n, v, *([] if a.mode == 'air' else [a]))
    except Exception as e:
        v.check('run', False, f'{type(e).__name__}: {e}', 'read the message; if an arm goal was rejected, the reason is in it. '
                'Run `s02_gripper.py park` before moving on')
        res = {'error': str(e)}
    save(f'gripper_{a.mode}.json', res)
    return finish(n, v)


if __name__ == '__main__':
    raise SystemExit(main())
