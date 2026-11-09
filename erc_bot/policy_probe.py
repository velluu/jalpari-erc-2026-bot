"""SIMULATION ONLY. Fast probe for the learned stages in erc-gym: skips the column and book search by reading the
episode's layout.json and gz ground truth, then runs LearnedPick (and optionally LearnedPlace) exactly as the mission
would. Used for the observation and action parity records and the teacher plumbing check (lane E).

  python3 -m erc_bot.policy_probe --digit 2 --label red --out OUT --sim-time --checkpoint pick.npz \
      [--truth gz] [--actions parity] [--place-checkpoint place.npz]

Writes OUT/policy_pick.jsonl (per control period: observation, action, command, measured joints, gz truth),
OUT/policy_pick_js.json (joint states at their own rate, parity runs) and OUT/mission.json (stages, result).
"""
import argparse, json, os, sys
import numpy as np
import rclpy
from .live import Live, Abort
from .arm import Arm
from .mission import Bot, WORDS
from .shelf import Shelf, PLATE_Z
from .stages import PickTask, load
from .policy_truth import GzTruth, ROBOT


class ProbeBot(Bot):
    def __init__(self, live, digit, label, out, model=''):
        self.r, self.digit, self.label = live, int(digit), label
        self.out, self.result, self.reason = out, 'running', ''
        self.target = WORDS[label.upper()] if label.upper() in WORDS else f'book_{label.lower()}'
        if model:                                   # the scripted stages look again with the detector
            from .perception import Eyes
            self.eyes = Eyes(live, model)
        else:
            self.eyes = None
        self.arm = Arm(live)
        self.shelf = Shelf()
        self.t0 = live.now()
        self.events = []

    gz = None
    target_name = None
    z0 = None

    def seen_in_hand(self, frames=6):
        return False

    def holding(self):
        """SIM ONLY: the target book lifted 1.5 cm or more off its spawn height (gz truth; the sim gripper reads
        0.79 on a 3 cm book, like air)."""
        try:
            return bool(self.gz.pose(self.target_name)[1][2] > self.z0 + 0.015)
        except Exception:
            return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--digit', type=int, required=True); ap.add_argument('--label', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--sim-time', action='store_true')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--truth', default=''); ap.add_argument('--actions', default='')
    ap.add_argument('--mode', default='topic'); ap.add_argument('--steps', default='')
    ap.add_argument('--place-checkpoint', default=''); ap.add_argument('--place-truth', default='')
    ap.add_argument('--pick-impl', default='policy', choices=('policy', 'scripted'))
    ap.add_argument('--place-impl', default='policy', choices=('policy', 'scripted'))
    ap.add_argument('--opt', action='append', default=[], help='extra key=value for the learned stages')
    ap.add_argument('--model', default='', help='detector ONNX (needed by the scripted stages and the fallback)')
    ap.add_argument('--place-opt', action='append', default=[], help='extra key=value for the learned place')
    a, _ = ap.parse_known_args(argv)
    lay = json.load(open(os.path.join(a.out, 'layout.json')))
    col = lay['markers'].index(a.digit) + 1
    lab = a.label.lower()
    cands = [b for b in lay['books'] if b['column'] == col and (b['name'].endswith('_' + lab) or b['name'].endswith('_white_' + lab.upper()))]
    if len(cands) != 1:
        sys.exit(f'target not unique: {cands}')
    tb = cands[0]; row = int(tb['row']) - 1
    rclpy.init()
    live = Live(name='erc_policy_probe', sim_time=a.sim_time)
    bot, res = None, {}
    result, reason = 'error', ''
    try:
        live.ready()
        gz = GzTruth()
        if not gz.wait((ROBOT, tb['name'])):
            raise Abort('gz truth not available')
        bot = ProbeBot(live, a.digit, a.label, a.out, a.model)
        bot.gz, bot.target_name, bot.z0 = gz, tb['name'], float(gz.pose(tb['name'])[1][2])
        bot.stage('START', probe=True, target=tb['name'], row=row, column=col, sim_only=True)
        bot.arm.park()
        live.head(0.0, 0.12)
        wTo = gz.world_T_odom(live.odom_T_base()); oTw = np.linalg.inv(wTo)
        book_w = gz.pose(tb['name'])[1] - np.array([0.08, 0.0, 0.0])                  # spine-face centre (truth)
        plate_w = np.array([2.755, lay['column_y'][col - 1], PLATE_Z])
        to_o = lambda p: (oTw @ np.r_[p, 1.0])[:3]
        n_o = oTw[:3, :3] @ np.array([1.0, 0, 0]); t_o = oTw[:3, :3] @ np.array([0, 1.0, 0])
        task = PickTask(bot.target, to_o(book_w), row, n_o[:2], t_o[:2], to_o(plate_w))
        bot.stage('TRUTH_TASK', book_w=np.round(book_w, 4).tolist(), plate_w=plate_w.tolist(), world_T_odom=np.round(wTo, 5).tolist())
        opts = [f'checkpoint={a.checkpoint}', f'mode={a.mode}']
        opts += [f'truth={a.truth}'] if a.truth else []
        opts += [f'actions={a.actions}'] if a.actions else []
        opts += [f'steps={a.steps}'] if a.steps else []
        opts += a.opt
        bot.picker = load('erc_bot.policy_runner:LearnedPick', bot, opts) if a.pick_impl == 'policy' else \
            load('erc_bot.stages:ScriptedPick', bot, [])
        bot.placer = bot.picker
        bot.stage('PICK', row=row, stage_impl=bot.picker.name)
        res = bot.picker.run(task) or {}
        bot.stage('PICK_DONE', **{k: v for k, v in res.items() if k != 'held'})
        if (a.place_checkpoint or a.place_impl == 'scripted') and not a.actions:
            if not bot.holding():
                raise Abort('pick did not lift the book (gz truth), place not run')
            bin_w = gz.pose('erc_collection_bin')[1]
            bin_o = to_o(np.r_[bin_w[:2], 0.95])
            if bot.eyes is not None:              # the mission's own bin search (turns away from the shelf, detector)
                try:
                    bin_o = bot.find_bin(n_o[:2])
                    bot.stage('BIN_VS_TRUTH', err=np.round(bin_o[:2] - to_o(bin_w)[:2], 3).tolist())
                except Abort as e:
                    bot.stage('BIN_NOT_FOUND', why=str(e)[:80])
                    bot.face(float(np.arctan2(-n_o[1], -n_o[0])))
            from .stages import PlaceTask
            popts = [f'checkpoint={a.place_checkpoint}'] + ([f'truth={a.place_truth}'] if a.place_truth else []) + a.place_opt
            bot.placer = load('erc_bot.policy_runner:LearnedPlace', bot, popts) if a.place_impl == 'policy' else \
                load('erc_bot.stages:ScriptedPlace', bot, [])
            bot.stage('PLACE', stage_impl=bot.placer.name)
            bot.placer.run(PlaceTask(bot.target, bin_o, n_o[:2]))
        bot.stage('DONE')
        result = 'done'
    except Abort as e:
        live.say(f'FAILED: {e}'); result, reason = 'aborted', str(e)
    except KeyboardInterrupt:
        result, reason = 'stopped', 'stopped by the harness'
    except Exception as e:
        import traceback; traceback.print_exc(); result, reason = 'crashed', f'{type(e).__name__}: {e}'
    finally:
        try:
            live.stop()
        except Exception:
            pass
        if bot is not None:
            bot.result, bot.reason = result, reason; bot.save()
        live.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if result == 'done' else 1


if __name__ == '__main__':
    sys.exit(main())
