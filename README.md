# erc-bot (version c7)

This is version **c7**: the c7 kit plus the live fixes made during the 6 Nov 2026 hardware session at MBRL, the code
that ran on the real TIAGo Pro that day (found the red book in column 3 row 2 and grasped it). Later versions are not
in this repository. The detector weights (`det_v4_736x1280.onnx`) are not included.

Team JALPARI's ERC 2026 Phase 2 solution for the live TIAGo Pro: find the shelf column from the digit plates, find
the target book (colour, or the spine word on column 2 in Round 2), pick it with the right arm, carry it, and lower it
into the collection bin.

It uses only what the live robot offers (ERC 2026 Live Phase Interface Guide): the `/erc/` controllers through the
organisers' checking layer, the compressed colour and aligned-depth camera, `/scan`, `/mobile_base_controller/odom`,
TF, `/robot_description`. No contact sensors, no walls, no map, no Nav2.

| module | does |
|---|---|
| `erc_bot/live.py` | interface: sensors, TF, actions through `/erc/`, every goal clamped to the live ranges and timed under the speed caps, rejections surfaced, laser-guarded base twist |
| `erc_bot/kinematics.py`, `erc_bot/arm.py` | arm IK from the live URDF, parking, checked tool poses (from the team's Phase 1 code) |
| `erc_bot/detector.py`, `erc_bot/perception.py` | the erc-gym YOLO detector (ONNX) on the camera, detections lifted to 3D with aligned depth, edge-cut boxes dropped |
| `erc_bot/shelf.py` | shelf line, inward normal and digit-to-column map from the plates, held in odom |
| `erc_bot/mission.py` | the mission: start, find column, find book, pick stage, find bin, place stage; `mission.json` with `--out` |
| `erc_bot/stages.py` | the stage interface and the scripted pick (with the carry posture) and place |

## Run

On the team laptop (ROS 2 Humble, `pip3 install onnxruntime==1.23.2 numpy==1.24.4`, `sudo apt install ros-humble-control-msgs python3-scipy`):

```bash
source tiago_dds_connect/tiago_env.sh
cd erc-bot
python3 -m erc_bot.mission --digit 2 --label red                          # Round 1, or Round 2 columns 1, 3, 5
python3 -m erc_bot.mission --digit 2 --label A                            # Round 2 column 2: A FUTURE, B ROBOT, C LIBRARY, D AGENT
```

The detector defaults to `/opt/model/det_v4_736x1280.onnx` (det-v4), `--model` points elsewhere. Against the erc-gym
simulation (live-robot profile) add `--sim-time`. `ERC_BOT_ORT_THREADS=4` caps the detector's
CPU threads when the bot shares a machine with a simulator.

## Grasp geometry

Torso height and reach per shelf row, the 0.24 m right-arm offset, the 9 cm pre-grasp, the 15 cm insertion in steps,
the 2.5 cm test lift and the upright carrying path are the team's Phase 1 values, which scored in the official sim.
The grasp check reads the gripper position (the real fingers stop early on a book, closing on air reads 0.79).

## Stage interface (learned policies)

The pick and the place are pluggable (`erc_bot/stages.py`). The mission does the perception around them (column from
the plates, target book, bin) and hands each stage a task:

| stage | gets | must leave |
|---|---|---|
| pick | `PickTask`: `target` (detector class), `book` (spine-face centre, odom), `row` (1 top to 4), `n` (shelf inward normal), `t` (tangent), `plate` | the book held and the arm clear of the shelf, ready to drive (the scripted pick ends in the upright carry posture). Return `{'held': True}` to skip the mission's own hold check |
| place | `PlaceTask`: `target`, `bin` (bin estimate, odom, median of several detections), `n` | the book released in the bin, the gripper open and above the rim |

```python
from erc_bot.stages import Stage
from erc_bot.live import Abort

class LearnedPick(Stage):
    name = 'rl-pick-v1'
    def __init__(self, bot, checkpoint='policy.onnx', **opts):
        super().__init__(bot, **opts)          # self.bot, self.r (Live), self.opts
        ...                                    # load the policy
    def run(self, task):
        ...                                    # observe through self.r / self.bot.eyes, act through self.r.send, self.bot.arm, self.bot.go
        raise Abort('why')                     # on failure: the harness tables the reason
        return {'held': True}
```

```bash
python3 -m erc_bot.mission --digit 2 --label red --pick my_pkg.policy:LearnedPick --pick-opt checkpoint=/opt/policies/pick.onnx
```

Every command still goes through `Live.send` (the organisers' checking layer, live ranges, speed caps) or `Live.velocity`
(laser stop). `bot.stage(name, **kw)` logs a sub-stage into `mission.json`. Abort raises carry the failure reason.

## Scored evaluation (erc-gym)

With `--out DIR` the bot writes `DIR/mission.json` at every stage: result (`done`, `aborted`, `crashed`, `stopped`),
failure reason, last stage, mission time and the sim time of every stage. erc-gym's `gymctl.py run` reads it into
`summary.md` and `summary.json` next to the scorer's ground truth: per-stage reach, one failure reason per episode,
collisions, time to place, and breakdowns by target row, digit and label.

Z8 (Docker, live-robot profile, 4 at once):

```bash
cd ~/erc && python3 gym/tools/gymctl.py run --name bot-r1 --round 1 --seeds 1-3 --tasks cross:1-5:red,green,blue,yellow \
  --profile real --concurrency 4 --timeout 330 --cpus-per-episode 8 \
  --mount $HOME/erc/bot:/opt/bot --mount $HOME/erc/results/det-v4/final:/opt/model \
  --solution "cd /opt/bot && ERC_BOT_ORT_THREADS=3 python3 -m erc_bot.mission --digit {digit} --label {label} --sim-time --out {out}"
```

A learned stage is scored on the same seeds by adding `--pick module:Class` (and its mount) to `--solution`.
