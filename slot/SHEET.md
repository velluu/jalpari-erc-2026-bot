# JALPARI robot slot: Fri 6 Nov, 13:00 to 14:00, Mohammed Bin Rashid Library

You run this alone. You do not need to understand the code: run one command per step, read the last lines it prints
(`STEP n PASSED` or `STEP n FAILED ...`, then `NEXT: ...`) and do what NEXT says. Every result is saved.

**Rules of the room:** arrive 12:30 (the slot is cancelled if nobody is there by 13:10). Nobody within 1.5 m of the robot
while a command runs. No cables on the floor. Ctrl-C stops any command, the robot then holds still.

## 12:30 to 12:55: laptop setup (no robot motion)
1. Plug the Ethernet cable into the robot's expansion panel (the organisers show where).
2. Open ONE terminal and type, line by line:
   ```
   source ~/tiago_dds_connect/tiago_env.sh
   source ~/erc-bot-kit/kit_env.sh
   cd ~/erc-bot-kit
   bash check.sh
   ```
   `check.sh` must end with `READY`. If it says `rmem` FAILED: `sudo cp dds/60-erc-dds.conf /etc/sysctl.d/ && sudo sysctl --system`,
   then type the two `source` lines again. If tiago_env.sh says "No interface with 10.68.0.x": the cable or the laptop's
   wired network is not up, ask the organisers.
3. Ask the organisers which shelf digit and book the robot should fetch. If it is not digit 2 / red, type
   `export SLOT_DIGIT=<digit> SLOT_LABEL=<colour or A/B/C/D>` (example: `export SLOT_DIGIT=4 SLOT_LABEL=blue`).
4. Photograph the camera exposure and white balance values the organisers post.

## 13:00: the steps (minutes are robot time measured in the simulator dry run, see the end)

| # | type | takes | what happens |
|---|---|---|---|
| 1 | `bash slot/step.sh 1` | under 1 min | starts the recorder, checks every topic (the laser `/scan` too: step 8 re-anchors on it), the clock and the network buffer |
| 2 | `bash slot/step.sh 2` | 1 min | saves the robot's description, checks joint limits, torso and head move a little |
| 3 | `bash slot/step.sh 3` | 1 min | gripper closes and opens, then the right arm reaches forward with the gripper open |
| 4 | robot still: one person slides a **competition book** (coloured spine or a word book, 3 cm) between the open fingers, spine out, steps back 1.5 m. Then `bash slot/step.sh 4`, answer `y`. When asked, with a tape from the floor: the BOTTOM edge of the held book, in cm (do not touch the robot; Enter skips) | 2 min plus the book | gripper closes on the book, reads it, the head looks at it (HOLD by sight), checks the bot's hang estimate against the tape (sets a deeper release if needed), opens (the book drops) |
| 5 | `bash slot/step.sh 5` | 4 min | small head, torso, arm and gripper moves measure the robot's delay. Arms park at the end |
| 6 | `bash slot/step.sh 6` | 1 min | camera sample, then the base drives 30 cm and turns (keep 1 m clear) |
| 7 | `bash slot/step.sh 7` | 1 min | two head sweeps from where the robot stands (only the head moves) |
| 8 | `bash slot/step.sh 8` | 4 min | the full mission, slowly. It picks scripted or learned stages itself and prints why. The bot re-anchors on the shelf with the laser (/scan, about 4 s per pick); the end report says how often the laser saw the shelf panel |
| 9 | `bash slot/step.sh 9` | 1 min | stops the recorders, packs everything into ONE file and prints its name |

## Clock rules (they beat the table)
- Started late or running late: steps 5, 6 and 7 skip themselves when it is too late (after 13:45, 13:35, 13:40) and
  print `NEXT: bash slot/step.sh 8`. Just follow NEXT.
- **Fallback:** if at 13:40 step 8 has not started, or step 1 still fails at 13:35 after the fixes below: type
  `bash slot/step.sh fallback`. It records, runs one scripted mission and packs the file (about 5 minutes).
- At 13:52 at the latest, whatever is running: Ctrl-C it and type `bash slot/step.sh 9`.

## When a step says FAILED
Every FAILED line comes with `what to do:`. Do that, then follow NEXT. These cases change the plan:

| step | FAILED item | do |
|---|---|---|
| 1 | `rate ...` (incl. `rate /scan`), `action servers` or `robot_description` | `ros2 topic list`. Empty or short: unplug and replug the cable, type the two `source` lines again, rerun step 1. Twice: ask the organisers whether the robot's software is up |
| 1 | `rmem_max`, `cyclone profile` or `cyclone buffer test` | the sysctl line from setup, the two `source` lines, rerun step 1 |
| 1 | `clock offset` | `sudo timedatectl set-ntp false`, then `sudo date -s @$(python3 slot/s00_preflight.py --robot-epoch)`, rerun step 1 |
| 3 | `present pose` | NEXT sends you to step 5 (no book test) |
| 4 | `close on book` (reads like air) | the book slipped out: slide it in deeper, run step 4 once more, then go on |
| 8 | `mission` | before 13:47: put the book back on the shelf, run step 8 once more. Later: step 9 |
| 8 | `laser anchors` | not blocking: write it on the sheet for the lead. If a drive went wrong near the shelf: `ERC_BOT_ANCHOR=0 bash slot/step.sh 8` |
| 4 | `hold hang margin` | not blocking: step 8 already uses the deeper release it set. Write the line on the sheet |
| any | a Python error with `Traceback` | Ctrl-C if it still runs, then run the NEXT step anyway (the error is in the logs) |

Anything else that FAILED is a measurement: the plan goes on.

## After the slot
Step 9 prints `COLLECTED: /home/<you>/jalpari_slot_<date>_<time>.tar.gz`. Send that one file to Karth today
(Drive or WhatsApp document), plus the photo of the camera values. If there is a USB stick, also copy the folder
`~/slot_out` (the full camera recordings, several GB).

## Time budget
Measured in the simulator dry runs of this sheet (robot clock): steps 1 to 4 and 8 on this kit's code (c4s,
2026-11-06, 2 seeds), steps 5, 6, 7 and 9 on kit-20261105-2149. Step 1 5 s (plus about 15 s for the network buffer
test, which only runs on the laptop), step 2 44 s, step 3 56 to 57 s, step 4 11 s (plus the book hand-over and the
tape), step 5 234 s, step 6 53 s, step 7 51 s, step 8 192 s (scripted mission at half speed with the laser anchors,
one seed run to the end), step 9 10 s. Robot time for the
whole sheet: about 11 minutes, plus a few seconds of start-up per command and the book hand-over. Started 20 minutes
late (13:20), steps 1 to 9 still end before 13:45, which leaves one retry of step 8.
