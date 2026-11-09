# JALPARI bot kit (team laptop, Ubuntu 22.04, ROS 2 Humble)

Everything the laptop needs for the hardware slots and the finals, offline: the bot (`erc_bot/`), the slot scripts
(`slot/`, run sheet `slot/SHEET.md`), the detector (`models/`, det-v4), Python wheels, the .deb files, the recording
tools (`record/`), and later the learned-stage runner and policies (`extras/`). `VERSION` says what is inside,
`MANIFEST.md5` proves the copy is intact.

## Once, at home (10 minutes, no internet needed)
```bash
tar xzf erc-bot-kit-<version>.tar.gz -C ~        # makes ~/erc-bot-kit
cd ~/erc-bot-kit
bash install.sh                                   # apt .debs (sudo), then pip wheels into your user site
sudo cp dds/60-erc-dds.conf /etc/sysctl.d/ && sudo sysctl --system   # CRITICAL: net.core.rmem_max 4 MB for camera frames
bash check.sh                                     # must end with READY
```
Without that sysctl CycloneDDS drops whole camera frames (lane C: 0 Hz at the default 212992) and the bot stops
with "interface not ready". `kit_env.sh` and `slot/s00_preflight.py` both refuse loudly until it is set.
The organisers' DDS profile: `git clone https://github.com/dfl-rlab/tiago_dds_connect ~/tiago_dds_connect`, or offline
`cp -r dds/tiago_dds_connect ~/` (copy of commit b4c960d, 2026-10-28).

## Every terminal at the robot
```bash
source ~/tiago_dds_connect/tiago_env.sh
source ~/erc-bot-kit/kit_env.sh                   # 4 MB Cyclone receive-buffer profile, PYTHONPATH, model, results folder
cd ~/erc-bot-kit
```
Then follow `slot/SHEET.md` minute by minute. The bot alone: `python3 -m erc_bot.mission --digit 2 --label red --model $ERC_BOT_MODEL`.

## A newer bot
The lead sends `erc-bot-code-<version>.tar.gz` (and its `.md5`). Put both anywhere, then:
```bash
bash ~/erc-bot-kit/update.sh ~/Downloads/erc-bot-code-<version>.tar.gz     # verifies, swaps code, runs check.sh
bash ~/erc-bot-kit/update.sh --rollback                                     # back to the previous version
```
Models, wheels and .debs stay. `head -1 VERSION` shows the version now in use. Say it when reporting a run.

## If something fails
- `check.sh` prints one FAIL line per problem with the fix after `->`. Send the whole output to Karth if the fix does not work.
- numpy must stay 1.24.4 (Humble's OpenCV breaks under numpy 2): never `pip install --upgrade` in this user.
- Nothing in the kit sends anything anywhere. Results stay in `slot_out/<date>/` next to the kit.
