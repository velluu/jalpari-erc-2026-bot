#!/bin/bash
# Offline install of the bot kit on the team laptop (Ubuntu 22.04, ROS 2 Humble). Run once, from the kit folder:
#   bash install.sh
# 1. apt: the .deb files in debs/ (control_msgs, the CycloneDDS RMW and what they need), skipped when already installed
# 2. pip (user site, no internet): numpy 1.24.4, onnxruntime 1.23.2, scipy 1.11.4 from wheels/; opencv-python-headless
#    only when `import cv2` fails (Humble desktop already has OpenCV)
# 3. bash check.sh
K=$(cd "$(dirname "$0")" && pwd); cd "$K"
[ -f /opt/ros/humble/setup.bash ] || { echo "FAIL install: no /opt/ros/humble, install ROS 2 Humble first"; exit 1; }
source /opt/ros/humble/setup.bash
set -u      # after the ROS setup script, which reads unset variables (AMENT_TRACE_SETUP_FILES)
SUDO=""; [ "$(id -u)" != 0 ] && SUDO=sudo
need=""
while read -r p; do case "$p" in ''|'#'*) continue;; esac
  dpkg -s "$p" >/dev/null 2>&1 || need="$need $p"
done < apt.txt
if [ -n "$need" ]; then
  echo "install: apt packages missing:$need, installing from debs/ (offline)"
  $SUDO dpkg -i --skip-same-version debs/*.deb >/tmp/kit_dpkg.log 2>&1 || { echo "FAIL install: dpkg, see /tmp/kit_dpkg.log; online fix: sudo apt install$need"; exit 1; }
else
  echo "install: apt packages present"
fi
PIP="python3 -m pip"
python3 -m pip --version >/dev/null 2>&1 || PIP="python3 $(ls wheels/pip-*.whl | head -1)/pip"
$PIP install --user --no-index --find-links wheels -r requirements.txt || { echo "FAIL install: pip"; exit 1; }
if ! python3 -c "import cv2" 2>/dev/null; then
  echo "install: no OpenCV in this Python, installing opencv-python-headless from wheels/"
  $PIP install --user --no-index --find-links wheels --no-deps opencv-python-headless || { echo "FAIL install: opencv wheel"; exit 1; }
fi
echo "install: done, now: bash check.sh"
