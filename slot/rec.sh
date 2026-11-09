#!/bin/bash
# Slot recorder. Terminal R1 for the whole hour (light, about 0.3 MB/s), terminal R2 only while the camera matters:
#   bash slot/rec.sh light slot1_light       joints, TF, scan, odom, cmd_vel, every /erc/ topic incl. action status and
#                                            feedback, controller state topics if the robot shows them, camera_info
#   bash slot/rec.sh cam slot1_sweep_pos1    compressed colour and aligned depth (about 2 to 6 MB/s), plus joints and TF
# Ctrl-C stops it cleanly. Bags land in $SLOT_OUT/bags (default ./slot_out/bags). 2 GB splits.
MODE=${1:?light or cam}; NAME=${2:-${MODE}_$(date +%H%M%S)}
OUT=${SLOT_OUT:-$PWD/slot_out}/bags; mkdir -p "$OUT"
if [ "${ROS_DOMAIN_ID:-0}" != "2" ] && [ "${ERC_SLOT_SIM:-}" != "1" ]; then
  echo "FAIL rec: ROS_DOMAIN_ID is ${ROS_DOMAIN_ID:-unset} -> source tiago_dds_connect/tiago_env.sh first"; exit 1
fi
C=head_front_camera   # no leading slash: the regex below is anchored at ^/
case $MODE in
  light) RE="^/(joint_states|tf|tf_static|scan|mobile_base_controller/odom|cmd_vel|erc/.*|erc_bot/status|.*controller_state|$C/color/camera_info)\$" ;;
  cam)   RE="^/(joint_states|tf|tf_static|erc/rejections|erc_bot/status|$C/color/camera_info|$C/color/image_raw/compressed|$C/aligned_depth_to_color/image_raw/compressedDepth)\$" ;;
  *) echo "FAIL rec: mode light or cam"; exit 1 ;;
esac
[ -e "$OUT/$NAME" ] && { echo "FAIL rec: $OUT/$NAME exists -> pick another name"; exit 1; }
echo "recording $MODE to $OUT/$NAME (Ctrl-C to stop)"
SIMARG=""; [ "${ERC_SLOT_SIM:-}" = "1" ] && SIMARG="--use-sim-time"
exec ros2 bag record -o "$OUT/$NAME" -b 2000000000 --include-hidden-topics $SIMARG -e "$RE"
