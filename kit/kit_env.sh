# Source in every terminal AFTER tiago_dds_connect/tiago_env.sh:   source kit_env.sh
# DDS: switches CYCLONEDDS_URI to dds/cyclone_tiago_rcvbuf.xml (the organisers' cyclone_tiago.xml, commit b4c960d,
# plus SocketReceiveBufferSize min 4MB, lane C) when net.core.rmem_max allows 4 MB; otherwise keeps the organisers'
# profile and prints the fix (with min 4MB Cyclone refuses to start below that, so the switch waits for the sysctl).
# Also: erc_bot, slot, extras/* on PYTHONPATH, ERC_BOT_MODEL, ERC_POLICIES, SLOT_OUT.
K=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RMEM=$(cat /proc/sys/net/core/rmem_max 2>/dev/null || echo 0)
if [ -z "${ROBOT_IFACE:-}" ] || [ "${ROS_DOMAIN_ID:-}" != 2 ]; then
  echo -e "\033[31mkit_env: source ~/tiago_dds_connect/tiago_env.sh first (ROBOT_IFACE and domain 2 unset)\033[0m"
elif [ "$RMEM" -ge 4194304 ]; then
  export CYCLONEDDS_URI="file://$K/dds/cyclone_tiago_rcvbuf.xml"
  echo "kit_env: CycloneDDS 4 MB receive buffer (rmem_max $RMEM), iface $ROBOT_IFACE"
else
  echo -e "\033[31m!!! kit_env: net.core.rmem_max is $RMEM: camera frames will be lost and the bot can stop with 'interface not ready'.\033[0m"
  echo -e "\033[31m!!! Fix once: sudo cp $K/dds/60-erc-dds.conf /etc/sysctl.d/ && sudo sysctl --system   then source both env files again.\033[0m"
fi
for x in "$K"/extras/*/; do [ -d "$x" ] && PYTHONPATH="$x:${PYTHONPATH:-}"; done
export PYTHONPATH="$K:$K/slot:${PYTHONPATH:-}"
export ERC_POLICIES="$K/policies"
export ERC_BOT_MODEL="$K/models/$(cat "$K/models/DEFAULT" 2>/dev/null || ls "$K"/models/ | grep onnx | head -1)"   # chosen at build time (models/DEFAULT)
export SLOT_OUT=${SLOT_OUT:-$K/../slot_out/$(date +%Y%m%d)}
mkdir -p "$SLOT_OUT"
echo "kit_env: $(head -1 "$K/VERSION" 2>/dev/null), model $(basename "$ERC_BOT_MODEL"), results in $SLOT_OUT"
