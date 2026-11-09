#!/bin/bash
# Drop a newer bot into this kit without repacking by hand:   bash update.sh /path/erc-bot-code-<ver>.tar.gz
# Verifies the tarball's own MANIFEST.md5, keeps the current code in .prev/<old version>, swaps in erc_bot, slot,
# extras and the scripts, rewrites MANIFEST.md5 and VERSION (models, wheels, debs stay), then runs check.sh.
# Roll back: bash update.sh --rollback
set -euo pipefail
K=$(cd "$(dirname "$0")" && pwd)
PARTS="erc_bot slot policies extras dds install.sh check.sh update.sh kit_env.sh apt.txt requirements.txt README.md"
if [ "${1:-}" = --rollback ]; then
  P=$(ls -td "$K"/.prev/*/ | head -1); [ -n "$P" ] || { echo "FAIL update: nothing to roll back"; exit 1; }
  for x in $PARTS; do rm -rf "$K/$x"; [ -e "$P/$x" ] && cp -a "$P/$x" "$K/"; done
  cp "$P/VERSION" "$K/VERSION"; (cd "$K" && find . -path ./.prev -prune -o -type f ! -name MANIFEST.md5 -print | sort | xargs md5sum > MANIFEST.md5)
  rm -rf "$P"; echo "rolled back to $(head -1 "$K/VERSION")"; exec bash "$K/check.sh"
fi
TGZ=$(realpath "${1:?code tarball}")
[ -f "$TGZ.md5" ] && { (cd "$(dirname "$TGZ")" && md5sum -c "$(basename "$TGZ").md5" >/dev/null) || { echo "FAIL update: tarball md5 differs, copy it again"; exit 1; }; }
T=$(mktemp -d); tar xzf "$TGZ" -C "$T"; N=$T/erc-bot-kit
(cd "$N" && md5sum --quiet -c MANIFEST.md5) || { echo "FAIL update: files in the tarball do not match its manifest"; rm -rf "$T"; exit 1; }
OLD=$(head -1 "$K/VERSION" | awk '{print $2}'); mkdir -p "$K/.prev/$OLD"
for x in $PARTS VERSION; do [ -e "$K/$x" ] && cp -a "$K/$x" "$K/.prev/$OLD/"; done
for x in $PARTS; do [ -e "$N/$x" ] && { rm -rf "$K/$x"; cp -a "$N/$x" "$K/"; }; done
{ head -1 "$N/VERSION"; tail -n +2 "$N/VERSION"; grep -E "^model " "$K/.prev/$OLD/VERSION" || true; echo "updated $(date -Is) from $OLD"; } > "$K/VERSION"
(cd "$K" && find . -path ./.prev -prune -o -type f ! -name MANIFEST.md5 -print | sort | xargs md5sum > MANIFEST.md5)
rm -rf "$T"
echo "updated to $(head -1 "$K/VERSION") (previous kept in .prev/$OLD)"
exec bash "$K/check.sh"
