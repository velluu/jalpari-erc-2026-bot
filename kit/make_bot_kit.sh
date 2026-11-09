#!/bin/bash
# Build the team-laptop bot kit from a COMMITTED erc-bot tree (git archive of REF, never the working directory), tag
# that commit kit-YYYYMMDD-HHMM, and stamp the commit, tag and every model / policy md5 into VERSION and README.md.
#   bash kit/make_bot_kit.sh full OUTDIR --model det.onnx [--model pose.onnx] [--default-model NAME.onnx] \
#        --wheels DIR --debs DIR [--policies DIR] [--extra NAME=DIR] [--ref REF] [--no-tag]
#   bash kit/make_bot_kit.sh code OUTDIR [--policies DIR] [--extra NAME=DIR] [--ref REF] [--no-tag]   drop-in update
# full -> OUTDIR/erc-bot-kit-<tag>.tar.gz, code -> OUTDIR/erc-bot-code-<tag>.tar.gz, each with a .md5 next to it.
# ERC_KIT_STAMP=YYYYMMDD-HHMM (Gulf time) when this machine's clock is off. Uncommitted changes are listed, not shipped.
set -euo pipefail
MODE=${1:?full or code}; OUT=$(realpath -m "${2:?outdir}"); shift 2
B=$(cd "$(dirname "$0")/.." && pwd)            # erc-bot checkout
G=${ERC_GYM:-$HOME/projects/erc-gym}
MODELS=(); DEFAULT=""; WHEELS=""; DEBS=""; POL=""; EXTRAS=(); REF=HEAD; TAG=1
while [ $# -gt 0 ]; do case $1 in
  --model) MODELS+=("$2"); shift 2;; --default-model) DEFAULT=$2; shift 2;; --wheels) WHEELS=$2; shift 2;;
  --debs) DEBS=$2; shift 2;; --policies) POL=$2; shift 2;; --extra) EXTRAS+=("$2"); shift 2;;
  --ref) REF=$2; shift 2;; --no-tag) TAG=0; shift;; *) echo "unknown $1"; exit 1;; esac; done
C=$(git -C "$B" rev-parse --verify "$REF^{commit}")
DIRTY=$(git -C "$B" status --porcelain | wc -l)
STAMP=${ERC_KIT_STAMP:-$(TZ=Asia/Dubai date +%Y%m%d-%H%M)}
NAME_TAG=kit-$STAMP
if [ $TAG = 1 ]; then
  git -C "$B" tag "$NAME_TAG" "$C" 2>/dev/null || [ "$(git -C "$B" rev-parse "$NAME_TAG^{commit}")" = "$C" ] || { echo "tag $NAME_TAG exists on another commit"; exit 1; }
fi
VER="$NAME_TAG-$(git -C "$B" rev-parse --short "$C")"
T=$(mktemp -d); K=$T/erc-bot-kit; mkdir -p $K $T/src
git -C "$B" archive "$C" erc_bot slot kit | tar x -C $T/src
[ -d $T/src/erc_bot ] && [ -d $T/src/slot ] && [ -d $T/src/kit ] || { echo "commit $C lacks erc_bot/, slot/ or kit/"; exit 1; }
cp -r $T/src/erc_bot $K/
rsync -a --exclude hpc $T/src/slot/ $K/slot/
cp $T/src/kit/{install.sh,check.sh,update.sh,kit_env.sh,apt.txt,requirements.txt,README.md} $K/; cp -r $T/src/kit/dds $K/dds
mkdir -p $K/policies $K/extras
[ -n "$POL" ] && cp "$POL"/*.npz "$POL"/*.json $K/policies/ 2>/dev/null || true
for e in "${EXTRAS[@]+"${EXTRAS[@]}"}"; do n=${e%%=*}; d=${e#*=}; rsync -a --exclude __pycache__ --exclude '*.pyc' "$d/" "$K/extras/$n/"; done
if [ "$MODE" = full ]; then
  [ ${#MODELS[@]} -gt 0 ] && [ -d "$WHEELS" ] && [ -d "$DEBS" ] || { echo "full kit needs --model, --wheels and --debs"; exit 1; }
  mkdir -p $K/models $K/wheels $K/debs $K/record
  for m in "${MODELS[@]}"; do cp "$m" $K/models/; done
  echo "${DEFAULT:-$(basename "${MODELS[0]}")}" > $K/models/DEFAULT
  [ -f "$K/models/$(cat $K/models/DEFAULT)" ] || { echo "default model $(cat $K/models/DEFAULT) not among --model"; exit 1; }
  cp "$WHEELS"/*.whl $K/wheels/; cp "$DEBS"/*.deb $K/debs/
  cp "$G"/tools/{record_session.sh,bag_to_frames.py} "$G"/gym/perception_node.py $K/record/ 2>/dev/null || true
fi
{
  echo "erc-bot-kit $VER ($MODE)"
  echo "erc-bot commit $C, tag $NAME_TAG$([ $TAG = 1 ] || echo ' (not tagged)')"
  echo "built $STAMP Gulf on $(hostname), working tree had $DIRTY uncommitted paths, none of them shipped"
  for e in "${EXTRAS[@]+"${EXTRAS[@]}"}"; do echo "extra ${e%%=*} from ${e#*=} $(git -C "${e#*=}" rev-parse --short HEAD 2>/dev/null || echo nogit)"; done
  for f in $K/models/*.onnx; do [ -f "$f" ] && echo "model $(basename "$f") md5 $(md5sum < "$f" | cut -c1-32)$([ "$(basename "$f")" = "$(cat $K/models/DEFAULT 2>/dev/null)" ] && echo ' (default)')"; done
  n=0; for f in $K/policies/*.npz; do [ -f "$f" ] && { n=1; echo "policy $(basename "$f") md5 $(md5sum < "$f" | cut -c1-32)"; }; done
  [ $n = 0 ] && echo "policies: none (scripted stages only)"
  echo "files $(find $K -type f | wc -l)"
} > $K/VERSION
{ echo; echo "## This build"; echo '```'; cat $K/VERSION; echo '```'; } >> $K/README.md
(cd $K && find . -type f ! -name MANIFEST.md5 | sort | xargs md5sum > MANIFEST.md5)
mkdir -p "$OUT"
NAME=erc-bot-$([ "$MODE" = full ] && echo kit || echo code)-${VER#kit-}.tar.gz
tar czf "$OUT/$NAME" --owner=0 --group=0 --numeric-owner -C $T erc-bot-kit
rm -rf $T
(cd "$OUT" && md5sum "$NAME" > "$NAME.md5")     # relative name, so md5sum -c works wherever the pair is copied
echo "$OUT/$NAME $(du -h "$OUT/$NAME" | cut -f1) md5 $(cut -c1-32 "$OUT/$NAME.md5") commit $C tag $NAME_TAG"
[ "$DIRTY" -gt 0 ] && echo "note: $DIRTY uncommitted paths in $B were not shipped (git status)"
exit 0
