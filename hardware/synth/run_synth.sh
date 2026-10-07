#!/usr/bin/env bash
# run_synth.sh — run the Vivado jobs of one or more synthesis bundles (gen_synth.py output),
# either on this machine or on a remote Vivado host.
#
# Usage: run_synth.sh <bundle root> [<bundle name> ...]     default: every bundle under the root
#
# Settings come from hardware/synth/vivado.env (start from vivado.env.example) or from the
# environment:
#   VIVADO_MODE        local or remote
#   VIVADO_SETTINGS    Vivado's settings64.sh (on the remote host in remote mode)
#   VIVADO_LICENSE     licence server, optional
#   VIVADO_JOBS        parallel Vivado processes, default 4
#   VIVADO_HOST        remote mode: ssh host
#   VIVADO_REMOTE_DIR  remote mode: work directory on the host (use a local scratch disk)
#
# Each tcl/boardN.tcl of a bundle runs in runs/boardN/. Boards with an existing
# runs/boardN/summary.json are skipped; delete the file to rerun. In remote mode the bundle is
# copied to the host and the reports (summary.json, *.rpt, vivado.log) are copied back.
# Then collect the results:
#   python hardware/synth/collect_synth.py <bundle root>
#
# One board can also be run by hand on any machine with Vivado:
#   cd <bundle>/runs/boardN && vivado -mode batch -source ../../tcl/boardN.tcl
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$HERE/vivado.env" ] && { set -a; . "$HERE/vivado.env"; set +a; }
ROOT="$(readlink -f "${1:?usage: run_synth.sh <bundle root> [name...]}")"; shift
MODE="${VIVADO_MODE:?set VIVADO_MODE=local|remote in hardware/synth/vivado.env}"
SETTINGS="${VIVADO_SETTINGS:?set VIVADO_SETTINGS=/path/to/Vivado/<ver>/settings64.sh}"
JOBS="${VIVADO_JOBS:-4}"
LICENSE="${VIVADO_LICENSE:-}"
NAMES=("$@")
if [ ${#NAMES[@]} -eq 0 ]; then NAMES=($(ls "$ROOT")); fi

# Runs on the Vivado machine: <root> <jobs> <settings> <licence> <bundle names...>.
LOOP='
ROOT=$1; JOBS=$2; SETTINGS=$3; LICENSE=$4; shift 4
export LC_ALL=C; unset LANGUAGE
[ -n "$LICENSE" ] && export XILINXD_LICENSE_FILE="$LICENSE"
source "$SETTINGS"
run_one() {   # <bundle dir> <board tcl>
    local B=$1 T=$2 b R t0 rc
    b=$(basename "$T" .tcl); R="$B/runs/$b"
    if [ -f "$R/summary.json" ]; then echo "skip $(basename "$B")/$b (done)"; return 0; fi
    mkdir -p "$R"; cd "$R"
    t0=$SECONDS
    vivado -mode batch -source "$T" -log vivado.log -journal vivado.jou > stdout.txt 2>&1
    rc=$?
    echo "$(basename "$B")/$b: rc=$rc $((SECONDS - t0))s $(grep -o "SYNTH_DONE.*" vivado.log || grep -m1 "^ERROR" vivado.log || true)"
}
export -f run_one
for n in "$@"; do ls "$ROOT/$n"/tcl/*.tcl | sed "s|^|$ROOT/$n |"; done \
    | xargs -P "$JOBS" -L 1 bash -c '"'"'run_one "$@"'"'"' _
'

case "$MODE" in
  local)
    bash -c "$LOOP" _ "$ROOT" "$JOBS" "$SETTINGS" "$LICENSE" "${NAMES[@]}"
    ;;
  remote)
    HOST="${VIVADO_HOST:?set VIVADO_HOST for remote mode}"
    REMOTE="${VIVADO_REMOTE_DIR:?set VIVADO_REMOTE_DIR for remote mode}"
    for n in "${NAMES[@]}"; do
        rsync -a --exclude runs "$ROOT/$n/" "$HOST:$REMOTE/$n/"
    done
    ssh "$HOST" "bash -s" -- "$REMOTE" "$JOBS" "$SETTINGS" "$LICENSE" "${NAMES[@]}" <<< "$LOOP"
    for n in "${NAMES[@]}"; do
        mkdir -p "$ROOT/$n/runs"
        rsync -a --include='*/' --include='summary.json' --include='*.rpt' --include='vivado.log' \
              --exclude='*' "$HOST:$REMOTE/$n/runs/" "$ROOT/$n/runs/"
    done
    ;;
  *) echo "VIVADO_MODE must be local or remote (got '$MODE')" >&2; exit 1 ;;
esac
