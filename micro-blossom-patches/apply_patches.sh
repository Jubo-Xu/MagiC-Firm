#!/usr/bin/env bash
# apply_patches.sh <checkout-dir> — apply MagiC-Firm's micro-blossom patches
# to a checkout (main /data clone or a worker clone).
#
# Idempotent and cheap: skips when the patch marker is already present, so
# repeated calls never touch mtimes (no spurious cargo rebuilds). Otherwise
# resets the touched paths to pristine and applies every patch in order.
#
# Patches:
#   0001 — optional non-uniform layer arrival schedule (LAYER_SCHEDULE_NS env;
#          absent -> behavior unchanged, verified bit-identical). Table-based:
#          per-layer ready-time offsets programmed into hardware BEFORE the
#          timed window (bus subaddresses 96/100/104), so scheduled runs add
#          zero in-window traffic. Touches the Scala bus, Rust driver,
#          simulator shim, embedded binding, bare-metal binding, benchmark main.
set -euo pipefail
CHECKOUT="${1:?usage: apply_patches.sh <micro-blossom-checkout>}"
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"

# version marker: the v2 (table-based) patch defines setup_load_stall_schedule
if grep -q "setup_load_stall_schedule" "$CHECKOUT/src/cpu/embedded/src/mains/benchmark_decoding.rs" 2>/dev/null; then
    echo "micro-blossom patches: already applied in $CHECKOUT"
    exit 0
fi
git -C "$CHECKOUT" checkout -- src/
for p in "$PATCH_DIR"/*.patch; do
    git -C "$CHECKOUT" apply "$p"
    echo "applied $(basename "$p") in $CHECKOUT"
done
# the Scala change invalidates the assembled simulation host jar; force
# reassembly on next ensure_ready
rm -f "$CHECKOUT/target/scala-2.12/microblossom.jar"
