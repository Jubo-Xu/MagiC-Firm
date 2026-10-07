#!/usr/bin/env bash
# apply_patches.sh — apply our local modifications to the pristine
# magic-state-cultivation submodule checkout (upstream/).
#
# Idempotent: resets upstream/ to its pinned pristine state first, then
# applies every patch in patches/ in order. Run after cloning
# (git clone --recurse-submodules) and after any submodule update.
#
# The patches are the ONLY changes we carry against upstream:
#   0001 — int() casts for sinter stats (numpy>=2 ints are not JSON
#          serializable and repr as np.int64(...), breaking sinter output)
#   0002 — timed gap-decode methods on CompiledDesaturationSampler
#          (decode_det_set_with_time; serial/parallel decoder-latency model)

set -euo pipefail
cd "$(dirname "$0")/upstream"

git checkout -- src
for p in ../patches/*.patch; do
    git apply "$p"
    echo "applied $(basename "$p")"
done
echo "OK: upstream patched ($(git diff --stat | tail -1))"
