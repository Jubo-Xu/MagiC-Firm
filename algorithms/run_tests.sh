#!/usr/bin/env bash
# run_tests.sh — Run all algorithm tests.
#
# Usage (from the algorithms/ directory):
#   bash run_tests.sh
#
# Runs every pytest file under test_python/ in the conda environment named
# below (conda must be on the PATH). Exit code: 0 if all pass, 1 if any fail.

set -euo pipefail

ALGORITHMS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$ALGORITHMS_DIR/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-magicfirm}"      # override: CONDA_ENV=<name> bash run_tests.sh

# cultiv lives in magic_state_cultivation/upstream/src — add to PYTHONPATH so tests
# that import it work without a pip install.
CULTIV_SRC="$REPO_ROOT/magic_state_cultivation/upstream/src"
PYTHON="conda run -n $CONDA_ENV env PYTHONPATH=${CULTIV_SRC}:${ALGORITHMS_DIR}:${PYTHONPATH:-} python"

PASS=0
FAIL=0
FAILURES=()

run_step() {
    local label="$1"; shift
    printf "  %-55s" "$label ..."
    if "$@" > /tmp/test_out.txt 2>&1; then
        echo "PASS"
        (( PASS++ )) || true
    else
        echo "FAIL"
        (( FAIL++ )) || true
        FAILURES+=("$label")
        cat /tmp/test_out.txt | sed 's/^/    /'
    fi
}

echo ""
echo "================================================================"
echo "Python unit tests (pytest)"
echo "================================================================"

cd "$ALGORITHMS_DIR"
for f in test_python/test_*.py; do
    [ -e "$f" ] || continue
    run_step "$f" $PYTHON -m pytest -q --tb=short "$f"
done

TOTAL=$(( PASS + FAIL ))
echo ""
echo "================================================================"
echo "Results: $PASS / $TOTAL passed"
if [ ${#FAILURES[@]} -gt 0 ]; then
    echo "Failed:"
    for f in "${FAILURES[@]}"; do
        echo "  - $f"
    done
    exit 1
fi
echo "All tests passed."
