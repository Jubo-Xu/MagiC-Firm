#!/usr/bin/env bash
# setup_local.sh — machine-local setup for MagiC-Firm on hosts where large
# generated files should live on a local disk instead of (quota-limited,
# network-mounted) $HOME. Mirrors the mb4msc "clone onto local disk" advice.
#
# Usage:
#   ./setup_local.sh /data/$USER
#
# What it does (idempotent — safe to re-run):
#   1. Creates <data>/MagiC-Firm_outputs and symlinks out/ and
#      hardware/emulator/build into it, so builds/results land on local disk.
#   2. Ensures the magic_state_cultivation/upstream submodule is checked out
#      and applies our patches (magic_state_cultivation/apply_patches.sh).
#   3. Clones micro-blossom onto <data> at the commit pinned by this repo,
#      deinits the local submodule checkout, and replaces the micro-blossom
#      path with a symlink to that clone — so its heavyweight cargo/sbt/
#      Verilator builds never touch $HOME. Two git protections make the
#      symlink invisible to git:
#        - submodule.micro-blossom.ignore=all   (silences status/diff)
#        - update-index --skip-worktree         (so `git add -A` can't stage
#                                                the symlink over the pin)
#      To intentionally update the micro-blossom pin later:
#        git update-index --no-skip-worktree micro-blossom, update, re-run this.
#   4. If docker + the micro-blossom toolchain image are available, starts the
#      long-lived 'mb-magicfirm' container that mb_toolchain.py runs RTL
#      generation / Verilator builds / simulator executions in.
#
# If you cloned this repo directly onto a local disk and are happy building
# in place, you don't need step 3: just `git submodule update --init` and
# run magic_state_cultivation/apply_patches.sh.

set -euo pipefail

if [ $# -gt 1 ]; then
    echo "usage: $0 [<local-data-dir>]   e.g. $0 /data/\$USER   (no argument: keep everything in the repo)" >&2
    exit 1
fi
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

write_config() {   # <out dir> <micro-blossom dir> <container name>  -> .magicfirm.env (read by experiments/lib/paths.py)
    printf '%s\n' "# written by setup_local.sh; not committed. Environment variables override." \
        "MAGICFIRM_OUT=$1" "MAGICFIRM_MB_ROOT=$2" "MAGICFIRM_MB_CONTAINER=$3" > .magicfirm.env
    echo "config written to .magicfirm.env"
}

# --- 0. in-repo mode: no data dir; submodule + patches + config only --------
if [ $# -eq 0 ]; then
    if [ ! -f magic_state_cultivation/upstream/src/cultiv/__init__.py ]; then
        git submodule update --init magic_state_cultivation/upstream
    fi
    ./magic_state_cultivation/apply_patches.sh
    mkdir -p out
    write_config "$REPO_ROOT/out" "$REPO_ROOT/micro-blossom" "mb-magicfirm-$USER"
    echo "outputs -> $REPO_ROOT/out"
    exit 0
fi
DATA_DIR="$(readlink -f "$1")"

MB_URL="https://github.com/yuewuo/micro-blossom.git"
MB_PIN="$(git ls-files -s micro-blossom | awk '{print $2}')"
OUTS="$DATA_DIR/MagiC-Firm_outputs"

# --- 1. output dirs on local disk ------------------------------------------
mkdir -p "$OUTS/emulator_build" "$OUTS/out"
ln -sfn "$OUTS/out" out
ln -sfn "$OUTS/emulator_build" hardware/emulator/build
echo "outputs -> $OUTS"

# --- 2. cultivation submodule + patches ------------------------------------
if [ ! -f magic_state_cultivation/upstream/src/cultiv/__init__.py ]; then
    git submodule update --init magic_state_cultivation/upstream
fi
./magic_state_cultivation/apply_patches.sh

# --- 3. micro-blossom on local disk ----------------------------------------
if [ ! -d "$DATA_DIR/micro-blossom/.git" ]; then
    echo "cloning micro-blossom -> $DATA_DIR/micro-blossom"
    git clone "$MB_URL" "$DATA_DIR/micro-blossom"
fi
git -C "$DATA_DIR/micro-blossom" checkout -q "$MB_PIN"
echo "micro-blossom at pinned $(git -C "$DATA_DIR/micro-blossom" rev-parse --short HEAD)"
"$REPO_ROOT/micro-blossom-patches/apply_patches.sh" "$DATA_DIR/micro-blossom"

# Only deinit a real submodule checkout — never follow an existing symlink
# (git would try to absorb the /data clone's .git dir into our repo).
if [ ! -L micro-blossom ] && { [ -f micro-blossom/.git ] || [ -d micro-blossom/.git ]; }; then
    git submodule deinit -f micro-blossom
    rm -rf .git/modules/micro-blossom
fi
[ -d micro-blossom ] && [ ! -L micro-blossom ] && rmdir micro-blossom
ln -sfn "$DATA_DIR/micro-blossom" micro-blossom
git config submodule.micro-blossom.ignore all
git update-index --skip-worktree micro-blossom
echo "micro-blossom -> $DATA_DIR/micro-blossom (symlink, git-protected)"

# --- 4. micro-blossom toolchain container (optional; needs docker) ---------
# Long-lived idle container from the pinned toolchain image (Rust, sbt/
# SpinalHDL, Verilator). experiments/lib/mb_toolchain.py runs every hardware
# command (RTL generation, Verilator builds, simulator runs) inside it via
# `docker exec`; it also auto-starts the container, so this step is a
# convenience that surfaces docker/image problems at setup time instead of
# minutes into a first latency run. DATA_DIR is mounted at the identical
# path inside the container so absolute paths work in both worlds.
MB_IMAGE="micro-blossom:latest"
MB_CONTAINER="mb-magicfirm-$USER"
write_config "$OUTS/out" "$DATA_DIR/micro-blossom" "$MB_CONTAINER"

mb_docker() {
    if docker info >/dev/null 2>&1; then
        docker "$@"
    else
        sg docker -c "docker $*"
    fi
}

if ! mb_docker info >/dev/null 2>&1; then
    echo "NOTE: docker not usable (no group access?) — micro-blossom hardware"
    echo "      runs unavailable until it is. Everything else is set up."
elif ! mb_docker image inspect "$MB_IMAGE" >/dev/null 2>&1; then
    echo "NOTE: docker image '$MB_IMAGE' not found on this machine. Build it once with:"
    echo "      docker build -t $MB_IMAGE micro-blossom/"
elif mb_docker container inspect -f '{{.State.Running}}' "$MB_CONTAINER" 2>/dev/null | grep -q true; then
    echo "toolchain container '$MB_CONTAINER' already running"
else
    mb_docker rm -f "$MB_CONTAINER" >/dev/null 2>&1 || true
    mb_docker run -d --name "$MB_CONTAINER" -v "$DATA_DIR:$DATA_DIR" "$MB_IMAGE" sleep infinity >/dev/null
    echo "toolchain container '$MB_CONTAINER' started ($MB_IMAGE, $DATA_DIR mounted)"
fi

echo "OK: local setup complete"
