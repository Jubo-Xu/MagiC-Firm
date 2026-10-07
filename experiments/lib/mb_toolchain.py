# mb_toolchain.py — run micro-blossom's hardware toolchain (Rust, sbt/
# SpinalHDL, Verilator) inside the pinned docker container.
#
# Everything process/docker-related lives here; the rest of the project
# calls plain Python functions. Design follows micro-blossom's own model:
# the simulator self-generates RTL and Verilator-compiles it at every
# launch (SpinalSim paradigm; ~minutes, softened by ccache), so there is
# no per-graph RTL cache — only one-time global preparation
# (ensure_ready: cargo build + sbt assembly) and per-launch execution
# with a unique simWorkspace name.
#
# Layout of decoder artifact dirs (on /data via the out/ symlink):
#   out/mb/<decoder_key>/
#     graph.json, graph.hash      — decoder identity (mb_graph.graph_hash)
#     runs/<run_id>/              — scratch per execution, deleted on
#                                   success unless keep_runs
#     latency.json, ...           — durable distilled results

import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import time
import uuid

import numpy as np

import paths

REPO = paths.REPO
MB_ROOT = paths.MB_ROOT                                # micro-blossom checkout (config / env / submodule)
BLOSSOM_CRATE = MB_ROOT / "src" / "cpu" / "blossom"
OUT_MB = paths.OUT / "mb"                              # resolved path: valid inside the container too

MB_IMAGE = "micro-blossom:latest"
MB_CONTAINER = paths.MB_CONTAINER
# Mounted at the identical path inside the container so absolute paths are
# valid in both worlds.
MB_MOUNT = MB_ROOT.parent                              # e.g. /data/<user>
# Per-worker checkouts for parallel campaigns (each simulator launch needs an
# exclusive checkout: embedded.defects slot, cargo target, simWorkspace are
# all in-tree). Created on demand by ensure_worker().
WORKERS_DIR = MB_MOUNT / "micro-blossom-workers"


def _crate(root: pathlib.Path) -> pathlib.Path:
    """Blossom crate dir of a checkout (main MB_ROOT or a worker root).
    [used: internally wherever a checkout-relative path is needed]"""
    return root / "src" / "cpu" / "blossom"


class ToolchainError(RuntimeError):
    pass


def _docker(*args: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a docker CLI command, falling back to `sg docker -c` when the
    current session lacks the docker group (pre-relogin sessions).
    [used: internally by every container interaction]"""
    cmd = ["docker", *args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 and "permission denied" in (r.stderr or "").lower():
        joined = " ".join(shlex.quote(a) for a in cmd)
        r = subprocess.run(["sg", "docker", "-c", joined], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise ToolchainError(f"docker {' '.join(args[:3])}... failed:\n{r.stderr[-2000:]}")
    return r


def _ensure_container() -> None:
    """Start the long-lived toolchain container if it isn't running.
    [used: automatically before any in-container command; a no-op check
    after the first call — setup_local.sh step 4 normally did this already]"""
    r = _docker("container", "inspect", "-f", "{{.State.Running}}", MB_CONTAINER, check=False)
    if r.returncode == 0 and r.stdout.strip() == "true":
        return
    _docker("rm", "-f", MB_CONTAINER, check=False)
    mounts = ["-v", f"{MB_MOUNT}:{MB_MOUNT}"]
    if not paths.OUT.is_relative_to(MB_MOUNT):
        mounts += ["-v", f"{paths.OUT}:{paths.OUT}"]
    _docker("run", "-d", "--name", MB_CONTAINER, *mounts, MB_IMAGE, "sleep", "infinity")


_CONTAINER_PATH = "/root/.cargo/bin:/root/.local/share/coursier/bin:/usr/local/bin:/usr/bin:/bin"


def _exec(cmd: str, *, cwd: pathlib.Path, env: dict[str, str] | None = None,
          timeout: int | None = None) -> str:
    """Run one shell command inside the container; return stdout.

    `cwd` must be under the container mount (i.e. on /data). Raises
    ToolchainError with captured output on failure.
    [used: internally by every toolchain command (builds, parser, simulator)]
    """
    _ensure_container()
    # SpinalHDL's Verilator backend compiles with `make -j<availableProcessors>`;
    # inside the container that is the whole host (128 cores), so N parallel
    # workers would launch N x 128 compile jobs and swamp the machine. Cap the
    # processor count the JVM (and hence make -j) sees per worker.
    # MB_COMPILE_CPUS is set by run_mb_characterization from --workers.
    # Two layers: the JVM flag (for anything that consults availableProcessors)
    # and, decisively, a `make` wrapper in out/mb/bin (first on PATH) that
    # rewrites -jN / --jobs=N to the cap — SpinalHDL's cpuCount does not come
    # from availableProcessors, so the flag alone left `make -j128` in place.
    compile_cpus = os.environ.get("MB_COMPILE_CPUS", "16")
    base_env = {"PATH": f"{OUT_MB / 'bin'}:{_CONTAINER_PATH}",
                "MB_COMPILE_CPUS": compile_cpus,
                # MB_JAVA_OPTS: JVM flags for the SpinalHDL/Verilator simulation host.
                # Default ParallelGC: OpenJDK 11's collectors occasionally SIGSEGV during
                # elaboration of large hard-coded-weight designs (G1 more often than
                # ParallelGC); run_mb_characterization retries such shards.
                "JAVA_TOOL_OPTIONS": (f"-XX:ActiveProcessorCount={compile_cpus} "
                                      f"{os.environ.get('MB_JAVA_OPTS', '-XX:+UseParallelGC')}").strip()}
    env_args = []
    for k, v in {**base_env, **(env or {})}.items():
        env_args += ["-e", f"{k}={v}"]
    # Run everything at low scheduling priority (MB_NICE, default 19): the
    # RTL compiles and simulations are throughput jobs on a shared machine.
    nice = os.environ.get("MB_NICE", "19")
    r = _docker("exec", *env_args, "-w", str(cwd), MB_CONTAINER,
                "nice", "-n", nice, "bash", "-c", cmd, check=False)
    if r.returncode != 0:
        raise ToolchainError(
            f"command failed in container (cwd={cwd}):\n  {cmd}\n"
            f"--- stdout tail ---\n{r.stdout[-1500:]}\n--- stderr tail ---\n{r.stderr[-1500:]}")
    return r.stdout


# ---------------------------------------------------------------- readiness

def ensure_ready(*, root: pathlib.Path = MB_ROOT, quiet: bool = False) -> None:
    """One-time preparation of a micro-blossom checkout (main or worker):
    cargo release build (micro_blossom + embedded_simulator) and
    sbt assembly (microblossom.jar for the simulation host).
    [used: once per checkout, before any decoder work — environment
    initialization; cheap existence check afterwards]"""
    crate = _crate(root)
    need_cargo = not (crate / "target" / "release" / "micro_blossom").exists() \
        or not (crate / "target" / "release" / "embedded_simulator").exists()
    need_jar = not (root / "target" / "scala-2.12" / "microblossom.jar").exists()
    if need_cargo:
        if not quiet:
            print(f"mb_toolchain: cargo build --release in {root.name} (first time, ~minutes)...")
        _exec("cargo build --release", cwd=crate)
    if need_jar:
        if not quiet:
            print(f"mb_toolchain: sbt assembly in {root.name} (first time, ~minutes)...")
        _exec("sbt assembly", cwd=root)


# ------------------------------------------------------- decoder directories

def decoder_dir(key: str) -> pathlib.Path:
    """Artifact directory out/mb/<key>/ for one decoder configuration.
    [used: by prepare_decoder and result writers, any time a decoder's
    files are addressed]"""
    d = OUT_MB / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def prepare_decoder(key: str, initializer: dict, positions: list[dict]) -> pathlib.Path:
    """Materialize a decoder's identity in out/mb/<key>/: write the
    initializer as a syndromes-format header, run `micro_blossom parser`
    to produce the MicroBlossomSingle graph.json the simulator loads, and
    stamp graph.hash. Software-only and fast — no RTL work happens here.
    [used: once per decoder configuration (instantiation time); cached
    no-op when graph.hash already matches]"""
    import mb_graph  # local import; algorithms/ is on sys.path for callers

    d = decoder_dir(key)
    h = mb_graph.graph_hash(initializer)
    hash_file = d / "graph.hash"
    graph_json = d / "graph.json"
    if graph_json.exists() and hash_file.exists() and hash_file.read_text() == h:
        return graph_json

    ensure_ready()
    header = d / "graph.syndromes"
    mb_graph.write_syndrome_file(header, initializer, positions)
    micro_blossom = BLOSSOM_CRATE / "target" / "release" / "micro_blossom"
    _exec(f"{micro_blossom} parser {header} --graph-file {graph_json}",
          cwd=BLOSSOM_CRATE)
    hash_file.write_text(h)
    return graph_json


# ------------------------------------------------------------- simulation

def _sim_lock(root: pathlib.Path = MB_ROOT):
    """Advisory PER-CHECKOUT lock serializing simulator launches: within one
    checkout, the embedded.defects slot, cargo target, and simWorkspace are
    shared, so launches must queue. Different worker checkouts have
    different locks and run truly concurrently.
    [used: internally by run_simulator / run_benchmark_decoding]"""
    import contextlib, fcntl

    @contextlib.contextmanager
    def lock():
        lock_dir = OUT_MB / ".locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        with open(lock_dir / f"sim_{root.name}", "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield
    return lock()


def run_simulator(key: str, main: str, *, extra_env: dict[str, str] | None = None,
                  root: pathlib.Path = MB_ROOT,
                  keep_run: bool = False, quiet: bool = False) -> tuple[str, pathlib.Path]:
    """Launch one embedded_simulator process for decoder <key>: the JVM
    host regenerates RTL for graph.json and Verilator-compiles it (the
    slow per-launch step), then the embedded main `main` drives the
    simulated hardware. Returns (stdout, run_dir); run_dir is deleted on
    success unless keep_run, always kept on failure.
    [used: once per simulator launch — a whole latency campaign or a
    MicroBlossomMatching lifetime; NOT per shot]"""
    d = decoder_dir(key)
    graph_json = d / "graph.json"
    if not graph_json.exists():
        raise ToolchainError(f"decoder {key} not prepared (no graph.json); call prepare_decoder first")
    ensure_ready(root=root)

    run_dir = d / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir.mkdir(parents=True)
    env = {"EMBEDDED_BLOSSOM_MAIN": main, "MANUALLY_COMPILE_QEC": "1", **(extra_env or {})}
    sim = _crate(root) / "target" / "release" / "embedded_simulator"
    if not quiet:
        print(f"mb_toolchain: launching simulator [{main}] for {key} "
              f"(RTL+Verilator compile at startup, ~minutes on first build)...")
    try:
        with _sim_lock(root):
            _exec(f"{sim} {graph_json} > {run_dir}/stdout.txt 2> {run_dir}/stderr.txt",
                  cwd=_crate(root), env=env)
    except ToolchainError:
        raise  # run_dir kept for post-mortem (stdout/stderr are in it)
    stdout = (run_dir / "stdout.txt").read_text()
    if not keep_run:
        import shutil
        shutil.rmtree(run_dir)
    return stdout, run_dir


def chown_outputs() -> None:
    """Reclaim ownership of container-written (root-owned) files under the
    micro-blossom checkout and out/mb.
    [used: occasionally, after builds or before manual cleanup]"""
    uid, gid = os.getuid(), os.getgid()
    _exec(f"chown -R {uid}:{gid} {MB_ROOT}/target {MB_ROOT}/simWorkspace {WORKERS_DIR} {OUT_MB} 2>/dev/null || true",
          cwd=MB_MOUNT)

def pack_defects(syndromes_path: pathlib.Path, defects_path: pathlib.Path, *,
                 root: pathlib.Path = MB_ROOT) -> None:
    """Convert a syndromes file (with defect lines) into the binary .defects
    stream benchmark_decoding compiles in, via `micro_blossom parser`.
    [used: once per characterization run/shard, after writing the defect lists]"""
    ensure_ready(root=root)
    micro_blossom = _crate(root) / "target" / "release" / "micro_blossom"
    _exec(f"{micro_blossom} parser {syndromes_path} --defects-file {defects_path}",
          cwd=_crate(root))


def run_benchmark_decoding(key: str, defects_file: pathlib.Path, *,
                           num_layer_fusion: int,
                           measurement_cycle_ns: int = 1000,
                           frequency_hz: float = 100e6,
                           layer_schedule_ns: list[int] | None = None,
                           use_layer_fusion: bool = True,
                           max_round: int = 0,
                           root: pathlib.Path = MB_ROOT,
                           keep_run: bool = False,
                           quiet: bool = False) -> tuple[str, pathlib.Path]:
    """Run the embedded 'benchmark_decoding' latency benchmark for decoder
    <key> on a prepared defects file. Unlike run_simulator, this main bakes
    the defects and its parameters into the binary at COMPILE time
    (include_bytes! / option_env!), so it: (1) copies the defects file to
    the in-tree gitignored slot the embedded crate includes, (2) launches
    via `cargo run` so the crate rebuilds (cargo tracks both), then the
    usual self-building RTL simulation runs. The copy and run both happen
    under the sim lock: the slot is shared per checkout.
    Returns (stdout, run_dir); per-shot lines are parsed by
    parse_benchmark_latencies.
    [used: once per characterization campaign launch — NOT per shot]"""
    d = decoder_dir(key)
    graph_json = d / "graph.json"
    if not graph_json.exists():
        raise ToolchainError(f"decoder {key} not prepared; call prepare_decoder first")
    ensure_ready(root=root)

    run_dir = d / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir.mkdir(parents=True)
    env = {
        "EMBEDDED_BLOSSOM_MAIN": "benchmark_decoding",
        "MANUALLY_COMPILE_QEC": "1",
        "SUPPORT_LAYER_FUSION": "1",
        "SUPPORT_LOAD_STALL_EMULATOR": "1",
        "NUM_LAYER_FUSION": str(num_layer_fusion),
        "MEASUREMENT_CYCLE_NS": str(measurement_cycle_ns),
        "MICRO_BLOSSOM_FREQUENCY": f"{frequency_hz:g}",
        "MAX_ROUND": str(max_round),
    }
    if use_layer_fusion:
        env["USE_LAYER_FUSION"] = "1"
    if layer_schedule_ns is not None:
        # cumulative ns offsets from the first layer's arrival (patch 0001);
        # entry 0 must be 0, length must equal num_layer_fusion
        assert len(layer_schedule_ns) == num_layer_fusion and layer_schedule_ns[0] == 0
        env["LAYER_SCHEDULE_NS"] = ",".join(str(int(v)) for v in layer_schedule_ns)
    if not quiet:
        print(f"mb_toolchain: benchmark_decoding for {key} [{root.name}] "
              f"(cargo rebuild + RTL/Verilator compile, ~minutes)...")
    embedded_defects = root / "src" / "cpu" / "embedded" / "embedded.defects"
    try:
        with _sim_lock(root):
            # copy inside the container: the slot (like the rest of the
            # checkout's build outputs) is root-owned
            _exec(f"cp {defects_file} {embedded_defects}", cwd=MB_MOUNT)
            _exec(f"cargo run --release --bin embedded_simulator -- {graph_json} "
                  f"> {run_dir}/stdout.txt 2> {run_dir}/stderr.txt",
                  cwd=_crate(root), env=env)
    except ToolchainError:
        raise  # run_dir kept with logs
    stdout = (run_dir / "stdout.txt").read_text()
    if not keep_run:
        shutil.rmtree(run_dir)
    return stdout, run_dir


_BENCH_LINE = re.compile(r"^\[(\d+)\] time: ([0-9.]+)us, counter: (\d+), wall: ([0-9.]+)us")


def parse_benchmark_latencies(stdout: str) -> np.ndarray:
    """Extract per-entry hardware latencies (microseconds) from
    benchmark_decoding output, indexed by the tool's own [n] counter so
    line reordering or loss cannot shift the shot mapping; unseen indices
    are NaN for downstream detection. Raises if no lines at all were found.
    [used: by run_mb_characterization right after run_benchmark_decoding]"""
    entries = {}
    for line in stdout.splitlines():
        m = _BENCH_LINE.match(line)
        if m:
            entries[int(m.group(1))] = float(m.group(2))
    if not entries:
        raise ToolchainError("no per-shot latency lines found in benchmark output")
    n = max(entries) + 1
    out = np.full(n, np.nan)
    for i, v in entries.items():
        out[i] = v
    return out


def ensure_worker(idx: int, *, quiet: bool = False) -> pathlib.Path:
    """Create worker checkout w<idx> if missing — local `git clone` of the
    main /data checkout (hardlinked objects, cheap) pinned to its exact
    commit (the main checkout sits on a detached pin, so the clone's
    default branch cannot be trusted) — and make it build-ready (cargo +
    sbt, once per worker; container-wide ccache keeps rebuilt C++ cheap).
    [used: by parallel characterization before sharding; idempotent]"""
    root = WORKERS_DIR / f"w{idx}"
    if not (root / ".git").exists():
        if not quiet:
            print(f"mb_toolchain: creating worker checkout {root.name}...")
        pin = subprocess.run(["git", "-C", str(MB_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        WORKERS_DIR.mkdir(exist_ok=True)
        subprocess.run(["git", "clone", "-q", str(MB_ROOT), str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "checkout", "-q", pin], check=True)
    # clones carry committed HEAD, not the main checkout's patched tree —
    # apply MagiC-Firm patches (idempotent: skips when already applied).
    # Container builds leave root-owned artifacts (e.g. the stale jar the
    # script must delete); on failure, reclaim ownership and retry once.
    apply_cmd = [str(REPO / "micro-blossom-patches" / "apply_patches.sh"), str(root)]
    r = subprocess.run(apply_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        uid, gid = os.getuid(), os.getgid()
        _exec(f"chown -R {uid}:{gid} {root}", cwd=MB_MOUNT)
        r = subprocess.run(apply_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise ToolchainError(
                f"patch application failed in {root} even after chown:\n"
                f"{r.stdout[-800:]}\n{r.stderr[-800:]}")
    ensure_ready(root=root, quiet=quiet)
    return root
