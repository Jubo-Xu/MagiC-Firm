# attempt_stream.py — deterministic, extendable streams of circuit attempts.
#
# Attempt i of a trace is a pure function of (circuit, master seed, i), so
# tables measured separately over the same attempt indices join exactly.
#
# Files in trace_dir:
#   trace_s<S>.meta.json      identity + layout (TraceMeta)
#   trace_s<S>_chunk<k>.npz   attempts [k*C, (k+1)*C): packed dets + obs
#
# The stim version and circuit hash are stored and checked on open.

import dataclasses
import hashlib
import json
import os
import pathlib

import numpy as np
import stim

FORMAT_VERSION = 1
# Part of trace identity: attempt i is column i % B of FlipSimulator batch
# i // B. Changing it changes every shot.
BATCH_SIZE = 1_000
# Attempts per .npz chunk. Storage only; must be a multiple of BATCH_SIZE.
CHUNK_ATTEMPTS = 1_000_000


def _circuit_sha(circuit: stim.Circuit) -> str:
    return hashlib.sha256(str(circuit).encode()).hexdigest()[:16]


def batch_seed(master_seed: int, batch_index: int) -> int:
    """Seed of FlipSimulator batch `batch_index`. sha256-based, so stable
    across machines and Python versions."""
    digest = hashlib.sha256(f"magicfirm-attempts:{master_seed}:{batch_index}".encode()).digest()
    return int.from_bytes(digest[:8], "little") >> 1  # non-negative int64


@dataclasses.dataclass
class TraceMeta:
    """Identity and layout of a trace. Written at creation, verified on open."""
    format_version: int
    master_seed: int
    batch_size: int
    chunk_attempts: int
    stim_version: str
    circuit_sha: str
    num_detectors: int
    num_observables: int
    layer_det_ranges: list        # per stage-layer [start, end) detector ids
    layer_meas_ranges: list       # per stage-layer [start, end) measurement ids
    num_attempts: int             # grows on extension; only field that changes


def generate_batch(circuit: stim.Circuit, master_seed: int, batch_index: int,
                   batch_size: int = BATCH_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one batch. Returns (dets_packed, obs) with shapes
    (batch_size, ceil(D/8)) uint8, little bit order, and (batch_size,) uint8.
    Always runs the full circuit; postselection is left to consumers."""
    sim = stim.FlipSimulator(
        batch_size=batch_size,
        num_qubits=circuit.num_qubits,
        seed=batch_seed(master_seed, batch_index),
    )
    sim.do(circuit)
    dets = sim.get_detector_flips()               # (num_detectors, batch_size) bool
    obs = sim.get_observable_flips()              # (num_observables, batch_size) bool
    dets_packed = np.packbits(dets.T, axis=1, bitorder="little")
    obs_row = obs[0].astype(np.uint8) if obs.shape[0] else np.zeros(batch_size, np.uint8)
    return dets_packed, obs_row


class AttemptTrace:
    """Store for one trace: (circuit, master_seed) -> chunk files.
    [used: run_mb_characterization, RuntimeEstimator.attach_trace]"""

    def __init__(self, trace_dir: pathlib.Path, circuit: stim.Circuit, meta: TraceMeta):
        self.trace_dir = trace_dir
        self.circuit = circuit
        self.meta = meta
        self._chunk_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # ---------------- creation / opening ----------------

    @classmethod
    def create_or_open(cls, trace_dir, circuit, master_seed, *,
                       layer_det_ranges=None, layer_meas_ranges=None,
                       chunk_attempts: int = CHUNK_ATTEMPTS) -> "AttemptTrace":
        """Open an existing trace or create an empty one. Raises if the
        circuit, stim version or batch size differ from the stored trace."""
        trace_dir = pathlib.Path(trace_dir)
        meta_path = trace_dir / f"trace_s{master_seed}.meta.json"
        if meta_path.exists():
            meta = TraceMeta(**json.loads(meta_path.read_text()))
            if meta.circuit_sha != _circuit_sha(circuit):
                raise ValueError("trace was generated from a different circuit")
            if meta.stim_version != stim.__version__:
                raise ValueError(
                    f"trace generated with stim {meta.stim_version}, running "
                    f"{stim.__version__}: shots would differ; regenerate or pin stim")
            if meta.batch_size != BATCH_SIZE:
                raise ValueError(
                    f"trace generated with batch_size {meta.batch_size}, code "
                    f"uses {BATCH_SIZE}: shots would differ")
            trace = cls(trace_dir, circuit, meta)
            trace._update_num_attempts()
            return trace
        trace_dir.mkdir(parents=True, exist_ok=True)
        assert chunk_attempts % BATCH_SIZE == 0, "BATCH_SIZE must divide chunk_attempts"
        meta = TraceMeta(
            format_version=FORMAT_VERSION, master_seed=master_seed,
            batch_size=BATCH_SIZE, chunk_attempts=chunk_attempts,
            stim_version=stim.__version__, circuit_sha=_circuit_sha(circuit),
            num_detectors=circuit.num_detectors,
            num_observables=circuit.num_observables,
            layer_det_ranges=[list(r) for r in (layer_det_ranges or [])],
            layer_meas_ranges=[list(r) for r in (layer_meas_ranges or [])],
            num_attempts=0)
        meta_path.write_text(json.dumps(dataclasses.asdict(meta), indent=2))
        return cls(trace_dir, circuit, meta)

    # ---------------- generation ----------------

    def ensure(self, n_attempts: int, *, quiet: bool = False) -> None:
        """Grow the trace to at least n_attempts, in whole chunks. Append-only."""
        C = self.meta.chunk_attempts
        needed_chunks = -(-n_attempts // C)
        for k in range(self._num_chunks(), needed_chunks):
            if not quiet:
                print(f"attempt_stream: generating chunk {k} ({C} attempts)...")
            dets_parts, obs_parts = [], []
            for b in range(k * C // BATCH_SIZE, (k + 1) * C // BATCH_SIZE):
                d, o = generate_batch(self.circuit, self.meta.master_seed, b)
                dets_parts.append(d)
                obs_parts.append(o)
            # Write then rename: a kill leaves no partial chunk, and racing
            # writers produce identical content. The tmp name must end in
            # .npz, otherwise np.savez appends the suffix.
            tmp = self._chunk_path(k).with_name(
                f"{self._chunk_path(k).stem}.tmp{os.getpid()}.npz")
            np.savez_compressed(tmp,
                                dets_packed=np.concatenate(dets_parts),
                                obs=np.concatenate(obs_parts))
            os.replace(tmp, self._chunk_path(k))
        self._update_num_attempts()

    # ---------------- access ----------------

    def get_range(self, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        """Attempts [start, end) as (dets_packed, obs). Raises IndexError past
        the end of the trace."""
        if not (0 <= start <= end <= self.meta.num_attempts):
            raise IndexError(
                f"range [{start}, {end}) outside trace of {self.meta.num_attempts} "
                f"attempts — call ensure({end}) to extend")
        C = self.meta.chunk_attempts
        dets_parts, obs_parts = [], []
        for k in range(start // C, -(-end // C) if end else 0):
            dets, obs = self._load_chunk(k)
            lo = max(start - k * C, 0)
            hi = min(end - k * C, C)
            dets_parts.append(dets[lo:hi])
            obs_parts.append(obs[lo:hi])
        return np.concatenate(dets_parts), np.concatenate(obs_parts)

    @staticmethod
    def defect_hash(dets_packed_row: np.ndarray) -> str:
        """16-hex hash of one attempt's packed detector bits. Producers store
        it with their measurements; consumers verify it on join."""
        return hashlib.sha256(dets_packed_row.tobytes()).hexdigest()[:16]

    # ---------------- internals ----------------

    def _chunk_path(self, k: int) -> pathlib.Path:
        return self.trace_dir / f"trace_s{self.meta.master_seed}_chunk{k}.npz"

    def _num_chunks(self) -> int:
        """Number of contiguous chunk files on disk."""
        k = 0
        while self._chunk_path(k).exists():
            k += 1
        return k

    def _load_chunk(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        if k not in self._chunk_cache:
            if len(self._chunk_cache) >= 4:
                self._chunk_cache.pop(next(iter(self._chunk_cache)))
            with np.load(self._chunk_path(k)) as z:
                self._chunk_cache[k] = (z["dets_packed"], z["obs"])
        return self._chunk_cache[k]

    def _update_num_attempts(self) -> None:
        self.meta.num_attempts = self._num_chunks() * self.meta.chunk_attempts
        meta_path = self.trace_dir / f"trace_s{self.meta.master_seed}.meta.json"
        tmp = meta_path.with_name(f"{meta_path.name}.tmp{os.getpid()}")
        tmp.write_text(json.dumps(dataclasses.asdict(self.meta), indent=2))
        os.replace(tmp, meta_path)


class ShotCursor:
    """Sequential reader: next() -> (dets_packed_row, obs, index). `index` is
    the join key. start=k, step=K gives worker k of K a disjoint partition.
    [used: RuntimeEstimator trace mode, run_eval_partial_mask]"""

    def __init__(self, trace: AttemptTrace, start: int = 0, step: int = 1):
        self.trace = trace
        self.next_index = start
        self.step = int(step)

    def next(self) -> tuple[np.ndarray, int, int]:
        i = self.next_index
        if i >= self.trace.meta.num_attempts:
            raise RuntimeError(
                f"attempt trace exhausted at index {i}; call "
                f"AttemptTrace.ensure({i + 1}) to extend (append-only) and retry")
        dets, obs = self.trace.get_range(i, i + 1)
        self.next_index = i + self.step
        return dets[0], int(obs[0]), i
