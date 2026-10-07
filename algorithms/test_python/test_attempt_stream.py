# Tests for algorithms/attempt_stream.py: determinism, chunked storage,
# append-only extension, identity verification, cursor access.

import pathlib
import sys

import numpy as np
import pytest
import stim

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import attempt_stream as ats


def small_circuit(rounds: int = 3) -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=3, rounds=rounds,
        after_clifford_depolarization=0.01)


def test_generate_batch_deterministic():
    c = small_circuit()
    d1, o1 = ats.generate_batch(c, 42, 0, batch_size=100)
    d2, o2 = ats.generate_batch(c, 42, 0, batch_size=100)
    d3, _ = ats.generate_batch(c, 42, 1, batch_size=100)
    assert np.array_equal(d1, d2) and np.array_equal(o1, o2)
    assert not np.array_equal(d1, d3)
    assert d1.any()
    assert d1.shape == (100, -(-c.num_detectors // 8))


def test_batch_seed_stable():
    # sha256-based: must be identical across machines/runs (never builtin hash)
    assert ats.batch_seed(42, 0) == ats.batch_seed(42, 0)
    assert ats.batch_seed(42, 0) != ats.batch_seed(42, 1)
    assert ats.batch_seed(42, 0) != ats.batch_seed(43, 0)
    assert 0 <= ats.batch_seed(1, 1) < 2**63


def test_trace_roundtrip_extension_and_identity(tmp_path):
    c = small_circuit()
    t = ats.AttemptTrace.create_or_open(tmp_path, c, 42, chunk_attempts=2000)
    t.ensure(3000, quiet=True)
    assert t.meta.num_attempts == 4000          # rounds up to whole chunks

    d1, o1 = t.get_range(1500, 2500)            # crosses a chunk boundary
    assert d1.shape[0] == 1000 and o1.shape[0] == 1000

    t2 = ats.AttemptTrace.create_or_open(tmp_path, c, 42)
    d2, o2 = t2.get_range(1500, 2500)
    assert np.array_equal(d1, d2) and np.array_equal(o1, o2)

    t2.ensure(5000, quiet=True)                 # extension preserves old data
    assert t2.meta.num_attempts == 6000
    d3, _ = t2.get_range(1500, 2500)
    assert np.array_equal(d1, d3)

    with pytest.raises(IndexError):
        t2.get_range(5999, 6001)

    with pytest.raises(ValueError):             # different circuit rejected
        ats.AttemptTrace.create_or_open(tmp_path, small_circuit(rounds=5), 42)


def test_cursor(tmp_path):
    c = small_circuit()
    t = ats.AttemptTrace.create_or_open(tmp_path, c, 7, chunk_attempts=1000)
    t.ensure(1000, quiet=True)
    cur = ats.ShotCursor(t, start=998)
    row0, obs0, i0 = cur.next()
    row1, obs1, i1 = cur.next()
    assert (i0, i1) == (998, 999)
    ref, refo = t.get_range(998, 1000)
    assert np.array_equal(row0, ref[0]) and np.array_equal(row1, ref[1])
    assert obs0 == int(refo[0])
    with pytest.raises(RuntimeError, match="ensure"):
        cur.next()


def test_defect_hash():
    a = np.array([1, 2, 3], dtype=np.uint8)
    b = np.array([1, 2, 4], dtype=np.uint8)
    assert ats.AttemptTrace.defect_hash(a) == ats.AttemptTrace.defect_hash(a.copy())
    assert ats.AttemptTrace.defect_hash(a) != ats.AttemptTrace.defect_hash(b)
    assert len(ats.AttemptTrace.defect_hash(a)) == 16
