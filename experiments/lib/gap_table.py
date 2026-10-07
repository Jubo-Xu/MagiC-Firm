# gap_table.py — per-shot gap tables: sampling, gating-rule statistics and the
# input fingerprints that guard table and result reuse.
#
# A table stores, per post-selected shot, the complete gap gc, the partial gap
# gp (-1 without a partial decoder) and err (complete prediction != observable).
# Any gating rule's accept rate and LER follow from it without re-decoding.

import hashlib
import pathlib

import numpy as np

import paths
from circuit_folder import load_circuit_folder, compile_samplers

GAP_TABLE_DIR = paths.OUT / "gap_tables"       # <circuit>/<partial tag>_s<seed>.npz (+ .meta.json)


def rule_stats(gc, gp, err, *, tc, tl=None, th=None):
    """(accepted, errors, ler) of a gating rule over the table (numpy arrays).
    th None: complete decoder only, accept gc >= tc. Else two-stage: accept
    gp > th, or gp >= tl and gc >= tc."""
    if th is None:
        acc = gc >= tc
    else:
        fast = gp > th
        acc = fast | ((gp >= (tl or 0)) & ~fast & (gc >= tc))
    n = int(acc.sum()); e = int((acc & err).sum())
    return n, e, (e / n if n else float("nan"))


def mask_fingerprint(folder, partial):
    """sha of the decoder inputs a table depends on: the complete DEM and, for
    a partial tag, its mask and contracted DEM. Stored in the meta, checked on reuse."""
    h = hashlib.sha256((folder / "desaturated_with_obs.dem").read_bytes())
    if partial:
        h.update((folder / partial / "mask_bool.npy").read_bytes()); h.update((folder / partial / "contracted.dem").read_bytes())
    return h.hexdigest()[:16]


def result_fingerprint(folder, partial, *, mb_shots, mb_seed, mb_freq_hz, control_latency=None):
    """Every input an estimator result depends on: the mask fingerprint plus the
    sha of the MB characterization file(s) and of the control-latency profile.
    Recorded in the result JSON (config.inputs) and compared before reuse."""
    folder = pathlib.Path(folder)
    def sha(path):
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16] if path and pathlib.Path(path).exists() else None
    mbdir = paths.RESULT / "mb_characterization"
    ftag = f"{mb_freq_hz / 1e6:g}MHz"
    fp = {"mask_sha": mask_fingerprint(folder, partial),
          "mb_complete_sha": sha(mbdir / f"{folder.name}__complete__s{mb_seed}_n{mb_shots}_{ftag}_realsched.json"),
          "mb_partial_sha": sha(mbdir / f"{folder.name}__{partial}__s{mb_seed}_n{mb_shots}_{ftag}_realsched.json") if partial else None,
          "control_latency_sha": sha(control_latency)}
    return fp


def decode_chunk(folder, partial, seed, chunk, want, batch):
    """Sample until `want` post-selected shots are collected for this chunk
    (sampler seed = seed * 1_000_003 + chunk, so chunks are reproducible).
    Returns (gc, gp, err, attempts, passed); gaps are integer dB, rounded up."""
    cf = load_circuit_folder(folder, partial)
    complete, part = compile_samplers(cf)
    sampler = cf.gap_circuit.compile_detector_sampler(seed=seed * 1_000_003 + chunk)
    gcs, gps, errs = [], [], []
    attempts = passed = 0          # post-selection acceptance is counted BEFORE truncation
    got = 0
    while got < want:
        dets, obs = sampler.sample(batch, separate_observables=True, bit_packed=True)
        keep = ~np.any(dets & complete._discard_mask, axis=1)
        attempts += batch; passed += int(keep.sum())
        dets, obs = dets[keep], obs[keep][:, 0]
        if not len(dets):
            continue
        take = min(len(dets), want - got)
        dets, obs = dets[:take], obs[:take]
        # partial first: it repacks (non-mutating); the complete decode mutates in place
        if part is not None:
            _, gp = part._decode_batch_overwrite_last_byte(dets)
            gps.append(np.ceil(gp).astype(np.int16))
        pred, gc = complete._decode_batch_overwrite_last_byte(dets)
        gcs.append(np.ceil(gc).astype(np.int16))
        errs.append((pred.astype(bool) != obs.astype(bool)))
        got += take
    gc = np.concatenate(gcs); err = np.concatenate(errs)
    gp = np.concatenate(gps) if gps else np.full(len(gc), -1, np.int16)
    return gc, gp, err, attempts, passed
