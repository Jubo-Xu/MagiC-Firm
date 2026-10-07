# circuit_folder.py — the circuit-folder contract as one loader.
#
# Layout (experiments/data/circuits/<name>/, bootstrapped by
# get_desaturated_dem_with_obs_detector.py; partial subfolders by
# gen_partial_mask_dem.py):
#   <stem>.stim                  original noisy circuit
#   original.dem
#   desaturated_with_obs.dem     complete gap DEM (matchable, clipped, +obs det)
#   gap_circuit.stim             circuit + record-less padding DETECTORs (trace key)
#   postselected_detectors.json  self-describing: circuit_file, n_circuit_dets,
#                                n_gap_dets, obs_detector_index, postselected_detectors
#   partial_<tag>/               config.json, mask_bool.npy, contracted.dem,
#                                kept.npy, old_to_new.json, stats.json
#
# The estimator script, run_mb_characterization and the compiler must agree
# on where every artifact lives and on the decoder key
# "<folder>__complete" / "<folder>__<partial tag>"; this module is that
# agreement for the estimator side (the characterization script keeps its
# own resolve_config with identical conventions).

import dataclasses
import json
import pathlib
from typing import Optional

import numpy as np
import sinter
import stim

from partial_desaturation_sampler import PartialDesaturationSampler
from naming import freq_tag


@dataclasses.dataclass
class CircuitFolder:
    """One circuit folder (+ optional partial subfolder), loaded and verified.
    [used: run_runtime_estimation]"""
    folder: pathlib.Path
    stem: str                        # circuit filename without .stim
    circuit: stim.Circuit            # original noisy circuit
    gap_circuit: stim.Circuit        # + record-less padding detectors (trace key)
    postselected: set[int]           # original detector ids
    partial: Optional[str]           # partial tag or None (complete only)
    mask_bool: Optional[np.ndarray]  # (num_detectors,) when partial
    weight_cutoff: Optional[float]
    connectivity_fix_type: Optional[str]

    @property
    def decoder_key(self) -> str:
        """== run_mb_characterization's decoder key for this variant."""
        return f"{self.folder.name}__{self.partial or 'complete'}"

    def _key(self, complete: bool) -> str:
        return f"{self.folder.name}__complete" if complete else self.decoder_key

    def trace_dir(self, out_root) -> pathlib.Path:
        return pathlib.Path(out_root) / "traces" / self.folder.name

    def mb_result_json(self, result_dir, *, seed: int, shots: int, freq_hz: float,
                       complete: bool = False) -> pathlib.Path:
        """Mode A input: experiments/result/mb_characterization/<key>__s<seed>_n<shots>_<f>MHz_realsched.json"""
        return (pathlib.Path(result_dir)
                / f"{self._key(complete)}__s{seed}_n{shots}_{freq_tag(freq_hz)}_realsched.json")

    def mb_latency_cache(self, out_root, *, seed: int, freq_hz: float,
                         complete: bool = False) -> pathlib.Path:
        """Mode B input: out/mb/<key>/latency_cache_s<seed>_<f>MHz_realsched.npz"""
        return (pathlib.Path(out_root) / "mb" / self._key(complete)
                / f"latency_cache_s{seed}_{freq_tag(freq_hz)}_realsched.npz")


def load_circuit_folder(folder, partial: Optional[str] = None) -> CircuitFolder:
    """Load a circuit folder (+ optional partial subfolder). Verifies the
    self-describing jsons against the files they describe.
    [used: run_runtime_estimation, once per run]"""
    folder = pathlib.Path(folder).resolve()
    ps_doc = json.loads((folder / "postselected_detectors.json").read_text())
    circuit = stim.Circuit.from_file(folder / ps_doc["circuit_file"])
    gap_circuit = stim.Circuit.from_file(folder / "gap_circuit.stim")
    if int(ps_doc["n_circuit_dets"]) != circuit.num_detectors:
        raise ValueError(f"{folder.name}: postselected_detectors.json n_circuit_dets "
                         f"{ps_doc['n_circuit_dets']} != circuit {circuit.num_detectors}")
    if int(ps_doc["n_gap_dets"]) != gap_circuit.num_detectors:
        raise ValueError(f"{folder.name}: postselected_detectors.json n_gap_dets "
                         f"{ps_doc['n_gap_dets']} != gap_circuit {gap_circuit.num_detectors}")
    name = ps_doc["circuit_file"]
    stem = name[: name.index(".stim")] if ".stim" in name else name

    mask = cutoff = fix = None
    if partial is not None:
        pdir = folder / partial
        if not pdir.is_dir():
            raise FileNotFoundError(f"partial subfolder not found: {pdir}")
        cfg = json.loads((pdir / "config.json").read_text())
        mask = np.load(pdir / "mask_bool.npy")
        if mask.shape != (circuit.num_detectors,):
            raise ValueError(f"{pdir.name}: mask_bool shape {mask.shape} != ({circuit.num_detectors},)")
        cutoff = float(cfg["weight_cutoff"])
        fix = str(cfg["connectivity_fix_type"])

    return CircuitFolder(
        folder=folder, stem=stem, circuit=circuit, gap_circuit=gap_circuit,
        postselected={int(d) for d in ps_doc["postselected_detectors"]},
        partial=partial, mask_bool=mask, weight_cutoff=cutoff,
        connectivity_fix_type=fix,
    )


def compile_samplers(cf: CircuitFolder):
    """(complete_compiled, partial_compiled | None) via PartialDesaturationSampler
    — one code path for both variants (Step A is deterministic). Asserts the
    compiled DEMs equal the folder's desaturated_with_obs.dem / contracted.dem,
    i.e. the estimator decodes on exactly the graphs MB was characterized on.
    [used: run_runtime_estimation, once per run (partial contraction ~minutes)]"""
    task = sinter.Task(circuit=cf.circuit,
                       detector_error_model=cf.circuit.detector_error_model())
    sampler = PartialDesaturationSampler()
    complete = sampler.compiled_sampler_for_task(task)
    ref = stim.DetectorErrorModel.from_file(cf.folder / "desaturated_with_obs.dem")
    if not complete.gap_dem_complete.approx_equals(ref, atol=1e-9):
        raise RuntimeError(f"{cf.folder.name}: compiled complete DEM != desaturated_with_obs.dem")

    partial = None
    if cf.partial is not None:
        partial = sampler.compiled_sampler_for_task(
            task, partial_mask_bool=cf.mask_bool, weight_cutoff=cf.weight_cutoff,
            connectivity_fix_type=cf.connectivity_fix_type)
        ref = stim.DetectorErrorModel.from_file(cf.folder / cf.partial / "contracted.dem")
        if not partial.gap_dem.approx_equals(ref, atol=1e-9):
            raise RuntimeError(f"{cf.folder.name}/{cf.partial}: compiled partial DEM != contracted.dem")
    return complete, partial
