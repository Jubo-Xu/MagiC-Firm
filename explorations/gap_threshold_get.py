import pathlib
import sys
from typing import Any, Optional

import numpy as np
import sinter
import stim
from matplotlib import pyplot as plt

src_path = pathlib.Path(__file__).parent.parent / "magic_state_cultivation" / "upstream" / "src"
assert src_path.exists()
sys.path.append(str(src_path))

import cultiv
import gen


class GapThreshotGenerate:
    def __init__(
        self,
        *,
        basis: str,
        gateset: str,
        circuit_type: str,
        noise_model: str,
        noise_strength: float,
        distance: int,
        d2: int = 15,
        injection_protocol: str = "unitary",
        r_in_escape: Optional[int] = None,
        sampler: Optional[Any] = None,
        target_attempts_per_kept: Optional[float] = None,
        threshold_shots: int = 1024 * 100
    ):
        self.distance = int(distance)
        self.sampler = sampler if sampler is not None else cultiv.DesaturationSampler()
        self.threshold_shots = int(threshold_shots)
        self.target_attempts_per_kept = target_attempts_per_kept

        ideal_circuit = self._build_ideal_circuit(
            basis=basis,
            circuit_type=circuit_type,
            injection_protocol=injection_protocol,
            r_in_escape=r_in_escape if r_in_escape is not None else distance,
            d1=distance,
            d2=d2,
        )
        self.circuit = self._add_noise(
            circuit=ideal_circuit,
            gateset=gateset,
            noise_model=noise_model,
            noise_strength=noise_strength,
        )
        self.task = sinter.Task(circuit=self.circuit, detector_error_model=self.circuit.detector_error_model())
        self.compiled = self.sampler.compiled_sampler_for_task(self.task)
        if not hasattr(self.compiled, "decode_det_set"):
            raise NotImplementedError("Selected sampler must provide decode_det_set for cycle-accurate emulation.")

        self.gap_threshold = self._auto_gap_threshold()


    
    def _auto_gap_threshold(self) -> int:
        stat = self.compiled.sample(self.threshold_shots)
        rows = self._threshold_rows(stat)
        if not rows:
            return 0
        target_attempts = self.target_attempts_per_kept
        if target_attempts is None:
            target_attempts = 4.0 if self.distance == 3 else 100.0
        valid = [r for r in rows if r["keep_rate"] > 0]
        chosen = min(valid, key=lambda r: abs(r["attempts_per_kept"] - target_attempts))
        return int(chosen["threshold"])
    
    @staticmethod
    def _threshold_rows(stat: sinter.AnonTaskStats) -> list[dict[str, float | int]]:
        bins: dict[int, dict[str, int]] = {}
        for key, count in stat.custom_counts.items():
            if not (key.startswith("C") or key.startswith("E")):
                continue
            gap = int(key[1:])
            bins.setdefault(gap, {"C": 0, "E": 0})
            bins[gap]["C" if key.startswith("C") else "E"] += int(count)
        if not bins:
            return []
        rows: list[dict[str, float | int]] = []
        for threshold in sorted(bins):
            below = 0
            for gap, ce in bins.items():
                if gap < threshold:
                    below += ce["C"] + ce["E"]
            keep = (stat.shots - (stat.discards + below)) / stat.shots
            attempts = float("inf") if keep <= 0 else 1 / keep
            rows.append({"threshold": threshold, "keep_rate": keep, "attempts_per_kept": attempts})
        return rows
    
    @staticmethod
    def _build_ideal_circuit(
        *,
        basis: str,
        circuit_type: str,
        injection_protocol: str,
        r_in_escape: int,
        d1: int,
        d2: int,
    ) -> stim.Circuit:
        if circuit_type == "end2end-inplace-distillation":
            return cultiv.make_end2end_cultivation_circuit(
                dcolor=d1,
                dsurface=d2,
                basis=basis,
                r_growing=r_in_escape,
                r_end=0,
                inject_style=injection_protocol,
            )
        if circuit_type == "inject+cultivate":
            return cultiv.make_inject_and_cultivate_circuit(
                dcolor=d1,
                basis=basis,
                inject_style=injection_protocol,
            )
        if circuit_type == "escape-to-big-matchable-code":
            return cultiv.make_escape_to_big_matchable_code_circuit(
                dcolor=d1,
                dsurface=d2,
                basis=basis,
                r_growing=r_in_escape,
                r_end=0,
            )
        raise NotImplementedError(f"Unsupported circuit_type={circuit_type}")
    
    @staticmethod
    def _add_noise(
        *,
        circuit: stim.Circuit,
        gateset: str,
        noise_model: str,
        noise_strength: float,
    ) -> stim.Circuit:
        if gateset == "cz":
            if noise_model != "circuit-level-SI1000":
                raise ValueError("For gateset='cz', noise_model must be 'circuit-level-SI1000'.")
            # Match project behavior for CZ+SI1000: transpile interactions to Z-basis first.
            circuit = gen.transpile_to_z_basis_interaction_circuit(circuit)
            model = gen.NoiseModel.si1000(noise_strength)
        else:
            if noise_model == "circuit-level-SI1000":
                model = gen.NoiseModel.si1000(noise_strength)
            elif noise_model == "uniform-depolarizing":
                model = gen.NoiseModel.uniform_depolarizing(noise_strength)
            else:
                raise ValueError(f"Unsupported noise_model={noise_model}")
        return model.noisy_circuit_skipping_mpp_boundaries(circuit)