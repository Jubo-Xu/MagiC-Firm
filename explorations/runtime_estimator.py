import pathlib
import sys
import tempfile
import time
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


class RuntimeEstimator:
    def __init__(
        self,
        *,
        epoch: int,
        basis: str,
        gateset: str,
        circuit_type: str,
        noise_model: str,
        noise_strength: float,
        distance: int,
        d2: int = 15,
        injection_protocol: str = "unitary",
        r_in_escape: Optional[int] = None,
        feedback_time_ns: float = 100.0,
        decoder_latency_ns: float = 10_000.0,
        use_measured_decode_time: bool = True,
        sampler: Optional[Any] = None,
        target_attempts_per_kept: Optional[float] = None,
        threshold_shots: int = 1024 * 100,
        wait_rounds_for_plot: int = 10,
        gate_times_ns: Optional[dict[str, float]] = None,
    ):
        if epoch <= 0:
            raise ValueError("epoch must be positive.")

        self.epoch = int(epoch)
        self.distance = int(distance)
        self.feedback_time_ns = float(feedback_time_ns)
        self.decoder_latency_ns = float(decoder_latency_ns)
        self.use_measured_decode_time = bool(use_measured_decode_time)
        self.wait_rounds_for_plot = int(wait_rounds_for_plot)
        self.gate_times_ns = gate_times_ns if gate_times_ns is not None else self._default_gate_times_ns()
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

        self.layers = self._circuit_to_layers(self.circuit)
        self.layer_gate_times_ns = np.array([self._layer_gate_time_ns(layer) for layer in self.layers], dtype=np.float64)
        self.postselected_detectors = self._find_postselected_detectors(self.circuit)
        self.detector_to_stage = self._build_detector_stage_index(self.layers)

        self.gap_threshold = self._auto_gap_threshold()

        self.stage_names_single_wait = self._build_stage_names(wait_rounds=2)
        self.wait_stage_idx = self.stage_names_single_wait.index("[wait for gap]")
        self.ready_stage_idx = self.stage_names_single_wait.index("ready")
        self.num_stages = len(self.stage_names_single_wait)

        self.records_first_reach = {
            "total": np.full((self.epoch, self.num_stages), np.nan, dtype=np.float64),
            "gate": np.full((self.epoch, self.num_stages), np.nan, dtype=np.float64),
            "feedback": np.full((self.epoch, self.num_stages), np.nan, dtype=np.float64),
            "decode": np.full((self.epoch, self.num_stages), np.nan, dtype=np.float64),
        }
        self.records_until_success = {
            "total": np.zeros((self.epoch, self.num_stages), dtype=np.float64),
            "gate": np.zeros((self.epoch, self.num_stages), dtype=np.float64),
            "feedback": np.zeros((self.epoch, self.num_stages), dtype=np.float64),
            "decode": np.zeros((self.epoch, self.num_stages), dtype=np.float64),
        }
        self.attempt_count = 0
        self.stage_pass_counts = np.zeros(self.num_stages, dtype=np.int64)
        self.decode_call_times_ns: list[float] = []
        self.accepted_actual_obs: list[int] = []
        self.accepted_predict_obs: list[int] = []
        self.logical_error_rate = 0
        self._simulated = False

    def estimate_runtime(
        self,
        gap_check: bool = True,
        *,
        partial_mask: bool = False,
        mask_bool: Optional[np.ndarray] = None,
        gap_threshold: Optional[float] = None,
        decoder_time_measure_mode = "serial",
    ) -> None:
        for k in self.records_first_reach:
            self.records_first_reach[k][:] = np.nan
            self.records_until_success[k][:] = 0.0
        self.attempt_count = 0
        self.stage_pass_counts[:] = 0
        self.decode_call_times_ns = []
        self.accepted_actual_obs = []
        self.accepted_predict_obs = []
        # Override the auto-computed gap threshold when caller passes one explicitly.
        effective_gap_threshold = float(gap_threshold) if gap_threshold is not None else self.gap_threshold

        decode_mask_bool: Optional[np.ndarray] = None
        if partial_mask:
            if mask_bool is None:
                raise ValueError("mask_bool is required when partial_mask=True.")
            if mask_bool.shape != (self.circuit.num_detectors,):
                raise ValueError(
                    f"mask_bool must have shape ({self.circuit.num_detectors},); got {mask_bool.shape}."
                )
            decode_mask_bool = np.asarray(mask_bool, dtype=np.bool_)

        for e in range(self.epoch):
            acc = {"total": 0.0, "gate": 0.0, "feedback": 0.0, "decode": 0.0}
            first_seen = np.zeros(self.num_stages, dtype=np.bool_)
            success = False

            while not success:
                det_set, fail_stage, actual_obs = self._run_single_attempt_until_postselect()
                self.attempt_count += 1

                if fail_stage is not None:
                    for s in range(fail_stage):
                        self.stage_pass_counts[s] += 1
                    for s in range(fail_stage + 1):
                        self._add_gate_stage(acc, s)
                        self._record_stage(e, s, acc, first_seen)
                    self._add_feedback(acc)
                    continue

                for s in range(len(self.layers)):
                    self.stage_pass_counts[s] += 1
                for s in range(len(self.layers)):
                    self._add_gate_stage(acc, s)
                    self._record_stage(e, s, acc, first_seen)

                det_set_for_decode = det_set
                if decode_mask_bool is not None:
                    det_set_for_decode = {d for d in det_set if d < len(decode_mask_bool) and decode_mask_bool[d]}
                pred_obs, gap, decode_ns = self._decode_with_timing(det_set_for_decode, measure_mode=decoder_time_measure_mode)
                self._add_decode(acc, decode_ns)
                self._record_stage(e, self.wait_stage_idx, acc, first_seen)
                self.stage_pass_counts[self.wait_stage_idx] += 1
                if gap_check:
                    if gap >= effective_gap_threshold:
                        self._record_stage(e, self.ready_stage_idx, acc, first_seen)
                        self.stage_pass_counts[self.ready_stage_idx] += 1
                        if actual_obs is None:
                            raise ValueError("Expected actual observable for accepted attempt, but got None.")
                        self.accepted_actual_obs.append(int(actual_obs))
                        if partial_mask:
                            pred_obs, _ = self.compiled.decode_det_set(det_set)
                        self.accepted_predict_obs.append(int(pred_obs))
                        success = True
                    else:
                        self._add_feedback(acc)
                else:
                    self._record_stage(e, self.ready_stage_idx, acc, first_seen)
                    self.stage_pass_counts[self.ready_stage_idx] += 1
                    if actual_obs is None:
                        raise ValueError("Expected actual observable for accepted attempt, but got None.")
                    self.accepted_actual_obs.append(int(actual_obs))
                    self.accepted_predict_obs.append(int(pred_obs))
                    success = True
        errs = [a != p for a, p in zip(self.accepted_actual_obs, self.accepted_predict_obs)]
        assert len(errs) == self.epoch
        self.logical_error_rate = float(np.mean(errs))
        self._simulated = True

    def calculate_mean(self, record_type: str = "until_success") -> dict[str, np.ndarray]:
        records = self._get_records(record_type)
        fn = np.nanmean if record_type == "first_reach" else np.mean
        return {k: fn(v, axis=0) for k, v in records.items()}

    ## Distribution calculation and visualization
    def calculate_distribution(
        self,
        *,
        stage_name: str,
        time_type: str,
        record_type: str = "until_success",
    ) -> np.ndarray:
        records = self._get_records(record_type)
        if time_type not in records:
            raise ValueError(f"time_type must be one of {list(records.keys())}")
        if stage_name not in self.stage_names_single_wait:
            raise ValueError(f"Unknown stage_name={stage_name}")
        idx = self.stage_names_single_wait.index(stage_name)
        values = records[time_type][:, idx]
        return values[~np.isnan(values)]

    def report_average_decode_latency_ns(self) -> float:
        if not self.decode_call_times_ns:
            raise ValueError("No decode timing data. Run estimate_runtime() first.")
        avg = float(np.mean(self.decode_call_times_ns))
        print(f"Average decoding latency: {avg:.3f} ns")
        return avg

    def plot_distribution_svg(
        self,
        *,
        stage_name: str,
        time_type: str,
        path: pathlib.Path,
        record_type: str = "until_success",
        bins: int = 40,
    ) -> None:
        values = self.calculate_distribution(stage_name=stage_name, time_type=time_type, record_type=record_type)
        if len(values) == 0:
            raise ValueError("No values available for this stage/time_type/record_type selection.")
        mean = float(np.mean(values))
        var = float(np.var(values))
        std = float(np.sqrt(var))

        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5), dpi=120)
        # Density histogram for readable distribution shape.
        ax.hist(values, bins=bins, density=True, alpha=0.35, color="#4c78a8", edgecolor="white", linewidth=0.8)

        # Lightweight KDE-like smoothing from histogram density (no extra deps).
        hist_y, hist_edges = np.histogram(values, bins=bins, density=True)
        hist_x = 0.5 * (hist_edges[:-1] + hist_edges[1:])
        if len(hist_y) >= 3:
            kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
            kernel /= np.sum(kernel)
            smooth_y = np.convolve(hist_y, kernel, mode="same")
            ax.plot(hist_x, smooth_y, color="#1f2a44", linewidth=2.0, label="Smoothed density")

        # Mean and +-1sigma markers.
        ax.axvline(mean, color="#d62728", linewidth=2.0, linestyle="-", label="Mean")
        ax.axvline(mean - std, color="#d62728", linewidth=1.5, linestyle="--", alpha=0.9, label="Mean ± 1σ")
        ax.axvline(mean + std, color="#d62728", linewidth=1.5, linestyle="--", alpha=0.9)

        stats_text = f"n={len(values)}\nmean={mean:.3f} ns\nstd={std:.3f} ns\nvar={var:.3f} ns²"
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999", alpha=0.9),
        )

        ax.set_xlabel(f"{time_type} time (ns)")
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution at stage '{stage_name}' ({record_type})")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(path, format="svg")
        plt.close(fig)

    def plot_distribution_notebook(
        self,
        *,
        stage_name: str,
        time_type: str,
        record_type: str = "until_success",
        bins: int = 40,
    ):
        try:
            from IPython.display import SVG
        except ImportError as ex:
            raise ImportError("IPython is required for notebook display.") from ex

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            self.plot_distribution_svg(
                stage_name=stage_name,
                time_type=time_type,
                path=tmp_path,
                record_type=record_type,
                bins=bins,
            )
            return SVG(data=tmp_path.read_text())
        finally:
            tmp_path.unlink(missing_ok=True)

    ## Lifetime calculation and visualization
    def plot_life_time_notebook(self, record_type: str = "until_success"):
        try:
            from IPython.display import SVG
        except ImportError as ex:
            raise ImportError("IPython is required for notebook display.") from ex

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            self.plot_life_time_svg(path=tmp_path, record_type=record_type)
            return SVG(data=tmp_path.read_text())
        finally:
            tmp_path.unlink(missing_ok=True)

    def plot_life_time_svg(self, *, path: pathlib.Path, record_type: str = "until_success") -> None:
        if not self._simulated:
            raise ValueError("Run estimate_runtime() first.")

        # For this lifetime plot, use first-reach means for time traces.
        means = self.calculate_mean(record_type="first_reach")

        # Survival proportion from estimator pass-rate counts.
        surv_stage = self._survival_by_attempt()

        # Shift to stage-entry for intermediate stages, but keep ready at its own
        # accumulated value so plotted ready matches recorded ready mean.
        total_entry = self._shift_to_stage_entry_with_ready(means["total"])
        gate_entry = self._shift_to_stage_entry_with_ready(means["gate"])
        feedback_entry = self._shift_to_stage_entry_with_ready(means["feedback"])
        decode_entry = self._shift_to_stage_entry_with_ready(means["decode"])

        gate_plus_feedback_entry = gate_entry + feedback_entry
        total_from_components = gate_plus_feedback_entry + decode_entry
        total_entry = np.maximum(total_entry, total_from_components)

        expanded_names, expanded_surv, expanded_gf, expanded_g = self._expand_wait_rounds_for_plot(
            stage_names=self.stage_names_single_wait,
            total=surv_stage,
            gate_plus_feedback=gate_plus_feedback_entry,
            gate_only=gate_entry,
            wait_rounds=self.wait_rounds_for_plot,
        )
        _, expanded_total, _, _ = self._expand_wait_rounds_for_plot(
            stage_names=self.stage_names_single_wait,
            total=total_entry,
            gate_plus_feedback=total_entry,
            gate_only=total_entry,
            wait_rounds=self.wait_rounds_for_plot,
        )

        # Draw each stage value across its full interval [k, k+1],
        # so the ready stage envelope uses the ready-stage value itself.
        x_s, y_s = self._make_interval_series(expanded_surv)
        x_t, y_total = self._make_interval_series(expanded_total)
        _, y_gf = self._make_interval_series(expanded_gf)
        _, y_g = self._make_interval_series(expanded_g)

        fig, ax = plt.subplots(1, 1, figsize=(10.24, 5.12), dpi=100)
        line_survival, = ax.plot(x_s, y_s, label="Surviving Shots", color="C0", linestyle="--", linewidth=2)
        ax.fill_between(x_s, 0, y_s, alpha=0.2, color="C0")
        ax.set_ylim(0, 1.01)
        ax.set_xlim(0, max(x_s))
        ax.set_yticks([x * 0.1 for x in range(11)])
        ax.set_ylabel("Survival Proportion")

        ax_r = ax.twinx()
        line_time_no_dec, = ax_r.plot(
            x_t,
            y_gf,
            label="Gate+Feedback Runtime (ns)",
            color="#7570b3",
            linewidth=2.8,
        )
        line_time_gate, = ax_r.plot(
            x_t,
            y_g,
            label="Gate-Only Runtime (ns)",
            color="#d62728",
            linewidth=2.8,
        )
        ax_r.fill_between(x_t, 0, y_g, alpha=0.28, color="#d62728", label="Gate Execution Time (immutable)")
        ax_r.fill_between(x_t, y_g, y_gf, alpha=0.2, color="#9467bd", label="Accumulated Feedback Time (ns)")
        ax_r.fill_between(x_t, y_gf, y_total, alpha=0.2, color="#2ca02c", label="Accumulated Decoder Time (ns)")
        line_time_total, = ax_r.plot(
            x_t,
            y_total,
            label="Total Runtime (ns)",
            color="#1b9e77",
            linewidth=3.1,
            zorder=5,
        )
        ax_r.set_ylabel("Execution Time (ns)")

        ax.set_xticks(range(len(expanded_names) + 1), [""] * (len(expanded_names) + 1))
        ax.set_xticks([e + 0.5 for e in range(len(expanded_names))], expanded_names, rotation=90, minor=True)
        ax.xaxis.set_tick_params(length=0, which="minor")
        ax.grid()

        lines = [line_survival, line_time_total, line_time_no_dec, line_time_gate]
        labels_for_legend = [line.get_label() for line in lines]
        labels_for_legend.append("Gate Execution Time (immutable)")
        labels_for_legend.append("Accumulated Feedback Time (ns)")
        labels_for_legend.append("Accumulated Decoder Time (ns)")
        handles = [
            line_survival,
            line_time_total,
            line_time_no_dec,
            line_time_gate,
            plt.Rectangle((0, 0), 1, 1, fc="#d62728", alpha=0.28),
            plt.Rectangle((0, 0), 1, 1, fc="#9467bd", alpha=0.2),
            plt.Rectangle((0, 0), 1, 1, fc="#2ca02c", alpha=0.2),
        ]
        ax.legend(handles, labels_for_legend, loc="upper right")

        ax.set_title(
            f"Life of a Fault-Distance-{self.distance} Cultivation\n"
            f"(runtime estimator, epochs={self.epoch}, first_reach means, wait_rounds={self.wait_rounds_for_plot})"
        )
        fig.tight_layout()
        fig.savefig(path, format="svg")
        plt.close(fig)

    @staticmethod
    def _shift_to_stage_entry(values: np.ndarray) -> np.ndarray:
        arr = np.array(values, dtype=np.float64)
        if len(arr) == 0:
            return arr
        out = np.zeros_like(arr)
        out[1:] = arr[:-1]
        return out

    @staticmethod
    def _shift_to_stage_entry_with_ready(values: np.ndarray) -> np.ndarray:
        arr = np.array(values, dtype=np.float64)
        n = len(arr)
        if n == 0:
            return arr
        if n == 1:
            return np.array([arr[0]], dtype=np.float64)
        out = np.zeros_like(arr)
        if n > 2:
            out[1:-1] = arr[:-2]
        out[-1] = arr[-1]
        return out

    @staticmethod
    def _make_interval_series(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs: list[float] = []
        ys: list[float] = []
        for i, v in enumerate(values):
            xs.append(float(i))
            ys.append(float(v))
            xs.append(float(i + 1))
            ys.append(float(v))
        return np.array(xs), np.array(ys)

    def _run_single_attempt_until_postselect(self) -> tuple[set[int], Optional[int], Optional[int]]:
        sim = stim.FlipSimulator(batch_size=1, num_qubits=self.circuit.num_qubits)
        sim.clear()
        det_fired: set[int] = set()
        cur_det = 0
        fail_stage: Optional[int] = None
        for stage_i, layer in enumerate(self.layers):
            sim.do(layer)
            while cur_det < sim.num_detectors:
                fired = bool(sim.get_detector_flips(detector_index=cur_det)[0])
                if fired:
                    det_fired.add(cur_det)
                    if cur_det in self.postselected_detectors and fail_stage is None:
                        fail_stage = stage_i
                cur_det += 1
            if fail_stage is not None:
                break

        actual_obs: Optional[int] = None
        if fail_stage is None and self.circuit.num_observables > 0:
            actual_obs = int(sim.get_observable_flips(observable_index=0)[0])
        return det_fired, fail_stage, actual_obs

    def _record_stage(self, epoch_idx: int, stage_idx: int, acc: dict[str, float], first_seen: np.ndarray) -> None:
        for k in ["total", "gate", "feedback", "decode"]:
            self.records_until_success[k][epoch_idx, stage_idx] = acc[k]
        if not first_seen[stage_idx]:
            first_seen[stage_idx] = True
            for k in ["total", "gate", "feedback", "decode"]:
                self.records_first_reach[k][epoch_idx, stage_idx] = acc[k]

    def _add_gate_stage(self, acc: dict[str, float], stage_idx: int) -> None:
        t = float(self.layer_gate_times_ns[stage_idx])
        acc["gate"] += t
        acc["total"] += t

    def _add_feedback(self, acc: dict[str, float]) -> None:
        acc["feedback"] += self.feedback_time_ns
        acc["total"] += self.feedback_time_ns

    def _add_decode(self, acc: dict[str, float], decode_time_ns: float) -> None:
        acc["decode"] += decode_time_ns
        acc["total"] += decode_time_ns

    def _decode_with_timing(self, det_set: set[int], measure_mode: str = "serial") -> tuple[Any, int, float]:
        # t0 = time.perf_counter_ns()
        # obs, gap = self.compiled.decode_det_set(det_set)
        # t1 = time.perf_counter_ns()
        # measured_ns = float(max(t1 - t0, 0))
        obs, gap, measured_ns = self.compiled.decode_det_set_with_time(det_set, measure_mode=measure_mode)
        decode_ns = measured_ns if self.use_measured_decode_time else self.decoder_latency_ns
        self.decode_call_times_ns.append(decode_ns)
        return obs, int(gap), float(decode_ns)

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

    def _get_records(self, record_type: str) -> dict[str, np.ndarray]:
        if not self._simulated:
            raise ValueError("Run estimate_runtime() first.")
        if record_type == "first_reach":
            return self.records_first_reach
        if record_type == "until_success":
            return self.records_until_success
        raise ValueError("record_type must be 'first_reach' or 'until_success'")

    @staticmethod
    def _circuit_to_layers(circuit: stim.Circuit) -> list[stim.Circuit]:
        cur_layer = stim.Circuit()
        prev_layers: list[stim.Circuit] = []
        saw_two_qubit_gate = False
        saw_measurement = False
        for inst in circuit.flattened():
            data = stim.GateData(inst.name)
            if inst.name == "QUBIT_COORDS":
                continue
            elif data.is_two_qubit_gate:
                if saw_measurement:
                    saw_measurement = False
                    prev_layers.append(cur_layer)
                    cur_layer = stim.Circuit()
                saw_two_qubit_gate = True
                cur_layer.append(inst)
            elif data.produces_measurements:
                cur_layer.append(inst)
                saw_measurement = True
            elif data.is_reset or data.produces_measurements:
                if saw_two_qubit_gate:
                    saw_two_qubit_gate = False
                    saw_measurement = False
                    prev_layers.append(cur_layer)
                    cur_layer = stim.Circuit()
                cur_layer.append(inst)
            else:
                cur_layer.append(inst)
        if len(cur_layer):
            prev_layers.append(cur_layer)
        return prev_layers

    def _layer_gate_time_ns(self, layer: stim.Circuit) -> float:
        tick_slice_max = 0.0
        total = 0.0
        saw_gate = False
        for inst in layer.flattened():
            if inst.name == "TICK":
                total += tick_slice_max
                tick_slice_max = 0.0
                saw_gate = False
                continue
            t = self._inst_duration_ns(inst)
            if t > 0:
                saw_gate = True
            if t > tick_slice_max:
                tick_slice_max = t
        if saw_gate or tick_slice_max > 0:
            total += tick_slice_max
        return float(total)

    def _inst_duration_ns(self, inst: stim.CircuitInstruction) -> float:
        name = inst.name
        if name in ["QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "TICK"]:
            return 0.0
        if name in self.gate_times_ns:
            return float(self.gate_times_ns[name])
        data = stim.GateData(name)
        if data.produces_measurements:
            return float(self.gate_times_ns["MEAS"])
        if data.is_reset:
            return float(self.gate_times_ns["RESET"])
        if data.is_two_qubit_gate:
            return float(self.gate_times_ns["2Q"])
        if data.is_single_qubit_gate:
            return float(self.gate_times_ns["1Q"])
        return float(self.gate_times_ns["DEFAULT"])

    @staticmethod
    def _find_postselected_detectors(circuit: stim.Circuit) -> set[int]:
        result: set[int] = set()
        coords = circuit.get_detector_coordinates()
        for det, coord in coords.items():
            if len(coord) == 3 or coord[-1] == -9 or (len(coord) > 4 and (coord[4] == 0 or coord[4] == 4)):
                result.add(det)

        # B2 rule (DEM-driven): both endpoints of any error with single-basis,
        # exactly 2 detectors, and non-zero obs_mask. These mark fault-distance-3
        # logical-error mechanisms that any decoder is fundamentally unable to
        # resolve, so the protocol discards them. Mirrors the rule in
        # CompiledDesaturationSampler.from_task and clip_dem_as_desaturation.
        dem = circuit.detector_error_model().flattened()
        det_bases: dict[int, str] = {}
        for d in range(dem.num_detectors):
            c = coords.get(d, [])
            if len(c) <= 4 or c[4] == -9:
                det_bases[d] = '!'
            else:
                det_bases[d] = 'XXXZZZXZ'[int(c[4])]
        for inst in dem:
            if inst.type != 'error':
                continue
            targets = inst.targets_copy()
            if any(t.is_separator() for t in targets):
                continue
            det_set: set[int] = set()
            obs_mask = 0
            for t in targets:
                if t.is_relative_detector_id():
                    det_set.add(t.val)
                elif t.is_logical_observable_id():
                    obs_mask |= 1 << t.val
            if len(det_set) == 2 and obs_mask:
                if len({det_bases.get(d, '!') for d in det_set}) == 1:
                    result.update(det_set)
        return result

    @staticmethod
    def _build_detector_stage_index(layers: list[stim.Circuit]) -> dict[int, int]:
        # Detector indices become available in-order while simulating layers.
        # This helper maps detector index -> first layer index where it is visible.
        sim = stim.FlipSimulator(batch_size=1, num_qubits=max((layer.num_qubits for layer in layers), default=0))
        sim.clear()
        mapping: dict[int, int] = {}
        cur_det = 0
        for stage_i, layer in enumerate(layers):
            sim.do(layer)
            while cur_det < sim.num_detectors:
                mapping[cur_det] = stage_i
                cur_det += 1
        return mapping

    def _build_stage_names(self, *, wait_rounds: int) -> list[str]:
        names = [
            "Encode T",
            "Stabilize",
            "Check T",
            "Check T",
            *(
                [
                    "Stabilize",
                    "Stabilize",
                    "Stabilize",
                    "Check T",
                    "Check T",
                ]
                * (self.distance == 5)
            ),
            *(["Stabilize"] * self.distance),
            "Escaped!",
            *(["[wait for gap]"] * max(wait_rounds - 1, 0)),
            "ready",
        ]
        return names

    @staticmethod
    def _expand_wait_rounds_for_plot(
        *,
        stage_names: list[str],
        total: np.ndarray,
        gate_plus_feedback: np.ndarray,
        gate_only: np.ndarray,
        wait_rounds: int,
    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
        names = list(stage_names)
        t = np.array(total, dtype=np.float64)
        gf = np.array(gate_plus_feedback, dtype=np.float64)
        g = np.array(gate_only, dtype=np.float64)

        if "[wait for gap]" not in names:
            return names, t, gf, g
        wait_idx = names.index("[wait for gap]")
        current_waits = names.count("[wait for gap]")
        target_waits = max(wait_rounds - 1, 0)
        extra = target_waits - current_waits
        if extra <= 0:
            return names, t, gf, g

        insert_at = wait_idx + current_waits
        wait_name = "[wait for gap]"
        wait_t = t[wait_idx]
        wait_gf = gf[wait_idx]
        wait_g = g[wait_idx]
        for _ in range(extra):
            names.insert(insert_at, wait_name)
            t = np.insert(t, insert_at, wait_t)
            gf = np.insert(gf, insert_at, wait_gf)
            g = np.insert(g, insert_at, wait_g)
            insert_at += 1
        return names, t, gf, g

    @staticmethod
    def _make_step_series(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs: list[float] = []
        ys: list[float] = []
        for i, v in enumerate(values):
            if i > 0:
                xs.append(i)
                ys.append(ys[-1])
            xs.append(i)
            ys.append(float(v))
        return np.array(xs), np.array(ys)

    def _survival_by_attempt(self) -> np.ndarray:
        if self.attempt_count <= 0:
            return np.zeros(self.num_stages, dtype=np.float64)
        return self.stage_pass_counts.astype(np.float64) / float(self.attempt_count)

    @staticmethod
    def _default_gate_times_ns() -> dict[str, float]:
        return {
            "1Q": 20.0,
            "2Q": 40.0,
            "MEAS": 400.0,
            "RESET": 200.0,
            "DEFAULT": 20.0,
            "H": 20.0,
            "S": 20.0,
            "S_DAG": 20.0,
            "X": 20.0,
            "Y": 20.0,
            "Z": 20.0,
            "RX": 20.0,
            "RY": 20.0,
            "R": 200.0,
            "MX": 400.0,
            "MY": 400.0,
            "M": 400.0,
            "MPP": 400.0,
            "CX": 40.0,
            "CZ": 40.0,
        }

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
