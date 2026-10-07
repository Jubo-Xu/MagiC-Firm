"""Cycle-accurate runtime estimator for the magic-state cultivation protocol.

All times are in ns. A stage is one measurement round; the hardware stages are
followed by the plot-only "[wait for gap]" stages and "ready". Time is charged
as gate, feedback or gap_decode, and accumulates over the failed attempts of
an epoch (one accepted shot). Gating modes and tiers: see estimate_runtime.
"""
import json
import pathlib
import sys
from typing import Any, Iterable, Optional

import numpy as np
import stim

src_path = pathlib.Path(__file__).parent.parent / "magic_state_cultivation" / "upstream" / "src"
assert src_path.exists()
sys.path.append(str(src_path))

from layer_schedule import round_schedule, has_terminal_mpp, detector_last_measurements


class RuntimeEstimator:
    def __init__(
        self,
        *,
        circuit_generator=None,
        circuit: Optional[stim.Circuit] = None,
        complete_compiled,
        partial_compiled=None,
        postselected_detectors: Optional[Iterable[int]] = None,
        epoch: int = 1000,
        feedback_time_ns: float = 100.0,
        use_measured_decode_time: bool = True,
        decoder_latency_ns: float = 10_000.0,
        partial_decoder_latency_ns: float = 500.0,
        gate_times_ns: Optional[dict[str, float]] = None,
        wait_rounds: int = 2,
        boundary_round_as_previous: bool = True,
    ):
        if epoch <= 0:
            raise ValueError("epoch must be positive.")
        # wait_stage_idx needs at least one "[wait for gap]" stage.
        if wait_rounds < 2:
            raise ValueError("wait_rounds must be >= 2 (gives at least 1 [wait for gap] stage).")

        self.epoch = int(epoch)
        self.feedback_time_ns = float(feedback_time_ns)
        # Profile from set_control_latency; replaces feedback_time_ns when set.
        self.control_latency: dict[str, Any] | None = None
        self._ps_feedback_by_round: np.ndarray | None = None
        self._delivered = False          # delivery charged once per attempt
        self.use_measured_decode_time = bool(use_measured_decode_time)
        self.decoder_latency_ns = float(decoder_latency_ns)
        self.partial_decoder_latency_ns = float(partial_decoder_latency_ns)
        self.gate_times_ns = (
            gate_times_ns if gate_times_ns is not None else self._default_gate_times_ns()
        )
        self.wait_rounds = int(wait_rounds)

        # The generator's d1 is used only in the plot title.
        if (circuit is None) == (circuit_generator is None):
            raise ValueError("give exactly one of circuit= or circuit_generator=")
        if circuit_generator is not None:
            if circuit_generator.ideal_circuit is None:
                raise ValueError("circuit_generator.ideal_circuit is None. Call generate() first.")
            circuit = circuit_generator.noisy_circuit or circuit_generator.ideal_circuit
            params = circuit_generator.params
            self.distance: Optional[int] = int(params.d1) if params.d1 is not None else None
        else:
            self.distance = None
        self.circuit_generator = circuit_generator
        self.circuit = circuit

        self.complete_compiled = complete_compiled
        self.partial_compiled = partial_compiled

        # A given set (the circuit folder's postselected_detectors.json, shared
        # with run_mb_characterization and the control-system compiler) must
        # equal the estimator's own rule, else ValueError.
        own_rule = self._find_postselected_detectors(self.circuit)
        if postselected_detectors is None:
            self.postselected_detectors: set[int] = own_rule
            self.postselect_source = "own_rule"
        else:
            given = {int(d) for d in postselected_detectors}
            if given != own_rule:
                raise ValueError(
                    "postselected_detectors disagrees with the estimator's own rule "
                    f"(tags + B2): only-in-given={sorted(given - own_rule)[:8]}, "
                    f"only-in-own={sorted(own_rule - given)[:8]}"
                )
            self.postselected_detectors = given
            self.postselect_source = "folder_json"

        # Hardware stage r is measurement round r, the convention of
        # layer_schedule and the compiler's timing.py. It closes with the
        # round's measurement slice, so detector d's data exists at
        # round_completion_ns[detector_to_stage[d]].
        self.boundary_round_as_previous = bool(boundary_round_as_previous)
        round_ns, meas_round_ids = round_schedule(
            self.circuit, self.gate_times_ns,
            boundary_round_as_previous=self.boundary_round_as_previous,
        )
        self.round_completion_ns: np.ndarray = np.asarray(round_ns, dtype=np.float64)
        self.num_hardware_stages: int = len(round_ns)
        self.layer_gate_times_ns: np.ndarray = np.diff(self.round_completion_ns, prepend=0.0)
        self.gate_cumsum_ns: np.ndarray = self.round_completion_ns
        self.gate_to_complete_ns: float = float(self.round_completion_ns[-1])
        self.round_meas_counts: list[int] = np.bincount(
            meas_round_ids, minlength=self.num_hardware_stages).tolist()
        self.has_terminal_mpp: bool = has_terminal_mpp(self.circuit)

        # A detector's stage is the round of its last contributing measurement.
        # Its DETECTOR declaration can trail by a round.
        det_last = detector_last_measurements(self.circuit)
        self.detector_to_stage: dict[int, int] = {
            d: int(meas_round_ids[m]) for d, m in enumerate(det_last) if m is not None
        }

        self.stage_names: list[str] = self._stage_names_from_rounds(
            self.round_meas_counts, wait_rounds=self.wait_rounds,
        )
        # The first "[wait for gap]" stage follows the last hardware stage.
        self.wait_stage_idx: int = self.stage_names.index("[wait for gap]")
        self.ready_stage_idx: int = self.stage_names.index("ready")
        self.num_stages: int = len(self.stage_names)
        assert self.wait_stage_idx == self.num_hardware_stages

        # Array views for the vectorised per-attempt derivation.
        nd = self.circuit.num_detectors
        self._det_stage_arr = np.full(nd, -1, dtype=np.int64)
        for d, s in self.detector_to_stage.items():
            self._det_stage_arr[d] = s
        self._postselect_mask = np.zeros(nd, dtype=np.bool_)
        self._postselect_mask[sorted(self.postselected_detectors)] = True
        assert self._det_stage_arr[self._postselect_mask].min() >= 0

        # last_kept_layer: last round holding a detector kept by the partial
        # mask. gate_to_round_m_ns: gate time up to the end of that round.
        self.last_kept_layer: Optional[int] = None
        self.gate_to_round_m_ns: Optional[float] = None
        if partial_compiled is not None:
            mask = getattr(partial_compiled, "partial_mask_bool", None)
            if mask is None:
                raise ValueError(
                    "partial_compiled must expose `partial_mask_bool` "
                    "(e.g., be a CompiledPartialDesaturationSampler in partial mode)."
                )
            kept_dets = [d for d in range(len(mask)) if bool(mask[d])]
            if not kept_dets:
                raise ValueError("partial_mask_bool selects no detectors.")
            missing = [d for d in kept_dets if d not in self.detector_to_stage]
            if missing:
                raise ValueError(
                    f"partial_mask references detector indices with no measurement "
                    f"round (record-less DETECTORs?): e.g. {missing[:5]}."
                )
            self.last_kept_layer = max(self.detector_to_stage[d] for d in kept_dets)
            self.gate_to_round_m_ns = float(self.gate_cumsum_ns[self.last_kept_layer])

        # Accumulated ns per epoch and stage, shape (epoch, num_stages):
        # first_reach at the first pass of a stage, until_success at the last.
        self.records_first_reach: dict[str, np.ndarray] = {
            k: np.full((self.epoch, self.num_stages), np.nan, dtype=np.float64)
            for k in ("total", "gate", "feedback", "gap_decode")
        }
        self.records_until_success: dict[str, np.ndarray] = {
            k: np.zeros((self.epoch, self.num_stages), dtype=np.float64)
            for k in ("total", "gate", "feedback", "gap_decode")
        }

        self.attempt_count: int = 0
        self.stage_pass_counts: np.ndarray = np.zeros(self.num_stages, dtype=np.int64)

        self.partial_decode_call_times_ns: list[float] = []
        self.complete_decode_call_times_ns: list[float] = []
        self.accepted_decision_times_ns: list[float] = []
        self.accepted_prediction_times_ns: list[float] = []
        self.accepted_actual_obs: list[int] = []
        self.accepted_predict_obs: list[int] = []
        # Join key of each accepted shot: its trace attempt index, or None
        # when simulating.
        self.accepted_attempt_indices: list[Optional[int]] = []

        # "simulate" (fresh FlipSimulator passes) or "trace:..." (attach_trace).
        self._trace = None
        self._cursor = None
        self.attempt_source: str = "simulate"
        self._last_attempt_index: Optional[int] = None
        self._last_row: Optional[np.ndarray] = None   # packed gap-circuit row (trace mode)

        # Latency mode: "measured" (pymatching wall-clock) or "constant"
        # (decoder_latency_ns, partial_decoder_latency_ns) from the constructor;
        # "external_constant" and "external_per_shot" from
        # set_decoder_latency_constants and set_decoder_latency_table.
        # latency_source is the label recorded in the statistics.
        self._latency_mode: str = "measured" if self.use_measured_decode_time else "constant"
        self.latency_source: str = self._latency_mode
        self.latency_provenance: dict[str, Any] = {}
        self._latency_const: dict[str, dict[str, float]] = {}
        self._latency_table: dict[str, dict[str, np.ndarray]] = {}

        # tierN_count: attempts in that gap_p region, in both 2-stage modes (see
        # estimate_runtime). In PCP tier 1 only rejects and tier 3 only accepts.
        self.tier1_count:        int = 0
        self.tier1_accept_count: int = 0
        self.tier1_reject_count: int = 0
        self.tier2_count:        int = 0
        self.tier2_accept_count: int = 0
        self.tier2_reject_count: int = 0
        self.tier3_count:        int = 0
        self.tier3_accept_count: int = 0
        self.tier3_reject_count: int = 0

        self.logical_error_rate: float = 0.0
        self._simulated: bool = False

        # Filled by collect_statistics(); reset to None by estimate_runtime().
        self.statistics: Optional[dict[str, Any]] = None

    # ==================================================================
    # ====================== SECTION 1: SIMULATION =====================
    # ==================================================================

    # ------------------------------------------------------------------
    # 1.1  Simulation entry point
    # ------------------------------------------------------------------

    def estimate_runtime(
        self,
        *,
        gap_check: bool = True,
        gap_threshold: Optional[float] = None,
        gap_threshold_low: Optional[float] = None,
        gap_threshold_high: Optional[float] = None,
        gap_threshold_partial: Optional[float] = None,
        decoder_time_measure_mode: str = "serial",
        record_decoder_call_times: bool = False,
    ) -> None:
        """Simulate until `self.epoch` shots are accepted. Results are stored
        on the instance: records_*, accepted_* lists (length epoch), tier
        counters, stage_pass_counts and logical_error_rate.

        The thresholds given select the gating mode. Bad combinations raise
        ValueError, and both 2-stage modes need partial_compiled.
          no_gating: gap_check=False. Every postselect-passing shot is
            accepted; the complete decoder gives the prediction.
          single_stage: gap_threshold only. Accept if gap_c >= gap_threshold.
          two_stage (PCP): also gap_threshold_low and gap_threshold_high. The
            partial decoder rejects gap_p < T_low (tier 1) and accepts
            gap_p > T_high (tier 3). In between (tier 2) the complete decoder
            gates with gap_c >= gap_threshold.
          two_stage_inverted (CPC): also gap_threshold_partial, with T_low <=
            T_partial <= T_high. The complete decoder gates tiers 1 and 3; in
            tier 2 the partial decoder accepts if gap_p >= T_partial.

        An attempt decided by the partial decoder is charged gate time only
        through last_kept_layer. If it is accepted, the complete decoder still
        runs for the prediction, so its prediction time is later than its
        decision time.

        decoder_time_measure_mode: "serial" sums the two gap decodes,
          "parallel" takes their max.
        record_decoder_call_times: log every decode latency in
          {partial,complete}_decode_call_times_ns. Diagnostic only.
        """
        # ----------- mode resolution and validation -----------
        if not gap_check:
            if any(x is not None for x in (
                gap_threshold, gap_threshold_low, gap_threshold_high, gap_threshold_partial,
            )):
                raise ValueError("Cannot specify thresholds when gap_check=False.")
            mode = "no_gating"
        elif gap_threshold_partial is not None:
            if any(x is None for x in (gap_threshold, gap_threshold_low, gap_threshold_high)):
                raise ValueError(
                    "Inverted 2-stage gating requires all of gap_threshold, "
                    "gap_threshold_low, gap_threshold_high, and gap_threshold_partial."
                )
            if self.partial_compiled is None:
                raise ValueError(
                    "partial_compiled must be provided to RuntimeEstimator for "
                    "inverted 2-stage gating."
                )
            if not (gap_threshold_low <= gap_threshold_partial <= gap_threshold_high):
                raise ValueError(
                    f"Inverted 2-stage requires "
                    f"gap_threshold_low ({gap_threshold_low}) <= "
                    f"gap_threshold_partial ({gap_threshold_partial}) <= "
                    f"gap_threshold_high ({gap_threshold_high})."
                )
            mode = "two_stage_inverted"
        elif gap_threshold_low is not None or gap_threshold_high is not None:
            if gap_threshold_low is None or gap_threshold_high is None:
                raise ValueError(
                    "Both gap_threshold_low and gap_threshold_high must be set "
                    "for 2-stage gating."
                )
            if gap_threshold is None:
                raise ValueError(
                    "gap_threshold (T_complete) is required for 2-stage gating."
                )
            if self.partial_compiled is None:
                raise ValueError(
                    "partial_compiled must be provided to RuntimeEstimator for "
                    "2-stage gating."
                )
            if not (gap_threshold_low <= gap_threshold_high):
                raise ValueError(
                    f"gap_threshold_low ({gap_threshold_low}) must be <= "
                    f"gap_threshold_high ({gap_threshold_high})."
                )
            mode = "two_stage"
        else:
            if gap_threshold is None:
                raise ValueError("gap_threshold is required for single-stage gating.")
            mode = "single_stage"

        T_complete = float(gap_threshold) if gap_threshold is not None else None
        T_low      = float(gap_threshold_low) if gap_threshold_low is not None else None
        T_high     = float(gap_threshold_high) if gap_threshold_high is not None else None
        T_partial  = float(gap_threshold_partial) if gap_threshold_partial is not None else None

        # ----------- reset state -----------
        self.statistics = None
        for k in self.records_first_reach:
            self.records_first_reach[k][:] = np.nan
            self.records_until_success[k][:] = 0.0
        self.attempt_count = 0
        self.stage_pass_counts[:] = 0
        self.partial_decode_call_times_ns = []
        self.complete_decode_call_times_ns = []
        self.accepted_decision_times_ns = []
        self.accepted_prediction_times_ns = []
        self.accepted_actual_obs = []
        self.accepted_predict_obs = []
        self.accepted_attempt_indices = []
        self.tier1_count = 0
        self.tier1_accept_count = 0
        self.tier1_reject_count = 0
        self.tier2_count = 0
        self.tier2_accept_count = 0
        self.tier2_reject_count = 0
        self.tier3_count = 0
        self.tier3_accept_count = 0
        self.tier3_reject_count = 0

        last_layer_idx = self.num_hardware_stages - 1

        # ----------- main loop -----------
        for e in range(self.epoch):
            acc: dict[str, float] = {"total": 0.0, "gate": 0.0, "feedback": 0.0, "gap_decode": 0.0}
            first_seen = np.zeros(self.num_stages, dtype=np.bool_)
            success = False

            while not success:
                det_set, fail_stage, actual_obs = self._run_single_attempt_until_postselect()
                self.attempt_count += 1
                self._delivered = False

                # ---- postselect failure (mid-protocol) ----
                if fail_stage is not None:
                    for s in range(fail_stage):
                        self.stage_pass_counts[s] += 1
                    for s in range(fail_stage + 1):
                        self._add_gate_stage(acc, s)
                        self._record_stage(e, s, acc, first_seen)
                    self._add_feedback(acc, "postselect", fail_stage)
                    continue

                # ---- postselect passed: classify by mode ----
                if actual_obs is None:
                    raise ValueError(
                        "Expected actual observable for postselect-passing attempt, got None."
                    )

                if mode == "no_gating":
                    self._charge_stages_through(e, acc, first_seen, last_layer_idx)
                    self._record_wait_stages(e, acc, first_seen)

                    pred_obs, _gap_c, complete_ns = self._complete_decode(
                        det_set, decoder_time_measure_mode,
                    )
                    if record_decoder_call_times:
                        self.complete_decode_call_times_ns.append(complete_ns)
                    self._charge_decode(acc, complete_ns)

                    self._record_ready(e, acc, first_seen)

                    self._record_accept(
                        decision_ns=acc["total"], prediction_ns=acc["total"],
                        actual_obs=actual_obs, pred_obs=pred_obs,
                    )
                    success = True
                    continue

                if mode == "single_stage":
                    self._charge_stages_through(e, acc, first_seen, last_layer_idx)
                    self._record_wait_stages(e, acc, first_seen)

                    pred_obs, gap_c, complete_ns = self._complete_decode(
                        det_set, decoder_time_measure_mode,
                    )
                    if record_decoder_call_times:
                        self.complete_decode_call_times_ns.append(complete_ns)
                    self._charge_decode(acc, complete_ns)

                    if gap_c >= T_complete:
                        self._record_ready(e, acc, first_seen)
                        self._record_accept(
                            decision_ns=acc["total"], prediction_ns=acc["total"],
                            actual_obs=actual_obs, pred_obs=pred_obs,
                        )
                        success = True
                    else:
                        self._add_feedback(acc, "gap")
                    continue

                # ---- 2-stage modes (standard PCP or inverted CPC) ----
                if mode == "two_stage":
                    acc_total_at_attempt_start = acc["total"]

                    gap_p, partial_ns = self._partial_decode(det_set, decoder_time_measure_mode)
                    if record_decoder_call_times:
                        self.partial_decode_call_times_ns.append(partial_ns)

                    if gap_p < T_low:
                        # Tier 1: reject on the partial gap.
                        self._charge_stages_through(e, acc, first_seen, self.last_kept_layer)
                        self._count_stages_no_gate(
                            e, acc, first_seen,
                            self.last_kept_layer + 1, last_layer_idx,
                        )
                        self._record_wait_stages(e, acc, first_seen)
                        self._charge_decode(acc, partial_ns)
                        self.tier1_count += 1
                        self.tier1_reject_count += 1
                        self._add_feedback(acc, "gap")
                        continue

                    if gap_p > T_high:
                        # Tier 3: accept on the partial gap.
                        self._charge_stages_through(e, acc, first_seen, self.last_kept_layer)
                        self._count_stages_no_gate(
                            e, acc, first_seen,
                            self.last_kept_layer + 1, last_layer_idx,
                        )
                        self._record_wait_stages(e, acc, first_seen)
                        self._charge_decode(acc, partial_ns)
                        self._record_ready(e, acc, first_seen)

                        # The complete decoder runs alongside the rest of the
                        # circuit, for the prediction only. TODO: no gap is needed
                        # here, so one decode of the matchable DEM would do.
                        pred_obs, _gap_c, complete_ns = self._complete_decode(
                            det_set, decoder_time_measure_mode,
                        )
                        if record_decoder_call_times:
                            self.complete_decode_call_times_ns.append(complete_ns)

                        self.tier3_count += 1
                        self.tier3_accept_count += 1
                        self._record_accept(
                            decision_ns=acc["total"],
                            prediction_ns=acc_total_at_attempt_start + self.gate_to_complete_ns + self._deliver_ns() + complete_ns,
                            actual_obs=actual_obs, pred_obs=pred_obs,
                        )
                        success = True
                        continue

                    # Tier 2: the complete decoder gates.
                    pred_obs, gap_c, complete_ns = self._complete_decode(
                        det_set, decoder_time_measure_mode,
                    )
                    if record_decoder_call_times:
                        self.complete_decode_call_times_ns.append(complete_ns)

                    self._charge_stages_through(e, acc, first_seen, last_layer_idx)
                    self._record_wait_stages(e, acc, first_seen)
                    self._charge_decode(acc, complete_ns)
                    self.tier2_count += 1

                    if gap_c >= T_complete:
                        self.tier2_accept_count += 1
                        self._record_ready(e, acc, first_seen)
                        self._record_accept(
                            decision_ns=acc["total"], prediction_ns=acc["total"],
                            actual_obs=actual_obs, pred_obs=pred_obs,
                        )
                        success = True
                    else:
                        self.tier2_reject_count += 1
                        self._add_feedback(acc, "gap")
                    continue

                # ---- inverted 2-stage (CPC) ----
                if mode == "two_stage_inverted":
                    acc_total_at_attempt_start = acc["total"]

                    gap_p, partial_ns = self._partial_decode(det_set, decoder_time_measure_mode)
                    if record_decoder_call_times:
                        self.partial_decode_call_times_ns.append(partial_ns)

                    if gap_p < T_low or gap_p > T_high:
                        # Tier 1 or 3: the complete decoder gates on gap_c.
                        pred_obs, gap_c, complete_ns = self._complete_decode(
                            det_set, decoder_time_measure_mode,
                        )
                        if record_decoder_call_times:
                            self.complete_decode_call_times_ns.append(complete_ns)

                        self._charge_stages_through(e, acc, first_seen, last_layer_idx)
                        self._record_wait_stages(e, acc, first_seen)
                        self._charge_decode(acc, complete_ns)

                        is_low = (gap_p < T_low)
                        if is_low:
                            self.tier1_count += 1
                        else:
                            self.tier3_count += 1

                        if gap_c >= T_complete:
                            if is_low:
                                self.tier1_accept_count += 1
                            else:
                                self.tier3_accept_count += 1
                            self._record_ready(e, acc, first_seen)
                            self._record_accept(
                                decision_ns=acc["total"], prediction_ns=acc["total"],
                                actual_obs=actual_obs, pred_obs=pred_obs,
                            )
                            success = True
                        else:
                            if is_low:
                                self.tier1_reject_count += 1
                            else:
                                self.tier3_reject_count += 1
                            self._add_feedback(acc, "gap")
                        continue

                    # Tier 2: the partial decoder gates on gap_p vs T_partial.
                    self.tier2_count += 1

                    if gap_p >= T_partial:
                        # Accept, charged like PCP tier 3.
                        self._charge_stages_through(e, acc, first_seen, self.last_kept_layer)
                        self._count_stages_no_gate(
                            e, acc, first_seen,
                            self.last_kept_layer + 1, last_layer_idx,
                        )
                        self._record_wait_stages(e, acc, first_seen)
                        self._charge_decode(acc, partial_ns)
                        self._record_ready(e, acc, first_seen)

                        pred_obs, _gap_c, complete_ns = self._complete_decode(
                            det_set, decoder_time_measure_mode,
                        )
                        if record_decoder_call_times:
                            self.complete_decode_call_times_ns.append(complete_ns)

                        self.tier2_accept_count += 1
                        self._record_accept(
                            decision_ns=acc["total"],
                            prediction_ns=acc_total_at_attempt_start + self.gate_to_complete_ns + self._deliver_ns() + complete_ns,
                            actual_obs=actual_obs, pred_obs=pred_obs,
                        )
                        success = True
                    else:
                        # Reject, charged like PCP tier 1.
                        self._charge_stages_through(e, acc, first_seen, self.last_kept_layer)
                        self._count_stages_no_gate(
                            e, acc, first_seen,
                            self.last_kept_layer + 1, last_layer_idx,
                        )
                        self._record_wait_stages(e, acc, first_seen)
                        self._charge_decode(acc, partial_ns)
                        self.tier2_reject_count += 1
                        self._add_feedback(acc, "gap")
                    continue

        # ---- LER over the accepted shots ----
        errs = [
            int(a) != int(p)
            for a, p in zip(self.accepted_actual_obs, self.accepted_predict_obs)
        ]
        if len(errs) != self.epoch:
            raise AssertionError(
                f"Expected {self.epoch} accepted shots; got {len(errs)}."
            )
        self.logical_error_rate = float(np.mean(errs)) if errs else 0.0
        self._simulated = True

    # ------------------------------------------------------------------
    # 1.2  Time accounting and per-stage records
    # ------------------------------------------------------------------

    def _charge_stages_through(
        self,
        e: int,
        acc: dict[str, float],
        first_seen: np.ndarray,
        last_layer: int,
    ) -> None:
        """Charge gate times for stages 0..last_layer (inclusive) and record them."""
        for s in range(last_layer + 1):
            self.stage_pass_counts[s] += 1
            self._add_gate_stage(acc, s)
            self._record_stage(e, s, acc, first_seen)

    def _count_stages_no_gate(
        self,
        e: int,
        acc: dict[str, float],
        first_seen: np.ndarray,
        stage_start: int,
        stage_end_inclusive: int,
    ) -> None:
        """Mark stages [stage_start, stage_end_inclusive] as passed at the
        current acc, adding no gate time. For attempts decided by the partial
        decoder at last_kept_layer: the hardware keeps running the remaining
        rounds, but they do not delay the decision."""
        for s in range(stage_start, stage_end_inclusive + 1):
            self.stage_pass_counts[s] += 1
            self._record_stage(e, s, acc, first_seen)

    def _record_wait_stages(self, e: int, acc: dict[str, float], first_seen: np.ndarray) -> None:
        """Mark every "[wait for gap]" stage passed at the current acc. They
        are plot-only and take no time; decoder latency is added after them."""
        for s in range(self.wait_stage_idx, self.ready_stage_idx):
            self.stage_pass_counts[s] += 1
            self._record_stage(e, s, acc, first_seen)

    def _record_ready(self, e: int, acc: dict[str, float], first_seen: np.ndarray) -> None:
        self.stage_pass_counts[self.ready_stage_idx] += 1
        self._record_stage(e, self.ready_stage_idx, acc, first_seen)

    def _record_accept(self, *, decision_ns: float, prediction_ns: float,
                       actual_obs: int, pred_obs: int) -> None:
        """Per-accepted-shot scalars, plus the attempt index (join key)."""
        self.accepted_decision_times_ns.append(float(decision_ns))
        self.accepted_prediction_times_ns.append(float(prediction_ns))
        self.accepted_actual_obs.append(int(actual_obs))
        self.accepted_predict_obs.append(int(pred_obs))
        self.accepted_attempt_indices.append(self._last_attempt_index)

    def _record_stage(
        self,
        e: int,
        s: int,
        acc: dict[str, float],
        first_seen: np.ndarray,
    ) -> None:
        for k in ("total", "gate", "feedback", "gap_decode"):
            self.records_until_success[k][e, s] = acc[k]
        if not first_seen[s]:
            first_seen[s] = True
            for k in ("total", "gate", "feedback", "gap_decode"):
                self.records_first_reach[k][e, s] = acc[k]

    def _add_gate_stage(self, acc: dict[str, float], stage_idx: int) -> None:
        t = float(self.layer_gate_times_ns[stage_idx])
        acc["gate"] += t
        acc["total"] += t

    def set_control_latency(self, profile: dict[str, Any]) -> None:
        """Replace feedback_time_ns by a control-latency profile
        (algorithms/control_latency.py). Fields, in ns:
          ps_feedback_ns[stage]  postselect reject in the stage's rounds
                                 (det_time_range): up to the stage board, verdict
                                 to the root, abort broadcast to every leaf
          gap_feedback_ns        decoder verdict from the root to the leaves
          deliver_ns             last measurement at the leaves to the root
                                 decoder, charged once per decoded attempt
        Raises unless the profile's stages cover exactly the estimator's rounds."""
        n = self.num_hardware_stages
        if profile.get("meas_times") != n:
            raise ValueError(f"control-latency profile has {profile.get('meas_times')} rounds, "
                             f"estimator has {n}")
        by_round = np.full(n, np.nan)
        for st in profile["stages"]:
            a, b = st["det_time_range"]
            by_round[a:b + 1] = float(profile["ps_feedback_ns"][str(st["stage"])])
        if np.isnan(by_round).any():
            raise ValueError("control-latency profile stages do not cover every round")
        self._ps_feedback_by_round = by_round
        self.control_latency = profile

    def _add_feedback(self, acc: dict[str, float], kind: str = "gap",
                      round_idx: int | None = None) -> None:
        """Classical round trip before the next attempt: `kind` 'postselect' (a
        stage board rejected in round `round_idx`) or 'gap' (decoder verdict at
        the root). Fixed feedback_time_ns unless a control-latency profile is set."""
        if self.control_latency is None:
            ns = self.feedback_time_ns
        elif kind == "postselect":
            ns = float(self._ps_feedback_by_round[round_idx])
        else:
            ns = float(self.control_latency["gap_feedback_ns"])
        acc["feedback"] += ns
        acc["total"] += ns

    def _charge_decode(self, acc: dict[str, float], ns: float) -> None:
        """Charge decoder latency as gap_decode. With a control-latency
        profile, the first decode of an attempt also charges deliver_ns as
        feedback; the partial and complete decoders share that delivery."""
        if self.control_latency is not None and not self._delivered:
            d = float(self.control_latency["deliver_ns"])
            acc["feedback"] += d
            acc["total"] += d
            self._delivered = True
        acc["gap_decode"] += ns
        acc["total"] += ns

    def _deliver_ns(self) -> float:
        return float(self.control_latency["deliver_ns"]) if self.control_latency is not None else 0.0

    # ------------------------------------------------------------------
    # 1.3  Decoder latency and decoder calls
    # ------------------------------------------------------------------

    def set_decoder_latency_constants(self, *, complete: dict[str, float],
                                      partial: Optional[dict[str, float]] = None,
                                      provenance: Optional[dict[str, Any]] = None,
                                      source: str = "external_constant") -> None:
        """Constant decoder latency per decode call, supplied by the caller
        (e.g. means measured on a hardware decoder). `complete` / `partial`
        map each measure mode to ns: {"serial": ns, "parallel": ns} — 'serial'
        = ON+OFF gap decodes back to back, 'parallel' = max of the two;
        decoder_time_measure_mode picks one at decode time. Exact for the
        AVERAGE prep time (latency enters additively; the path choice depends
        on the gap, not the latency); per-shot correlation and tails need a
        per-attempt table. `provenance` and `source` are recorded verbatim in
        the statistics. If the latencies are residuals measured after the last
        syndrome layer arrives, verify the clocks agree with check_round_clock."""
        given = {"complete": complete, "partial": partial}
        self._latency_const = {w: {m: float(v[m]) for m in ("serial", "parallel")}
                               for w, v in given.items() if v is not None}
        if self.partial_compiled is not None and "partial" not in self._latency_const:
            raise ValueError("partial_compiled is set but no partial latency given")
        self.latency_provenance = dict(provenance or {})
        self._latency_mode, self.latency_source = "external_constant", source

    def check_round_clock(self, *, first_round: int, last_round: int,
                          span_ns: float, label: str = "latency") -> None:
        """Raise unless rounds [first_round, last_round] of this estimator's
        clock span `span_ns` (within 1 ns). For decoder latencies measured as
        a residual from the arrival of the last syndrome layer: the estimator
        adds them after the last needed round, so the measurement's arrival
        schedule and the round-completion times must sit on the same clock
        (same gate times and boundary-round policy)."""
        span = self.round_completion_ns[last_round] - self.round_completion_ns[first_round]
        if abs(span - span_ns) > 1.0:
            raise ValueError(
                f"{label} was measured on a different round clock: it spans {span_ns} ns, "
                f"this estimator's rounds [{first_round}, {last_round}] span {span:.0f} ns "
                f"(gate times / boundary-round policy differ)")

    @property
    def trace_identity(self) -> Optional[dict[str, Any]]:
        """Identity of the attached trace (circuit hash, master seed, batch
        size), or None when simulating. Lets a caller check that an external
        per-attempt table was measured on the same trace."""
        if self._trace is None:
            return None
        m = self._trace.meta
        return {"circuit_sha": m.circuit_sha, "master_seed": m.master_seed,
                "batch_size": m.batch_size}

    def set_decoder_latency_table(self, *, complete: dict[str, np.ndarray],
                                  partial: Optional[dict[str, np.ndarray]] = None,
                                  provenance: Optional[dict[str, Any]] = None,
                                  source: str = "external_per_shot") -> None:
        """Per-attempt decoder latency supplied by the caller. Each table holds
        equal-length arrays: attempt_indices (strictly increasing), on_ns and
        off_ns (latency of the two gap decodes), defect_hashes
        (AttemptTrace.defect_hash of each shot). Requires attach_trace(): the
        join key is the attempt index and every lookup verifies the shot's
        defect hash, so mispairing is impossible to miss. `provenance` and
        `source` are recorded verbatim in the statistics."""
        if self._trace is None:
            raise RuntimeError("attach_trace() first: per-attempt latency joins on attempt index")
        keys = ("attempt_indices", "on_ns", "off_ns", "defect_hashes")
        self._latency_table = {}
        for which, t in (("complete", complete), ("partial", partial)):
            if t is None:
                continue
            t = {k: np.asarray(t[k]) for k in keys}
            if not np.all(np.diff(t["attempt_indices"]) > 0):
                raise ValueError(f"{which} latency table: attempt_indices not strictly increasing")
            self._latency_table[which] = t
        if self.partial_compiled is not None and "partial" not in self._latency_table:
            raise ValueError("partial_compiled is set but no partial latency table given")
        self.latency_provenance = dict(provenance or {})
        self._latency_mode, self.latency_source = "external_per_shot", source

    def _latency_table_lookup_ns(self, which: str, measure_mode: str) -> float:
        from attempt_stream import AttemptTrace
        t = self._latency_table[which]
        i = self._last_attempt_index
        pos = int(np.searchsorted(t["attempt_indices"], i))
        if pos >= len(t["attempt_indices"]) or t["attempt_indices"][pos] != i:
            raise RuntimeError(
                f"attempt {i} is not in the {which} latency table (last measured attempt "
                f"{int(t['attempt_indices'][-1])}): extend the table to cover it")
        if t["defect_hashes"][pos] != AttemptTrace.defect_hash(self._last_row):
            raise RuntimeError(f"defect-hash mismatch for attempt {i}: table/trace mispaired")
        on, off = float(t["on_ns"][pos]), float(t["off_ns"][pos])
        return on + off if measure_mode == "serial" else max(on, off)

    def _latency_ns(self, which: str, measured_ns: float, measure_mode: str) -> float:
        """Decoder latency charged for one decode call, per latency mode."""
        mode = self._latency_mode
        if mode == "measured":
            return float(measured_ns)
        if mode == "constant":
            return self.decoder_latency_ns if which == "complete" else self.partial_decoder_latency_ns
        if mode == "external_constant":
            return self._latency_const[which][measure_mode]
        if mode == "external_per_shot":
            return self._latency_table_lookup_ns(which, measure_mode)
        raise ValueError(f"unknown latency mode {mode!r}")

    def _partial_decode(
        self,
        det_set: set[int],
        measure_mode: str,
    ) -> tuple[float, float]:
        """Returns (gap, decode_ns). The partial decoder's prediction is
        discarded: it comes from a sparsified syndrome."""
        if self.partial_compiled is None:
            raise RuntimeError(
                "_partial_decode called but partial_compiled is None."
            )
        _obs, gap, measured_ns = self.partial_compiled.decode_det_set_with_time(
            det_set, measure_mode=measure_mode,
        )
        decode_ns = self._latency_ns("partial", measured_ns, measure_mode)
        return float(gap), float(decode_ns)

    def _complete_decode(
        self,
        det_set: set[int],
        measure_mode: str,
    ) -> tuple[bool, float, float]:
        """Returns (prediction, gap, decode_ns)."""
        obs, gap, measured_ns = self.complete_compiled.decode_det_set_with_time(
            det_set, measure_mode=measure_mode,
        )
        decode_ns = self._latency_ns("complete", measured_ns, measure_mode)
        return bool(obs), float(gap), float(decode_ns)

    # ------------------------------------------------------------------
    # 1.4  Attempt source
    # ------------------------------------------------------------------

    def attach_trace(self, trace_dir, gap_circuit: stim.Circuit, *,
                     master_seed: int = 0, start: int = 0, step: int = 1) -> None:
        """Replay the shot trace (attempt_stream) instead of simulating, so
        per-shot tables join on attempt index. The trace is keyed by the gap
        circuit: the original circuit plus the record-less padding DETECTORs
        of desaturation. Its first circuit.num_detectors bits are the original
        detectors. The trace is extended when exhausted. start=k, step=K gives
        worker k of K a disjoint partition."""
        from attempt_stream import AttemptTrace, ShotCursor
        if (gap_circuit.num_measurements != self.circuit.num_measurements
                or gap_circuit.num_observables != self.circuit.num_observables
                or gap_circuit.num_detectors < self.circuit.num_detectors):
            raise ValueError("gap_circuit is not the estimator's circuit plus padding detectors")
        self._trace = AttemptTrace.create_or_open(trace_dir, gap_circuit, master_seed)
        self._cursor = ShotCursor(self._trace, start, step)
        self.attempt_source = f"trace:{self._trace.meta.circuit_sha}:s{master_seed}"
        if step > 1:
            self.attempt_source += f":k{start}/{step}"

    def _next_attempt_bits(self) -> tuple[np.ndarray, int, Optional[int]]:
        """Returns (fired detector ids, observable bit, attempt index or None)
        from a fresh FlipSimulator pass or the attached trace. Always the
        whole circuit, with no early abort."""
        if self._cursor is None:
            sim = stim.FlipSimulator(batch_size=1, num_qubits=self.circuit.num_qubits)
            sim.do(self.circuit)
            fired = np.flatnonzero(sim.get_detector_flips()[:, 0])
            obs = int(sim.get_observable_flips()[0, 0]) if self.circuit.num_observables else 0
            self._last_row = None
            return fired, obs, None
        if self._cursor.next_index >= self._trace.meta.num_attempts:
            self._trace.ensure(self._cursor.next_index + 1)
        row, obs, idx = self._cursor.next()
        self._last_row = row
        bits = np.unpackbits(row, bitorder="little")[: self.circuit.num_detectors]
        return np.flatnonzero(bits), int(obs), idx

    def _run_single_attempt_until_postselect(
        self,
    ) -> tuple[set[int], Optional[int], Optional[int]]:
        """One attempt. Returns (det_fired, fail_stage, actual_obs): all fired
        detectors, the earliest stage (round) in which a postselected detector
        fired (None if the attempt passes), and the observable bit of a
        passing attempt (None otherwise)."""
        fired, obs, idx = self._next_attempt_bits()
        self._last_attempt_index = idx
        det_fired = {int(d) for d in fired}

        fail_stage: Optional[int] = None
        ps_fired = fired[self._postselect_mask[fired]]
        if len(ps_fired):
            fail_stage = int(self._det_stage_arr[ps_fired].min())

        actual_obs: Optional[int] = None
        if fail_stage is None and self.circuit.num_observables > 0:
            actual_obs = obs
        return det_fired, fail_stage, actual_obs

    # ------------------------------------------------------------------
    # 1.5  Circuit pre-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _find_postselected_detectors(circuit: stim.Circuit) -> set[int]:
        """Postselected detectors: those matching the coordinate-tag rules
        (color-code injection and cultivation), plus the B2 rule: both
        endpoints of every single-basis 2-detector DEM error that flips an
        observable (fault-distance-3 logical errors)."""
        result: set[int] = set()
        coords = circuit.get_detector_coordinates()
        for det, coord in coords.items():
            if (
                len(coord) == 3
                or coord[-1] == -9
                or (len(coord) > 4 and (coord[4] == 0 or coord[4] == 4))
            ):
                result.add(det)

        dem = circuit.detector_error_model().flattened()
        det_bases: dict[int, str] = {}
        for d in range(dem.num_detectors):
            c = coords.get(d, [])
            if len(c) <= 4 or c[4] == -9:
                det_bases[d] = "!"
            else:
                det_bases[d] = "XXXZZZXZ"[int(c[4])]
        for inst in dem:
            if inst.type != "error":
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
                if len({det_bases.get(d, "!") for d in det_set}) == 1:
                    result.update(det_set)
        return result

    @staticmethod
    def load_postselected_detectors(folder, circuit: Optional[stim.Circuit] = None) -> set[int]:
        """Read <folder>/postselected_detectors.json (original-circuit detector
        ids), stored as a dict or a plain list. With `circuit`, raises if a
        dict json was made for a different detector count."""
        doc = json.loads((pathlib.Path(folder) / "postselected_detectors.json").read_text())
        if isinstance(doc, dict):
            if circuit is not None and int(doc["n_circuit_dets"]) != circuit.num_detectors:
                raise ValueError(
                    f"postselected_detectors.json is for a circuit with "
                    f"{doc['n_circuit_dets']} detectors; got {circuit.num_detectors}"
                )
            ids = doc["postselected_detectors"]
        else:
            ids = doc
        return {int(d) for d in ids}

    @staticmethod
    def _stage_names_from_rounds(meas_counts: list[int], *, wait_rounds: int) -> list[str]:
        """Stage labels from the round sizes alone: "r<k> escape" for rounds
        with at least half the measurements of the largest round (the terminal
        MPP included), "r<k> cultivate" for the others, then (wait_rounds - 1)
        x "[wait for gap]" and "ready". Injection is not a stage: it is the
        unmeasured prefix of r0 (unitary) or rounds r0-r1 (degenerate)."""
        big = max(meas_counts)
        names = [f"r{r} {'escape' if m >= big / 2 else 'cultivate'}"
                 for r, m in enumerate(meas_counts)]
        names += ["[wait for gap]"] * (max(int(wait_rounds), 0) - 1)
        names.append("ready")
        return names

    @staticmethod
    def _default_gate_times_ns() -> dict[str, float]:
        # Google superconducting-hardware gate times, shared by algorithms/.
        # The control-system compiler keeps its own copy.
        from gate_times import GOOGLE_GATE_TIMES_NS
        return dict(GOOGLE_GATE_TIMES_NS)

    # ==================================================================
    # ================ SECTION 2: STATISTICS COLLECTION ================
    # ==================================================================

    # ------------------------------------------------------------------
    # 2.1  Statistics and raw export
    # ------------------------------------------------------------------

    def collect_statistics(self) -> dict[str, Any]:
        """Summarise a finished run into a dict, also stored on
        self.statistics: per-stage stats over epochs, per-shot and per-call
        scalar stats, tier counts, pass and accept rates, LER."""
        if not self._simulated:
            raise ValueError("Run estimate_runtime() first.")

        stats: dict[str, Any] = {}

        # Per-stage stats over epochs, shape (num_stages,). A stage never
        # reached is NaN in first_reach and 0 in until_success.
        stats["records_first_reach"] = {
            k: {
                "mean":   np.nanmean(arr, axis=0),
                "std":    np.nanstd(arr,  axis=0),
                "var":    np.nanvar(arr,  axis=0),
                "median": np.nanpercentile(arr, 50, axis=0),
                "p95":    np.nanpercentile(arr, 95, axis=0),
                "p99":    np.nanpercentile(arr, 99, axis=0),
                "min":    np.nanmin(arr, axis=0),
                "max":    np.nanmax(arr, axis=0),
            }
            for k, arr in self.records_first_reach.items()
        }
        stats["records_until_success"] = {
            k: {
                "mean":   np.mean(arr, axis=0),
                "std":    np.std(arr,  axis=0),
                "var":    np.var(arr,  axis=0),
                "median": np.percentile(arr, 50, axis=0),
                "p95":    np.percentile(arr, 95, axis=0),
                "p99":    np.percentile(arr, 99, axis=0),
                "min":    np.min(arr, axis=0),
                "max":    np.max(arr, axis=0),
            }
            for k, arr in self.records_until_success.items()
        }

        for name in (
            "accepted_decision_times_ns",
            "accepted_prediction_times_ns",
            "accepted_actual_obs",
            "accepted_predict_obs",
        ):
            stats[name] = self._scalar_stats(getattr(self, name))

        # Provenance of the shots, the postselected set and the latencies.
        stats["attempt_source"] = self.attempt_source
        stats["postselect_source"] = self.postselect_source
        stats["latency_source"] = self.latency_source
        stats["latency_provenance"] = self.latency_provenance
        stats["control_latency"] = self.control_latency
        # Coverage summary only; the full index list is in export_raw().
        idx = [i for i in self.accepted_attempt_indices if i is not None]
        stats["accepted_attempt_indices"] = (
            {"count": len(idx), "first": int(min(idx)), "last": int(max(idx))}
            if idx else None
        )

        # Per-call decoder times, empty unless record_decoder_call_times=True.
        for name in ("partial_decode_call_times_ns", "complete_decode_call_times_ns"):
            stats[name] = self._scalar_stats(getattr(self, name))

        # In PCP mode tier1_accept and tier3_reject are always zero.
        stats["tier_counts"] = {
            "tier1":         int(self.tier1_count),
            "tier1_accept":  int(self.tier1_accept_count),
            "tier1_reject":  int(self.tier1_reject_count),
            "tier2":         int(self.tier2_count),
            "tier2_accept":  int(self.tier2_accept_count),
            "tier2_reject":  int(self.tier2_reject_count),
            "tier3":         int(self.tier3_count),
            "tier3_accept":  int(self.tier3_accept_count),
            "tier3_reject":  int(self.tier3_reject_count),
        }

        n_attempts = max(int(self.attempt_count), 1)
        stage_pass_rate = self.stage_pass_counts.astype(np.float64) / n_attempts
        stats["attempt_count"]      = int(self.attempt_count)
        stats["stage_pass_counts"]  = self.stage_pass_counts.copy()
        stats["stage_pass_rate"]    = stage_pass_rate
        stats["stage_reject_rate"]  = 1.0 - stage_pass_rate

        n_accepted = len(self.accepted_actual_obs)
        n_rejected = int(self.attempt_count) - n_accepted
        stats["accept_count"] = n_accepted
        stats["reject_count"] = n_rejected
        stats["accept_rate"]  = n_accepted / n_attempts
        stats["reject_rate"]  = n_rejected / n_attempts

        stats["logical_error_rate"] = float(self.logical_error_rate)
        stats["epoch"]              = int(self.epoch)

        self.statistics = stats
        return stats

    def export_raw(self) -> dict[str, np.ndarray]:
        """Raw state of a finished run as flat numpy arrays (np.savez-able):
        records_{first_reach,until_success}_<kind> of shape (epoch, stage),
        per-accepted-shot scalars, attempt indices (-1 when simulating),
        per-call decode times and counters. Enough to recompute
        collect_statistics(); input of merge_raw.
        [used: run_runtime_estimation --save-raw and --workers]"""
        if not self._simulated:
            raise ValueError("Run estimate_runtime() first.")
        raw: dict[str, np.ndarray] = {}
        for k, arr in self.records_first_reach.items():
            raw[f"records_first_reach_{k}"] = arr.copy()
        for k, arr in self.records_until_success.items():
            raw[f"records_until_success_{k}"] = arr.copy()
        raw["accepted_decision_times_ns"] = np.asarray(self.accepted_decision_times_ns, np.float64)
        raw["accepted_prediction_times_ns"] = np.asarray(self.accepted_prediction_times_ns, np.float64)
        raw["accepted_actual_obs"] = np.asarray(self.accepted_actual_obs, np.uint8)
        raw["accepted_predict_obs"] = np.asarray(self.accepted_predict_obs, np.uint8)
        raw["accepted_attempt_indices"] = np.asarray(
            [-1 if i is None else int(i) for i in self.accepted_attempt_indices], np.int64)
        raw["partial_decode_call_times_ns"] = np.asarray(self.partial_decode_call_times_ns, np.float64)
        raw["complete_decode_call_times_ns"] = np.asarray(self.complete_decode_call_times_ns, np.float64)
        raw["stage_pass_counts"] = self.stage_pass_counts.copy()
        raw["tier_counts"] = np.asarray([
            self.tier1_count, self.tier1_accept_count, self.tier1_reject_count,
            self.tier2_count, self.tier2_accept_count, self.tier2_reject_count,
            self.tier3_count, self.tier3_accept_count, self.tier3_reject_count,
        ], np.int64)
        raw["attempt_count"] = np.asarray(self.attempt_count, np.int64)
        raw["stage_names"] = np.asarray(self.stage_names)
        return raw

    def merge_raw(self, raws: list[dict]) -> None:
        """Load the union of several export_raw() dicts as if this estimator
        had run all their epochs: records and per-shot lists are concatenated
        along the epoch axis, counters are summed. Statistics do not depend on
        shard order. Raises if a shard has a different stage layout.
        [used: run_runtime_estimation --workers]"""
        for r in raws:
            if [str(s) for s in r["stage_names"]] != self.stage_names:
                raise ValueError("shard stage layout differs from this estimator")

        def cat(key):
            return np.concatenate([np.asarray(r[key]) for r in raws])

        for k in self.records_first_reach:
            self.records_first_reach[k] = cat(f"records_first_reach_{k}")
            self.records_until_success[k] = cat(f"records_until_success_{k}")
        self.epoch = int(self.records_until_success["total"].shape[0])
        self.accepted_decision_times_ns = cat("accepted_decision_times_ns").tolist()
        self.accepted_prediction_times_ns = cat("accepted_prediction_times_ns").tolist()
        self.accepted_actual_obs = cat("accepted_actual_obs").astype(int).tolist()
        self.accepted_predict_obs = cat("accepted_predict_obs").astype(int).tolist()
        self.accepted_attempt_indices = [None if i < 0 else int(i)
                                         for i in cat("accepted_attempt_indices")]
        self.partial_decode_call_times_ns = cat("partial_decode_call_times_ns").tolist()
        self.complete_decode_call_times_ns = cat("complete_decode_call_times_ns").tolist()
        self.stage_pass_counts = sum(np.asarray(r["stage_pass_counts"]) for r in raws)
        (self.tier1_count, self.tier1_accept_count, self.tier1_reject_count,
         self.tier2_count, self.tier2_accept_count, self.tier2_reject_count,
         self.tier3_count, self.tier3_accept_count, self.tier3_reject_count,
         ) = (int(x) for x in sum(np.asarray(r["tier_counts"]) for r in raws))
        self.attempt_count = int(sum(int(r["attempt_count"]) for r in raws))
        errs = [a != p for a, p in zip(self.accepted_actual_obs, self.accepted_predict_obs)]
        self.logical_error_rate = float(np.mean(errs)) if errs else 0.0
        self._simulated = True
        self.statistics = None

    @staticmethod
    def _scalar_stats(seq) -> dict[str, float]:
        """count/mean/std/var/median/p95/p99/min/max of a 1-D sequence, as
        Python floats. All NaN, with count 0, for an empty input."""
        if not seq:
            return {
                "count":  0,
                "mean":   float("nan"),
                "std":    float("nan"),
                "var":    float("nan"),
                "median": float("nan"),
                "p95":    float("nan"),
                "p99":    float("nan"),
                "min":    float("nan"),
                "max":    float("nan"),
            }
        arr = np.asarray(seq, dtype=np.float64)
        return {
            "count":  int(arr.size),
            "mean":   float(arr.mean()),
            "std":    float(arr.std()),
            "var":    float(arr.var()),
            "median": float(np.percentile(arr, 50)),
            "p95":    float(np.percentile(arr, 95)),
            "p99":    float(np.percentile(arr, 99)),
            "min":    float(arr.min()),
            "max":    float(arr.max()),
        }

    # ------------------------------------------------------------------
    # 2.2  Headline summary
    # ------------------------------------------------------------------

    def print_statistics(self) -> None:
        """Print the headline summary. Collects the statistics if needed;
        also works on statistics loaded from JSON."""
        if getattr(self, "statistics", None) is None:
            if not self._simulated:
                raise ValueError(
                    "No statistics available. Call estimate_runtime() then "
                    "collect_statistics(), or load_statistics_from_json(...)."
                )
            self.collect_statistics()
        s = self.statistics

        ready_idx = self.ready_stage_idx
        avg_at_ready = {
            k: float(s["records_until_success"][k]["mean"][ready_idx])
            for k in ("total", "gate", "feedback", "gap_decode")
        }

        print("=== Runtime Statistics ===")
        print(f"  epoch (= accepted shots)      : {s['epoch']}")
        print(f"  total attempts                : {s['attempt_count']}")

        print(f"\n  Average execution time per epoch (records_until_success at 'ready'):")
        print(f"    total       : {avg_at_ready['total']:.1f} ns")
        print(f"    gate        : {avg_at_ready['gate']:.1f} ns")
        print(f"    feedback    : {avg_at_ready['feedback']:.1f} ns")
        print(f"    gap_decode  : {avg_at_ready['gap_decode']:.1f} ns")

        dec_s  = s["accepted_decision_times_ns"]
        pred_s = s["accepted_prediction_times_ns"]
        print(f"\n  Decision-ready time   : "
              f"mean={dec_s['mean']:.1f} ns, median={dec_s['median']:.1f}, "
              f"p95={dec_s['p95']:.1f}, p99={dec_s['p99']:.1f}, max={dec_s['max']:.1f}")
        print(f"  Prediction-ready time : "
              f"mean={pred_s['mean']:.1f} ns, median={pred_s['median']:.1f}, "
              f"p95={pred_s['p95']:.1f}, p99={pred_s['p99']:.1f}, max={pred_s['max']:.1f}")

        n_acc = s["accept_count"]
        n_rej = s["reject_count"]
        n_att = s["attempt_count"]
        print(f"\n  Accept rate : {n_acc}/{n_att} = {s['accept_rate']:.6f}")
        print(f"  Reject rate : {n_rej}/{n_att} = {s['reject_rate']:.6f}")

        print(f"\n  Logical error rate (LER) : {s['logical_error_rate']:.6f}")

    # ------------------------------------------------------------------
    # 2.3  Save statistics to JSON
    # ------------------------------------------------------------------

    def save_statistics_to_json(self, filepath) -> pathlib.Path:
        """Write self.statistics as JSON, collecting it first if needed.
        Creates parent directories. Returns the path written."""
        if getattr(self, "statistics", None) is None:
            if not self._simulated:
                raise ValueError(
                    "No statistics to save. Call estimate_runtime() first."
                )
            self.collect_statistics()
        path = pathlib.Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._stats_to_jsonable(self.statistics)
        with path.open("w") as f:
            json.dump(payload, f, indent=2)
        return path

    @staticmethod
    def _stats_to_jsonable(obj):
        """Recursively convert numpy arrays/scalars to JSON-serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): RuntimeEstimator._stats_to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [RuntimeEstimator._stats_to_jsonable(x) for x in obj]
        return obj

    # ------------------------------------------------------------------
    # 2.4  Load statistics from JSON
    # ------------------------------------------------------------------

    def load_statistics_from_json(self, filepath) -> dict[str, Any]:
        """Load a saved stats dict into self.statistics and return it. The
        per-stage stats and stage_pass_* become numpy arrays again. The raw
        per-shot lists are not restored and _simulated is not set: the result
        serves print_statistics() and the plots only."""
        path = pathlib.Path(filepath)
        with path.open("r") as f:
            stats = json.load(f)

        _record_stat_keys = ("mean", "std", "var", "median", "p95", "p99", "min", "max")
        for view_key in ("records_first_reach", "records_until_success"):
            if view_key not in stats:
                continue
            for sub in stats[view_key].values():
                for stat_key in _record_stat_keys:
                    if stat_key in sub:
                        sub[stat_key] = np.asarray(sub[stat_key], dtype=np.float64)

        if "stage_pass_counts" in stats:
            stats["stage_pass_counts"] = np.asarray(stats["stage_pass_counts"], dtype=np.int64)
        if "stage_pass_rate" in stats:
            stats["stage_pass_rate"] = np.asarray(stats["stage_pass_rate"], dtype=np.float64)
        if "stage_reject_rate" in stats:
            stats["stage_reject_rate"] = np.asarray(stats["stage_reject_rate"], dtype=np.float64)

        self.statistics = stats
        return stats

    # ==================================================================
    # ===================== SECTION 3: PLOTTING ========================
    # ==================================================================

    # ------------------------------------------------------------------
    # 3.0  Section colors and shared constants
    # ------------------------------------------------------------------
    _SECTION_COLORS: dict[str, str] = {
        "gate":       "#d62728",  # red
        "feedback":   "#9467bd",  # purple
        "gap_decode": "#2ca02c",  # green
        "total":      "#1b9e77",  # darker green outline
    }

    _RECORD_TYPES = ("until_success", "first_reach")
    _STAT_KEYS    = ("mean", "std", "var", "median", "p95", "p99", "min", "max")
    _SECTIONS     = ("total", "gate", "feedback", "gap_decode")

    # ------------------------------------------------------------------
    # 3.1  Plot helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _shift_to_stage_entry_with_ready(values: np.ndarray) -> np.ndarray:
        """Shift per-stage values so each stage shows the time at its start
        (the end of the previous stage). The last stage, `ready`, keeps its
        own accumulated value."""
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
        """Staircase (x, y) series: value i spans x=i to x=i+1."""
        xs: list[float] = []
        ys: list[float] = []
        for i, v in enumerate(values):
            xs.append(float(i))
            ys.append(float(v))
            xs.append(float(i + 1))
            ys.append(float(v))
        return np.array(xs), np.array(ys)

    def _ensure_stats_for_plot(self) -> dict[str, Any]:
        """Collect statistics if needed. Raises if nothing was run or loaded."""
        if getattr(self, "statistics", None) is None:
            if not self._simulated:
                raise ValueError(
                    "No statistics available. Call estimate_runtime() then "
                    "collect_statistics(), or load_statistics_from_json(...)."
                )
            self.collect_statistics()
        return self.statistics

    def _resolve_stage(self, stage: Any) -> int:
        """Stage index from an int index or a stage name."""
        if isinstance(stage, str):
            try:
                return self.stage_names.index(stage)
            except ValueError as ex:
                raise ValueError(
                    f"Stage name {stage!r} not found. Available: "
                    f"{list(dict.fromkeys(self.stage_names))}."
                ) from ex
        idx = int(stage)
        if not (0 <= idx < self.num_stages):
            raise ValueError(
                f"Stage index {idx} out of range [0, {self.num_stages})."
            )
        return idx

    # ------------------------------------------------------------------
    # 3.2  Lifetime plot
    # ------------------------------------------------------------------

    def plot_lifetime(
        self,
        *,
        record_type: str = "until_success",
        save_path: Optional[pathlib.Path] = None,
        ax=None,
    ):
        """Plot the survival proportion (left axis) and the stacked gate,
        feedback and gap_decode time in ns (right axis) over all stages.
        record_type is "until_success" (time including all failed retries) or
        "first_reach". Draws into `ax` if given and saves to `save_path` if
        given. Returns the Figure."""
        import matplotlib.pyplot as plt

        if record_type not in self._RECORD_TYPES:
            raise ValueError(
                f"record_type must be one of {self._RECORD_TYPES}; got {record_type!r}."
            )
        s = self._ensure_stats_for_plot()

        view = s["records_" + record_type]
        total_mean    = view["total"]["mean"]
        gate_mean     = view["gate"]["mean"]
        feedback_mean = view["feedback"]["mean"]
        decode_mean   = view["gap_decode"]["mean"]

        n_attempts = max(int(s["attempt_count"]), 1)
        surv_stage = np.asarray(s["stage_pass_counts"], dtype=np.float64) / n_attempts

        # Each stage is plotted at its entry time; `ready` keeps its own value.
        total_entry    = self._shift_to_stage_entry_with_ready(total_mean)
        gate_entry     = self._shift_to_stage_entry_with_ready(gate_mean)
        feedback_entry = self._shift_to_stage_entry_with_ready(feedback_mean)
        decode_entry   = self._shift_to_stage_entry_with_ready(decode_mean)

        gate_plus_feedback = gate_entry + feedback_entry
        total_from_components = gate_plus_feedback + decode_entry
        # Float drift can leave total below the sum of its components.
        total_entry = np.maximum(total_entry, total_from_components)

        x_s, y_s = self._make_interval_series(surv_stage)
        x_t, y_total = self._make_interval_series(total_entry)
        _,   y_gf    = self._make_interval_series(gate_plus_feedback)
        _,   y_g     = self._make_interval_series(gate_entry)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10.24, 5.12), dpi=100)
        else:
            fig = ax.figure

        # Left axis: survival proportion.
        line_surv, = ax.plot(x_s, y_s, label="Surviving Shots",
                             color="C0", linestyle="--", linewidth=2)
        ax.fill_between(x_s, 0, y_s, alpha=0.2, color="C0")
        ax.set_ylim(0, 1.01)
        ax.set_xlim(0, max(x_s) if len(x_s) else 1)
        ax.set_yticks([x * 0.1 for x in range(11)])
        ax.set_ylabel("Survival Proportion")

        # Right axis: execution time.
        ax_r = ax.twinx()
        line_gf, = ax_r.plot(x_t, y_gf, label="Gate+Feedback Runtime (ns)",
                             color="#7570b3", linewidth=2.8)
        line_g, = ax_r.plot(x_t, y_g, label="Gate-Only Runtime (ns)",
                            color=self._SECTION_COLORS["gate"], linewidth=2.8)
        ax_r.fill_between(x_t, 0, y_g, alpha=0.28,
                          color=self._SECTION_COLORS["gate"])
        ax_r.fill_between(x_t, y_g, y_gf, alpha=0.2,
                          color=self._SECTION_COLORS["feedback"])
        ax_r.fill_between(x_t, y_gf, y_total, alpha=0.2,
                          color=self._SECTION_COLORS["gap_decode"])
        line_total, = ax_r.plot(x_t, y_total, label="Total Runtime (ns)",
                                color=self._SECTION_COLORS["total"],
                                linewidth=3.1, zorder=5)
        ax_r.set_ylabel("Execution Time (ns)")

        # Stage labels sit on minor ticks, centred on each stage.
        n_stages = self.num_stages
        ax.set_xticks(range(n_stages + 1), [""] * (n_stages + 1))
        ax.set_xticks(
            [e + 0.5 for e in range(n_stages)],
            self.stage_names,
            rotation=90, minor=True,
        )
        ax.xaxis.set_tick_params(length=0, which="minor")
        ax.grid()

        handles = [
            line_surv, line_total, line_gf, line_g,
            plt.Rectangle((0, 0), 1, 1, fc=self._SECTION_COLORS["gate"], alpha=0.28),
            plt.Rectangle((0, 0), 1, 1, fc=self._SECTION_COLORS["feedback"], alpha=0.2),
            plt.Rectangle((0, 0), 1, 1, fc=self._SECTION_COLORS["gap_decode"], alpha=0.2),
        ]
        labels = [
            line_surv.get_label(), line_total.get_label(),
            line_gf.get_label(), line_g.get_label(),
            "Gate Execution Time (immutable)",
            "Accumulated Feedback Time",
            "Accumulated Decoder Time",
        ]
        ax.legend(handles, labels, loc="upper right")

        ax.set_title(
            f"Lifetime of a {'fault-distance-' + str(self.distance) + ' ' if self.distance else ''}"
            f"cultivation  "
            f"(epoch={s['epoch']}, record_type={record_type!r})"
        )
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path)
        return fig

    # ------------------------------------------------------------------
    # 3.3  Per-stage plot
    # ------------------------------------------------------------------

    def plot_stage_time(
        self,
        *,
        stage,
        record_type: str = "until_success",
        statistic: str = "mean",
        section: str = "total",
        save_path: Optional[pathlib.Path] = None,
        ax=None,
    ):
        """Plot one statistic of one section at a single stage, in ns.

        stage: int index in [0, num_stages) or a name in self.stage_names.
        record_type: "until_success" or "first_reach".
        statistic: one of _STAT_KEYS. section: one of _SECTIONS.

        section="total" with statistic="mean" draws a stacked bar of gate,
        feedback and gap_decode, which sum to the total mean. With another
        statistic it draws four side-by-side bars (the three parts and the
        total), because non-mean statistics are not additive. Any other
        section draws a single bar. Returns the Figure."""
        import matplotlib.pyplot as plt

        if record_type not in self._RECORD_TYPES:
            raise ValueError(
                f"record_type must be one of {self._RECORD_TYPES}; got {record_type!r}."
            )
        if statistic not in self._STAT_KEYS:
            raise ValueError(
                f"statistic must be one of {self._STAT_KEYS}; got {statistic!r}."
            )
        if section not in self._SECTIONS:
            raise ValueError(
                f"section must be one of {self._SECTIONS}; got {section!r}."
            )
        s = self._ensure_stats_for_plot()
        stage_idx = self._resolve_stage(stage)
        view = s["records_" + record_type]

        if ax is None:
            fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=100)
        else:
            fig = ax.figure

        stage_name = self.stage_names[stage_idx]

        if section == "total" and statistic == "mean":
            gate_v = float(view["gate"]["mean"][stage_idx])
            fb_v   = float(view["feedback"]["mean"][stage_idx])
            dec_v  = float(view["gap_decode"]["mean"][stage_idx])

            ax.bar(0, gate_v,
                   color=self._SECTION_COLORS["gate"], label="gate")
            ax.bar(0, fb_v, bottom=gate_v,
                   color=self._SECTION_COLORS["feedback"], label="feedback")
            ax.bar(0, dec_v, bottom=gate_v + fb_v,
                   color=self._SECTION_COLORS["gap_decode"], label="gap_decode")
            ax.set_xticks([0], [stage_name])
            ax.legend(loc="upper right")

        elif section == "total":
            sections = ["gate", "feedback", "gap_decode", "total"]
            values = [float(view[sec][statistic][stage_idx]) for sec in sections]
            colors = [self._SECTION_COLORS[sec] for sec in sections]
            xs = np.arange(len(sections))
            ax.bar(xs, values, color=colors)
            ax.set_xticks(xs, sections)
            ax.set_xlabel(f"section (at stage {stage_name!r})")

        else:
            val = float(view[section][statistic][stage_idx])
            ax.bar(0, val, color=self._SECTION_COLORS[section], label=section)
            ax.set_xticks([0], [stage_name])
            ax.legend(loc="upper right")

        ax.set_ylabel(f"{statistic} time (ns)")
        ax.set_title(
            f"Stage {stage_name!r}  ({record_type}, section={section}, "
            f"statistic={statistic})"
        )
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path)
        return fig
