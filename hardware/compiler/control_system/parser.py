"""parser.py — single-pass scanner for the detector-construction compiler.

Scans a *flattened* stim circuit once, in order, building the raw intermediate
structures (leading underscore) that later stages turn into the IR.

Structures
----------
_physical_qubits_coord_raw[q]           = (x, y)
_measurement_rec_raw_flat[i]            = (qubit, time)     absolute record list
_measurement_rec_raw[t]                 = [(qubit, t), …]   same, grouped by time
_logical_observable_terms               = [[(qubit, pauli), …], …]
_detector_coord_raw[d]                  = [(x, y), new_t, old_t, remaining]
_detector_to_measurement_raw[new_t]     = {det_idx: {meas_time: [qubit, …]}}
_measurement_to_detector_raw[meas_time] = {qubit: {det_new_t: [det_idx, …]}}
_stages                                 = [[det_idx, …], …]   (split at old_t gaps)

`time` for measurements is an ordering index: +1 each time a new group of
measurements appears after a TICK (multiple TICKs count once).  Independent of
the detector time.

Terminal MPP  (assumed the final measurement instruction)
---------------------------------------------------------
Format `MPP obs_0 … obs_{m-1}  stab_0 … stab_{n-1}` (observables first, n = the
number of stabilizer terms = ``num_stab``).  The MPP records and the terminal
DETECTOR annotations are NOT copied from the previous round; instead we keep the
circuit's own detector indexing (so ``det_idx == stim detector index``) and only
recover the physical qubit each MPP record corresponds to:

  * in the MPP branch we occupy one *empty* placeholder slot in the two
    measurement raw lists for EVERY MPP record — the ``m`` observables first,
    then the ``n`` stabilizers — so that negative-rec indexing stays aligned
    with stim.  The qubits are unknown here (``None``) and filled later.  We
    assert ``n <= (#measurements in the previous round)``.

  * after the scan, ``_process_terminal_detectors`` walks the terminal DETECTOR
    annotations in emission order (keeping their float x,y coords verbatim) and
    fills the one empty MPP slot each detector references:
      - >=2 recs (normal weight-2 / composite weight-3): copy the qubit of the
        previous-round record listed right before the MPP term;
      - 1 rec (boundary): match by nearest (x, y) to a single-qubit detector
        among the LAST ``num_tdet`` detectors of the previous round.
    We assert ``num_tdet <= (#previous-round detectors)`` and that each terminal
    detector references exactly one MPP record.

num_stab and num_tdet are treated as independent quantities.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import stim

PauliFactor = Tuple[int, str]


class DetectorConstructionScanner:
    def __init__(self, circuit: stim.Circuit, *, num_observable: Optional[int] = None):
        self.raw_circuit = circuit
        self.num_observable = num_observable   # None -> auto-detect leading mixed-basis products

        self._physical_qubits_coord_raw: List[Optional[Tuple[float, float]]] = []
        self._measurement_rec_raw_flat: List[Tuple[Optional[int], int]] = []
        self._measurement_rec_raw: List[List[Tuple[Optional[int], int]]] = []
        self._logical_observable_terms: List[List[PauliFactor]] = []
        self._detector_coord_raw: List[list] = []
        self._detector_to_measurement_raw: List[dict] = []
        self._measurement_to_detector_raw: List[dict] = []
        self._stages: List[List[int]] = [[]]
        self._terminal_meas_time: Optional[int] = None
        self._terminal_det_time: Optional[int] = None
        self._last_round_det_time: Optional[int] = None
        self._num_stab: int = 0
        self._num_mpp_rec: int = 0                       # m + n MPP records
        self._terminal_dets_raw: List[tuple] = []        # (xy, old_t, remaining, recvals)
        self._logical_observable_recs: List[list] = []   # OBSERVABLE_INCLUDE -> [(qubit, t), …]

        # refined (public) structures, produced by refine()
        self.detector_coord: List[list] = []             # [new_xy, old_xy, new_t, old_t, remaining]
        self.measurement_rec: List[List[Tuple[int, int]]] = []   # observable slots removed
        self.detector_to_measurement: List[dict] = []
        self.measurement_to_detector: List[dict] = []
        self.channels: dict = {}                         # new_xy -> [det_idx ordered by new_t]
        self.channel_support: dict = {}                  # new_xy -> sorted [qubit]
        self.detector_window: List[Tuple[int, int]] = [] # det_idx -> (min_meas_time, max_meas_time)
        self.stages: List[dict] = []                     # [{detectors, det_time_range}]

    @classmethod
    def from_file(cls, path: str, *, num_observable: Optional[int] = None) -> "DetectorConstructionScanner":
        return cls(stim.Circuit.from_file(path), num_observable=num_observable)

    # ------------------------------------------------------------------ #
    def scan(self) -> None:
        circ = self.raw_circuit.flattened()

        coords: dict = {}
        mrr_flat = self._measurement_rec_raw_flat
        mrr = self._measurement_rec_raw
        obs_terms = self._logical_observable_terms
        det_coord = self._detector_coord_raw
        d2m = self._detector_to_measurement_raw
        m2d = self._measurement_to_detector_raw
        stages = self._stages
        terminal_dets = self._terminal_dets_raw

        meas_t = -1
        tick_since = False
        terminal_mpp = False
        old_to_new_t: dict = {}
        prev_old_t: Optional[float] = None
        cur_det_time = -1

        def grow(lst, i):
            while len(lst) <= i:
                lst.append({})

        for inst in circ:
            name = inst.name

            if name == "QUBIT_COORDS":
                a = inst.gate_args_copy()
                for t in inst.targets_copy():
                    coords[t.value] = (a[0], a[1])

            elif name == "TICK":
                tick_since = True

            elif name == "MPP":
                if terminal_mpp:
                    raise ValueError("only one terminal MPP is supported")
                products = self._split_mpp_products(inst)
                m = self.num_observable if self.num_observable is not None \
                    else self._count_leading_mixed(products)
                n = len(products) - m
                last_round = mrr[-1] if mrr else []
                if n > len(last_round):
                    raise ValueError(
                        f"terminal MPP has {n} stabilizers but previous layer has "
                        f"only {len(last_round)} measurements")
                for prod in products[:m]:                    # observables -> logical only
                    obs_terms.append(prod)

                self._num_stab = n
                self._num_mpp_rec = m + n
                self._last_round_det_time = cur_det_time

                # occupy one EMPTY placeholder slot per MPP record (observables
                # first, then stabilizers) so negative-rec indexing stays aligned
                # with stim.  qubits are unknown now; filled from the terminal
                # DETECTORs.  slots are mutable lists shared by mrr and mrr_flat.
                meas_t += 1
                tick_since = False
                self._terminal_meas_time = meas_t
                new_layer = [[None, meas_t] for _ in range(m + n)]
                mrr.append(new_layer)
                mrr_flat.extend(new_layer)
                terminal_mpp = True

            elif name[0] == "M":                             # normal measurement
                if terminal_mpp:
                    raise ValueError("measurement after terminal MPP")
                if tick_since or not mrr:
                    meas_t += 1; tick_since = False; mrr.append([])
                for t in inst.targets_copy():
                    if t.is_combiner:
                        continue
                    mrr[-1].append((t.value, meas_t))
                    mrr_flat.append((t.value, meas_t))

            elif name == "OBSERVABLE_INCLUDE":
                recs = [mrr_flat[t.value] for t in inst.targets_copy()
                        if t.is_measurement_record_target]
                self._logical_observable_recs.append(list(recs))

            elif name == "DETECTOR":
                a = inst.gate_args_copy()
                if len(a) < 3:
                    raise ValueError(f"DETECTOR needs >=3 coords, got {a}")
                xy = (a[0], a[1]); old_t = a[2]; remaining = tuple(a[3:])
                if terminal_mpp:
                    recvals = [t.value for t in inst.targets_copy()
                               if t.is_measurement_record_target]
                    terminal_dets.append((xy, old_t, remaining, recvals))
                    continue
                if old_t not in old_to_new_t:
                    if prev_old_t is not None and old_t > prev_old_t + 1:
                        stages.append([])                    # gap -> new stage
                    old_to_new_t[old_t] = len(old_to_new_t)
                    prev_old_t = old_t
                new_t = old_to_new_t[old_t]
                cur_det_time = new_t
                det_idx = len(det_coord)
                det_coord.append([xy, new_t, old_t, remaining])
                stages[-1].append(det_idx)
                grow(d2m, new_t)
                d2m[new_t].setdefault(det_idx, {})
                for tgt in inst.targets_copy():
                    if not tgt.is_measurement_record_target:
                        continue
                    q, mt = mrr_flat[tgt.value]
                    d2m[new_t][det_idx].setdefault(mt, []).append(q)
                    grow(m2d, mt)
                    m2d[mt].setdefault(q, {}).setdefault(new_t, []).append(det_idx)

        # --- resolve the terminal detectors (at the end of the scan) ---
        if terminal_mpp:
            self._process_terminal_detectors(grow)

        n_q = (max(coords) + 1) if coords else 0
        self._physical_qubits_coord_raw = [coords.get(q) for q in range(n_q)]

    # ------------------------------------------------------------------ #
    def _process_terminal_detectors(self, grow) -> None:
        """Resolve the qubit each terminal MPP record measures.

        A terminal detector is the same stabilizer as some previous-round
        detector, one round later — but the ancilla may *move* between rounds
        (diagonal detectors), so the terminal record is NOT necessarily on the
        same qubit as the record before it.  We recover the true qubit by
        matching the terminal detector's *shape* — its records as
        (qubit, meas_time - min_meas_time) — against the previous round: the
        unique previous-round detector whose shape equals the terminal's shape
        with the MPP record as a wildcard supplies that record's qubit.
        """
        d2m = self._detector_to_measurement_raw
        m2d = self._measurement_to_detector_raw
        det_coord = self._detector_coord_raw
        mrr = self._measurement_rec_raw
        mrr_flat = self._measurement_rec_raw_flat
        stages = self._stages
        terminal_dets = self._terminal_dets_raw
        n_mpp = self._num_mpp_rec
        last_t = self._last_round_det_time
        term_t = last_t + 1
        self._terminal_det_time = term_t

        prev_layer = d2m[last_t]
        num_tdet = len(terminal_dets)
        assert num_tdet <= len(prev_layer), (
            f"#terminal DETECTOR annotations ({num_tdet}) > #previous-round "
            f"detectors ({len(prev_layer)})")

        # --- index the previous round by shape (multi-rec) or by remaining
        #     coords (single-rec), so a terminal detector matches exactly one ---
        def norm_recs(adj):
            t0 = min(adj)
            return frozenset((q, mt - t0) for mt, qs in adj.items() for q in qs), t0

        # every previous-round detector is an eligible template (r2=3 circuits
        # only ever needed the last num_tdet — the dropped ones sat at the front
        # and were single-rec — but in r2=0 circuits a front detector can be a
        # real template; the shape/tag indices below keep matches unique).
        prev_ids = list(prev_layer.keys())

        shape_index = {}            # (shape_without_top, top_pos, remaining) -> qubit
        shape_index_notag = {}      # (shape_without_top, top_pos) -> qubit | None (ambiguous)
        single_index = {}           # remaining -> [(coord, qubit)]   [r == 1 prev dets]
        for pid in prev_ids:
            adj = prev_layer[pid]
            remaining = tuple(det_coord[pid][3])
            recs = [(q, mt) for mt, qs in adj.items() for q in qs]
            if len(recs) == 1:
                single_index.setdefault(remaining, []).append((det_coord[pid][0], recs[0][0]))
                continue
            norm, t0 = norm_recs(adj)
            top_pos = max(p for _q, p in norm)
            tops = [q for q, p in norm if p == top_pos]
            if len(tops) != 1:
                # r2=0 circuits end on the last escape round, where a few boundary
                # detectors carry TWO records at their top time (e.g. d3d15 r2=0
                # dets 583/590). A terminal detector replaces exactly one top
                # record, so such detectors can never be terminal templates:
                # leave them out of the index (an unmatched terminal detector
                # still trips the `key in shape_index` assertion below).
                continue
            top_q = tops[0]
            key = (frozenset(r for r in norm if r != (top_q, top_pos)), top_pos, remaining)
            assert key not in shape_index, \
                f"previous-round detectors collide on shape {key}"
            shape_index[key] = top_q
            # tag-free fallback index (see the lookup below); ambiguous -> None
            nk = key[:2]
            shape_index_notag[nk] = None if nk in shape_index_notag else top_q

        grow(d2m, term_t)
        last_mt = self._terminal_meas_time - 1

        def placeholder_qubit(xy, remaining, prev, exclude):
            """Qubit label for a terminal detector's MPP placeholder record.
            `prev` = its real (qubit, meas_time) records; `exclude` = labels already
            taken at the terminal round (labels must be unique per meas-time)."""
            if prev and all(mt == last_mt for _q, mt in prev):
                # The terminal detector compares the MPP product with the SAME
                # stabilizer's ancilla measurement(s) in the last real round: the
                # placeholder qubit is that ancilla (in every shape-matched r2=3
                # case the template's top qubit is exactly this record's qubit).
                # This also covers r2=0 circuits, whose last real round is a
                # growth round with no shape-shifted template; a merged stabilizer
                # (two last-round records) takes the lowest ancilla deterministically.
                return min(q for q, _mt in prev)
            if prev:
                # shape match against the previous round
                t0 = min(mt for _q, mt in prev)
                mpp_pos = self._terminal_meas_time - t0
                real_norm = frozenset((q, mt - t0) for q, mt in prev)
                key = (real_norm, mpp_pos, remaining)
                if key in shape_index:
                    return shape_index[key]
                # r2=0 circuits: the round before the MPP is the last ESCAPE
                # round, whose few boundary detectors carry different trailing
                # coordinate tags (e.g. (3,3)) than their terminal counterparts
                # ((3,7)). Fall back to the tag-free shape, which must be unique.
                qfill = shape_index_notag.get(key[:2])
                assert qfill is not None, (
                    f"terminal detector at {xy} (shape {real_norm}, top {mpp_pos}, "
                    f"coords {remaining}) has no unique previous-round match")
                return qfill
            # single-rec boundary detector: no measurement to link on, so
            # match by nearest coordinate among same-`remaining` single-rec
            # previous-round detectors (the one geometric fallback), skipping
            # labels already taken by other terminals at the terminal round.
            cands = [c for c in single_index.get(remaining, []) if c[1] not in exclude]
            assert cands, (
                f"1-rec terminal detector at {xy} coords {remaining}: no "
                f"single-rec previous-round detector to match")
            ranked = sorted(cands, key=lambda c: (c[0][0] - xy[0]) ** 2
                                                + (c[0][1] - xy[1]) ** 2)
            if len(ranked) > 1:
                d0 = (ranked[0][0][0] - xy[0]) ** 2 + (ranked[0][0][1] - xy[1]) ** 2
                d1 = (ranked[1][0][0] - xy[0]) ** 2 + (ranked[1][0][1] - xy[1]) ** 2
                assert d0 < d1, (
                    f"1-rec terminal detector at {xy}: nearest single-rec "
                    f"previous-round detector is a tie (dist^2={d0})")
            return ranked[0][1]

        # Placeholder labels in two passes: terminals with real records first
        # (their label is fixed by their own ancilla), then pure-MPP boundary
        # terminals, which must avoid the labels already taken.
        parsed = []
        for (xy, old_t, remaining, recvals) in terminal_dets:
            remaining = tuple(remaining)
            # MPP records are the last n_mpp records => rec value in [-n_mpp, -1]
            mpp_recs = [rv for rv in recvals if rv >= -n_mpp]
            assert len(mpp_recs) == 1, (
                f"terminal detector at {xy} references {len(mpp_recs)} MPP "
                f"records (expected exactly 1); recvals={recvals}")
            prev = [mrr_flat[rv] for rv in recvals if rv < -n_mpp]   # (qubit, meas_time)
            parsed.append((xy, old_t, remaining, mpp_recs[0], prev, recvals))
        fills = [None] * len(parsed)
        used = set()
        for ti, (xy, _ot, remaining, _rv, prev, _rc) in enumerate(parsed):
            if prev:
                fills[ti] = placeholder_qubit(xy, remaining, prev, used)
                assert fills[ti] not in used, (
                    f"terminal detector at {xy}: placeholder qubit {fills[ti]} already used")
                used.add(fills[ti])
        for ti, (xy, _ot, remaining, _rv, prev, _rc) in enumerate(parsed):
            if not prev:
                fills[ti] = placeholder_qubit(xy, remaining, prev, used)
                used.add(fills[ti])

        for ti, (xy, old_t, remaining, rv_e, prev, recvals) in enumerate(parsed):
            qfill = fills[ti]
            assert qfill is not None

            # fill the shared placeholder slot (updates mrr and mrr_flat at once)
            mrr_flat[rv_e][0] = qfill

            # register the terminal detector; det_idx continues stim's order
            det_idx = len(det_coord)
            det_coord.append([xy, term_t, old_t, remaining])
            stages[-1].append(det_idx)
            d2m[term_t].setdefault(det_idx, {})
            for rv in recvals:
                q, mt = mrr_flat[rv]
                d2m[term_t][det_idx].setdefault(mt, []).append(q)
                grow(m2d, mt)
                m2d[mt].setdefault(q, {}).setdefault(term_t, []).append(det_idx)

        # normalize the terminal measurement slots back to tuples
        mrr[-1] = [tuple(s) for s in mrr[-1]]
        for i in range(len(mrr_flat) - n_mpp, len(mrr_flat)):
            mrr_flat[i] = tuple(mrr_flat[i])

    # ------------------------------------------------------------------ #
    def refine(self) -> None:
        """Build the public compiler-facing structures from the raw ones.

        * detector_coord  — canonicalise the spatial coordinate so detectors on
          the same qubit support share it (fixes the float terminal coords),
          keeping the original coord alongside.
        * measurement_rec — drop the logical-observable placeholder slots.
        * detector_to_measurement / measurement_to_detector — copied verbatim
          (already clean: terminal slots were filled with real qubits).
        * channels / channel_support / detector_window — precomputed helpers.
        """
        d2m = self._detector_to_measurement_raw
        m2d = self._measurement_to_detector_raw
        raw = self._detector_coord_raw

        def signature(det_idx, det_t):
            return frozenset(q for qs in d2m[det_t][det_idx].values() for q in qs)

        # --- 1) canonical spatial coordinate (only terminal detectors change) ---
        sig_to_xy = {}
        if self._last_round_det_time is not None:
            for did in d2m[self._last_round_det_time]:
                sig = signature(did, self._last_round_det_time)
                assert sig not in sig_to_xy, \
                    f"previous-round detectors share a qubit signature: {sig}"
                sig_to_xy[sig] = raw[did][0]

        term_t = self._terminal_det_time

        def terminal_channel(did, xy):
            """Channel a terminal detector continues: the previous-round detector
            with the same qubit signature (r2>0: always exists). In r2=0 circuits
            the last real round is a GROWTH round, so a terminal may only have a
            strict-superset match (e.g. {10} -> {4,10}); take the unique smallest
            one, nearest coordinate on ties. Returns None when nothing matches."""
            sig = signature(did, term_t)
            if sig in sig_to_xy:
                return sig_to_xy[sig]
            supers = [s for s in sig_to_xy if sig < s]
            if not supers:
                return None
            kmin = min(len(s) for s in supers)
            ranked = sorted((((sig_to_xy[s][0] - xy[0]) ** 2 + (sig_to_xy[s][1] - xy[1]) ** 2), s)
                            for s in supers if len(s) == kmin)
            if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
                return None
            return sig_to_xy[ranked[0][1]]

        # Terminal channel assignment, two passes: a channel may carry at most one
        # detector per det-time, so terminals whose inherited channel is claimed by
        # another terminal too (two stabilizers merged by one growth detector, e.g.
        # {7} and {13} both -> {7,13}) keep their own coordinate as a fresh channel.
        assign = {}                     # terminal did -> channel coordinate
        if term_t is not None:
            for did, (xy, new_t, _ot, _rem) in enumerate(raw):
                if new_t == term_t:
                    c = terminal_channel(did, xy)
                    assign[did] = xy if c is None else c
            while True:                 # resolve contested channels to a fixed point
                holders = {}
                for did, c in assign.items():
                    holders.setdefault(c, []).append(did)
                contested = {c: ds for c, ds in holders.items() if len(ds) > 1}
                if not contested:
                    break
                for c, ds in contested.items():
                    # the terminal whose own coordinate IS this channel keeps it
                    # (none or one such); every other claimant falls back to its
                    # own coordinate (unique among terminals).
                    for did in ds:
                        if raw[did][0] != c:
                            assign[did] = raw[did][0]
        self.detector_coord = []
        for did, (xy, new_t, old_t, remaining) in enumerate(raw):
            new_xy = assign.get(did, xy)
            self.detector_coord.append([new_xy, xy, new_t, old_t, remaining])

        # --- 2) measurement_rec without the observable placeholder slots ---
        self.measurement_rec = [[s for s in layer if s[0] is not None]
                                for layer in self._measurement_rec_raw]

        # --- 3) adjacency lists are already clean; copy verbatim ---
        self.detector_to_measurement = [dict(layer) for layer in d2m]
        self.measurement_to_detector = [dict(layer) for layer in m2d]

        # --- 4) channels, per-channel qubit support, per-detector window ---
        self.channels = {}
        self.channel_support = {}
        self.detector_window = [None] * len(self.detector_coord)
        for did, (new_xy, _oxy, new_t, _ot, _rem) in enumerate(self.detector_coord):
            adj = d2m[new_t][did]
            self.detector_window[did] = (min(adj), max(adj))
            self.channels.setdefault(new_xy, []).append((new_t, did))
            sup = self.channel_support.setdefault(new_xy, set())
            for qs in adj.values():
                sup.update(qs)
        # order each channel by detector time; freeze supports to sorted lists
        self.channels = {xy: [did for _t, did in sorted(v)]
                         for xy, v in self.channels.items()}
        self.channel_support = {xy: sorted(s) for xy, s in self.channel_support.items()}

        # --- 5) stages: detectors grouped by stage, with detector-time range ---
        #     (postselect/matchable split is left to the compiler + its rule)
        self.stages = []
        for det_ids in self._stages:
            if not det_ids:
                continue
            times = [self.detector_coord[d][2] for d in det_ids]
            self.stages.append({
                "detectors": list(det_ids),
                "det_time_range": (min(times), max(times)),
            })

    # ------------------------------------------------------------------ #
    def dump_refined(self, path: str) -> None:
        """Write the refined structures to `path` for manual inspection."""

        def fmt(v):
            return int(v) if float(v).is_integer() else v

        def xy_str(xy):
            return f"({fmt(xy[0])}, {fmt(xy[1])})"

        lines: List[str] = []

        # detector_coord: new vs old spatial coordinate
        lines.append("# DETECTOR_COORD  new_xy | old_xy | new_t old_t (remaining)  det_idx")
        for d, (new_xy, old_xy, new_t, old_t, rem) in enumerate(self.detector_coord):
            tag = "" if new_xy == old_xy else "   <-- remapped"
            rems = ", ".join(str(fmt(r)) for r in rem)
            lines.append(f"DETECTOR_COORD {xy_str(new_xy)} | {xy_str(old_xy)} | "
                         f"{new_t} {old_t} ({rems}) {d}{tag}")
        lines.append("")

        # channels: spatial coord -> ordered detectors + selector support
        lines.append("# CHANNEL  new_xy  support=[qubits]  detectors=[det_idx by time]")
        for xy in sorted(self.channels, key=lambda p: (p[0], p[1])):
            sup = self.channel_support[xy]
            dets = self.channels[xy]
            lines.append(f"CHANNEL {xy_str(xy)}  n={len(sup)}  support={sup}")
            lines.append(f"        detectors={dets}")
        lines.append("")

        # detector_window: (min_meas_time, max_meas_time) and span
        lines.append("# DETECTOR_WINDOW  det_idx: (min_meas_time, max_meas_time)  span")
        for d, (lo, hi) in enumerate(self.detector_window):
            lines.append(f"DETECTOR_WINDOW {d}: ({lo}, {hi})  span={hi - lo + 1}")
        lines.append("")

        # measurement_rec (observable slots removed)
        lines.append("# MEASUREMENT_REC(t)  qubits (observable slots removed)")
        for t, layer in enumerate(self.measurement_rec):
            lines.append(f"MEASUREMENT_REC({t}) " + " ".join(str(q) for q, _ in layer))

        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------ #
    def dump_raw(self, path: str) -> None:
        """Write every raw structure to `path` in a human-checkable format."""

        def fmt(v):                                  # 0.0 -> 0, 3.5 -> 3.5
            return int(v) if float(v).is_integer() else v

        def coord_str(xy, new_t, old_t, remaining):
            rem = ", ".join(str(fmt(r)) for r in remaining)
            return f"(({fmt(xy[0])}, {fmt(xy[1])}), {new_t}, {old_t}, ({rem}))"

        def prod_str(term):                          # [(q,pauli),...] -> "X77*Z93"
            return "*".join(f"{p}{q}" for q, p in term)

        lines: List[str] = []

        # 1) physical qubit coordinates (stim QUBIT_COORDS format)
        for q, xy in enumerate(self._physical_qubits_coord_raw):
            if xy is not None:
                lines.append(f"QUBIT_COORDS({fmt(xy[0])}, {fmt(xy[1])}) {q}")
        lines.append("")

        # 2) measurement coordinates, grouped by measurement time
        m_obs = self._num_mpp_rec - self._num_stab   # #leading observable records
        term_mt = self._terminal_meas_time
        for t, layer in enumerate(self._measurement_rec_raw):
            toks = []
            for j, slot in enumerate(layer):
                if t == term_mt and j < m_obs:       # logical observable record
                    toks.append(f"l{j}|{prod_str(self._logical_observable_terms[j])}")
                else:
                    toks.append(str(slot[0]))
            lines.append(f"MEASUREMENT_COORDS({t}) " + " ".join(toks))
        lines.append("")

        # 3) detector coordinates (QUBIT_COORDS-like)
        for d, (xy, new_t, old_t, remaining) in enumerate(self._detector_coord_raw):
            lines.append(f"DETECTOR_COORDS{coord_str(xy, new_t, old_t, remaining)} {d}")
        lines.append("")

        # 4) detector -> measurement, grouped by detector time, one line per meas time
        d2m = self._detector_to_measurement_raw
        for t in range(len(d2m)):
            layer = d2m[t]
            if not layer:
                continue
            for det_idx, by in layer.items():
                xy, new_t, old_t, remaining = self._detector_coord_raw[det_idx]
                header = f"DETECTOR{coord_str(xy, new_t, old_t, remaining)} "
                for k, mt in enumerate(sorted(by)):
                    group = " ".join(f"({q}, {mt})" for q in by[mt])
                    lines.append((header if k == 0 else " " * len(header)) + group)
            lines.append("")

        # 5) measurement -> detector, grouped by measurement time, one line per det time
        m2d = self._measurement_to_detector_raw
        for mt in range(len(m2d)):
            layer = m2d[mt]
            if not layer:
                continue
            for q in sorted(layer):
                by = layer[q]
                header = f"MEASUREMENT({q}, {mt}) "
                for k, dt in enumerate(sorted(by)):
                    group = " ".join(f"({d}, {dt})" for d in by[dt])
                    lines.append((header if k == 0 else " " * len(header)) + group)
            lines.append("")

        # 6) stages
        for i, st in enumerate(self._stages):
            lines.append(f"STAGE({i}) " + ", ".join(str(d) for d in st))
        lines.append("")

        # 7) logical observables — Pauli products (from MPP) or measurement
        #    records (from OBSERVABLE_INCLUDE), whichever the circuit provides
        if self._logical_observable_terms:
            for i, term in enumerate(self._logical_observable_terms):
                lines.append(f"LOGICAL_OBSERVABLE({i}) {prod_str(term)}")
        else:
            for i, recs in enumerate(self._logical_observable_recs):
                toks = " ".join(f"({q}, {t})" for (q, t) in recs)
                lines.append(f"LOGICAL_OBSERVABLE({i}) {toks}")

        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pauli_char(t: stim.GateTarget) -> str:
        if t.is_x_target: return "X"
        if t.is_y_target: return "Y"
        if t.is_z_target: return "Z"
        raise ValueError(f"non-Pauli MPP target: {t!r}")

    @classmethod
    def _split_mpp_products(cls, inst: stim.CircuitInstruction) -> List[List[PauliFactor]]:
        toks = inst.targets_copy()
        products: List[List[PauliFactor]] = []
        i, n = 0, len(toks)
        while i < n:
            prod = [(toks[i].value, cls._pauli_char(toks[i]))]; i += 1
            while i < n and toks[i].is_combiner:
                i += 1; prod.append((toks[i].value, cls._pauli_char(toks[i]))); i += 1
            products.append(prod)
        return products

    @staticmethod
    def _count_leading_mixed(products: List[List[PauliFactor]]) -> int:
        m = 0
        for p in products:
            if len({pauli for _, pauli in p}) > 1:
                m += 1
            else:
                break
        return m


# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <circuit.stim>")
    path = sys.argv[1]

    s = DetectorConstructionScanner.from_file(path)
    s.scan()
    circ = stim.Circuit.from_file(path).flattened()

    print(f"input: {path.split('/')[-1]}")
    n_obs = len(s._logical_observable_terms) or len(s._logical_observable_recs)
    print(f"  qubits                  : {len(s._physical_qubits_coord_raw)}")
    print(f"  measurement records     : {len(s._measurement_rec_raw_flat)}")
    print(f"  measurement layers      : {len(s._measurement_rec_raw)}")
    print(f"  detectors               : {len(s._detector_coord_raw)}")
    print(f"  detector time layers    : {len(s._detector_to_measurement_raw)}")
    print(f"  logical observables     : {n_obs}")
    print(f"  stages                  : {len(s._stages)}  sizes={[len(st) for st in s._stages]}")
    print(f"  num_stab={s._num_stab}  num_mpp_rec={s._num_mpp_rec}")
    print(f"  terminal meas_time={s._terminal_meas_time}  det_time={s._terminal_det_time}")

    # 1) detector index consistency with stim (each stim detector -> one entry, in order)
    assert len(s._detector_coord_raw) == circ.num_detectors, \
        (len(s._detector_coord_raw), circ.num_detectors)
    print(f"\n  detector count == stim num_detectors: "
          f"{len(s._detector_coord_raw)} == {circ.num_detectors}  OK")

    # 2) if the circuit ends in a terminal MPP, every MPP slot must be filled
    tt = s._terminal_det_time
    if tt is not None:
        unfilled = sum(1 for by in s._detector_to_measurement_raw[tt].values()
                       for qs in by.values() for q in qs if q is None)
        print(f"  terminal detectors ({len(s._detector_to_measurement_raw[tt])}) "
              f"with unfilled (None) qubit: {unfilled}")
        assert unfilled == 0
        print("  OK")
    else:
        print("  (no terminal MPP; all detectors are real)")

    out = sys.argv[2] if len(sys.argv) > 2 else path.rsplit("/", 1)[-1] + ".raw.txt"
    s.dump_raw(out)
    print(f"\n  raw dump written to {out}")

    s.refine()
    remapped = sum(1 for r in s.detector_coord if r[0] != r[1])
    print(f"  channels                : {len(s.channels)}")
    print(f"  max channel support (n) : {max((len(v) for v in s.channel_support.values()), default=0)}")
    print(f"  max detector span       : {max((hi - lo + 1 for lo, hi in s.detector_window), default=0)}")
    print(f"  remapped spatial coords : {remapped}")
    out2 = out[:-8] + ".refined.txt" if out.endswith(".raw.txt") else out + ".refined.txt"
    s.dump_refined(out2)
    print(f"  refined dump written to {out2}")
