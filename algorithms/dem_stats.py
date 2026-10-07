# dem_stats.py — DEM summary statistics for pipeline reporting: component
# counts by detector arity and degree measures.

from typing import Any

import stim


def dem_stats(dem: stim.DetectorErrorModel) -> dict[str, Any]:
    """Summary statistics of a DEM's structure.

    A fault node is one `error` instruction. A component is one
    separator-delimited part of it, with duplicate detector targets
    XOR-cancelled: `error(p) D0 D1 ^ D2 D3` is one fault node and two edges.

    Returned keys:
      num_detectors, num_observables
      num_fault_nodes        error instructions
      num_components         components (== num_fault_nodes if none decomposed)
      n_zero_det             components touching 0 detectors
      n_boundary_edges       components touching exactly 1 detector
      n_edges                components touching exactly 2 detectors
      n_hyperedges           components touching >= 3 detectors
      n_obs_flipping         fault nodes flipping at least one observable
      max_detector_degree    max over detectors of components touching it
      avg_detector_degree    mean of that count over all detectors
      max_fault_node_degree  max detectors touched by a single component
      detector_degree, fault_node_degree
                             distribution (see _dist) of those two counts
    [used: gen_partial_mask_dem, run_eval_partial_mask]"""
    flat = dem.flattened()
    n_det = flat.num_detectors

    det_degree = [0] * n_det
    num_fault_nodes = 0
    num_components = 0
    n_zero_det = 0
    n_boundary_edges = 0
    n_edges = 0
    n_hyperedges = 0
    n_obs_flipping = 0
    max_fault_node_degree = 0
    arities: list[int] = []          # per component: number of distinct detectors

    for inst in flat:
        if inst.type != "error":
            continue
        num_fault_nodes += 1

        components: list[list[Any]] = [[]]
        flips_obs = False
        for t in inst.targets_copy():
            if t.is_separator():
                components.append([])
            elif t.is_relative_detector_id():
                components[-1].append(t.val)
            elif t.is_logical_observable_id():
                flips_obs = True
        if flips_obs:
            n_obs_flipping += 1

        for comp in components:
            num_components += 1
            det_parity: dict[int, int] = {}
            for d in comp:
                det_parity[d] = det_parity.get(d, 0) ^ 1
            det_set = [d for d, par in det_parity.items() if par]

            k = len(det_set)
            if k == 0:
                n_zero_det += 1
            elif k == 1:
                n_boundary_edges += 1
            elif k == 2:
                n_edges += 1
            else:
                n_hyperedges += 1
            if k > max_fault_node_degree:
                max_fault_node_degree = k
            arities.append(k)
            for d in det_set:
                det_degree[d] += 1

    def _dist(values: list[int]) -> dict[str, float]:
        """min/mean/median/max/p95/p99 of an integer distribution."""
        if not values:
            return {k: 0.0 for k in ("min", "mean", "median", "max", "p95", "p99")}
        a = sorted(values)
        q = lambda f: float(a[min(len(a) - 1, int(round(f * (len(a) - 1))))])
        return {"min": float(a[0]), "mean": float(sum(a) / len(a)), "median": q(0.5),
                "max": float(a[-1]), "p95": q(0.95), "p99": q(0.99)}

    return {
        "detector_degree": _dist(det_degree),
        "fault_node_degree": _dist(arities),
        "num_detectors": int(n_det),
        "num_observables": int(flat.num_observables),
        "num_fault_nodes": int(num_fault_nodes),
        "num_components": int(num_components),
        "n_zero_det": int(n_zero_det),
        "n_boundary_edges": int(n_boundary_edges),
        "n_edges": int(n_edges),
        "n_hyperedges": int(n_hyperedges),
        "n_obs_flipping": int(n_obs_flipping),
        "max_detector_degree": int(max(det_degree, default=0)),
        "avg_detector_degree": float(sum(det_degree) / n_det) if n_det else 0.0,
        "max_fault_node_degree": int(max_fault_node_degree),
    }


def print_dem_stats(stats: dict[str, Any], *, label: str = "") -> None:
    """Print a dem_stats() dict as a three-line block, prefixed by [label]."""
    tag = f"[{label}] " if label else ""
    print(
        f"{tag}detectors={stats['num_detectors']} "
        f"obs={stats['num_observables']} "
        f"fault_nodes={stats['num_fault_nodes']} "
        f"components={stats['num_components']}"
    )
    print(
        f"  edges: boundary={stats['n_boundary_edges']} "
        f"pair={stats['n_edges']} "
        f"hyper={stats['n_hyperedges']} "
        f"zero={stats['n_zero_det']} "
        f"obs_flipping={stats['n_obs_flipping']}"
    )
    print(
        f"  degree: det max={stats['max_detector_degree']} "
        f"avg={stats['avg_detector_degree']:.1f}, "
        f"fault_node max={stats['max_fault_node_degree']}"
    )
