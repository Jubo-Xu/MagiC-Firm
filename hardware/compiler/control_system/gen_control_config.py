#!/usr/bin/env python3
"""gen_control_config.py — build a distributed control-board tree for a circuit
folder and compile it, then report the per-link message widths.

Topology parameters (the physical facts of the deployment):
    --q       max physical qubits wired to one leaf board (coordinate-local clusters)
    --fanout  max children per router / root board (GT lanes minus the uplink)

Construction: physical qubits are clustered by recursive coordinate bisection
(longest axis, median split) until every cluster has <= q qubits -> leaves;
leaves are grouped by the same bisection on their centroids into <= fanout
children per router, level by level, until one root remains. Kernels and
raw_out are sized in two passes like gen_and_test_configs.tighten (generous
first pass, then sized to the compiled need + 1 headroom).

Outputs (under --out-dir, default <circuit folder>/control_system/):
    config_q<q>_f<fanout>.json      compiler config (postselect list inlined)
    links_q<q>_f<fanout>.json       per-board link widths for the latency model:
and the serializer output (manifest, report, thousands of regfiles) under
<repo>/out/control_system/<circuit>/compiled/ (bulk disk, regenerable):
        B_up  = 2*D_OUT + 2*RAW + 3   (fwd_det+valid, fwd_raw+valid, det/raw finish, attempt)
        B_ps  = 2                     (stage-board fast path: ps_out + attempt)
        B_down= 6                     (event: valid, type[2], payload[2], attempt)

Usage:
    python gen_control_config.py <circuit folder> --q 14 --fanout 29 [--out-dir DIR] [--no-compile]
Run from this directory (the compiler modules use relative imports).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compiler import Compiler
from config import CompilerConfig
from gen_and_test_configs import _collapse_int_arrays, tighten
from parser import DetectorConstructionScanner

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
def bisect(items, key_xy, cap):
    """Recursively split `items` (each with a 2-D position key_xy(item)) along the
    longer axis at the median until every group has <= cap items."""
    if len(items) <= cap:
        return [list(items)]
    xs = [key_xy(i)[0] for i in items]
    ys = [key_xy(i)[1] for i in items]
    axis = 0 if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else 1
    srt = sorted(items, key=lambda i: (key_xy(i)[axis], key_xy(i)[1 - axis]))
    mid = len(srt) // 2
    return bisect(srt[:mid], key_xy, cap) + bisect(srt[mid:], key_xy, cap)


def build_tree(coords, q, fanout, kern, postselect):
    qubits = [i for i, xy in enumerate(coords) if xy is not None]
    leaf_groups = bisect(qubits, lambda i: coords[i], q)
    leaves = [{"board_id": i, "qubits": sorted(g), "kernels": dict(kern), "raw_out": 99999}
              for i, g in enumerate(leaf_groups)]
    centroid = {b["board_id"]: (sum(coords[i][0] for i in b["qubits"]) / len(b["qubits"]),
                                sum(coords[i][1] for i in b["qubits"]) / len(b["qubits"]))
                for b in leaves}
    layers = [{"level": 0, "type": "leaf", "boards": leaves}]
    current = leaves
    level = 1
    while len(current) > 1:
        groups = bisect([b["board_id"] for b in current], lambda bid: centroid[bid], fanout)
        is_root = len(groups) == 1
        boards = []
        for gi, g in enumerate(groups):
            bid = 1000 * level + gi
            board = {"board_id": bid, "children": sorted(g), "kernels": dict(kern)}
            if not is_root:
                board["raw_out"] = 99999
            centroid[bid] = (sum(centroid[c][0] for c in g) / len(g), sum(centroid[c][1] for c in g) / len(g))
            boards.append(board)
        layers.append({"level": level, "type": "root" if is_root else "router", "boards": boards})
        current, level = boards, level + 1
    if len(layers) == 1:                       # a single leaf: make it a monolithic root
        layers[0]["type"] = "root"
        for b in layers[0]["boards"]:
            b.pop("raw_out", None)
    return {"name": f"tree_q{q}_f{fanout}", "version": 1, "hardware": {"layers": layers},
            "postselect": sorted(postselect)}


def tighten_raw(cfg, comp):
    """Size raw_out to the compiled forwarding need (+1 headroom, min 1)."""
    for layer in cfg["hardware"]["layers"]:
        for b in layer["boards"]:
            if "raw_out" in b:
                need = len(comp.pl.forward_normal[b["board_id"]]) + len(comp.pl.forward_obs[b["board_id"]])
                b["raw_out"] = need + 1


def link_table(manifest, cfg):
    boards = {b["board_id"]: b for b in manifest["boards"]}
    parent, children = {}, {}
    for layer in cfg["hardware"]["layers"]:
        for b in layer["boards"]:
            children[b["board_id"]] = list(b.get("children", []))
            for c in b.get("children", []):
                parent[c] = b["board_id"]
    for bid, b in boards.items():
        b["children"] = children.get(bid, [])
    # detector lines out of a board (physical layout): pass (all children's lines)
    # ++ construct (all k kernels, idle included) — mirrors RTL D_OUT = D_IN + K
    def d_out(bid):
        b = boards[bid]
        return sum(d_out(c) for c in (b.get("children") or [])) + b["kernels_avail"]

    def d_used(bid):
        b = boards[bid]
        return sum(d_used(c) for c in (b.get("children") or [])) + b["kernels_used"]

    rows = []
    for bid, b in sorted(boards.items()):
        if bid not in parent:
            up = None
        else:
            raw = b["forward_normal"] + b["forward_obs"]
            up = {"D_OUT": d_out(bid), "D_used": d_used(bid), "RAW": raw,
                  "B_up": 2 * d_out(bid) + 2 * raw + 3, "B_up_used": 2 * d_used(bid) + 2 * raw + 3}
        rows.append({"board_id": bid, "type": b["type"], "level": b["level"], "parent": parent.get(bid),
                     "children": b.get("children"), "stage_board": bool(b.get("postselect_stages")),
                     "up": up})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="circuit folder (experiments/data/circuits/<config>)")
    ap.add_argument("--q", type=int, default=14, help="max qubits per leaf board")
    ap.add_argument("--fanout", type=int, default=29, help="max children per router/root")
    ap.add_argument("--out-dir", default=None, help="default <folder>/control_system")
    ap.add_argument("--no-compile", action="store_true", help="write the config only")
    args = ap.parse_args()

    folder = os.path.abspath(args.folder)
    name = os.path.basename(folder.rstrip("/"))
    stim_path = os.path.join(folder, f"{name}.stim")
    out_dir = args.out_dir or os.path.join(folder, "control_system")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(folder, "postselected_detectors.json")) as f:
        postselect = sorted(json.load(f)["postselected_detectors"])

    s = DetectorConstructionScanner.from_file(stim_path)
    s.scan(); s.refine()
    coords = s._physical_qubits_coord_raw
    kern = {"count": 256, "n": 16, "h": 8}
    cfg = build_tree(coords, args.q, args.fanout, kern, postselect)
    comp1 = Compiler(CompilerConfig(cfg), s)                 # pass 1: generous
    tighten(cfg, comp1); tighten_raw(cfg, comp1)             # size to need
    cfg["name"] = f"tree_q{args.q}_f{args.fanout}"
    cfg_path = os.path.join(out_dir, f"config_q{args.q}_f{args.fanout}.json")
    with open(cfg_path, "w") as f:
        f.write(_collapse_int_arrays(json.dumps(cfg, indent=2)))
    comp = Compiler(CompilerConfig.from_file(cfg_path), s)   # pass 2: sized config
    problems = {b: comp.programs[b].problems for b in comp.programs if comp.programs[b].problems}
    layers = cfg["hardware"]["layers"]
    print(f"{name}: {sum(len(l['boards']) for l in layers)} boards in {len(layers)} layers "
          f"({' -> '.join(f'{len(l['boards'])} {l['type']}' for l in layers)}); feasible={not problems}")
    if problems:
        for b, p in problems.items():
            print(f"  board {b}: {p}")
        sys.exit(1)
    if args.no_compile:
        return
    # serializer output = thousands of small regfiles per tree -> keep it on the bulk
    # disk (<repo>/out -> /data), not in the circuit folder (home file quota)
    comp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "out",
                            "control_system", name, "compiled")
    os.makedirs(comp_dir, exist_ok=True)
    r = subprocess.run([PY, os.path.join(HERE, "cli.py"), stim_path, cfg_path, "--out", comp_dir],
                       cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:]); sys.exit(1)
    mans = sorted(glob.glob(os.path.join(comp_dir, f"{cfg['name']}__*", "manifest.json")), key=os.path.getmtime)
    manifest = json.load(open(mans[-1]))
    rows = link_table(manifest, cfg)
    links = {"circuit": name, "q": args.q, "fanout": args.fanout, "layers": len(layers),
             "boards_per_layer": [len(l["boards"]) for l in layers], "compiled": os.path.dirname(mans[-1]),
             "stage_boards": manifest["stage_boards"],
             # postselect stages as det-time (= measurement-round) ranges, for the
             # estimator's per-round feedback lookup (its rounds index the same way)
             "meas_times": manifest["meas_times"],
             "stages": [{"stage": i, "det_time_range": list(st["det_time_range"]), "detectors": len(st["detectors"])}
                        for i, st in enumerate(s.stages)],
             "B_ps": 2, "B_down": 6, "boards": rows}
    with open(os.path.join(out_dir, f"links_q{args.q}_f{args.fanout}.json"), "w") as f:
        json.dump(links, f, indent=1)
    print(f"  stage boards: {manifest['stage_boards']}")
    print(f"  {'board':>6} {'type':<7} {'lvl':>3} {'parent':>6} {'D_OUT':>5} {'RAW':>4} {'B_up':>5} {'words':>5}")
    for row in rows:
        u = row["up"]
        print(f"  {row['board_id']:>6} {row['type']:<7} {row['level']:>3} {str(row['parent']):>6} "
              + (f"{u['D_OUT']:>5} {u['RAW']:>4} {u['B_up']:>5} {-(-u['B_up'] // 64):>5}" if u else "   (root)"))


if __name__ == "__main__":
    main()
