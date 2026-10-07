"""Control-system latency model. Derives the feedback and delivery latencies (ns)
charged by the runtime estimator from a compiled board tree (links JSON written by
hardware/compiler/control_system/gen_control_config.py) and the link parameters.

Link model (one duplex point-to-point GT lane per parent-child path, no header):
    T_word    = n / M     n-bit frame at M Gb/s, e.g. 66 bit / 10 Gb/s = 6.6 ns
    L_link(B) = L_fixed + (ceil(B / m) - 1) * T_word
        B message bits, m payload bits per word, L_fixed = one-word message latency.
        No queueing term: one message per round, rounds are >= 500 ns apart.
    --legacy-link selects L_fixed + ceil(B / m) * T_word instead (one T_word more
    per hop). The paper's campaign profiles (latency_q14_f29.json) use it.
Per-board constants in cycles (RTL, gen_cb_tb.py --latency-trace); ns = cycles / f_board:
    leaf forward 3, router forward 2, root output 2, stage post-select verdict 3,
    root fast-path ps_in -> discard -> abort event 2 (root-own stage: 1), event
    relay 1 per level down.

Paths (level 0 = leaves, level D-1 = root, which hosts the decoder):
    deliver_ns         last measurement of a round at the leaves -> syndrome round at
                       the root decoder: sum over levels of board forward + uplink(B_up)
    verdict_down_ns    root abort event -> every leaf: (D-1) * [downlink(1 word) + relay]
    ps_feedback_ns[s]  post-select stage s, hosted at level l_s: up to that board
                       + its verdict + fast path to the root (if l_s < root)
                       + root discard/abort + verdict_down_ns
    gap_feedback_ns    root abort + verdict_down_ns; decoder latency is not included

Usage:
    python algorithms/control_latency.py <circuit folder> [--q 14 --fanout 29]
        [--gbps 10] [--frame 66] [--payload 64] [--l-fixed-ns 157] [--f-board-mhz 100]
        [--save]           # write control_system/latency_q<q>_f<fanout>.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

CYCLES = dict(leaf=3, router=2, root=2, stage_verdict=3, root_fastpath=2, root_own=1, relay=1)


def link_ns(bits, *, gbps=10.0, frame=66, payload=64, l_fixed_ns=157.0, fixed_is_one_word=True):
    """Latency in ns of a point-to-point GT link for a `bits`-bit message.
    fixed_is_one_word=True: l_fixed_ns covers the first word, so the result is
    l_fixed_ns + (words - 1) * T_word. False (--legacy-link): l_fixed_ns + words * T_word."""
    t_word = frame / gbps            # ns per frame at gbps Gb/s
    words = math.ceil(bits / payload)
    return l_fixed_ns + (words - 1 if fixed_is_one_word else words) * t_word


def profile(links, *, gbps=10.0, frame=66, payload=64, l_fixed_ns=157.0, f_board_mhz=100.0,
            cycles=CYCLES, fixed_is_one_word=True):
    """Latency profile (ns) for one compiled tree; `links` = links_q<q>_f<f>.json dict."""
    ns = lambda c: c / f_board_mhz * 1e3
    L = lambda b: link_ns(b, gbps=gbps, frame=frame, payload=payload, l_fixed_ns=l_fixed_ns,
                          fixed_is_one_word=fixed_is_one_word)
    boards = {b["board_id"]: b for b in links["boards"]}
    depth = links["layers"]
    root = next(b for b in links["boards"] if b["parent"] is None)["board_id"]

    def path_up(bid):
        """Worst case over `bid`'s subtree: last input at its leaves -> `bid` has
        forwarded (root: output)."""
        b = boards[bid]
        if not b["children"]:
            return ns(cycles["leaf"])
        below = max(path_up(c) + L(boards[c]["up"]["B_up"]) for c in b["children"])
        return below + ns(cycles["root"] if bid == root else cycles["router"])

    deliver = path_up(root)
    down = (depth - 1) * (L(links["B_down"]) + ns(cycles["relay"]))
    ps = {}
    for stage, bid in links["stage_boards"].items():
        b = boards[bid]
        # up to the stage board's inputs; the verdict replaces its forward latency
        if b["children"]:
            up = max(path_up(c) + L(boards[c]["up"]["B_up"]) for c in b["children"])
        else:
            up = 0.0
        if bid == root:
            t = up + ns(cycles["stage_verdict"]) + ns(cycles["root_own"])
        else:
            t = up + ns(cycles["stage_verdict"]) + L(links["B_ps"]) + ns(cycles["root_fastpath"])
        ps[int(stage)] = t + down
    # tag: q/fanout, plus gbps, l_fixed_ns, f_board_mhz when any is non-default
    tag = f"q{links['q']}f{links['fanout']}"
    if (gbps, l_fixed_ns, f_board_mhz) != (10.0, 157.0, 100.0):
        tag += f"g{gbps:g}L{l_fixed_ns:g}c{f_board_mhz:g}"
    return {
        "tag": tag,
        "circuit": links["circuit"], "q": links["q"], "fanout": links["fanout"], "layers": depth,
        "boards_per_layer": links["boards_per_layer"],
        "link": dict(gbps=gbps, frame=frame, payload=payload, l_fixed_ns=l_fixed_ns,
                     t_word_ns=frame / gbps), "f_board_mhz": f_board_mhz, "cycles": dict(cycles),
        "deliver_ns": deliver, "verdict_down_ns": down,
        "gap_feedback_ns": ns(cycles["root_own"]) + down,
        "ps_feedback_ns": ps, "stage_boards": links["stage_boards"],
        "meas_times": links.get("meas_times"), "stages": links.get("stages"),
        "max_uplink_words": max(math.ceil(b["up"]["B_up"] / payload) for b in links["boards"] if b["up"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--q", type=int, default=14)
    ap.add_argument("--fanout", type=int, default=29)
    ap.add_argument("--gbps", type=float, default=10.0)
    ap.add_argument("--frame", type=int, default=66)
    ap.add_argument("--payload", type=int, default=64)
    ap.add_argument("--l-fixed-ns", type=float, default=157.0)
    ap.add_argument("--f-board-mhz", type=float, default=100.0)
    ap.add_argument("--legacy-link", action="store_true",
                    help="legacy link form L_fixed + ceil(B/m)*T_word (the campaign profiles); default: one-word-fixed")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    folder = pathlib.Path(a.folder)
    links = json.loads((folder / "control_system" / f"links_q{a.q}_f{a.fanout}.json").read_text())
    p = profile(links, gbps=a.gbps, frame=a.frame, payload=a.payload, l_fixed_ns=a.l_fixed_ns, fixed_is_one_word=not a.legacy_link,
                f_board_mhz=a.f_board_mhz)
    print(f"{p['circuit']}: {p['layers']} layers {p['boards_per_layer']}, max uplink {p['max_uplink_words']} words")
    print(f"  deliver_ns      {p['deliver_ns']:7.1f}   (last measurement -> root decoder)")
    print(f"  verdict_down_ns {p['verdict_down_ns']:7.1f}   (root abort -> leaves)")
    print(f"  gap_feedback_ns {p['gap_feedback_ns']:7.1f}")
    for s, t in sorted(p["ps_feedback_ns"].items()):
        print(f"  ps_feedback_ns[stage {s} @ board {p['stage_boards'][str(s)]}] {t:7.1f}")
    if a.save:
        out = folder / "control_system" / f"latency_{p['tag']}.json"
        out.write_text(json.dumps(p, indent=1))
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
