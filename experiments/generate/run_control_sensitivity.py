#!/usr/bin/env python3
"""Sensitivity of the preparation time to the control-system parameters.

For each circuit and each combination of
    Q (qubits per leaf) x link rate (Gb/s) x link fixed latency (ns) x board clock (MHz)
build the board tree (gen_control_config, cached per Q), derive the latency
profile (control_latency), and run the estimator's two operating points
(complete-only single tc35, two-stage pcp@th* from --summary) at --epochs.
Results: experiments/result/runtime_estimation/control_sensitivity.{json,md}.

    python experiments/generate/run_control_sensitivity.py [--only PATTERN ...] [--epochs 1000000] [--workers 32]
        [--q 14 20] [--gbps 10 25] [--l-fixed 100 157 300] [--clock 100 200]
"""
import argparse
import itertools
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702
import campaign as rec                                                     # noqa: E402
from control_latency import profile as make_profile             # noqa: E402

REPO = rec.REPO
PY = sys.executable
COMPILER = REPO / "hardware" / "compiler" / "control_system"
DEFAULT_CIRCUITS = ["end2end_d1=3_d2=15_r1=3_r2=0_p=0.001_inj=unitary_b=Y",
                    "end2end_d1=3_d2=21_r1=3_r2=0_p=0.001_inj=unitary_b=Y",
                    "end2end_d1=5_d2=15_r1=5_r2=0_p=0.001_inj=unitary_b=Y"]


def ensure_links(name, q, fanout):
    links = rec.CIRCUITS / name / "control_system" / f"links_q{q}_f{fanout}.json"
    if not links.exists():
        r = subprocess.run([PY, "gen_control_config.py", str(rec.CIRCUITS / name), "--q", str(q), "--fanout", str(fanout)],
                           cwd=COMPILER, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"gen_control_config failed for {name} q{q}: {r.stdout[-800:]} {r.stderr[-800:]}")
    return json.loads(links.read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--summary", default="summary.json", help="where th* per circuit comes from")
    ap.add_argument("--epochs", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--fanout", type=int, default=29)
    ap.add_argument("--q", type=int, nargs="+", default=[14, 20])
    ap.add_argument("--gbps", type=float, nargs="+", default=[10, 25])
    ap.add_argument("--l-fixed", type=float, nargs="+", default=[100, 157, 300])
    ap.add_argument("--clock", type=float, nargs="+", default=[100, 200])
    ap.add_argument("--legacy-link", action="store_true",
                    help="link latency L_fixed + ceil(B/m)*T_word (the form of the stored campaign profiles)")
    args = ap.parse_args()
    ref = json.loads((rec.RESULT / args.summary).read_text())
    out_path = rec.RESULT / "control_sensitivity.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    circuits = [c for c in DEFAULT_CIRCUITS if not args.only or any(p in c for p in args.only)]

    for name in circuits:
        r = ref[name]; f = float(r["frequency_hz"]); th = int(r["th_star"]); tag_mask = r["partial_tag"]
        for q, gbps, lfix, clk in itertools.product(args.q, args.gbps, args.l_fixed, args.clock):
            links = ensure_links(name, q, args.fanout)
            prof = make_profile(links, gbps=gbps, l_fixed_ns=lfix, f_board_mhz=clk, fixed_is_one_word=not args.legacy_link)
            ppath = rec.CIRCUITS / name / "control_system" / f"latency_{prof['tag']}.json"
            ppath.write_text(json.dumps(prof, indent=1))
            key = f"{name}|{prof['tag']}"
            if key in results:
                continue
            a = argparse.Namespace(epochs=args.epochs, workers=args.workers, cs=True, cs_tag="_cs" + prof["tag"],
                                   cs_q=q, cs_f=args.fanout)
            # run_estimator/expected_path expect profile_path(name, args): override to this profile
            rec.profile_path = lambda n, ar, _p=ppath: _p
            log_path = rec.RESULT / name / "sensitivity.log"
            bp = rec.expected_path(name, "complete", f"single_tc{rec.TC:g}", a)
            if not bp.exists():
                bp = rec.run_estimator(name, f, a, ["--mode", "single", "--tc", rec.TC], log_path)
            pp = rec.expected_path(name, tag_mask, f"pcp_tc{rec.TC:g}_tl0_th{th}", a)
            if not pp.exists():
                pp = rec.run_estimator(name, f, a, ["--partial", tag_mask, "--mode", "pcp", "--tc", rec.TC, "--tl", 0, "--th", th], log_path)
            b, p = rec.summarize(bp), rec.summarize(pp)
            results[key] = dict(circuit=name, q=q, fanout=args.fanout, gbps=gbps, l_fixed_ns=lfix, clock_mhz=clk,
                                layers=prof["layers"], boards_per_layer=prof["boards_per_layer"],
                                deliver_ns=prof["deliver_ns"], gap_feedback_ns=prof["gap_feedback_ns"],
                                ps_feedback_ns=prof["ps_feedback_ns"], th_star=th,
                                baseline=b, partial=p, reduction=1 - p["prep_us"] / b["prep_us"])
            rec.log(f"{name[8:36]} {prof['tag']:<22} tree {prof['boards_per_layer']} deliver {prof['deliver_ns']:.0f} "
                    f"ps {min(prof['ps_feedback_ns'].values()):.0f}-{max(prof['ps_feedback_ns'].values()):.0f} | "
                    f"complete {b['prep_us']:.2f} partial {p['prep_us']:.2f} -> {results[key]['reduction']:.1%}")
            out_path.write_text(json.dumps(results, indent=1))

    lines = ["| circuit | Q | Gb/s | L_fixed ns | clock MHz | tree | deliver ns | ps feedback ns | complete µs | partial µs | reduction |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, v in sorted(results.items()):
        lines.append(f"| {v['circuit'][8:36]} | {v['q']} | {v['gbps']:g} | {v['l_fixed_ns']:g} | {v['clock_mhz']:g} | "
                     f"{'/'.join(map(str, v['boards_per_layer']))} | {v['deliver_ns']:.0f} | "
                     f"{min(v['ps_feedback_ns'].values()):.0f}–{max(v['ps_feedback_ns'].values()):.0f} | "
                     f"{v['baseline']['prep_us']:.2f} | {v['partial']['prep_us']:.2f} | {v['reduction']:.1%} |")
    (rec.RESULT / "control_sensitivity.md").write_text("\n".join(lines) + "\n")
    rec.log("SENSITIVITY DONE")


if __name__ == "__main__":
    main()
