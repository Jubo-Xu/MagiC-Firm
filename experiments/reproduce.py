#!/usr/bin/env python3
"""Entry point for reproducing the paper's results.

    python experiments/reproduce.py figures [--only NAME ...] [--out-dir DIR]
        Redraw every figure and table from the stored results under
        experiments/result/. Minutes; needs only the Python environment.
    python experiments/reproduce.py quick [--out-dir DIR] [--workers N]
        The method end to end on one small circuit (d_cultiv 3, d_escape 11,
        p 1e-3): circuit folder, detector scores, mask and contracted DEM,
        preparation-time estimation and a gap table. About five minutes with
        8 workers; writes under <out>/quick/ and never touches the paper's data.
    python experiments/reproduce.py full [--stage NAME ...] [--dry-run] [--workers N]
        The paper's campaigns with the paper's parameters, in order, each
        stage resumable: circuits, essentials, masks, control, estimation,
        gap_tables, thresholds, ablation, figures. Days of compute, about
        70 GB under MAGICFIRM_OUT; the masks stage and the masked-syndrome
        part of ablation need the micro-blossom toolchain container. The
        FPGA synthesis (Vivado) is a separate flow under hardware/.

Each figure or table is produced by one script under experiments/figures/,
which can also be run on its own (see its --help).
"""
import argparse
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lib"))
import paths                                                               # noqa: E402

FIG = paths.EXPERIMENTS / "figures"
R = paths.RESULT
FLAGSHIP = "end2end_d1=3_d2=15_r1=3_r2=0_p=0.001_inj=unitary_b=Y"

# name, what it shows, script, arguments (the output folder is appended via `out_flag`)
FIGURES = [
    ("gap_shift_and_threshold_sweep", "detector gap-shift maps and preparation time vs t_c, with the consumption pipeline",
     FIG / "fig_gap_shift_and_threshold_sweep.py",
     ["--sens", R / "gap_sensitivity" / FLAGSHIP / "overall_gap_sens_s0_n2000000.json",
      "--sweep", R / "runtime_estimation" / FLAGSHIP / "threshold_sweep_partial_qexp250_cl_wc15_fb3100_csq14f29.json",
      "--pipeline", FIG / "assets" / "pipeline.svg"], "--fig-dir"),
    ("mask_slices", "LAP base mask and its structural closure on one code slice",
     FIG / "fig_mask_slices.py", [], "--out-dir"),
    ("mask_size_sensitivity", "gap agreement vs retained-detector fraction",
     FIG / "fig_mask_size_sensitivity.py", [], "--fig-dir"),
    ("dem_contraction_ablation", "decode latency and gap agreement: complete, masked, contracted",
     FIG / "fig_dem_contraction_ablation.py", [], "--fig-dir"),
    ("iso_ler_scaling", "preparation time at matched LER vs d_escape and p",
     FIG / "fig_iso_ler_scaling.py", [], "--out"),
    ("pareto_frontiers", "preparation time vs LER frontiers and their sensitivity",
     FIG / "fig_pareto_frontiers.py", [], "--out"),
    ("table_mask_strategies", "geometric vs LAP vs LAP + closure at matched size",
     FIG / "table_mask_strategies.py", sorted((R / "mask_ablation").glob("heuristics_*_N336.json")), "--out-dir"),
    ("table_control_latency", "compiled control trees and worst-case latencies",
     FIG / "table_control_latency.py", [], "--out-dir"),
    ("table_fpga_resources", "per-role FPGA resources and Fmax",
     FIG / "table_fpga_resources.py", [], "--out-dir"),
]


def run_figures(args) -> int:
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    todo = [f for f in FIGURES if not args.only or f[0] in args.only]
    unknown = set(args.only or []) - {f[0] for f in FIGURES}
    if unknown:
        sys.exit(f"unknown figure name(s): {sorted(unknown)}; choose from {[f[0] for f in FIGURES]}")
    failed = []
    for name, what, script, extra, out_flag in todo:
        cmd = [sys.executable, str(script), *map(str, extra), out_flag, str(out)]
        log = out / f"{name}.log"
        r = subprocess.run(cmd, cwd=paths.REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log.write_text(" ".join(cmd) + "\n\n" + r.stdout)
        status = "ok" if r.returncode == 0 else f"FAILED (rc {r.returncode}, see {log.name})"
        print(f"{name:32s} {status}")
        if r.returncode:
            failed.append(name); print(r.stdout[-1500:])
    print(f"\noutputs in {out}")
    for name, what, *_ in todo:
        if name not in failed:
            print(f"  {name:32s} {what}")
    return 1 if failed else 0


# ---------------------------------------------------------------- quick
GEN = paths.EXPERIMENTS / "generate"
QUICK_D1, QUICK_D2, QUICK_P = 3, 11, 0.001
QUICK_NAME = f"end2end_d1={QUICK_D1}_d2={QUICK_D2}_r1={QUICK_D1}_r2=0_p={QUICK_P:g}_inj=unitary_b=Y"
QUICK_KQ, QUICK_TAG = 200, "partial_qexp200_cl_wc15"
QUICK_ESSENTIALS_SHOTS, QUICK_EPOCHS, QUICK_TABLE_SHOTS = 200_000, 200_000, 1_000_000
TC, TH = 35, 45
# decoder latencies (parallel mean, us) measured on the RTL simulator for this circuit
# at 77 MHz; used as constants when the stored characterization is absent
FALLBACK_LATENCY_US = {"complete": 4.34587, "partial": 2.566745}


def sh(cmd, log, env=None, cwd=None):
    """Run one stage; append its command and output to `log`; raise on failure."""
    import os
    cmd = [str(c) for c in cmd]
    with open(log, "a") as lf:
        lf.write("$ " + " ".join(cmd) + "\n"); lf.flush()
        r = subprocess.run(cmd, cwd=cwd or paths.REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                           env={**os.environ, **(env or {})})
        lf.write(r.stdout + "\n")
    if r.returncode:
        print(r.stdout[-2000:]); raise SystemExit(f"stage failed (rc {r.returncode}); log: {log}")
    return r.stdout


def quick_latency_us():
    """Stored micro-blossom means for the quick circuit, else the fallback."""
    import json
    mb = R / "mb_characterization"
    files = {"complete": mb / f"{QUICK_NAME}__complete__s0_n1000_77MHz_realsched.json",
             "partial": mb / f"{QUICK_NAME}__{QUICK_TAG}__s0_n1000_77MHz_realsched.json"}
    if all(f.exists() for f in files.values()):
        return {k: json.loads(f.read_text())["latency"]["parallel"]["mean_us"] for k, f in files.items()}, "stored MB characterization"
    return dict(FALLBACK_LATENCY_US), "built-in constants"


def estimator_summary(path):
    import json
    j = json.loads(pathlib.Path(path).read_text())
    ready = j["config"]["estimator"]["stage_names"].index("ready")
    us = j["records_until_success"]
    return {k: us[k]["mean"][ready] / 1000 for k in ("total", "gate", "feedback", "gap_decode")} | {
        "ler": j["logical_error_rate"], "accept": j["accept_rate"], "epochs": j["epoch"]}


def run_quick(args) -> int:
    import json, time
    import numpy as np
    from gap_table import rule_stats
    W = args.workers
    Q = (args.out_dir or paths.OUT / "quick").resolve(); (Q / "logs").mkdir(parents=True, exist_ok=True)
    env = {"MAGICFIRM_OUT": str(Q)}           # traces, gap tables and caches of this run live under Q
    folder = Q / "circuits" / QUICK_NAME
    stem = folder / f"{QUICK_NAME}.stim"
    print(f"quick reproduction: {QUICK_NAME}, {W} workers, workspace {Q}\n")
    t0 = time.time()
    def stage(n, what, done, cmd, log, **kw):
        if done():
            print(f"[{n}/6] {what:38s} exists, skip"); return
        t = time.time(); sh(cmd, Q / "logs" / log, env=env, **kw)
        print(f"[{n}/6] {what:38s} {time.time() - t:5.0f} s")
    py = sys.executable
    stage(1, "circuit folder + gap DEM", lambda: (folder / "postselected_detectors.json").exists(),
          [py, GEN / "gen_circuit_folders.py", "--d1", QUICK_D1, "--d2", QUICK_D2, "--p", QUICK_P, "--out-root", Q / "circuits"], "1_circuit.log")
    ess = folder / f"{QUICK_NAME}_logical_ambiguity_shots{QUICK_ESSENTIALS_SHOTS}.json"
    stage(2, f"logical-ambiguity scores, {QUICK_ESSENTIALS_SHOTS:,} shots", ess.exists,
          [py, paths.ALGORITHMS / "gen_essentials.py", "--circuit", stem, "--out", folder, "--type", "logical_ambiguity",
           "--shots", QUICK_ESSENTIALS_SHOTS, "--workers", W], "2_essentials.log")
    stage(3, f"mask K_Q={QUICK_KQ} + closure + contraction", lambda: (folder / QUICK_TAG / "contracted.dem").exists(),
          [py, paths.ALGORITHMS / "gen_partial_mask_dem.py", "--out", folder, "--base-size", QUICK_KQ, "--q-type", "exp",
           "--closure", "--contract", "--weight-cutoff", 15.0, "--essentials-logical-ambiguity", ess], "3_mask.log")
    lat, lat_src = quick_latency_us()
    print(f"      decoder latency: complete {lat['complete']:.2f} us, partial {lat['partial']:.2f} us ({lat_src})")
    est_dir = Q / "estimation"
    common = ["--epochs", QUICK_EPOCHS, "--latency", "constant", "--decoder-ns", lat["complete"] * 1000,
              "--partial-decoder-ns", lat["partial"] * 1000, "--trace", "--workers", W, "--out-root", Q, "--result-dir", est_dir]
    w = f"_w{W}" if W > 1 else ""
    single = est_dir / QUICK_NAME / f"complete__single_tc{TC}__lat-constant_trace-s0_e{QUICK_EPOCHS}{w}.json"
    pcp = est_dir / QUICK_NAME / f"{QUICK_TAG}__pcp_tc{TC}_tl0_th{TH}__lat-constant_trace-s0_e{QUICK_EPOCHS}{w}.json"
    stage(4, f"estimator, complete only, t_c={TC}", single.exists,
          [py, GEN / "run_runtime_estimation.py", folder, "--mode", "single", "--tc", TC, *common], "4_single.log")
    stage(5, f"estimator, two-stage, t_c={TC} t_h={TH}", pcp.exists,
          [py, GEN / "run_runtime_estimation.py", folder, "--partial", QUICK_TAG, "--mode", "pcp", "--tc", TC, "--tl", 0, "--th", TH, *common], "5_pcp.log")
    table = Q / "gap_tables" / QUICK_NAME / f"{QUICK_TAG}_s0.npz"
    stage(6, f"gap table, {QUICK_TABLE_SHOTS:,} shots", table.exists,
          [py, GEN / "collect_gap_table.py", folder, "--partial", QUICK_TAG, "--shots", QUICK_TABLE_SHOTS, "--seed", 0, "--workers", W], "6_gap_table.log")
    # summary
    s1, s2 = estimator_summary(single), estimator_summary(pcp)
    z = np.load(table); gc, gp, err = z["gc"], z["gp"], z["err"]
    n1, e1, l1 = rule_stats(gc, gp, err, tc=TC); n2, e2, l2 = rule_stats(gc, gp, err, tc=TC, tl=0, th=TH)
    red = 1 - s2["total"] / s1["total"]
    summary = {"circuit": QUICK_NAME, "mask": QUICK_TAG, "latency_us": lat, "latency_source": lat_src,
               "complete": {**s1, "table_ler": l1, "table_accept_errors": [n1, e1]},
               "two_stage": {**s2, "th": TH, "table_ler": l2, "table_accept_errors": [n2, e2]},
               "reduction": red, "elapsed_s": time.time() - t0}
    (Q / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nscheme       prep (us)   gate   feedback  gap-decode   LER (table, {len(gc):,} shots)")
    for name, s, l in (("complete", s1, l1), ("two-stage", s2, l2)):
        print(f"{name:12s} {s['total']:8.2f} {s['gate']:7.2f} {s['feedback']:8.2f} {s['gap_decode']:10.2f}   {l:.2e}")
    print(f"\ntwo-stage reduces mean preparation time by {red:.1%} (t_c={TC}, t_h={TH}); "
          f"summary in {Q / 'summary.json'}; {summary['elapsed_s']:.0f} s")
    return 0


# ---------------------------------------------------------------- full
import json
import shutil

COMPILER = paths.REPO / "hardware" / "compiler" / "control_system"
CS_Q, CS_F = 14, 29
ESS_SHOTS = 2_000_000
FLAGSHIPS = {3: "end2end_d1=3_d2=15_r1=3_r2=0_p=0.001_inj=unitary_b=Y", 5: "end2end_d1=5_d2=15_r1=5_r2=0_p=0.001_inj=unitary_b=Y"}
# threshold sweeps of the design-space figures: (circuit, mask tag); their gap tables
# hold 100M shots (d_cultiv 3) or 1B shots (d_cultiv 5); every other campaign circuit 20M
SWEEPS = [(f"end2end_d1=3_d2=15_r1=3_r2=0_p={p}_inj=unitary_b=Y", f"partial_qexp{k}_cl_wc15")
          for p, ks in (("0.0005", (100, 250)), ("0.0009", (250,)), ("0.001", (250, 350, 600))) for k in ks] + \
         [("end2end_d1=5_d2=15_r1=5_r2=0_p=0.0005_inj=unitary_b=Y", "partial_qexp250_cl_wc15"),
          ("end2end_d1=5_d2=15_r1=5_r2=0_p=0.001_inj=unitary_b=Y", "partial_qexp450_cl_wc15")]
GAP_TABLE_SHOTS = {(n, t): (1_000_000_000 if n.startswith("end2end_d1=5_") else 100_000_000) for n, t in SWEEPS}
GAP_CHUNK = 100_000                  # post-selected shots per chunk; part of a table's identity
MASK_SIZE_SWEEP = {3: (250, 350, 450, 600, 800), 5: (250, 350, 450, 550, 700)}
STAGES = ["circuits", "essentials", "masks", "control", "estimation", "gap_tables", "thresholds", "ablation", "figures"]


def mb_toolchain_available() -> bool:
    if not shutil.which("docker"):
        return False
    r = subprocess.run(["docker", "image", "inspect", "micro-blossom:latest"], capture_output=True)
    return r.returncode == 0


def campaign_circuits():
    """(name, mask tag, clock) of every circuit with a chosen mask, from the mask campaign."""
    f = R / "mask_campaign" / "summary.json"
    if not f.exists():
        return []
    return [(n, r["partial_tag"], r["frequency_hz"]) for n, r in sorted(json.loads(f.read_text()).items())
            if r.get("partial_tag")]


def profile_of(name):
    return paths.CIRCUITS / name / "control_system" / f"latency_q{CS_Q}_f{CS_F}.json"


def full_stage_commands(stage, W, mb_ok):
    """[(description, command, done-check or None, cwd)] for one stage.
    A None done-check means the driver resumes by itself."""
    py, alg = sys.executable, paths.ALGORITHMS
    C = paths.CIRCUITS
    out = []
    if stage == "circuits":
        out.append(("circuit folders, paper grid", [py, GEN / "gen_circuit_folders.py"], None, None))
    elif stage == "essentials":
        for f in sorted(C.glob("end2end_*")):
            ess = f / f"{f.name}_logical_ambiguity_shots{ESS_SHOTS}.json"
            out.append((f"essentials {f.name}", [py, alg / "gen_essentials.py", "--circuit", f / f"{f.name}.stim", "--out", f,
                        "--type", "logical_ambiguity", "--shots", ESS_SHOTS, "--workers", W], ess.exists, None))
    elif stage == "masks":
        cmd = [py, GEN / "run_mask_campaign.py", "--eval-workers", W, "--mb-workers", min(W, 8)]
        if not mb_ok:
            cmd.append("--no-mb")
        out.append(("mask campaign: sweep K_Q, select, generate, characterize on micro-blossom" + ("" if mb_ok else " (toolchain missing: --no-mb)"), cmd, None, None))
    elif stage == "control":
        for n, _, _ in campaign_circuits() or [(f.name, None, None) for f in sorted(C.glob("end2end_*"))]:
            links = C / n / "control_system" / f"links_q{CS_Q}_f{CS_F}.json"
            out.append((f"board tree {n}", [py, COMPILER / "gen_control_config.py", C / n, "--q", CS_Q, "--fanout", CS_F], links.exists, COMPILER))
            out.append((f"latency profile {n}", [py, alg / "control_latency.py", C / n, "--q", CS_Q, "--fanout", CS_F, "--legacy-link", "--save"],
                        profile_of(n).exists, None))
    elif stage == "estimation":
        cs = ["--cs", f"{CS_Q},{CS_F}"]
        out.append(("baseline + two-stage sweep, 1M epochs", [py, GEN / "run_estimation_campaign.py", *cs, "--epochs", 1_000_000, "--workers", W], None, None))
        out.append(("operating points, 10M epochs", [py, GEN / "run_estimation_campaign.py", *cs, "--reuse-th-from", R / "runtime_estimation" / "summary.json",
                    "--epochs", 10_000_000, "--workers", W], None, None))
    elif stage == "gap_tables":
        pairs = [(n, t) for n, t, _ in campaign_circuits()] + [p for p in SWEEPS if p not in {(a, b) for a, b, _ in campaign_circuits()}]
        for n, tag in pairs:
            shots = GAP_TABLE_SHOTS.get((n, tag), 20_000_000)
            out.append((f"gap table {n} {tag} {shots // 1_000_000}M", [py, GEN / "collect_gap_table.py", C / n, "--partial", tag, "--shots", shots,
                        "--seed", 0, "--chunk", GAP_CHUNK, "--workers", W], None, None))
    elif stage == "thresholds":
        cs = ["--cs", f"{CS_Q},{CS_F}"]
        out.append(("select t_h* from the gap tables", [py, GEN / "select_th_from_tables.py", *cs, "--rel-tol", 0.25, "--tc-grid", 29, 31, 33, 35], None, None))
        out.append(("LER-matched baselines", [py, GEN / "run_tc_match.py", *cs, "--epochs", 10_000_000], None, None))
        for n, tag in SWEEPS:
            base = [py, GEN / "collect_threshold_sweep.py", C / n, "--partial", tag, "--feedback-ns", 3100, "--control-latency", profile_of(n),
                    "--epochs", 1_000_000, "--workers", W]
            out.append((f"threshold sweep {n} {tag}", base + ["--tc-grid", 30, 35, 40, 45, 50, 55, 60, 65, 67], None, None))
            out.append((f"two-stage envelope {n} {tag}", base + ["--modes", "pcp", "--th-grid", 30, 35, 40, 45, 50, 55, 60, 65, 67, "cap"], None, None))
    elif stage == "ablation":
        for d1, n in FLAGSHIPS.items():
            out.append((f"gap sensitivity {n}", [py, GEN / "run_gap_sensitivity.py", C / n, "--shots", 2_000_000, "--seed", 0, "--workers", W, "--cmap", "black_red"],
                        (R / "gap_sensitivity" / n / "overall_gap_sens_s0_n2000000.json").exists, None))
        for d2 in (13, 15, 17):
            n = f"end2end_d1=3_d2={d2}_r1=3_r2=0_p=0.001_inj=unitary_b=Y"
            out.append((f"mask strategies {n}", [py, GEN / "run_mask_heuristics.py", C / n, "--ref", "partial_qexp250_cl_wc15", "--size", 336,
                        "--shots", 200_000, "--workers", W], (R / "mask_ablation" / f"heuristics_{n}_N336.json").exists, None))
        for d1, n in FLAGSHIPS.items():
            for k in MASK_SIZE_SWEEP[d1]:
                tag = f"partial_qexp{k}_cl_wc15"
                out.append((f"mask {tag} {n}", [py, alg / "gen_partial_mask_dem.py", "--out", C / n, "--base-size", k, "--q-type", "exp", "--closure",
                            "--contract", "--weight-cutoff", 15.0], (C / n / tag / "contracted.dem").exists, None))
                out.append((f"mask accuracy {n} {tag}", [py, GEN / "collect_mask_ablation.py", C / n, "--partial", tag, "--shots", 200_000, "--seed", 0, "--workers", W],
                            (R / "mask_ablation" / "accuracy" / n / f"{tag}_s0.json").exists, None))
        for n, tag, f in campaign_circuits():
            if not n.startswith("end2end_d1=3_"):
                continue
            out.append((f"mask accuracy {n} {tag}", [py, GEN / "collect_mask_ablation.py", C / n, "--partial", tag, "--shots", 200_000, "--seed", 0, "--workers", W],
                        (R / "mask_ablation" / "accuracy" / n / f"{tag}_s0.json").exists, None))
            out.append((f"pymatching latency {n} {tag}", [py, GEN / "collect_pymatching_latency.py", C / n, "--partial", tag],
                        (R / "mask_ablation" / "pymatching" / f"{n}__{tag}_s0.json").exists, None))
            ftag = f"{f / 1e6:g}MHz"
            masked = R / "mb_characterization" / f"{n}__complete__s0_n1000_{ftag}_realsched_mask-{tag}.json"
            if mb_ok or masked.exists():
                out.append((f"micro-blossom, masked syndrome {n} {tag}", [py, GEN / "run_mb_characterization.py", C / n, "--mask-only", tag, "--shots", 1000,
                            "--seed", 0, "--frequency", f, "--workers", min(W, 8)], masked.exists, None))
        out.append(("mask-size table", [py, GEN / "collect_mask_size_table.py"], None, None))
        out.append(("control-parameter sensitivity, d_cultiv 3 flagship", [py, GEN / "run_control_sensitivity.py", "--only", FLAGSHIPS[3],
                    "--q", 14, 20, "--gbps", 10, 25, "--l-fixed", 100, 157, 300, "--clock", 100, 200, "--legacy-link", "--workers", W], None, None))
    elif stage == "figures":
        out.append(("paper figures and tables", [py, paths.EXPERIMENTS / "reproduce.py", "figures"], None, None))
    return out


def run_full(args) -> int:
    import time
    stages = args.stage or STAGES
    bad = [s for s in stages if s not in STAGES]
    if bad:
        sys.exit(f"unknown stage(s) {bad}; choose from {STAGES}")
    mb_ok = mb_toolchain_available()
    logs = paths.OUT / "full_logs"; logs.mkdir(parents=True, exist_ok=True)
    print(f"full reproduction: stages {stages}; {args.workers} workers; micro-blossom toolchain "
          f"{'available' if mb_ok else 'NOT available'}; logs under {logs}\n")
    for stage in stages:
        cmds = full_stage_commands(stage, args.workers, mb_ok)
        print(f"== {stage} ({len(cmds)} commands)")
        for what, cmd, done, cwd in cmds:
            if args.dry_run:
                print(f"   {what}\n      $ " + " ".join(map(str, cmd))); continue
            if done is not None and done():
                print(f"   {what:70s} done"); continue
            t = time.time()
            sh(cmd, logs / f"{stage}.log", cwd=cwd)
            print(f"   {what:70s} {time.time() - t:6.0f} s")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("figures", help="redraw every figure and table from stored results")
    f.add_argument("--only", nargs="+", metavar="NAME", help="subset: " + " ".join(x[0] for x in FIGURES))
    f.add_argument("--out-dir", type=pathlib.Path, default=R / "figures")
    q = sub.add_parser("quick", help="the method end to end on one small circuit")
    q.add_argument("--out-dir", type=pathlib.Path, default=None, help="workspace (default <MAGICFIRM_OUT>/quick)")
    q.add_argument("--workers", type=int, default=8)
    fu = sub.add_parser("full", help="the paper's campaigns, resumable")
    fu.add_argument("--stage", nargs="+", metavar="NAME", help="subset, in order: " + " ".join(STAGES))
    fu.add_argument("--dry-run", action="store_true", help="print every command, run nothing")
    fu.add_argument("--workers", type=int, default=32,
                    help="estimator workers; the stored results were produced with 32 (the trace partition, and so "
                         "the exact shot set, depends on it; results with another count are statistically equivalent "
                         "and are reused when present)")
    args = ap.parse_args()
    paths.setup_imports()
    if args.cmd == "figures":
        sys.exit(run_figures(args))
    if args.cmd == "quick":
        sys.exit(run_quick(args))
    sys.exit(run_full(args))


if __name__ == "__main__":
    main()
