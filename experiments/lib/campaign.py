# campaign.py — helpers shared by the campaign drivers: running one estimator
# job as a subprocess, deciding whether a stored result can be reused, result
# file naming and summaries, and the mask-campaign essentials lookup.

import datetime
import json
import pathlib
import re
import subprocess
import sys

import paths
from naming import freq_tag

REPO = paths.REPO
PY = sys.executable
CIRCUITS = paths.CIRCUITS
RESULT = paths.RESULT / "runtime_estimation"
MB_RESULT = paths.RESULT / "mb_characterization"
MASK_SUMMARY = paths.RESULT / "mask_campaign" / "summary.json"
ESTIMATOR = paths.EXPERIMENTS / "generate" / "run_runtime_estimation.py"

TC = 35.0                            # complete-gap threshold of every campaign run
SIGMA = 2.0                          # Poisson sigma of the legacy th* rule
MB_SHOTS, MB_SEED = 1000, 0          # the campaign's MB characterization (run_runtime_estimation defaults)

# mask campaign
WC = 15.0                            # contraction weight cutoff
ESSENTIALS = ("logical_ambiguity",)  # the Q_exp base is the only essentials consumer


def log(msg):
    print(f"[{datetime.datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def profile_path(name, args):
    """Control-latency profile of a circuit for --cs Q,F."""
    return CIRCUITS / name / "control_system" / f"latency_q{args.cs_q}_f{args.cs_f}.json"


def expected_path(name, variant, mode_tag, args):
    """Result file of this job (mirrors run_runtime_estimation's naming). The
    worker count only changes the trace partition, so a result produced with
    a different --workers is statistically equivalent; such a file is returned
    when the exact name is absent, instead of re-running the job."""
    d = RESULT / name
    base = f"{variant}__{mode_tag}__lat-mb_trace-s0_e{args.epochs}"
    exact = d / f"{base}{f'_w{args.workers}' if args.workers > 1 else ''}{args.cs_tag}.json"
    if exact.exists():
        return exact
    for cand in sorted(d.glob(f"{base}*{args.cs_tag}.json")):
        if re.fullmatch(r"(_w\d+)?" + re.escape(args.cs_tag), cand.name[len(base):-len(".json")]):
            return cand
    return exact


def run_estimator(name, f, args, mode_args, log_path):
    """Run one estimator job as a subprocess; return the result JSON path."""
    cmd = [PY, ESTIMATOR, CIRCUITS / name, *mode_args,
           "--epochs", args.epochs, "--latency", "mb", "--mb-freq", f"{f:g}", "--trace",
           "--workers", args.workers]
    if args.cs:
        cmd += ["--control-latency", profile_path(name, args)]
    with open(log_path, "a") as lf:
        lf.write(f"\n$ {' '.join(map(str, cmd))}\n"); lf.flush()
        r = subprocess.run([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, cwd=REPO)
        lf.write(r.stdout)
    m = re.search(r"^saved (\S+\.json)", r.stdout, re.M)
    if r.returncode != 0 or not m:
        raise RuntimeError(f"estimator failed (rc {r.returncode}); see {log_path}")
    return REPO / m.group(1)


def _same_path(stored, wanted) -> bool:
    """Stored path (absolute or repo-relative) names the wanted file (None matches None)."""
    if not stored and not wanted:
        return True
    if not stored or not wanted:
        return False
    return paths.rel(pathlib.Path(stored) if pathlib.Path(stored).is_absolute() else REPO / stored) == paths.rel(wanted)


def valid(path, name, variant, f, args, mode_args=None):
    """True if a stored estimator result was produced from the CURRENTLY requested
    inputs: the fingerprint (complete DEM, mask, contracted DEM, MB files at the
    requested clock/seed/shots, control profile) must equal config.inputs, and the
    stored latency, estimator and gating settings must equal the request. The
    file name alone is never sufficient."""
    path = pathlib.Path(path)
    if not path.exists():
        return False
    from gap_table import result_fingerprint
    cfg = json.loads(path.read_text()).get("config", {})
    partial = None if variant == "complete" else variant
    ctrl = profile_path(name, args) if args.cs else None
    want = result_fingerprint(CIRCUITS / name, partial, mb_shots=MB_SHOTS, mb_seed=MB_SEED, mb_freq_hz=f,
                              control_latency=ctrl)
    if cfg.get("inputs") != want:
        return False
    lat, est = cfg.get("latency", {}), cfg.get("estimator", {})
    if (lat.get("source") != "mb" or lat.get("mb_freq_hz") != f or lat.get("mb_shots") != MB_SHOTS
            or lat.get("mb_seed") != MB_SEED or est.get("epochs") != args.epochs
            or _same_path(est.get("control_latency"), ctrl) is False):
        return False
    if mode_args is not None:                      # gating must be exactly the requested one
        g = cfg.get("gating", {}); m = dict(zip(mode_args[::2], mode_args[1::2]))
        want_g = {"mode": m.get("--mode"), "tc": float(m["--tc"]) if "--tc" in m else None,
                  "tl": float(m["--tl"]) if "--tl" in m else None, "th": float(m["--th"]) if "--th" in m else None}
        if any(g.get(k) != v for k, v in want_g.items()):
            return False
    return True


def summarize(path):
    """Preparation-time breakdown, LER and accept rate of one result file."""
    j = json.loads(path.read_text())
    ready = j["config"]["estimator"]["stage_names"].index("ready")
    us = j["records_until_success"]
    return {"file": paths.rel(path), "prep_us": us["total"]["mean"][ready] / 1000,
            "gate_us": us["gate"]["mean"][ready] / 1000, "feedback_us": us["feedback"]["mean"][ready] / 1000,
            "gap_decode_us": us["gap_decode"]["mean"][ready] / 1000,
            "p99_us": j["accepted_decision_times_ns"]["p99"] / 1000,
            "ler": j["logical_error_rate"], "errors": int(round(j["logical_error_rate"] * j["epoch"])),
            "accept": j["accept_rate"], "epochs": j["epoch"],
            "tier3_share": j["tier_counts"]["tier3"] / max(j["attempt_count"], 1)}


def write_md(summary, suffix=""):
    """summary<suffix>.md next to summary<suffix>.json."""
    rows = ["| circuit | f | mask | complete µs (gate/fb/decode) | th* | partial µs (gate/fb/decode) | reduction | LER complete / partial |",
            "|---|---|---|---|---|---|---|---|"]
    for name, r in sorted(summary.items(), key=lambda kv: (kv[1]["d1"], kv[1]["d2"], kv[1]["p"])):
        b, p = r["baseline"], r.get("partial")
        short = name.replace("end2end_", "").replace("_inj=unitary_b=Y", "")
        fmt = lambda s: f"{s['prep_us']:.2f} ({s['gate_us']:.1f}/{s['feedback_us']:.1f}/{s['gap_decode_us']:.1f})"
        rows.append(f"| {short} | {freq_tag(r['frequency_hz'])} | {r['partial_tag'].replace('partial_', '')} | {fmt(b)} | "
                    + (f"{r['th_star']} | {fmt(p)} | {r['reduction']:.1%} | {b['ler']:.1e} / {p['ler']:.1e} |" if p else "— | — | — | — |"))
    (RESULT / f"summary{suffix}.md").write_text("\n".join(rows) + "\n")


def pick_essentials(folder, kind):
    """Essentials file of `kind` in a circuit folder. If several exist, prefer the
    canonical threshold closest to 35, then the most shots."""
    files = sorted(folder.glob(f"*_{kind}_*.json"))
    if len(files) == 1:
        return files[0]
    def key(f):
        th = re.search(r"_th([0-9.]+)_", f.name); sh = re.search(r"_shots(\d+)", f.name)
        return (abs(float(th.group(1)) - 35) if th else 0.0, -(int(sh.group(1)) if sh else 0))
    return sorted(files, key=key)[0]


def essentials_flags(folder):
    """--essentials-<kind> <file> flags for gen_partial_mask_dem / run_eval_partial_mask."""
    return [x for kind in ESSENTIALS
            for x in (f"--essentials-{kind.replace('_', '-')}", pick_essentials(folder, kind))]
