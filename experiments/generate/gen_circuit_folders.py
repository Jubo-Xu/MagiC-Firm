#!/usr/bin/env python3
"""Generate circuit folders: the noisy end2end-inplace-distillation cultivation
circuit for every (d1, d2, p, injection) combination, bootstrapped with
algorithms/get_desaturated_dem_with_obs_detector.py (original.dem,
desaturated_with_obs.dem, gap_circuit.stim, postselected_detectors.json).

    python experiments/generate/gen_circuit_folders.py            # the paper grid (30 circuits)
    python experiments/generate/gen_circuit_folders.py --d1 3 --d2 11 --p 0.001 --out-root DIR

Defaults: p in {5e-4, 7e-4, 9e-4, 1e-3, 1.5e-3}, d1 = 3 (r1 = d1), d2 in
{11, 13, 15, 17, 19, 21} (r2 = 0), unitary injection, basis Y. Re-running
skips folders that already have postselected_detectors.json.
"""
import argparse
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702

from circuit_generator import CircuitGenerator

# the paper grid
P_LIST = [0.0005, 0.0007, 0.0009, 0.001, 0.0015]
D1_LIST = [3]
D2_LIST = [11, 13, 15, 17, 19, 21]
INJ_LIST = ["unitary"]
BASIS, GATESET = "Y", "css"
NOISE_MODEL = "uniform-depolarizing"

DESAT_CLI = paths.ALGORITHMS / "get_desaturated_dem_with_obs_detector.py"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p", type=float, nargs="+", default=P_LIST, help="physical error rates")
    ap.add_argument("--d1", type=int, nargs="+", default=D1_LIST, help="cultivation distances (r1 = d1)")
    ap.add_argument("--d2", type=int, nargs="+", default=D2_LIST, help="escape distances (r2 = 0)")
    ap.add_argument("--inj", nargs="+", default=INJ_LIST, choices=["unitary", "degenerate"], help="injection protocol")
    ap.add_argument("--out-root", type=pathlib.Path, default=paths.CIRCUITS, help="parent of the circuit folders")
    args = ap.parse_args()
    OUT_ROOT = args.out_root
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    combos = [(p, d1, d2, inj) for p in args.p for d1 in args.d1 for d2 in args.d2 for inj in args.inj]
    print(f"{len(combos)} circuit folders -> {OUT_ROOT}")
    done = skipped = failed = 0
    t_start = time.time()

    for i, (p, d1, d2, inj) in enumerate(combos, 1):
        name = f"end2end_d1={d1}_d2={d2}_r1={d1}_r2=0_p={p:g}_inj={inj}_b={BASIS}"
        folder = OUT_ROOT / name
        if (folder / "postselected_detectors.json").is_file():
            skipped += 1
            print(f"[{i}/{len(combos)}] {name}: exists, skip", flush=True)
            continue

        t0 = time.time()
        try:
            cg = CircuitGenerator.from_input_params(
                basis=BASIS, gateset=GATESET,
                circuit_type="end2end-inplace-distillation",
                noise_model=NOISE_MODEL, noise_strength=p,
                injection_protocol=inj,
                r_in_escape=d1, r_post_escape=0, d1=d1, d2=d2, HasNoise=True,
            )
            cg.generate()
            folder.mkdir(parents=True, exist_ok=True)
            stim_path = folder / f"{name}.stim"
            cg.noisy_circuit.to_file(stim_path)

            r = subprocess.run(
                [sys.executable, str(DESAT_CLI),
                 "--circuit", str(stim_path), "--out", str(folder)],
                capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"desaturation CLI failed:\n{r.stderr[-800:]}")
            for line in r.stdout.splitlines():
                if line.startswith("  WARNING"):
                    print(f"    {line.strip()}", flush=True)
            done += 1
            print(f"[{i}/{len(combos)}] {name}: OK "
                  f"({cg.noisy_circuit.num_detectors} dets, {time.time()-t0:.0f}s)",
                  flush=True)
        except Exception as e:
            failed += 1
            print(f"[{i}/{len(combos)}] {name}: FAILED — {type(e).__name__}: {e}",
                  flush=True)

    print(f"\ndone={done} skipped={skipped} failed={failed} "
          f"total {(time.time()-t_start)/60:.1f} min")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
