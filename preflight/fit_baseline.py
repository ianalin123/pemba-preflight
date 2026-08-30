"""Fit a tier-1 AnomalyModel from a directory of real probe traces (D1).

Usage: python -m preflight.fit_baseline data/d1_cold --out data/baseline_g1.json
Filenames follow g1_probe.py: {joint}_off{offset:.2f}_{timestamp}.npz
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from preflight.anomaly import AnomalyModel
from preflight.features import extract

FNAME = re.compile(r"(?P<joint>.+)_off(?P<off>-?\d+\.\d+)_\d+\.npz")


def load_features(npz_dir: Path) -> dict:
    baselines = defaultdict(list)
    for f in sorted(npz_dir.glob("*.npz")):
        m = FNAME.match(f.name)
        if not m:
            continue
        d = np.load(f)
        trace = {k: d[k] for k in ("t", "target", "q", "tau")}
        key = (m["joint"], float(m["off"]))
        baselines[key].append(extract(trace))
    return dict(baselines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/baseline_g1.json"))
    args = ap.parse_args()

    baselines = load_features(args.npz_dir)
    if not baselines:
        raise SystemExit(f"no parseable .npz traces in {args.npz_dir}")
    for key, feats in sorted(baselines.items()):
        cov = np.median([f["coverage"] for f in feats])
        print(f"  {key[0]} off={key[1]:+.2f}: {len(feats)} reps, median coverage={cov:.3f}")
    model = AnomalyModel.fit(baselines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"baseline saved → {args.out} ({len(baselines)} joint/pose keys)")


if __name__ == "__main__":
    main()
