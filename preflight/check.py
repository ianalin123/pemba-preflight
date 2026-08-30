"""Pemba Pre-Flight CLI — 60-second proprioceptive self-check for Unitree G1.

Usage:
    python -m preflight.check [--source sim|npz:PATH] [--joints J1 J2 ...]
        [--inject CLASS:JOINT:SEVERITY] [--baseline PATH] [--json OUT.json]
"""
import argparse
import json
import sys
from pathlib import Path

from preflight.anomaly import AnomalyModel, THRESHOLD
from preflight.features import extract
from preflight.protocol import G1_PROBES
from preflight.report import go_no_go, verdict_sentence

# FAULT_GRID copied from preflight/sim/corpus_modal.py to avoid pulling in
# the `modal` dependency at import time (modal is an optional extra).
# Keep in sync with corpus_modal.py if probe params change.
FAULT_GRID = {  # class → (param, low, high); severity in [0,1] maps linearly
    "friction": ("friction_add", 0.5, 6.0),
    "stiffness": ("damping_add", 1.0, 15.0),
    "obstruction": ("clamp_rad", 0.15, 0.02),  # smaller clamp = worse
    "derate": ("gain_x", 0.8, 0.3),
}

DEFAULT_JOINTS = ["left_knee_joint", "waist_yaw_joint", "left_shoulder_pitch_joint"]


def _build_fault(cls: str, severity: float):
    from preflight.sim.g1_probe_env import Fault
    if cls == "healthy":
        return Fault()
    param, lo, hi = FAULT_GRID[cls]
    value = lo + severity * (hi - lo)
    return Fault(cls=cls, severity=severity, **{param: value})


def _run_sim(probes, inject_map: dict) -> dict[tuple, dict]:
    """Return {(joint, offset): trace} for all (joint, pose) pairs.

    Uses randomize=True with seed=99 so the test probe is drawn from the
    same distribution as the randomized baseline — keeps tier-1 z-scores
    well-behaved for healthy joints while faults still deviate strongly.
    """
    from preflight.sim.g1_probe_env import Fault, run_probe
    traces = {}
    for p in probes:
        for pose in (p.neutral_offset, p.loaded_offset):
            fault = inject_map.get(p.joint, Fault())
            # randomize=True: match the domain-randomized distribution used
            # when building the quick-fit baseline.
            trace = run_probe(p.joint, pose_offset=pose, fault=fault,
                              seed=99, randomize=True)
            traces[(p.joint, pose)] = trace
    return traces


def _run_npz(source_path: str, probes) -> dict[tuple, dict]:
    """Load .npz files from directory; match joint+offset to probes."""
    import re
    import numpy as np
    traces = {}
    npz_dir = Path(source_path)
    wanted = {(p.joint, p.neutral_offset) for p in probes} | {(p.joint, p.loaded_offset) for p in probes}
    pattern = re.compile(r"^(?P<joint>.+)_off(?P<offset>-?\d+\.\d+)_.*\.npz$")
    for f in sorted(npz_dir.glob("*.npz")):
        m = pattern.match(f.name)
        if not m:
            continue
        joint = m.group("joint")
        offset = float(m.group("offset"))
        if (joint, offset) not in wanted:
            continue
        data = np.load(f)
        traces[(joint, offset)] = {k: data[k] for k in ("t", "target", "q", "tau")}
    return traces


def _fit_sim_baseline(probes) -> AnomalyModel:
    """Fit a throwaway tier-1 baseline from 20 healthy sim runs per (joint, pose).

    Uses 20 domain-randomized runs to get stable robust-MAD estimates across
    all 5 features. The MAD floor in AnomalyModel (+1e-6) is sufficient once
    the spread across seeds provides realistic between-run variance.
    """
    from preflight.sim.g1_probe_env import Fault, run_probe
    print("[check] No baseline found — fitting quick sim baseline (20 runs × joint × pose)…")
    baselines: dict = {}
    for p in probes:
        for pose in (p.neutral_offset, p.loaded_offset):
            key = (p.joint, pose)
            feats_list = []
            for seed in range(20):
                trace = run_probe(p.joint, pose_offset=pose, fault=Fault(),
                                  seed=seed, randomize=True)
                feats_list.append(extract(trace))
            baselines[key] = feats_list
    return AnomalyModel.fit(baselines)


def _load_tier2():
    """Load tier-2 model if present; return None if absent."""
    tier2_path = Path("data/tier2.pkl")
    if not tier2_path.exists():
        return None
    try:
        from joblib import load
        return load(tier2_path)
    except Exception as e:
        print(f"[check] Warning: could not load tier-2 model: {e}", file=sys.stderr)
        return None


def _classify_heuristic(feat: dict, z: float, inject_cls: str | None) -> tuple[str, float]:
    """Tier-2 fallback: heuristic classification when model is absent."""
    # Coverage collapse is the clearest obstruction signal.
    if feat["coverage"] < 0.5:
        return "obstruction", min(1.0, z / THRESHOLD)
    if inject_cls and inject_cls != "healthy":
        return inject_cls, min(1.0, z / THRESHOLD)
    return "unknown", min(1.0, z / THRESHOLD)


def main():
    parser = argparse.ArgumentParser(description="Pemba Pre-Flight check")
    parser.add_argument("--source", default="sim",
                        help="sim | npz:PATH")
    parser.add_argument("--joints", nargs="+", default=DEFAULT_JOINTS,
                        help="Joints to probe (sim/npz filter)")
    parser.add_argument("--inject", default=None,
                        help="CLASS:JOINT:SEVERITY — inject fault (sim only)")
    parser.add_argument("--baseline", default=None,
                        help="Path to AnomalyModel JSON baseline")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="Path for JSON results dump")
    args = parser.parse_args()

    # Resolve probes
    probes = [p for p in G1_PROBES if p.joint in args.joints]
    if not probes:
        sys.exit(f"[check] No matching probes for joints: {args.joints}")

    # Parse inject
    inject_map: dict = {}
    inject_cls_by_joint: dict[str, str] = {}
    if args.inject:
        parts = args.inject.split(":")
        if len(parts) != 3:
            sys.exit("[check] --inject must be CLASS:JOINT:SEVERITY")
        inj_cls, inj_joint, inj_sev = parts[0], parts[1], float(parts[2])
        inject_map[inj_joint] = _build_fault(inj_cls, inj_sev)
        inject_cls_by_joint[inj_joint] = inj_cls

    # Collect traces
    if args.source == "sim":
        print("[check] Running sim probes…")
        traces = _run_sim(probes, inject_map)
    elif args.source.startswith("npz:"):
        npz_path = args.source[4:]
        print(f"[check] Loading npz traces from {npz_path}…")
        traces = _run_npz(npz_path, probes)
    else:
        sys.exit(f"[check] Unknown --source: {args.source}")

    # Load or build tier-1 baseline
    if args.baseline:
        model = AnomalyModel.load(args.baseline)
    else:
        model = _fit_sim_baseline(probes)

    # Load tier-2 if available
    tier2 = _load_tier2()
    if tier2:
        print("[check] Tier-2 model loaded.")
    else:
        print("[check] No tier-2 model — using heuristic fallback for fault classification.")

    # Pipeline
    results = []
    joint_verdict: dict[str, tuple[str, str, float]] = {}  # joint → (joint, cls, severity)

    for p in probes:
        pose_results = []
        for pose in (p.neutral_offset, p.loaded_offset):
            key = (p.joint, pose)
            if key not in traces:
                print(f"[check] Warning: no trace for {p.joint} offset={pose:.2f}, skipping.")
                continue
            feat = extract(traces[key])
            ok, z = model.score(str(key), feat)

            if ok:
                cls, severity = "healthy", 0.0
            else:
                # Tier-2 classification
                if tier2:
                    import numpy as np
                    joints_list = tier2["joints"]
                    j_idx = joints_list.index(p.joint) if p.joint in joints_list else 0
                    vec = list(feat.values()) + [j_idx, pose]
                    cls = str(tier2["clf"].predict([vec])[0])
                    severity = float(tier2["reg"].predict([vec])[0])
                    severity = max(0.0, min(1.0, severity))
                else:
                    cls, severity = _classify_heuristic(feat, z, inject_cls_by_joint.get(p.joint))

            pose_results.append((cls, severity, feat, z))

        if not pose_results:
            continue

        # Merge poses: worst-case (non-healthy > healthy; higher severity wins)
        best = sorted(pose_results, key=lambda r: (0 if r[0] == "healthy" else 1, r[1]))[-1]
        final_cls, final_sev = best[0], best[1]

        sentence = verdict_sentence(p.joint, final_cls, final_sev)
        print(sentence)
        joint_verdict[p.joint] = (p.joint, final_cls, final_sev)

        # Collect per-pose data for JSON
        for i, (pose, (cls, sev, feat, z)) in enumerate(zip(
                (p.neutral_offset, p.loaded_offset), pose_results)):
            results.append({
                "joint": p.joint,
                "pose_offset": pose,
                "cls": cls,
                "severity": sev,
                "z_score": z,
                **feat,
            })

    verdicts = list(joint_verdict.values())
    summary = go_no_go(verdicts)
    print(summary)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"verdicts": verdicts, "details": results}, fh, indent=2)
        print(f"[check] Results written to {args.json_out}")

    sys.exit(0 if summary.startswith("Pre-flight PASS") else 1)


if __name__ == "__main__":
    main()
