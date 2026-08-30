import numpy as np
from preflight.features import extract
from preflight.protocol import AMP


def _trace(track=1.0, force=0.3, n=2000):
    t = np.arange(n) * 0.002
    tgt = AMP * np.sin(2 * np.pi * 0.5 * t)
    q = track * tgt
    tau = force * np.ones(n)
    return {"t": t, "target": tgt, "q": q, "tau": tau}


def test_healthy_vs_clamped_coverage():
    healthy = extract(_trace(track=0.95))
    clamped = extract(_trace(track=0.15))  # obstruction: barely moves
    assert healthy["coverage"] > 0.9
    assert clamped["coverage"] < 0.3
    assert clamped["rms_err"] > healthy["rms_err"] * 2


def test_feature_vector_stable_order():
    f = extract(_trace())
    assert list(f.keys()) == ["rms_err", "peak_err", "coverage", "rms_tau", "phase_lag"]
