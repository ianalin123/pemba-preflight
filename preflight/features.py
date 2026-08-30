"""Trace → features. Deliberately identical for sim, G1, SO-101: sim2real lives here."""
import numpy as np

FEATURE_NAMES = ["rms_err", "peak_err", "coverage", "rms_tau", "phase_lag"]


def extract(trace: dict) -> dict:
    t = np.asarray(trace["t"])
    tgt = np.asarray(trace["target"])
    q = np.asarray(trace["q"])
    tau = np.asarray(trace["tau"])
    half = len(t) // 2
    t, tgt, q, tau = t[half:], tgt[half:], q[half:], tau[half:]
    err = tgt - q
    span_t = tgt.max() - tgt.min()
    coverage = (q.max() - q.min()) / span_t if span_t > 1e-9 else 0.0
    # phase lag via cross-correlation peak (samples → seconds)
    tgt_c, q_c = tgt - tgt.mean(), q - q.mean()
    denom = np.sqrt((tgt_c ** 2).sum() * (q_c ** 2).sum())
    lag = 0.0
    if denom > 1e-9:
        xc = np.correlate(q_c, tgt_c, mode="full")
        lag = float((np.argmax(xc) - (len(tgt_c) - 1)) * (t[1] - t[0]))
    return {
        "rms_err": float(np.sqrt(np.mean(err ** 2))),
        "peak_err": float(np.max(np.abs(err))),
        "coverage": float(coverage),
        "rms_tau": float(np.sqrt(np.mean(tau ** 2))),
        "phase_lag": lag,
    }


def to_vector(f: dict) -> np.ndarray:
    return np.array([f[k] for k in FEATURE_NAMES], dtype=np.float32)
