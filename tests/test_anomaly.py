import numpy as np
from preflight.anomaly import AnomalyModel


def _feats(n, rms_err=0.03, coverage=0.99, rms_tau=0.4):
    rng = np.random.default_rng(0)
    return [{"rms_err": rms_err * rng.uniform(0.9, 1.1), "peak_err": 0.05,
             "coverage": coverage * rng.uniform(0.98, 1.0),
             "rms_tau": rms_tau * rng.uniform(0.9, 1.1), "phase_lag": 0.01}
            for _ in range(n)]


def test_flags_fault_not_healthy():
    m = AnomalyModel.fit({("knee", 0.0): _feats(20)})
    ok, z = m.score(("knee", 0.0), _feats(1, coverage=0.98)[0])
    bad, zb = m.score(("knee", 0.0), _feats(1, rms_err=0.13, coverage=0.4, rms_tau=3.0)[0])
    assert ok and not bad and zb > z
