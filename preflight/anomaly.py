"""Tier 1: per-(joint,pose) robust z-score against the robot's OWN healthy
baseline. Trained on real data only → immune to sim2real gap by construction."""
import json
import numpy as np
from preflight.features import FEATURE_NAMES

THRESHOLD = 6.0  # max |robust z|; tune on healthy holdout, favor low false-positive


class AnomalyModel:
    def __init__(self, stats: dict):
        self.stats = stats  # key -> (median[5], mad[5])

    @classmethod
    def fit(cls, baselines: dict) -> "AnomalyModel":
        stats = {}
        for key, feats in baselines.items():
            X = np.array([[f[k] for k in FEATURE_NAMES] for f in feats])
            med = np.median(X, axis=0)
            mad = np.median(np.abs(X - med), axis=0) * 1.4826 + 1e-6
            stats[str(key)] = (med.tolist(), mad.tolist())
        return cls(stats)

    def score(self, key, feat: dict) -> tuple[bool, float]:
        med, mad = self.stats[str(key)]
        x = np.array([feat[k] for k in FEATURE_NAMES])
        z = float(np.max(np.abs((x - np.array(med)) / np.array(mad))))
        return z <= THRESHOLD, z

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.stats, fh)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls(json.load(fh))
