import sys
from pathlib import Path
import numpy as np
from preflight.fit_baseline import load_features
from preflight.anomaly import AnomalyModel, THRESHOLD

b = load_features(Path(sys.argv[1] if len(sys.argv) > 1 else "data/d1_cold"))
fails = 0
total = 0
zs = []
for key, feats in sorted(b.items()):
    if len(feats) < 2:
        continue
    for i in range(len(feats)):
        train = {key: [f for j, f in enumerate(feats) if j != i]}
        m = AnomalyModel.fit(train)
        ok, z = m.score(key, feats[i])
        zs.append(z)
        total += 1
        if not ok:
            fails += 1
            print(f"  FAIL {key[0]} {key[1]:+.2f} rep{i}: z={z:.2f}")
zs = np.array(zs)
print(f"\nLOO holdout: {total - fails}/{total} healthy reps pass (threshold={THRESHOLD})")
print(f"z distribution: median={np.median(zs):.2f}  p90={np.percentile(zs, 90):.2f}  max={zs.max():.2f}")
