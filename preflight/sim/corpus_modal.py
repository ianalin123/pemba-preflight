"""D3 corpus on Modal: parallel probe rollouts with randomized faults → features
→ train tier-2 classifier + severity regressor. CPU-parallel plain MuJoCo
(each rollout is 2000 tiny steps; massive fan-out beats GPU vmap complexity —
switch to MJX only if throughput disappoints).

Run:  uv run --extra modal modal run preflight/sim/corpus_modal.py::gen
      uv run --extra modal modal run preflight/sim/corpus_modal.py::train
"""
import modal

app = modal.App("preflight-corpus")
vol = modal.Volume.from_name("preflight-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("mujoco==3.12.0", "numpy==2.5.2", "scikit-learn==1.9.0", "joblib")
    .run_commands(
        "git clone --depth 1 --filter=blob:none --sparse "
        "https://github.com/google-deepmind/mujoco_menagerie.git /menagerie && "
        "cd /menagerie && git sparse-checkout set unitree_g1")
    .env({"MUJOCO_GL": "disabled"})
    .add_local_python_source("preflight")
)

FAULT_GRID = {  # class → (param, low, high); severity in [0,1] maps linearly
    "friction": ("friction_add", 0.5, 6.0),
    "stiffness": ("damping_add", 1.0, 15.0),
    "obstruction": ("clamp_rad", 0.15, 0.02),  # smaller clamp = worse
    "derate": ("gain_x", 0.8, 0.3),
}


@app.function(image=image, cpu=2, timeout=1200)
def rollout_batch(args: list[dict]) -> list[dict]:
    import preflight.sim.g1_probe_env as env
    env.MJCF = "/menagerie/unitree_g1/g1.xml"
    from preflight.features import extract
    out = []
    for a in args:
        f = env.Fault(cls=a["cls"], severity=a["severity"], **a["params"])
        trace = env.run_probe(a["joint"], a["pose_offset"], f,
                              seed=a["seed"], randomize=True)
        feats = extract(trace)
        out.append({k: v for k, v in a.items() if k != "params"} | feats)
    return out


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def gen(n_per_class: int = 400):
    import json, random
    from preflight.protocol import G1_PROBES, FAULT_CLASSES
    jobs = []
    rnd = random.Random(0)
    for p in G1_PROBES:
        for off in (p.neutral_offset, p.loaded_offset):
            for cls in FAULT_CLASSES:
                for k in range(n_per_class // len(G1_PROBES)):
                    sev = rnd.random() if cls != "healthy" else 0.0
                    params = {}
                    if cls != "healthy":
                        name, lo, hi = FAULT_GRID[cls]
                        params[name] = lo + sev * (hi - lo)
                    jobs.append({"joint": p.joint, "pose_offset": off, "cls": cls,
                                 "severity": sev, "params": params,
                                 "seed": rnd.randrange(1 << 30)})
    rnd.shuffle(jobs)
    chunks = [jobs[i:i + 40] for i in range(0, len(jobs), 40)]
    rows = []
    for batch in rollout_batch.map(chunks):
        rows.extend(batch)
    with open("/data/corpus.json", "w") as fh:
        json.dump(rows, fh)
    vol.commit()
    print(f"corpus: {len(rows)} rows")


@app.function(image=image, volumes={"/data": vol}, timeout=1200)
def train():
    import json
    import numpy as np
    from joblib import dump
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    from preflight.features import FEATURE_NAMES
    from preflight.protocol import G1_PROBES
    rows = json.load(open("/data/corpus.json"))
    joints = sorted({p.joint for p in G1_PROBES})
    X = np.array([[r[k] for k in FEATURE_NAMES] + [joints.index(r["joint"]), r["pose_offset"]]
                  for r in rows], dtype=np.float32)
    y_cls = np.array([r["cls"] for r in rows])
    y_sev = np.array([r["severity"] for r in rows], dtype=np.float32)
    clf = HistGradientBoostingClassifier(max_iter=300)
    print("cls cv acc:", cross_val_score(clf, X, y_cls, cv=3).mean())
    clf.fit(X, y_cls)
    reg = HistGradientBoostingRegressor(max_iter=300).fit(X, y_sev)
    dump({"clf": clf, "reg": reg, "joints": joints}, "/data/tier2.pkl")
    vol.commit()
    print("saved /data/tier2.pkl")
