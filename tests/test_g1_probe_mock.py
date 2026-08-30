import subprocess, sys
from pathlib import Path


def test_mock_probe_end_to_end(tmp_path):
    r = subprocess.run(
        [sys.executable, "preflight/real/g1_probe.py", "--mock", "--reps", "1",
         "--joints", "left_knee_joint", "--out", str(tmp_path)],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert list(tmp_path.glob("left_knee_joint_*.npz"))
