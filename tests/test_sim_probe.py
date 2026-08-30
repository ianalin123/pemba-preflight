from preflight.sim.g1_probe_env import run_probe, Fault
from preflight.features import extract


def test_obstruction_separable_on_knee():
    healthy = extract(run_probe("left_knee_joint"))
    clamped = extract(run_probe("left_knee_joint", fault=Fault(cls="obstruction", clamp_rad=0.05)))
    assert healthy["coverage"] - clamped["coverage"] > 0.3
    assert clamped["rms_tau"] > healthy["rms_tau"] * 3


def test_derate_needs_loaded_pose():
    h = extract(run_probe("left_knee_joint", pose_offset=0.9))
    d = extract(run_probe("left_knee_joint", pose_offset=0.9, fault=Fault(cls="derate", gain_x=0.4)))
    assert d["rms_err"] > h["rms_err"] * 1.25
