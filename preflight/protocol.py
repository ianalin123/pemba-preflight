"""The probe protocol. This file is the single source of truth used by sim
corpus generation, the real G1, and the SO-101 rig. Change it in one place."""
from dataclasses import dataclass

PROBE_KP, PROBE_KD = 40.0, 2.0   # low-gain probe mode (faults invisible at stock kp=500)
AMP, FREQ, DURATION_S = 0.25, 0.5, 4.0
DT = 0.002
SETTLE_S = 1.0                   # discard transient; features from steady half


@dataclass(frozen=True)
class JointProbe:
    joint: str          # model/SDK joint name
    neutral_offset: float  # rad added to rest pose, pose 1
    loaded_offset: float   # rad, gravity-loaded pose 2 (derate visibility)


# G1: start with the joints that matter for walking + the waist-lock story.
G1_PROBES = [
    JointProbe("left_hip_pitch_joint", 0.0, 0.6),
    JointProbe("right_hip_pitch_joint", 0.0, 0.6),
    JointProbe("left_knee_joint", 0.0, 0.9),
    JointProbe("right_knee_joint", 0.0, 0.9),
    JointProbe("left_ankle_pitch_joint", 0.0, 0.4),
    JointProbe("right_ankle_pitch_joint", 0.0, 0.4),
    JointProbe("waist_yaw_joint", 0.0, 0.0),
    JointProbe("left_shoulder_pitch_joint", 0.0, 0.8),
    JointProbe("right_shoulder_pitch_joint", 0.0, 0.8),
    JointProbe("left_elbow_joint", 0.0, 0.7),
    JointProbe("right_elbow_joint", 0.0, 0.7),
]

FAULT_CLASSES = ["healthy", "friction", "stiffness", "obstruction", "derate"]


def target(t: float, q0: float) -> float:
    import math
    return q0 + AMP * math.sin(2 * math.pi * FREQ * t)
