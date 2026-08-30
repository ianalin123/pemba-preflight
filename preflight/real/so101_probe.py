"""Per-joint probe on the SO-101 arm via lerobot's FeetechMotorsBus.

Safety contract:
- one joint at a time, low P gain, small amplitude (protocol.AMP = 0.25 rad)
- SIGINT/SIGTERM and finally → hold current position with torque ON (a
  torqued-off arm falls), then restore the original P_Coefficient
- start/stop the script with the arm at rest or supported: the P_Coefficient
  restore needs a brief torque-off window (Feetech EEPROM write)

Usage:  python3 so101_probe.py --port /dev/ttyACM0 --out data/so101_run01
        python3 so101_probe.py --mock --reps 1 --out /tmp/so101_mock
"""
import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preflight.protocol import (DURATION_S, JointProbe, PROBE_KD,  # noqa: E402
                                PROBE_KP, target)

DT_SO101 = 0.02        # 50 Hz — serial bus can't do the G1's 500 Hz
RAMP_S = 2.0
COUNTS_PER_RAD = 4096 / (2 * math.pi)  # VERIFY on rig: STS3215 = 4096 counts/rev, 1 count ≈ 0.088°
COUNT_CENTER = 2048                     # VERIFY on rig: mid-range after lerobot-setup-motors homing
COUNT_EDGE_MARGIN = 96                  # keep goals away from the 0/4095 hard stops
PROBE_P_COEFFICIENT = 8                 # VERIFY on rig: stock P_Coefficient (lerobot so_follower writes 16; factory 32)
LOAD_SIGN_BIT = 0x400                   # VERIFY on rig: Present_Load = 10-bit magnitude, bit 10 = direction
LOAD_SCALE = 1000.0                     # tau normalized to roughly [-1, 1] (load is ‰ of stall torque)

SO101_MOTOR_IDS = {  # VERIFY on rig: default lerobot-setup-motors ID assignment (gripper=6, skipped)
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
}

SO101_PROBES = [
    JointProbe("shoulder_pan", 0.0, 0.0),   # yaw joint, no gravity-loaded pose
    JointProbe("shoulder_lift", 0.0, 0.5),
    JointProbe("elbow_flex", 0.0, 0.5),
    JointProbe("wrist_flex", 0.0, 0.4),
    JointProbe("wrist_roll", 0.0, 0.0),
]


class SO101Bus:
    """Thin wrapper so the same probe loop runs against mock or real bus.

    Works entirely in radians centered on COUNT_CENTER; all bus traffic uses
    normalize=False raw counts so no lerobot calibration file is needed.
    """

    def __init__(self, port: str):
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus
        motors = {
            name: Motor(id=mid, model="sts3215", norm_mode=MotorNormMode.RANGE_M100_100)
            for name, mid in SO101_MOTOR_IDS.items()
        }
        self._bus = FeetechMotorsBus(port=port, motors=motors)
        self._bus.connect()  # VERIFY on rig: pings all IDs; uncalibrated is OK since every read/write is normalize=False
        self._orig_p: dict[str, int] = {}
        self._bus.enable_torque()

    def read(self, joint: str) -> tuple[float, float]:
        raw_q = float(self._bus.read("Present_Position", joint, normalize=False))
        raw_load = int(self._bus.read("Present_Load", joint, normalize=False))
        sign = -1.0 if raw_load & LOAD_SIGN_BIT else 1.0
        tau = sign * (raw_load & (LOAD_SIGN_BIT - 1)) / LOAD_SCALE
        return (raw_q - COUNT_CENTER) / COUNTS_PER_RAD, tau

    def read_temp(self, joint: str) -> float:
        return float(self._bus.read("Present_Temperature", joint, normalize=False))

    def write_goal(self, joint: str, q_rad: float):
        counts = int(round(COUNT_CENTER + q_rad * COUNTS_PER_RAD))
        counts = max(COUNT_EDGE_MARGIN, min(4095 - COUNT_EDGE_MARGIN, counts))
        self._bus.write("Goal_Position", joint, counts, normalize=False)

    def lower_gains(self, joints: list[str]):
        # VERIFY on rig: P_Coefficient is EEPROM; lerobot so_follower writes it
        # inside torque_disabled(). Arm must be at rest for this brief window.
        with self._bus.torque_disabled():
            for j in joints:
                self._orig_p[j] = int(self._bus.read("P_Coefficient", j, normalize=False))
                self._bus.write("P_Coefficient", j, PROBE_P_COEFFICIENT, normalize=False)
        print(f"P lowered to {PROBE_P_COEFFICIENT} (originals: {self._orig_p})")

    def restore_gains(self):
        if not self._orig_p:
            return
        with self._bus.torque_disabled():  # <100 ms torque-off; hold() re-asserts goals right after
            for j, p in self._orig_p.items():
                self._bus.write("P_Coefficient", j, p, normalize=False)
        print(f"P restored: {self._orig_p}")
        self._orig_p = {}

    def hold(self, joints: list[str]):
        for j in joints:
            q, _ = self.read(j)
            self.write_goal(j, q)


class MockSO101:
    """Adapter over mock_lowlevel.MockRobot so the loop dry-runs without lerobot."""

    def __init__(self):
        from preflight.real.mock_lowlevel import MockRobot
        self._m = MockRobot()
        self._idx = {name: i for i, name in enumerate(SO101_MOTOR_IDS)}

    def read(self, joint):
        q, dq, tau, temp = self._m.read(self._idx[joint])
        return q, tau

    def read_temp(self, joint):
        return self._m.read(self._idx[joint])[3]

    def write_goal(self, joint, q_rad):
        i = self._idx[joint]
        self._m.command({i: q_rad}, {i: PROBE_KP}, {i: PROBE_KD}, dt=DT_SO101)

    def lower_gains(self, joints):
        print("mock: P lowered")

    def restore_gains(self):
        print("mock: P restored")

    def hold(self, joints):
        self._m.damp_all()


def probe_joint(bot, name: str, pose_offset: float, out: Path):
    q_start, _ = bot.read(name)
    if not math.isfinite(q_start) or not (-math.pi <= q_start <= math.pi):
        print(f"WARNING: {name} q_start={q_start!r} is non-finite or out of [-π, π] — holding and aborting")
        bot.hold([name])
        raise RuntimeError(f"Unsafe q_start={q_start!r} for joint {name}")
    q0 = q_start + pose_offset
    n_ramp = int(RAMP_S / DT_SO101)
    for i in range(n_ramp):
        a = min(1.0, i * DT_SO101 / RAMP_S)
        bot.write_goal(name, q_start + a * (q0 - q_start))
        time.sleep(DT_SO101)
    ts, tgts, qs, taus = [], [], [], []
    t0 = time.time()
    while (t := time.time() - t0) < DURATION_S:
        tgt = target(t, q0)
        bot.write_goal(name, tgt)
        q, tau = bot.read(name)
        ts.append(t); tgts.append(tgt); qs.append(q); taus.append(tau)
        time.sleep(DT_SO101)
    for i in range(n_ramp):
        a = min(1.0, i * DT_SO101 / RAMP_S)
        bot.write_goal(name, q0 + a * (q_start - q0))
        time.sleep(DT_SO101)
    temp = bot.read_temp(name)
    np.savez(out / f"{name}_off{pose_offset:.2f}_{int(time.time())}.npz",
             t=ts, target=tgts, q=qs, tau=taus, temp=np.full(len(ts), temp))
    print(f"  {name} off={pose_offset:+.2f} done, temp={temp:.0f}C, n={len(ts)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--out", default="data/so101")
    ap.add_argument("--joints", nargs="*", default=None)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    probes = [p for p in SO101_PROBES if args.joints is None or p.joint in args.joints]
    joints = [p.joint for p in probes]
    bot = None

    def safe_exit():
        if bot is None:
            return
        bot.hold(joints)        # torque stays ON — never drop the arm mid-air
        bot.restore_gains()
        bot.hold(joints)

    def on_sig(sig, frame):
        print(f"\n{signal.Signals(sig).name} → hold position + restore gains (torque stays ON)")
        safe_exit()
        sys.exit(1)
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)
    if hasattr(signal, "SIGHUP"):  # ssh drop must still restore gains
        signal.signal(signal.SIGHUP, on_sig)

    bot = MockSO101() if args.mock else SO101Bus(args.port)

    try:
        bot.lower_gains(joints)
        meta = {"start": time.time(), "reps": args.reps, "joints": joints,
                "dt": DT_SO101, "probe_p": PROBE_P_COEFFICIENT}
        for rep in range(args.reps):
            print(f"rep {rep+1}/{args.reps}")
            for p in probes:
                for off in (p.neutral_offset, p.loaded_offset):
                    probe_joint(bot, p.joint, off, out)
        (out / "meta.json").write_text(json.dumps(meta))
    finally:
        safe_exit()
        print("gains restored, arm holding position (torque ON)")


if __name__ == "__main__":
    main()
