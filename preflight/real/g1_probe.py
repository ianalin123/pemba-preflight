"""Per-joint probe on the real G1 via unitree_sdk2py. Runs ON THE ORIN.

Safety contract (gantry doc):
- run only after `robot dev-mode` (green face light confirmed on CCTV)
- SIGINT/SIGTERM → damping command to every joint, then exit
- amplitudes small (protocol.AMP=0.25 rad), low gains, one joint at a time

Usage:  python3 g1_probe.py --out data/d1_baseline_run01 [--joints left_knee_joint ...]
        python3 g1_probe.py --mock   # laptop dry-run against mock_lowlevel
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
from preflight.protocol import (AMP, DT, DURATION_S, FREQ, G1_PROBES,  # noqa: E402
                                PROBE_KD, PROBE_KP, target)

# G1 29-DoF joint index map (unitree_hg LowCmd motor order, from SDK g1 example)
G1_JOINT_INDEX = {
    "left_hip_pitch_joint": 0, "left_hip_roll_joint": 1, "left_hip_yaw_joint": 2,
    "left_knee_joint": 3, "left_ankle_pitch_joint": 4, "left_ankle_roll_joint": 5,
    "right_hip_pitch_joint": 6, "right_hip_roll_joint": 7, "right_hip_yaw_joint": 8,
    "right_knee_joint": 9, "right_ankle_pitch_joint": 10, "right_ankle_roll_joint": 11,
    "waist_yaw_joint": 12,
    "left_shoulder_pitch_joint": 15, "left_shoulder_roll_joint": 16,
    "left_shoulder_yaw_joint": 17, "left_elbow_joint": 18,
    "right_shoulder_pitch_joint": 22, "right_shoulder_roll_joint": 23,
    "right_shoulder_yaw_joint": 24, "right_elbow_joint": 25,
}
# ^ VERIFY INDICES against the SDK example on the Orin before first motion:
#   ls ~/unitree_sdk2_python/example/g1/  (g1_low_level_example.py has the map)
HOLD_KP, HOLD_KD = 60.0, 1.5   # non-probed joints hold current pose softly
DAMP_KD = 8.0


class Robot:
    """Thin wrapper so the same probe loop runs against mock or real SDK."""

    def __init__(self, iface: str = "eth0"):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC
        ChannelFactoryInitialize(0, iface)
        # release the built-in motion service or lowcmd is ignored (SDK example flow)
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        status, result = msc.CheckMode()
        while result is not None and result.get("name"):
            print(f"releasing motion mode {result['name']!r}…")
            msc.ReleaseMode()
            time.sleep(1.0)
            status, result = msc.CheckMode()
        self._crc = CRC()
        self._cmd = unitree_hg_msg_dds__LowCmd_()
        self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._pub.Init()
        self._state = None
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_state, 10)
        while self._state is None:
            time.sleep(0.05)
        self._mode_machine = self._state.mode_machine

    def _on_state(self, msg):
        self._state = msg

    def read(self, idx: int):
        ms = self._state.motor_state[idx]
        return ms.q, ms.dq, ms.tau_est, ms.temperature[0]

    def read_all_q(self):
        return [self._state.motor_state[i].q for i in range(29)]

    def command(self, targets: dict[int, float], kp: dict[int, float], kd: dict[int, float]):
        self._cmd.mode_pr = 0  # Mode.PR: series pitch/roll ankle control
        self._cmd.mode_machine = self._mode_machine
        for i in range(29):
            mc = self._cmd.motor_cmd[i]
            mc.mode = 1
            mc.q = targets.get(i, self._hold_q[i])
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = kp.get(i, HOLD_KP)
            mc.kd = kd.get(i, HOLD_KD)
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)

    def snapshot_hold(self):
        self._hold_q = self.read_all_q()

    def damp_all(self):
        self._cmd.mode_pr = 0
        self._cmd.mode_machine = self._mode_machine
        for i in range(29):
            mc = self._cmd.motor_cmd[i]
            mc.mode = 1
            mc.q = 0.0
            mc.dq = 0.0
            mc.tau = 0.0
            mc.kp = 0.0
            mc.kd = DAMP_KD
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)


def probe_joint(bot, name: str, pose_offset: float, out: Path):
    idx = G1_JOINT_INDEX[name]
    bot.snapshot_hold()
    q_start, *_ = bot.read(idx)
    if not math.isfinite(q_start) or not (-math.pi <= q_start <= math.pi):
        print(f"WARNING: {name} q_start={q_start!r} is non-finite or out of [-π, π] — damping and aborting")
        bot.damp_all()
        raise RuntimeError(f"Unsafe q_start={q_start!r} for joint {name}")
    q0 = q_start + pose_offset
    # ramp gently to q0 over 2 s
    for i in range(int(2.0 / DT)):
        a = min(1.0, i * DT / 2.0)
        bot.command({idx: q_start + a * (q0 - q_start)}, {idx: PROBE_KP}, {idx: PROBE_KD})
        time.sleep(DT)
    ts, tgts, qs, taus, temps = [], [], [], [], []
    t0 = time.time()
    while (t := time.time() - t0) < DURATION_S:
        tgt = target(t, q0)
        bot.command({idx: tgt}, {idx: PROBE_KP}, {idx: PROBE_KD})
        q, dq, tau, temp = bot.read(idx)
        ts.append(t); tgts.append(tgt); qs.append(q); taus.append(tau); temps.append(temp)
        time.sleep(DT)
    # return to start
    for i in range(int(2.0 / DT)):
        a = min(1.0, i * DT / 2.0)
        bot.command({idx: q0 + a * (q_start - q0)}, {idx: PROBE_KP}, {idx: PROBE_KD})
        time.sleep(DT)
    np.savez(out / f"{name}_off{pose_offset:.2f}_{int(time.time())}.npz",
             t=ts, target=tgts, q=qs, tau=taus, temp=temps)
    print(f"  {name} off={pose_offset:+.2f} done, temp={temps[-1]}C, n={len(ts)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/d1")
    ap.add_argument("--joints", nargs="*", default=None)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--iface", default="eth0")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    bot = None

    def on_sig(sig, frame):
        print(f"\n{signal.Signals(sig).name} → damping all joints")
        if bot is not None:
            for _ in range(10):          # repeated sends so a full damp write lands last
                bot.damp_all()
                time.sleep(0.02)
        sys.exit(1)
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    if args.mock:
        from preflight.real.mock_lowlevel import MockRobot
        bot = MockRobot()
    else:
        bot = Robot(args.iface)

    try:
        probes = [p for p in G1_PROBES if args.joints is None or p.joint in args.joints]
        meta = {"start": time.time(), "reps": args.reps, "joints": [p.joint for p in probes]}
        for rep in range(args.reps):
            print(f"rep {rep+1}/{args.reps}")
            for p in probes:
                for off in (p.neutral_offset, p.loaded_offset):
                    probe_joint(bot, p.joint, off, out)
        (out / "meta.json").write_text(json.dumps(meta))
    finally:
        bot.damp_all()
        print("robot damped")


if __name__ == "__main__":
    main()
