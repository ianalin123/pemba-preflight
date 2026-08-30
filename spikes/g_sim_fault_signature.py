"""G-SIM kill-test: do injected joint faults produce separable probe signatures
on the MuJoCo G1 under a small-amplitude per-joint probe protocol?

Protocol (mirrors real G1 driving mode): the model's own position servos are
driven with a small sinusoid via ctrl, with the probed joint's servo softened
to low gain (fault sensitivity). Log tracking error, achieved motion coverage,
and actuator force. Compare healthy vs:
  - friction fault (frictionloss +3 Nm)  ~ grit / dry friction
  - damping fault  (damping +8 Nms)      ~ cold-stiffened lubricant
  - range clamp    (obstruction at ±0.05 rad) ~ waist lock
  - torque derate  (servo gain x0.4)     ~ cold motor derating
"""

from pathlib import Path

import numpy as np
import mujoco

MJCF_PATH = str(Path(__file__).resolve().parent.parent / "mujoco_menagerie" / "unitree_g1" / "g1.xml")

PROBE_JOINTS = ["left_knee_joint", "waist_yaw_joint", "left_shoulder_pitch_joint"]
AMP, FREQ, DURATION = 0.25, 0.5, 4.0  # rad, Hz, s
PROBE_KP, PROBE_KD = 40.0, 2.0  # low-gain probe mode


def run_probe(joint_name, frictionloss_add=0.0, damping_add=0.0,
              range_clamp=None, gain_x=1.0):
    m = mujoco.MjModel.from_xml_path(MJCF_PATH)
    m.opt.timestep = 0.002
    d = mujoco.MjData(m)

    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
    assert jid >= 0 and aid >= 0, joint_name
    dofadr, qposadr = m.jnt_dofadr[jid], m.jnt_qposadr[jid]

    # soften the probed joint's servo to probe gains (fault sensitivity)
    kp, kd = PROBE_KP * gain_x, PROBE_KD * gain_x
    m.actuator_gainprm[aid][0] = kp
    m.actuator_biasprm[aid][1] = -kp
    m.actuator_biasprm[aid][2] = -kd

    # inject faults
    m.dof_frictionloss[dofadr] += frictionloss_add
    m.dof_damping[dofadr] += damping_add

    mujoco.mj_resetDataKeyframe(m, d, 0)
    m.opt.gravity[:] = 0  # suspended in harness
    q0 = float(d.qpos[qposadr])
    lo, hi = (q0 - range_clamp, q0 + range_clamp) if range_clamp is not None else (None, None)

    errs, forces, qs, targets = [], [], [], []
    for i in range(int(DURATION / m.opt.timestep)):
        t = i * m.opt.timestep
        target = q0 + AMP * np.sin(2 * np.pi * FREQ * t)
        d.ctrl[:] = m.key_ctrl[0]
        d.ctrl[aid] = target
        d.qfrc_applied[:] = 0
        if range_clamp is not None:
            q = float(d.qpos[qposadr])
            qc = np.clip(q, lo, hi)
            if q != qc:  # hard obstruction
                d.qfrc_applied[dofadr] = -800.0 * (q - qc) - 10.0 * float(d.qvel[dofadr])
        mujoco.mj_step(m, d)
        q = float(d.qpos[qposadr])
        errs.append(target - q)
        forces.append(float(d.actuator_force[aid]))
        qs.append(q)
        targets.append(target)

    errs, forces = np.array(errs), np.array(forces)
    qs, targets = np.array(qs), np.array(targets)
    half = len(errs) // 2
    coverage = (qs[half:].max() - qs[half:].min()) / (targets[half:].max() - targets[half:].min())
    return {
        "rms_err": float(np.sqrt(np.mean(errs[half:] ** 2))),
        "coverage": float(coverage),
        "rms_force": float(np.sqrt(np.mean(forces[half:] ** 2))),
    }


def main():
    conditions = {
        "healthy": {},
        "friction_+3Nm": {"frictionloss_add": 3.0},
        "damping_+8Nms": {"damping_add": 8.0},
        "clamp_0.05rad": {"range_clamp": 0.05},
        "derate_40%": {"gain_x": 0.4},
    }
    print(f"{'joint':<28} {'condition':<16} {'rms_err':>8} {'coverage':>9} {'rms_force':>10}  flags")
    n_separable, n_total = 0, 0
    for joint in PROBE_JOINTS:
        base = None
        for name, kw in conditions.items():
            r = run_probe(joint, **kw)
            flags = ""
            if base is not None:
                n_total += 1
                err_ratio = r["rms_err"] / max(base["rms_err"], 1e-9)
                cov_drop = base["coverage"] - r["coverage"]
                force_ratio = r["rms_force"] / max(base["rms_force"], 1e-9)
                sep = err_ratio > 1.3 or err_ratio < 0.7 or cov_drop > 0.15 or force_ratio > 1.3 or force_ratio < 0.7
                n_separable += sep
                flags = f"err x{err_ratio:.2f} cov -{cov_drop:.2f} F x{force_ratio:.2f} {'SEP' if sep else 'WEAK'}"
            print(f"{joint:<28} {name:<16} {r['rms_err']:>8.4f} {r['coverage']:>9.3f} {r['rms_force']:>10.2f}  {flags}")
            if name == "healthy":
                base = r
        print()
    print(f"VERDICT: {n_separable}/{n_total} fault conditions separable "
          f"{'— G-SIM PASSES' if n_separable == n_total else '— investigate weak cases'}")


if __name__ == "__main__":
    main()
