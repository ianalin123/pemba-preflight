"""G1 probe rollouts in MuJoCo with injectable faults. Pelvis pinned = harness."""
from pathlib import Path
from dataclasses import dataclass

import mujoco
import numpy as np

from preflight.protocol import PROBE_KP, PROBE_KD, DURATION_S, DT, target

MJCF = str(Path(__file__).resolve().parents[2] / "mujoco_menagerie/unitree_g1/g1.xml")


@dataclass
class Fault:
    cls: str = "healthy"          # protocol.FAULT_CLASSES
    friction_add: float = 0.0      # Nm      → "friction"
    damping_add: float = 0.0       # Nms     → "stiffness"
    clamp_rad: float | None = None  #        → "obstruction"
    gain_x: float = 1.0            #        → "derate"
    severity: float = 0.0          # 0..1 label for regression

def run_probe(joint: str, pose_offset: float = 0.0, fault: Fault = Fault(),
              seed: int = 0, randomize: bool = False) -> dict:
    m = mujoco.MjModel.from_xml_path(MJCF)
    m.opt.timestep = DT
    rng = np.random.default_rng(seed)
    if randomize:  # domain randomization for corpus
        m.body_mass[:] *= rng.uniform(0.9, 1.1, m.nbody)
        m.dof_frictionloss[:] *= rng.uniform(0.7, 1.5, m.nv)
    d = mujoco.MjData(m)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint)
    aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, joint)
    assert jid >= 0 and aid >= 0, joint
    dofadr, qposadr = m.jnt_dofadr[jid], m.jnt_qposadr[jid]

    kp, kd = PROBE_KP * fault.gain_x, PROBE_KD * fault.gain_x
    m.actuator_gainprm[aid][0] = kp
    m.actuator_biasprm[aid][1] = -kp
    m.actuator_biasprm[aid][2] = -kd
    m.dof_frictionloss[dofadr] += fault.friction_add
    m.dof_damping[dofadr] += fault.damping_add

    mujoco.mj_resetDataKeyframe(m, d, 0)
    base_qpos = d.qpos[:7].copy()
    q0 = float(d.qpos[qposadr]) + pose_offset
    lo = hi = None
    if fault.clamp_rad is not None:
        lo, hi = q0 - fault.clamp_rad, q0 + fault.clamp_rad

    ts, tgts, qs, taus = [], [], [], []
    for i in range(int(DURATION_S / DT)):
        t = i * DT
        d.ctrl[:] = m.key_ctrl[0]
        d.ctrl[aid] = target(t, q0)
        d.qfrc_applied[:] = 0
        if lo is not None:
            q = float(d.qpos[qposadr])
            qc = np.clip(q, lo, hi)
            if q != qc:
                d.qfrc_applied[dofadr] = -800.0 * (q - qc) - 10.0 * float(d.qvel[dofadr])
        d.qpos[:7] = base_qpos
        d.qvel[:6] = 0
        mujoco.mj_step(m, d)
        ts.append(t); tgts.append(d.ctrl[aid])
        qs.append(float(d.qpos[qposadr])); taus.append(float(d.actuator_force[aid]))
    return {"t": np.array(ts), "target": np.array(tgts), "q": np.array(qs), "tau": np.array(taus)}
