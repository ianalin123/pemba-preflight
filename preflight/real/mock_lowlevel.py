"""Mock of the Robot wrapper: 2nd-order joint responses, so g1_probe.py --mock
exercises timing, logging, ramps, SIGINT handling without hardware."""
import numpy as np


class MockRobot:
    def __init__(self):
        self.q = np.zeros(29)
        self.dq = np.zeros(29)
        self._hold_q = np.zeros(29)
        self._last = None

    def read(self, idx):
        return float(self.q[idx]), float(self.dq[idx]), float(0.1 * self.dq[idx]), 35.0

    def read_all_q(self):
        return self.q.tolist()

    def snapshot_hold(self):
        self._hold_q = self.q.copy()

    def command(self, targets, kp, kd, dt=0.002):
        for i, tgt in targets.items():
            acc = kp.get(i, 60.0) * (tgt - self.q[i]) - kd.get(i, 1.5) * self.dq[i]
            self.dq[i] += acc * dt / 0.05   # ~inertia 0.05
            self.q[i] += self.dq[i] * dt

    def damp_all(self):
        self.dq[:] = 0
