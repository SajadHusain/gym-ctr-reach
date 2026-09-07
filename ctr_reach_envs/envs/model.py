from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp

from ctr_reach_envs.envs.CTR_Python import Segment
from ctr_reach_envs.envs.model_utils import sample_parameters


class Model:
    """Simplified, unloaded quasi-static forward model from the supplied project."""

    def __init__(self, system_parameters):
        self.system_parameters = deepcopy(system_parameters)
        self.current_sys_parameters = deepcopy(system_parameters)
        self.r = np.empty((0, 3))
        self.r1 = np.empty((0, 3))
        self.r2 = np.empty((0, 3))
        self.r3 = np.empty((0, 3))

    def randomize_parameters(self, fraction: float, rng: np.random.Generator) -> None:
        for system_index, system in enumerate(self.system_parameters):
            self.current_sys_parameters[system_index] = [
                sample_parameters(tube, fraction, rng) for tube in system
            ]

    def forward_kinematics(self, joint, system: int):
        joint = np.asarray(joint, dtype=np.float64)
        if joint.shape != (6,) or not np.all(np.isfinite(joint)):
            raise ValueError("joint must be a finite six-element vector")

        beta = joint[:3]
        segment = Segment(*self.current_sys_parameters[system], beta)
        r_0 = np.zeros((3, 1))
        alpha_1_0 = joint[3]
        R_0 = np.array(
            [
                [np.cos(alpha_1_0), -np.sin(alpha_1_0), 0.0],
                [np.sin(alpha_1_0), np.cos(alpha_1_0), 0.0],
                [0.0, 0.0, 1.0],
            ]
        ).reshape(9, 1)
        alpha_0 = joint[3:].reshape(3, 1)
        self.r, _, tips = self._solve(
            system,
            np.zeros(3),
            alpha_0,
            r_0,
            R_0,
            segment,
            beta,
        )
        self.r1 = self.r[tips[1] : tips[0] + 1]
        self.r2 = self.r[tips[2] : tips[1] + 1]
        self.r3 = self.r[: tips[2] + 1]
        if not np.all(np.isfinite(self.r)):
            raise RuntimeError("Forward model returned non-finite backbone coordinates")
        return self.r[-1].copy()

    @staticmethod
    def _ode_eq(_, y, ux_0, uy_0, ei, gj):
        dydt = np.empty(18, dtype=np.float64)
        ux = np.empty(3, dtype=np.float64)
        uy = np.empty(3, dtype=np.float64)
        ei_sum = float(np.sum(ei))
        if ei_sum <= 0.0:
            raise RuntimeError("Invalid segment with zero total bending stiffness")

        for i in range(3):
            angle = y[3 + i] - y[3 : 6]
            ux[i] = np.sum(ei * (ux_0 * np.cos(angle) + uy_0 * np.sin(angle))) / ei_sum
            uy[i] = np.sum(ei * (-ux_0 * np.sin(angle) + uy_0 * np.cos(angle))) / ei_sum

        for i in range(3):
            if ei[i] == 0.0:
                dydt[i] = 0.0
                dydt[3 + i] = 0.0
            else:
                if gj[i] <= 0.0:
                    raise RuntimeError("Active tube segment has zero torsional stiffness")
                dydt[i] = (ei[i] / gj[i]) * (ux[i] * uy_0[i] - uy[i] * ux_0[i])
                dydt[3 + i] = y[i]

        uz = y[:3]
        R = y[9:].reshape(3, 3)
        u_hat = np.array(
            [
                [0.0, -uz[0], uy[0]],
                [uz[0], 0.0, -ux[0]],
                [-uy[0], ux[0], 0.0],
            ]
        )
        dr = R @ np.array([0.0, 0.0, 1.0])
        dR = (R @ u_hat).ravel()
        dydt[6:9] = dr
        dydt[9:18] = dR
        return dydt

    def _solve(self, system, uz_0, alpha_0, r_0, R_0, segmentation, beta):
        tubes = self.current_sys_parameters[system]
        lengths = np.empty(0)
        r = np.empty((0, 3))
        u_z = np.empty((0, 3))
        alpha = np.empty((0, 3))
        span = np.append([0.0], segmentation.S)

        for segment_index in range(len(segmentation.S)):
            y_0 = np.vstack((uz_0.reshape(3, 1), alpha_0, r_0, R_0)).ravel()
            start = float(span[segment_index])
            end = float(span[segment_index + 1] - 1e-6)
            if end <= start:
                continue
            s_eval = np.linspace(start, end, num=30)
            solution = solve_ivp(
                fun=lambda s, y: self._ode_eq(
                    s,
                    y,
                    segmentation.U_x[:, segment_index],
                    segmentation.U_y[:, segment_index],
                    segmentation.EI[:, segment_index],
                    segmentation.GJ[:, segment_index],
                ),
                t_span=(start, end),
                y0=y_0,
                t_eval=s_eval,
            )
            if not solution.success:
                raise RuntimeError(f"CTR ODE solve failed: {solution.message}")

            state = solution.y.T
            lengths = np.append(lengths, s_eval)
            u_z = np.vstack((u_z, state[:, :3]))
            alpha = np.vstack((alpha, state[:, 3:6]))
            r = np.vstack((r, state[:, 6:9]))
            r_0 = r[-1].reshape(3, 1)
            R_0 = state[-1, 9:].reshape(9, 1)
            uz_0 = u_z[-1].reshape(3, 1)
            alpha_0 = alpha[-1].reshape(3, 1)

        if r.size == 0:
            raise RuntimeError("CTR segmentation produced no active backbone")

        tube_tip_distances = np.array([tube.L for tube in tubes]) + beta
        tips = np.empty(3, dtype=np.int64)
        u_z_end = np.empty(3, dtype=np.float64)
        for index, tip_distance in enumerate(tube_tip_distances):
            sample = int(np.searchsorted(lengths, tip_distance - 1e-3, side="left"))
            sample = min(max(sample, 0), len(lengths) - 1)
            tips[index] = sample
            u_z_end[index] = u_z[sample, index]
        return r, u_z_end, tips

