"""Opt-in Gymnasium bridge; original registrations and training stay unchanged."""
from copy import deepcopy
from dataclasses import asdict

import numpy as np
from gymnasium.envs.registration import register, registry

from ctr_reach_envs.config import default_env_kwargs
from ctr_reach_envs.envs.ctr_reach_env import CtrReachEnv
from ctr_reach_envs.envs.model_utils import sample_parameters
from ctr_reach_envs.envs.obs import Obs

from .geometry import JointConstraints, TubeParameters
from .solver import EquilibriumSolver, SolverOptions


class EquilibriumModel:
    """Model-compatible adapter; pure ``solve`` does not alter rendered state.

    Each solve starts with zero torsion unless its caller explicitly supplies a
    shooting guess. This avoids adding hidden warm-start history to the Gym state.
    It does NOT prove a unique or elastically stable solution branch.
    """

    def __init__(self, systems, options=None):
        self.system_parameters = deepcopy(systems)
        self.current_sys_parameters = deepcopy(systems)
        self.options = SolverOptions() if options is None else options
        self.last_result = None
        self.r = self.r1 = self.r2 = self.r3 = np.empty((0, 3))
        self._build_solvers()

    def _build_solvers(self):
        self.solvers = [EquilibriumSolver([TubeParameters.from_legacy(t) for t in s], self.options)
                        for s in self.current_sys_parameters]

    def randomize_parameters(self, fraction, rng):
        if not np.isfinite(fraction) or not 0 <= fraction < 1:
            raise ValueError("domain_rand must be in [0, 1)")
        self.current_sys_parameters = [[sample_parameters(t, fraction, rng) for t in s]
                                       for s in self.system_parameters]
        self._build_solvers()
        self.last_result = None

    def solve(self, joints, system=0, initial_torsion=None):
        if not 0 <= system < len(self.solvers):
            raise IndexError("Invalid system index")
        return self.solvers[system].solve(joints, initial_torsion)

    def forward_kinematics(self, joints, system):
        result = self.solve(joints, system)
        # Commit only after all numerical checks succeeded.
        self.last_result = result
        self.r = result.position.copy()
        i1, i2, i3 = result.tube_tip_indices
        self.r1 = self.r[i2:i1+1].copy()
        self.r2 = self.r[i3:i2+1].copy()
        self.r3 = self.r[:i3+1].copy()
        return result.tip


class FeasibleObs(Obs):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraints = [JointConstraints(lengths, constrain_alpha=self.constrain_alpha)
                            for lengths in self.tube_lengths]

    def set_joints(self, joints, system):
        self.joints = self.constraints[system].project(joints)

    def set_action(self, action, system):
        action = np.asarray(action, dtype=float)
        if action.shape != self.joints.shape or not np.all(np.isfinite(action)):
            raise ValueError("action must match the joint vector and be finite")
        self.set_joints(self.joints + action, system)


class EquilibriumReachEnv(CtrReachEnv):
    """Experimental unloaded equilibrium environment, without a stability claim."""

    def __init__(self, *, solver_options=None, **kwargs):
        # Preserve the parent API and all its task/reward choices. The construction
        # below replaces both mechanical components before the first reset/step.
        super().__init__(**kwargs)
        options = SolverOptions(**({} if solver_options is None else solver_options))
        self.trig_obj = FeasibleObs(self.ctr_system_parameters, self.starting_joints,
                                   self.joint_representation, self.trig_obj.constrain_alpha)
        self.trig_obj.set_joints(self.starting_joints, 0)
        self.model = EquilibriumModel(self.ctr_system_parameters, options)
        self.starting_joints = self.trig_obj.joints.copy()
        self.starting_position = self.model.forward_kinematics(self.starting_joints, 0)
        self.achieved_goal = self.starting_position.copy()
        self.desired_goal = self.starting_position.copy()

    def _info(self, *, solver_failure):
        info = super()._info(solver_failure=solver_failure)
        info["mechanics_backend"] = "unloaded_equilibrium_bvp_v1"
        info["elastic_stability_certified"] = False
        if self.model.last_result is not None:
            info["equilibrium_residual"] = self.model.last_result.diagnostics["boundary_residual_scaled"]
        return info


ENV_ID = "CTR-Equilibrium-v0"


def register_equilibrium_env():
    if ENV_ID not in registry:
        register(id=ENV_ID, entry_point="ctr_reach_envs.mechanics.env:EquilibriumReachEnv",
                 kwargs={**default_env_kwargs(), "solver_options": asdict(SolverOptions())})
    return ENV_ID
