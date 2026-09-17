"""MPC-as-policy Q-learning (Gros & Zanon 2019/2020), using mpcrl 1.4.1.

The original CTR IVP and physical constraints are fixed. RL adjusts only the
internal cost approximation. This is nominal MPC, not robust/safe DPG.
"""
from dataclasses import dataclass
from collections import OrderedDict
import time

import casadi as cs
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc
from mpcrl import LstdQLearningAgent, LearnableParameter, LearnableParametersDict
from mpcrl.optim import GradientDescent

from ctr_reach_envs.envs.model import Model
from ctr_reach_envs.mechanics.geometry import JointConstraints
from .models import original_joint_step


@dataclass(frozen=True)
class LearningOptions:
    horizon: int = 2
    gamma: float = .95
    learning_rate: float = .001
    max_parameter_change: float = .02
    exploration_strength: float = .01
    tracking_scale_m: float = .01
    max_iterations: int = 120
    max_model_evaluations: int = 5000
    finite_difference_step: float = 1e-4
    solver_tolerance: float = 1e-4

    def __post_init__(self):
        for key in ("horizon", "max_iterations", "max_model_evaluations"):
            value = getattr(self, key)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not np.isfinite(self.gamma) or not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        for key in ("learning_rate", "max_parameter_change", "tracking_scale_m",
                    "finite_difference_step", "solver_tolerance"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if not np.isfinite(self.exploration_strength) or self.exploration_strength < 0:
            raise ValueError("exploration_strength must be finite and nonnegative")


class _TipJacobian(cs.Callback):
    """First derivative only; IPOPT uses limited-memory Hessians.

    Bound-aware finite differences query the complete nonlinear IVP at each
    candidate configuration. No fixed current-state Jacobian is rolled forward.
    """
    def __init__(self, name, parent):
        cs.Callback.__init__(self)
        self.parent = parent
        self.construct(name, {})

    def get_n_in(self): return 2  # q and the already evaluated tip
    def get_n_out(self): return 1
    def get_sparsity_in(self, i): return cs.Sparsity.dense(6 if i == 0 else 3, 1)
    def get_sparsity_out(self, i): return cs.Sparsity.dense(3, 6)

    def eval(self, args):
        q = np.asarray(args[0]).reshape(6)
        tip = np.asarray(args[1]).reshape(3)
        jac = np.empty((3, 6))
        for j in range(6):
            step = self.parent.options.finite_difference_step
            plus, minus = q.copy(), q.copy()
            plus[j] += step; minus[j] -= step
            if minus[j] >= self.parent.lower[j] and plus[j] <= self.parent.upper[j]:
                # First-order one-sided differences leave a systematic gradient
                # error that can prevent IPOPT's stationarity test converging.
                jac[:, j] = (self.parent.tip(plus)-self.parent.tip(minus))/(2*step)
            else:
                if plus[j] > self.parent.upper[j]: step = -step
                trial = q.copy(); trial[j] += step
                second = q.copy(); second[j] += 2*step
                if not self.parent.lower[j] <= second[j] <= self.parent.upper[j]:
                    raise RuntimeError("No admissible finite-difference perturbation")
                jac[:, j] = (-3*tip+4*self.parent.tip(trial)-self.parent.tip(second))/(2*step)
        return [cs.DM(jac)]


class IVPTipCallback(cs.Callback):
    """A CasADi callback retaining the exact existing forward-model implementation."""
    def __init__(self, forward, caps, lengths, constrain_alpha, options):
        cs.Callback.__init__(self)
        self.forward, self.caps, self.options = forward, np.asarray(caps), options
        self.lower = np.r_[(-np.asarray(lengths)+.001)/self.caps[:3], np.full(3, -np.inf)]
        self.upper = np.r_[np.zeros(3), np.full(3, np.inf)]
        if constrain_alpha:
            self.lower[3:], self.upper[3:] = -np.pi/self.caps[3:], np.pi/self.caps[3:]
        self.calls, self.seconds, self.solves, self._calls_at_start = 0, 0., 0, 0
        self.cache, self.children = OrderedDict(), []
        self.construct("ctr_tip_"+str(id(self)), {})

    def begin_solve(self):
        self._calls_at_start = self.calls
        self.solves += 1
        self.cache.clear()

    def tip(self, scaled_q):
        scaled_q = np.asarray(scaled_q, dtype=float).reshape(6)
        key = scaled_q.tobytes()
        if key not in self.cache:
            if self.calls-self._calls_at_start >= self.options.max_model_evaluations:
                raise RuntimeError("CTR IVP evaluation budget exceeded")
            self.calls += 1
            started = time.perf_counter()
            try:
                value = np.asarray(self.forward(scaled_q*self.caps), dtype=float).reshape(3)
                if not np.all(np.isfinite(value)): raise RuntimeError("Nonfinite CTR prediction")
                self.cache[key] = value/self.options.tracking_scale_m
                if len(self.cache) > 4096: self.cache.popitem(last=False)
            finally:
                self.seconds += time.perf_counter()-started
        return self.cache[key].copy()

    def get_n_in(self): return 1
    def get_n_out(self): return 1
    def get_sparsity_in(self, i): return cs.Sparsity.dense(6, 1)
    def get_sparsity_out(self, i): return cs.Sparsity.dense(3, 1)
    def eval(self, args): return [cs.DM(self.tip(args[0]))]
    def has_jacobian(self): return True

    def get_jacobian(self, name, inames, onames, opts):
        child = _TipJacobian(name, self)
        self.children.append(child)  # retain Python callbacks throughout NLP lifetime
        return child


def build_mpc(callback, constraints, options, name):
    """Normalized state = [q / joint_step_caps, goal / tracking_scale]."""
    horizon, caps = options.horizon, callback.caps
    # Joint dynamics are affine and exact. Eliminate their states rather than
    # asking a quasi-Newton solver to also optimize equality-constrained joints
    # and repeated goal coordinates. Nonlinear FK is still evaluated at every node.
    mpc = Mpc(Nlp(sym_type="MX", name=name), horizon, shooting="single")
    mpc.state("state", 9)
    action, _ = mpc.action("action", 6, lb=-1., ub=1.)
    mpc.set_affine_dynamics(np.eye(9), np.vstack((np.eye(6), np.zeros((3, 6)))))
    state = mpc.states["state"]
    ordering = cs.DM(constraints.A) @ cs.diag(cs.DM(caps[:3]))
    mpc.constraint("extension_order", ordering @ state[:3, 1:], "<=",
                   cs.repmat(cs.DM(constraints.b), 1, horizon))
    if constraints.constrain_alpha:
        upper = cs.repmat(cs.DM(callback.upper[3:]), 1, horizon)
        mpc.constraint("rotation_upper", state[3:6, 1:], "<=", upper)
        mpc.constraint("rotation_lower", state[3:6, 1:], ">=", -upper)
    stage = mpc.parameter("stage", (3, 1))
    move = mpc.parameter("move", (2, 1))
    terminal = mpc.parameter("terminal", (3, 1))
    offset = mpc.parameter("offset")
    perturbation = mpc.parameter("exploration", (6, 1))
    objective = offset + cs.dot(perturbation, action[:, 0])
    for k in range(horizon):
        error = callback(state[:6, k])-state[6:, k]
        effort = move[0]*cs.sumsqr(action[:3, k])+move[1]*cs.sumsqr(action[3:, k])
        objective += options.gamma**k*(cs.dot(stage, error**2)+effort)
    error = callback(state[:6, -1])-state[6:, -1]
    objective += options.gamma**horizon*cs.dot(terminal, error**2)
    mpc.minimize(objective)
    mpc.init_solver(dict(print_time=False, error_on_fail=False,
        ipopt=dict(print_level=0, sb="yes", max_iter=options.max_iterations,
                   tol=options.solver_tolerance, acceptable_iter=0,
                   constr_viol_tol=1e-9, compl_inf_tol=1e-6,
                   dual_inf_tol=options.solver_tolerance,
                   hessian_approximation="limited-memory", limited_memory_max_history=20,
                   bound_relax_factor=0.,
                   honor_original_bounds="yes")), solver="ipopt")
    return mpc


def td_residual(cost, q_value, next_value, gamma, terminated):
    """Success is absorbing; time-limit truncation retains its bootstrap."""
    return float(cost + (0. if terminated else gamma*next_value)-q_value)


class RelativeGradientDescent(GradientDescent):
    """Preserve a true relative update limit for small positive cost weights.

    mpcrl 1.4.1's base bound helper has an absolute 0.1 update floor, which
    overwhelms our 0.001 effort weights. The value offset alone uses unit scale
    so it can leave zero; all positive weights use their current magnitude.
    """
    def _get_update_bounds(self, theta, eps=None):
        scale = abs(theta).copy()
        scale[-1] = max(scale[-1], 1.)  # value offset is the final parameter
        radius = self.max_percentage_update*scale
        return (np.maximum(self.learnable_parameters.lb-theta, -radius),
                np.minimum(self.learnable_parameters.ub-theta, radius))


class MPCQLearner:
    """Thin CTR adapter around mpcrl's Q approximation, sensitivity and optimizer.

    A custom episodic loop is required: mpcrl 1.4.1's supplied continuing-task
    loop always bootstraps. Also, Bellman targets here use unperturbed V.
    """
    def __init__(self, forward, lengths, caps, constrain_alpha=False, options=None, parameters=None):
        self.options = options or LearningOptions()
        self.constraints = JointConstraints(lengths, constrain_alpha=constrain_alpha)
        self.callback = IVPTipCallback(forward, caps, lengths, constrain_alpha, self.options)
        self.caps = np.asarray(caps, dtype=float)
        # Separate graphs avoid copying native Python callback objects.
        v = build_mpc(self.callback, self.constraints, self.options, "ctr_value")
        q = build_mpc(self.callback, self.constraints, self.options, "ctr_quality")
        defaults = dict(stage=np.ones(3), move=np.full(2, .001), terminal=np.ones(3), offset=np.zeros(1))
        if parameters is not None:
            if set(parameters) != set(defaults): raise ValueError("Checkpoint parameter names differ")
            defaults = {k: np.asarray(parameters[k]).reshape(val.shape) for k, val in defaults.items()}
        bounds = dict(stage=(.001, 20.), move=(.00001, 2.), terminal=(.001, 40.), offset=(-100., 100.))
        pars = LearnableParametersDict([
            LearnableParameter(k, val.shape, val, lb=bounds[k][0], ub=bounds[k][1])
            for k, val in defaults.items()])
        self.agent = LstdQLearningAgent((v, q), update_strategy=1,
            discount_factor=self.options.gamma, learnable_parameters=pars,
            optimizer=RelativeGradientDescent(learning_rate=self.options.learning_rate,
                max_percentage_update=self.options.max_parameter_change, bound_consistency=True),
            fixed_parameters={"exploration": np.zeros((6, 1))}, experience=1,
            remove_bounds_on_initial_action=True)
        self.updates = 0

    @classmethod
    def from_env(cls, env, source_config, options=None, parameters=None):
        if source_config.get("segment_mode") != "continuous":
            raise ValueError("MPC-RL requires the saved continuous original-IVP model")
        if env.physics_observation["mode"] != "none" or len(env.ctr_system_parameters) != 1:
            raise ValueError("Use one fixed system and physics observations set to none")
        private = Model(env.ctr_system_parameters, **source_config["environment"]["model_options"])
        return cls(lambda q: private.forward_kinematics(q, 0), env.trig_obj.tube_lengths[0],
            env.n_substeps*env.action_scale, env.trig_obj.constrain_alpha, options, parameters)

    def parameters(self):
        return {k: p.value.reshape(-1).tolist() for k, p in self.agent.learnable_parameters.items()}

    def state(self, joints, goal):
        if not self.constraints.is_feasible(joints): raise ValueError("Infeasible source joints")
        return np.r_[np.asarray(joints)/self.caps, np.asarray(goal)/self.options.tracking_scale_m]

    def solve(self, state, *, action=None, perturbation=None):
        state = np.asarray(state, dtype=float).reshape(9)
        if not np.all(np.isfinite(state)): raise ValueError("Nonfinite MPC state")
        if action is not None and (np.shape(action) != (6,) or np.any(abs(action) > 1+1e-8)):
            raise ValueError("Q requires the executed normalized action")
        self.callback.begin_solve()
        started = time.perf_counter()
        # A feasible hold is always the initial guess. This avoids stale goals and
        # invalid full-extension guesses in a fresh nonlinear CTR solve.
        guess_action = np.zeros((6, self.options.horizon))
        if action is not None:
            guess_action[:, 0] = action
        guess = dict(action=guess_action)
        self.agent.fixed_parameters["exploration"] = np.asarray(
            np.zeros(6) if action is not None or perturbation is None else perturbation).reshape(6, 1)
        if action is None:
            _, sol = self.agent.state_value(state, deterministic=True, vals0=guess)
        else:
            sol = self.agent.action_value(state, action, vals0=guess)
        if not sol.success or not np.isfinite(sol.f):
            iterations = sol.stats.get("iterations", {})
            final = {key: values[-1] for key, values in iterations.items()
                     if key in ("inf_pr", "inf_du", "mu") and len(values)}
            raise RuntimeError(f"MPC solve failed: {sol.status}; residuals={final}; "
                               f"model_calls={self.callback.calls-self.callback._calls_at_start}")
        mpc = self.agent.V if action is None else self.agent.Q
        # Expose the derived single-shooting state trajectory alongside controls
        # for independent plant validation and diagnostics.
        sol.vals["state"] = sol.value(mpc.states["state"])
        states = np.asarray(sol.vals["state"])
        actions = np.asarray(sol.vals["action"])
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
            raise RuntimeError("Nonfinite optimized trajectory")
        residual = np.max(abs(np.diff(states, axis=1)-np.vstack((actions, np.zeros((3, self.options.horizon))))))
        if (residual > 1e-5 or np.max(abs(states[:, 0]-state)) > 1e-5
                or np.max(abs(actions)) > 1+1e-6
                or any(not self.constraints.is_feasible(s[:6]*self.caps, atol=1e-9) for s in states.T)):
            raise RuntimeError("MPC solution failed independent dynamics/joint validation")
        if action is not None and np.max(abs(actions[:, 0]-action)) > 1e-5:
            raise RuntimeError("Q did not fix the supplied first action")
        diagnostics = dict(status=sol.status, objective=float(sol.f),
            model_calls=self.callback.calls-self.callback._calls_at_start,
            solve_seconds=time.perf_counter()-started, dynamics_residual=float(residual))
        return np.clip(actions[:, 0], -1., 1.).astype(np.float32), sol, diagnostics

    def learn_transition(self, cost, sol_q, sol_v_next=None, *, terminated=False):
        if not sol_q.success or (not terminated and (sol_v_next is None or not sol_v_next.success)):
            raise ValueError("Learning requires converged Q and nonterminal V solutions")
        delta = td_residual(cost, sol_q.f, 0. if terminated else sol_v_next.f,
                            self.options.gamma, terminated)
        # mpcrl computes dQ/dtheta from the optimized NLP Lagrangian. Parameters
        # occur only in the cost, so no ODE second derivative is required.
        derivative = np.asarray(self.agent._sensitivity(sol_q)).reshape(-1)
        gradient = -delta*derivative
        if not np.isfinite(delta) or not np.all(np.isfinite(gradient)):
            raise RuntimeError("Nonfinite TD update")
        old = self.agent.learnable_parameters.value.copy()
        self.agent.store_experience(gradient)
        status = self.agent.update()
        new = self.agent.learnable_parameters.value
        if status is not None or not np.all(np.isfinite(new)):
            self.agent.learnable_parameters.update_values(old)
            raise RuntimeError(f"Parameter update failed: {status}")
        self.updates += 1
        return dict(td_error=delta, gradient_norm=float(np.linalg.norm(gradient)),
                    parameter_change_norm=float(np.linalg.norm(new-old)))

    def check_executed_action(self, env, action, solution):
        predicted = np.asarray(solution.vals["state"])[:6, 1]*self.caps
        actual = original_joint_step(env.trig_obj.joints, action, env.action_scale,
            env.n_substeps, env.trig_obj.tube_lengths[0], env.trig_obj.constrain_alpha)
        if np.max(abs((actual-predicted)/self.caps)) > 2e-5:
            raise RuntimeError("Optimized action disagrees with the original plant's projection")
