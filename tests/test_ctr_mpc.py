"""MPC behavior, paper boundary equations, and matched-plant evaluation tests."""
import csv
import hashlib
import json

import numpy as np
import pytest

from ctr_reach_envs.mechanics.geometry import TubeParameters, JointConstraints
from ctr_reach_envs.mechanics.solver import EquilibriumSolver
from ctr_reach_envs.mpc import MPCOptions, NonlinearMPC, PaperIVPModel, Prediction
from ctr_reach_envs.mpc.models import original_joint_step
from ctr_reach_envs.ivp.config import load_spec, resolve, make_env


class LinearPlant:
    torsion_count = 0

    def predict(self, q, z):
        return Prediction(np.array([q[0]+.2, q[1]*.01, 0.]), np.empty(0))


class TorsionPlant:
    torsion_count = 1

    def predict(self, q, z):
        return Prediction(np.array([q[0]+.2, .01*(q[1]+z[0]), 0.]),
                          np.array([z[0]-2*q[1]]))


def two_tube_solver():
    # Parameters in the authors' public Example.py, reproduced as factual data.
    return EquilibriumSolver([
        TubeParameters(.4, .2, .0007, .0011, 70e9, 10e9, 12.),
        TubeParameters(.3, .15, .0014, .0018, 70e9, 10e9, 6.)])


def configuration():
    spec, environment, provenance = load_spec()
    return resolve(spec, environment, segment_mode="continuous", seed=10,
                   physics={}, provenance=provenance)


def test_preview_changes_first_command_and_respects_horizon_step_caps():
    q = np.array([-.1, 0.]); plant = LinearPlant()
    tip = plant.predict(q, None).tip
    one = NonlinearMPC(plant, JointConstraints([.2]), [.01, .1], MPCOptions(horizon=1))
    assert one.plan(q, tip, tip).diagnostics["hold"]
    two = NonlinearMPC(plant, JointConstraints([.2]), [.01, .1], MPCOptions(horizon=2))
    result = two.plan(q, tip, np.array([tip, tip+np.array([.04, 0, 0])]))
    assert not result.diagnostics["hold"]
    assert result.joints[0]-q[0] > .009
    assert np.max(abs((result.joints-q)/two.caps)) <= 1+1e-8
    assert result.diagnostics["objective_selected"] < result.diagnostics["objective_hold"]


def test_unknown_base_torsion_is_optimized_and_boundary_enforced():
    q = np.array([-.1, 0.]); plant = TorsionPlant()
    controller = NonlinearMPC(plant, JointConstraints([.2]), [.01, .1])
    initial = plant.predict(q, np.zeros(1)).tip
    result = controller.plan(q, initial, initial+[0., .003, 0.], torsion_scaled=np.zeros(1))
    assert not result.diagnostics["hold"]
    assert result.torsion_scaled[0] > .15
    np.testing.assert_allclose(plant.predict(result.joints, result.torsion_scaled).residual, 0., atol=1e-5)
    assert result.diagnostics["boundary_residual_scaled"] <= controller.options.boundary_tolerance


@pytest.mark.parametrize("option", [dict(horizon=0), dict(max_model_evaluations=0),
    dict(tracking_scale_m=0), dict(finite_difference_step=np.nan), dict(move_weight=-1)])
def test_options_reject_invalid_inputs(option):
    with pytest.raises(ValueError): MPCOptions(**option)


def test_budget_and_model_disagreement_hold_without_unchecked_motion():
    q = np.array([-.1, 0.]); plant = LinearPlant(); tip = plant.predict(q, None).tip
    controller = NonlinearMPC(plant, JointConstraints([.2]), [.01, .1],
                              MPCOptions(max_model_evaluations=1))
    result = controller.plan(q, tip, tip+[.03, 0., 0.])
    np.testing.assert_array_equal(result.joints, q)
    assert result.diagnostics["hold"]
    assert result.diagnostics["model_evaluations"] == 1
    assert "budget" in result.diagnostics["status"]
    result = controller.plan(q, tip+[.1, 0., 0.], tip)
    assert result.diagnostics["hold"]
    assert "disagrees" in result.diagnostics["status"]


def test_free_tip_ivp_matches_independent_bvp_and_nonzero_shooting_residual():
    solver = two_tube_solver(); model = PaperIVPModel(solver)
    q = np.array([-.23, -.15, .5, -.3])
    eq = solver.solve(q)
    z = solver.scale*eq.base_torsional_strain
    prediction = model.predict(q, z)
    np.testing.assert_allclose(prediction.tip, eq.tip, atol=2e-8)
    np.testing.assert_allclose(prediction.residual, solver.scale*eq.distal_torsional_strain, atol=1e-7)
    assert np.max(abs(prediction.residual)) < 1e-7
    assert np.max(abs(model.predict(q, z+[.1, -.1]).residual)) > 1e-3


def test_straight_rod_boundary_residual_is_base_strain_at_own_tip():
    solver = EquilibriumSolver([TubeParameters(.2, 0., .001, .002, 50e9, 23e9, 0.)])
    prediction = PaperIVPModel(solver).predict([-.03, 1.2], [.6])
    np.testing.assert_allclose(prediction.tip, [0., 0., .17], atol=1e-10)
    np.testing.assert_allclose(prediction.residual, [.6], atol=1e-10)


def test_actual_two_tube_mpc_moves_toward_reachable_goal_and_solves_boundary():
    solver = two_tube_solver(); model = PaperIVPModel(solver)
    q = np.array([-.25, -.17, 0., 0.])
    eq = solver.solve(q)
    target = solver.solve(q+[.002, .001, .06, -.03]).tip
    controller = NonlinearMPC(model, solver.constraints, [.003, .003, .1, .1],
        MPCOptions(horizon=2, max_iterations=20, max_model_evaluations=400, base_separation_m=1e-5))
    result = controller.plan(q, eq.tip, target, torsion_scaled=solver.scale*eq.base_torsional_strain)
    assert not result.diagnostics["hold"], result.diagnostics
    actual = solver.solve(result.joints, initial_torsion=eq.base_torsional_strain)
    assert np.linalg.norm(actual.tip-target) < np.linalg.norm(eq.tip-target)
    np.testing.assert_allclose(actual.tip, result.predicted_tip, atol=1e-5)
    assert result.diagnostics["boundary_residual_scaled"] <= 1e-5
    assert solver.constraints.is_feasible(result.joints, atol=1e-10)


def test_original_action_substeps_match_environment_including_active_constraints():
    env = make_env(configuration(), evaluation=True, tolerance=.0015)
    rng = np.random.default_rng(2020)
    try:
        for _ in range(30):
            q = env.trig_obj.sample_goal(0)
            env.trig_obj.set_joints(q, 0)
            action = rng.uniform(-1., 1., 6).astype(np.float32)
            predicted = original_joint_step(q, action, env.action_scale, env.n_substeps,
                env.trig_obj.tube_lengths[0], env.trig_obj.constrain_alpha)
            for _ in range(env.n_substeps):
                env.trig_obj.set_action(action.astype(float)*env.action_scale, 0)
            np.testing.assert_array_equal(predicted, env.trig_obj.joints)
    finally:
        env.close()


def test_saved_three_tube_geometry_mpc_has_consistent_executed_prediction():
    from ctr_reach_envs.config import CTR_SYSTEMS_PARAMETERS
    solver = EquilibriumSolver([TubeParameters(**t) for t in CTR_SYSTEMS_PARAMETERS["ctr_0"].values()])
    q = np.array([-.28, -.2, -.1, 0., 0., 0.])
    eq = solver.solve(q)
    target = solver.solve(q+[.003, .001, .001, .06, -.03, .03]).tip
    controller = NonlinearMPC(PaperIVPModel(solver), solver.constraints, [.003]*3+[.1]*3,
        MPCOptions(horizon=2, max_iterations=25, max_model_evaluations=400, base_separation_m=1e-5))
    result = controller.plan(q, eq.tip, target, torsion_scaled=solver.scale*eq.base_torsional_strain)
    assert not result.diagnostics["hold"], result.diagnostics
    actual = solver.solve(result.joints, initial_torsion=eq.base_torsional_strain)
    assert np.linalg.norm(actual.tip-target) < np.linalg.norm(eq.tip-target)
    np.testing.assert_allclose(actual.tip, result.predicted_tip, atol=1e-5)
    assert solver.constraints.is_feasible(result.joints)


def test_evaluation_preserves_original_tasks_and_logs_prediction_costs(tmp_path):
    from evaluate_ctr_mpc import main
    config = configuration(); path = tmp_path/"source.json"
    path.write_text(json.dumps(config))
    env = make_env(config, evaluation=True, tolerance=.0015)
    expected = []
    try:
        for seed in (920000, 920001):
            env.reset(seed=seed)
            expected.append(hashlib.sha256(np.r_[env.trig_obj.joints, env.desired_goal].astype("<f8").tobytes()).hexdigest())
    finally:
        env.close()
    out = tmp_path/"evaluation"
    summary = main(["--config", str(path), "--episodes", "2", "--max-steps", "2",
        "--horizon", "1", "--max-iterations", "3", "--max-model-evaluations", "60", "--output-dir", str(out)])
    rows = list(csv.DictReader((out/"episodes.csv").open()))
    records = list(csv.DictReader((out/"controller.csv").open()))
    assert [r["task_fingerprint"] for r in rows] == expected
    assert summary["environment_fingerprint"] == config["environment_fingerprint"]
    assert summary["complete"] and summary["failures"] == 0
    assert summary["prediction_costs"]["ivp_calls"] > summary["decisions"]
    assert max(float(r["prediction_error_m"]) for r in records) < 1e-7
    assert not summary["physical_stability_guarantee"]
    with pytest.raises(SystemExit):
        main(["--config", str(path), "--output-dir", str(out)])


def test_paper_evaluator_records_failure_without_excluding_episode(tmp_path, monkeypatch):
    import evaluate_ctr_mpc as entry
    path = tmp_path/"source.json"; path.write_text(json.dumps(configuration()))

    def fail(*args, **kwargs):
        raise RuntimeError("injected BVP failure")

    monkeypatch.setattr(entry.EquilibriumSolver, "solve", fail)
    summary = entry.main(["--config", str(path), "--plant", "paper_bvp", "--episodes", "1",
        "--max-steps", "1", "--output-dir", str(tmp_path/"paper")])
    assert summary["episodes"] == summary["failures"] == 1
    assert summary["success_rate"] == 0.
    assert summary["mean_final_error_m"] is None
    assert summary["environment_fingerprint"] != summary["source_environment_fingerprint"]
