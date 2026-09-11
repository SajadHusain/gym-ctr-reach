"""Numerical underflow, action metrics, and matched-study reporting contracts."""
import csv
import warnings

import numpy as np
import pytest
from scipy.integrate import DOP853, solve_ivp

from ctr_reach_envs.mechanics.integration import ScaleSafeDOP853
from ctr_reach_envs.mechanics.rl_metrics import EpisodeMotion, summarize_motion
from run_physics_study import (ARMS, DEFAULTS, checkpoints, paired_seed_statistics,
                               report, validate_config, write_json)


@pytest.mark.parametrize("magnitude",[0.,1e-170,1e-160,1e-150,1.,1e150])
def test_scaled_error_norm_is_finite_and_homogeneous(magnitude):
    solver=ScaleSafeDOP853(lambda t,y:y,0.,[1.,2.],1.)
    k=np.random.default_rng(4).normal(size=solver.K.shape)
    expected=DOP853._estimate_error_norm(solver,k,.1,np.ones(2))
    with warnings.catch_warnings():
        warnings.simplefilter("error",RuntimeWarning)
        got=solver._estimate_error_norm(k*magnitude,.1,np.ones(2))
    assert np.isfinite(got)
    assert got==pytest.approx(expected*magnitude,rel=1e-12,abs=0.)


def test_normal_integration_matches_scipy_without_changing_tolerances():
    kwargs=dict(fun=lambda t,y:np.array([y[1],-y[0]]),t_span=(0.,3.),y0=[1.,0.],rtol=1e-9,atol=1e-11)
    a=solve_ivp(**kwargs,method=DOP853)
    b=solve_ivp(**kwargs,method=ScaleSafeDOP853)
    np.testing.assert_array_equal(a.t,b.t)
    np.testing.assert_array_equal(a.y,b.y)


def record(proposed,executed=None):
    a=np.asarray(proposed,dtype=float)
    b=a if executed is None else np.asarray(executed,dtype=float)
    return {"proposed_action":a,"executed_action":b,"applied_delta":b*np.array([.001,.05])}


def test_action_metrics_distinguish_command_from_execution_and_units():
    metric=EpisodeMotion(1)
    metric.add(record([1.,1.],[.5,.5]));metric.add(record([-1.,-1.],[-.5,-.5]))
    values=metric.summary()
    assert values["proposed_action_change_rms"]==2.
    assert values["executed_action_change_rms"]==1.
    assert values["translation_increment_change_rms_m"]==.001
    assert values["rotation_increment_change_rms_rad"]==.05
    assert values["executed_action_total_variation"]==pytest.approx(np.sqrt(2))
    assert values["executed_action_second_difference_rms"] is None


def test_single_step_and_reset_boundaries_do_not_invent_smoothness():
    a=EpisodeMotion(1);b=EpisodeMotion(1)
    a.add(record([1.,0.]));b.add(record([-1.,0.]))
    results=summarize_motion([a.summary(),b.summary()])
    assert results["executed_action_change_rms"]=={"mean":None,"episodes":0}
    assert results["action_difference_pairs"]["mean"]==0


def test_jacobian_tracking_does_not_penalize_nullspace_action_oscillation():
    # Straight one-tube position Jacobian: rotation does not move the tip.
    jacobian=np.array([[0.,0.],[0.,0.],[1.,0.]])
    a=np.array([.5,1.]);b=np.array([.5,-1.])
    np.testing.assert_array_equal(jacobian@a,jacobian@b)
    metric=EpisodeMotion(1)
    metric.add(record(a));metric.add(record(b))
    assert metric.summary()["executed_action_change_rms"]>1.


def test_study_has_only_joint_constrained_unassisted_arms():
    assert ARMS["ddpg"]=={"guidance":"none","physics_weight":0.}
    assert ARMS["jacobian"]["guidance"]=="none"
    assert set(ARMS)=={"ddpg","jacobian"}
    validate_config(DEFAULTS)
    assert checkpoints({**DEFAULTS,"total_timesteps":1100,"checkpoint_freq":500})==[0,500,1000,1100]
    with pytest.raises(ValueError,match="distinct"):
        validate_config({**DEFAULTS,"final_seed":DEFAULTS["eval_seed"]})


def test_uncertainty_uses_training_seeds_and_single_seed_has_no_interval():
    assert paired_seed_statistics([.1])["bootstrap_95"] is None
    result=paired_seed_statistics([.1,.2,.3])
    assert result["seeds"]==3 and result["mean_difference"]==pytest.approx(.2)
    assert result["seed_std"]==pytest.approx(.1)
    assert paired_seed_statistics([])["mean_difference"] is None


def fixture_evaluation(folder,arm,seed,fp="same"):
    checkpoint=folder/arm/f"seed_{seed}"/"checkpoints"/"step_000000002"
    out=checkpoint/"final_evaluation";out.mkdir(parents=True)
    write_json(checkpoint/"budget.json",{"gradient_updates":1,"training_seconds":2.,
              "physics_costs_including_resets_and_witnesses":{"equilibrium_calls":5}})
    fields=EpisodeMotion(1).summary()
    metrics={k:{"mean":0.,"episodes":1} for k in fields}
    write_json(out/"summary.json",{"complete":True,"episodes_per_mode":1,"checkpoint_timesteps":2,
        "first_seed":910000,"max_steps":60,"tolerance_m":.001,
        "modes":{"actor":{"success_rate":1.,"mean_error_m_on_completed_steps":.0005,"failures":0,
            "equilibrium_calls":3,"steps":2,"replaced_actions":0,"jacobian_fallbacks":0,"motion_all_episodes":metrics}}})
    with (out/"episodes.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["mode","seed","success","task_fingerprint","executed_action_change_rms"])
        writer.writeheader();writer.writerow(dict(mode="actor",seed=910000,success=True,task_fingerprint=fp,executed_action_change_rms=.2))


def test_report_checks_paired_tasks_and_marks_missing_runs(tmp_path):
    c={**DEFAULTS,"seeds":[1,2],"total_timesteps":2,"final_episodes":1,"task_profile":"legacy"}
    fixture_evaluation(tmp_path,"ddpg",1)
    fixture_evaluation(tmp_path,"jacobian",1)
    value=report({"config":c},tmp_path,final=True)
    assert not value["complete"] and len(value["missing_jobs"])==2
    assert value["contrasts"]["jacobian_minus_ddpg"]["success_rate"]["seeds"]==1
    fixture_evaluation(tmp_path,"ddpg",2,fp="different")
    with pytest.raises(ValueError,match="tasks differ"):
        report({"config":c},tmp_path,final=True)


def test_report_rejects_legacy_results_for_a_generalized_hold_study(tmp_path):
    c={**DEFAULTS,"seeds":[1],"total_timesteps":2,"final_episodes":1}
    fixture_evaluation(tmp_path,"ddpg",1)
    with pytest.raises(ValueError,match="task profile"):
        report({"config":c},tmp_path,final=True)


def test_generalized_study_window_validation():
    with pytest.raises(ValueError,match="Holding window"):
        validate_config({**DEFAULTS,"hold_steps":61})
    with pytest.raises(ValueError,match="goal step"):
        validate_config({**DEFAULTS,"goal_steps_min":10,"goal_steps_max":3})
