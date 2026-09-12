"""Public commands, objective attribution, and classical-control comparison."""
from copy import deepcopy
import json
import sys

import numpy as np
import pytest
import torch

from ctr_reach_envs.training.cli import mechanics_defaults, baseline_main, guided_main
from ctr_reach_envs.training.mechanics import arguments
from ctr_reach_envs.paper_config import paper_configuration
from ctr_reach_envs.mechanics.geometry import JointConstraints, TubeParameters
from ctr_reach_envs.mechanics.classical import constrained_jacobian_action, desired_displacement
from ctr_reach_envs.mechanics.rl_jacobian import JacobianDDPG, JacobianHerReplayBuffer, ProjectedJacobianLoss
from ctr_reach_envs.mechanics.simple_rl_env import JointConstrainedReachEnv
from ctr_reach_envs.mechanics.rl_policy import EquilibriumStateExtractor
from ctr_reach_envs.paper_policy import PaperMlpPolicy
from test_mechanics_jacobian import synthetic_sample


def test_two_commands_resolve_matched_mechanics_settings_and_preserve_paper(capsys):
    baseline_main(["--profile","paper","--dry-run"])
    paper=json.loads(capsys.readouterr().out)
    assert paper["profile"]=="paper-2024" and paper["curriculum_steps"]==1500000
    baseline_main(["--profile","mechanics","--dry-run"])
    base=json.loads(capsys.readouterr().out)
    guided_main(["--dry-run"])
    guided=json.loads(capsys.readouterr().out)
    differences={key for key in base if base[key]!=guided[key]}
    assert differences=={"physics_weight","output_dir"}
    spec=paper_configuration()
    assert base["batch_size"]==spec["batch_size"]==256
    assert base["train_freq"]==100 and base["gradient_steps"]==50
    assert base["learning_starts"]==0 and base["wait_for_completed_batch"]
    assert base["task_profile"]=="generalized_reach"
    assert guided["physics_anneal_steps"]==5000 and guided["physics_final_weight"]==0


def test_baseline_rejects_physics_and_partial_rollout_budgets():
    with pytest.raises(SystemExit):
        arguments(["--physics-weight",".1"],mechanics_defaults(True),baseline=True)
    with pytest.raises(SystemExit):
        arguments(["--total-timesteps","101"],mechanics_defaults())


def test_mechanics_only_and_checked_rl_ablations_are_explicit():
    a=arguments(["--ablation","mechanics-only"],mechanics_defaults())
    assert a.physics_final_weight==a.physics_weight>0
    b=arguments(["--ablation","rl-only-checked"],mechanics_defaults())
    assert b.physics_weight==b.physics_final_weight==0


def test_loss_units_do_not_change_target_cap_and_no_teacher_action_is_used():
    sample=synthetic_sample(.1708)
    action=torch.tensor([[.2,.3]],dtype=torch.float64,requires_grad=True)
    loss=ProjectedJacobianLoss([.2],gain=.4,cartesian_scale=.002,max_tip_step=.001)
    value=loss(action,sample)
    target=.4*float(sample.observations["desired_goal"][0,2]-sample.observations["achieved_goal"][0,2])
    predicted=.001*.2
    assert float(value.detach())==pytest.approx(((predicted-target)/.002)**2,abs=1e-8)
    grad=torch.autograd.grad(value,action)[0]
    assert grad[0,0].item()==pytest.approx(2*(predicted-target)*.001/.002**2,abs=1e-7)
    assert grad[0,1]==0  # Rotation lies in this straight tube's positional null space.
    scaled=ProjectedJacobianLoss([.2],gain=.4,cartesian_scale=.004,max_tip_step=.001)
    torch.testing.assert_close(scaled(action,sample),value/4)


def test_classical_qp_matches_scalar_solution_and_respects_joint_boundary():
    constraint=JointConstraints([.2])
    j=np.array([[0.,0.],[0.,0.],[1.,0.]])
    scale=np.array([.001,.05]); damping=.1
    action=constrained_jacobian_action(j,[-.03,0.],scale,constraint,[0.,0.,.0005],damping=damping)
    # A=.001/.002, b=(.5*.0005)/.002.
    assert action[0]==pytest.approx((.5*.125)/(.5**2+damping**2),rel=1e-5)
    assert abs(action[1])<1e-7
    boundary=constrained_jacobian_action(j,[0.,0.],scale,constraint,[0.,0.,.01])
    assert abs(boundary[0])<1e-6
    assert constraint.is_feasible(np.array([0.,0.])+scale*boundary,atol=1e-10)
    np.testing.assert_allclose(desired_displacement([.01,0,0]),[.002,0,0])


def tiny_model(objective="mechanics_only"):
    env=JointConstrainedReachEnv(tubes=[TubeParameters(.2,0.,.001,.002,50e9,23e9,0.)],
        task_profile="legacy",max_episode_steps=4,compute_jacobian=True)
    model=JacobianDDPG(PaperMlpPolicy,env,physics_lengths=[.2],physics_weight=.1,
        physics_final_weight=.1,actor_objective=objective,seed=15,device="cpu",
        learning_starts=20,buffer_size=100,batch_size=4,
        policy_kwargs={"net_arch":[16,16],"features_extractor_class":EquilibriumStateExtractor,
                       "features_extractor_kwargs":{"length_scale":.2}},
        replay_buffer_class=JacobianHerReplayBuffer,
        replay_buffer_kwargs={"copy_info_dict":True,"action_semantics":"proposal"})
    model.learn(8)
    return model


def test_mechanics_only_actor_update_is_independent_of_critic(monkeypatch):
    torch.set_num_threads(1)
    model=tiny_model()
    try:
        data=model.replay_buffer.sample(8)
        monkeypatch.setattr(model.replay_buffer,"sample",lambda *a,**kw:data)
        initial=deepcopy(model.policy.state_dict())
        actor_optimizer=deepcopy(model.actor.optimizer.state_dict())
        critic_optimizer=deepcopy(model.critic.optimizer.state_dict())
        model.train(1,8)
        expected={k:v.detach().clone() for k,v in model.actor.state_dict().items()}
        model.policy.load_state_dict(initial)
        model.actor.optimizer.load_state_dict(actor_optimizer)
        model.critic.optimizer.load_state_dict(critic_optimizer)
        with torch.no_grad():
            for p in model.critic.parameters():p.add_(.03)
            for p in model.critic_target.parameters():p.add_(.04)
        model.train(1,8)
        for k,v in expected.items():torch.testing.assert_close(model.actor.state_dict()[k],v,rtol=0,atol=0)
    finally:model.get_env().close()


def test_new_loss_settings_survive_checkpoint_and_zero_weight_is_rl_only(tmp_path):
    model=tiny_model("hybrid")
    try:
        model.physics_gain=.2;model.physics_cartesian_scale=.004;model.physics_max_tip_step=.001
        model.physics_weight=model.physics_final_weight=0.
        model.train(1,4)
        assert model.physics_update_count==0
        model.save(tmp_path/"model.zip")
        restored=JacobianDDPG.load(tmp_path/"model.zip",env=model.get_env())
        assert (restored.physics_gain,restored.physics_cartesian_scale,restored.physics_max_tip_step)==(.2,.004,.001)
        assert restored.actor_objective=="hybrid"
    finally:model.get_env().close()
