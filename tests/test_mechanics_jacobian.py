"""Finite bounds, projection derivatives, HER alignment and actor-loss ablations."""
from dataclasses import replace
import warnings
import numpy as np
import pytest
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv

from ctr_reach_envs.mechanics.geometry import JointConstraints
from ctr_reach_envs.mechanics.rl_env import EquilibriumReachEnv
from ctr_reach_envs.mechanics.rl_jacobian import (PhysicsSamples,ProjectedJacobianLoss,
                                               JacobianHerReplayBuffer,JacobianDDPG)
from ctr_reach_envs.mechanics.rl_policy import EquilibriumStateExtractor
from ctr_reach_envs.paper_policy import PaperMlpPolicy
from test_mechanics_rl import make_env,reset_options,batch


def synthetic_sample(goal_z=.18,jacobian=None):
    obs={"observation":torch.zeros(1,5),"achieved_goal":torch.tensor([[0.,0.,.17]]),
         "desired_goal":torch.tensor([[0.,0.,goal_z]])}
    if jacobian is None:jacobian=torch.tensor([[[0.,0.],[0.,0.],[1.,0.]]],requires_grad=True)
    return PhysicsSamples(obs,torch.zeros(1,2),obs,torch.zeros(1,1),-torch.ones(1,1),
                          jacobian,torch.tensor([[-.03,0.]],dtype=torch.float64),
                          torch.tensor([[.001,.05]],dtype=torch.float64))


def test_finite_bounds_preserve_observation_values_and_old_checkpoint_spaces():
    new=make_env()
    old=EquilibriumReachEnv(tubes=new.solver.tubes,legacy_observation_bounds=True,max_episode_steps=4)
    a,_=new.reset(seed=3,options=reset_options())
    b,_=old.reset(seed=3,options=reset_options())
    for key in a:
        np.testing.assert_array_equal(a[key],b[key])
        assert np.all(np.isfinite(new.observation_space[key].low))
        assert np.all(np.isfinite(new.observation_space[key].high))
    assert np.isinf(old.observation_space["observation"].high).any()
    from gymnasium.utils.env_checker import check_env
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(new,skip_render_check=True)


@pytest.mark.parametrize("system",["ctr_0","ctr_1","ctr_2","ctr_3"])
def test_unloaded_envelope_covers_each_robot(system):
    env=EquilibriumReachEnv(system)
    env.reset(seed=16)
    state=env.tracker.state.equilibrium
    assert np.all(abs(env.length*state.base_torsional_strain)<=env.torsion_observation_bound)
    for _ in range(2):
        obs,*_=env.step(np.full(6,.2,dtype=np.float32))
        assert env.observation_space.contains(obs)


def test_out_of_bounds_branch_state_is_rejected_not_clipped():
    env=make_env();env.reset(options=reset_options())
    state=env.tracker._current
    bad=replace(state.equilibrium,base_torsional_strain=np.array([1e6]))
    env.tracker._current=replace(state,equilibrium=bad)
    with pytest.raises(RuntimeError,match="no clipping"):
        env._observation()
    assert env._finished


def test_projected_actor_increment_matches_plant_constraints():
    lengths=[.3,.2,.1];c=JointConstraints(lengths);loss=ProjectedJacobianLoss(lengths)
    rng=np.random.default_rng(3)
    q=np.stack([c.project(np.r_[rng.uniform(-.3,0,3),rng.normal(size=3)]) for _ in range(16)])
    scales=np.tile([.02,.02,.02,.05,.05,.05],(16,1))
    actions=rng.uniform(-1,1,(16,6))
    actual=loss.projected_delta(torch.tensor(actions),torch.tensor(q),torch.tensor(scales)).numpy()
    expected=[]
    for state,action,scale in zip(q,actions,scales):
        dq=c.project(state+scale*action)-state
        expected.append(dq/max(1,np.max(abs(dq)/scale)))
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)


def test_projection_gradient_matches_finite_differences_on_an_active_face():
    loss=ProjectedJacobianLoss([.3,.2,.1])
    q=torch.tensor([[-.005,-.003,-.002,0.,0.,0.]],dtype=torch.float64)
    scale=torch.tensor([[.01,.01,.01,.05,.05,.05]],dtype=torch.float64)
    action=torch.tensor([[.9,-.8,.7,.1,.2,.3]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:loss.projected_delta(x,q,scale),(action,),eps=1e-6,atol=1e-5,rtol=1e-3)


def test_loss_gradients_follow_relabelled_goal_and_stop_at_solver_data():
    loss=ProjectedJacobianLoss([.2])
    sample=synthetic_sample()
    action=torch.zeros(1,2,requires_grad=True)
    loss(action,sample).backward()
    assert action.grad[0,0]<0
    assert sample.jacobian.grad is None
    reverse=synthetic_sample(.16)
    other=torch.zeros(1,2,requires_grad=True)
    loss(other,reverse).backward()
    assert other.grad[0,0]>0
    np.testing.assert_allclose(action.grad.numpy(),-other.grad.numpy(),rtol=1e-5)


def test_zero_jacobian_is_finite_without_inventing_a_descent_direction():
    sample=synthetic_sample(jacobian=torch.zeros(1,3,2))
    action=torch.zeros(1,2,requires_grad=True)
    value=ProjectedJacobianLoss([.2])(action,sample)
    value.backward()
    assert torch.isfinite(value)
    assert torch.count_nonzero(action.grad)==0


def test_her_physics_context_stays_aligned_without_solver_calls(monkeypatch):
    env=make_env(horizon=4);obs,_=env.reset(options=reset_options(.19))
    vec=DummyVecEnv([make_env])
    replay=JacobianHerReplayBuffer(20,vec.observation_space,vec.action_space,env=vec,
                                   copy_info_dict=True,device="cpu")
    for i in range(4):
        nxt,r,term,trunc,info=env.step([.4,0.])
        # Distinct synthetic labels expose any source-index permutation.
        info["physics"]["jacobian"][0,0]=info["q_before"][0]
        info["TimeLimit.truncated"]=trunc
        replay.add(batch(obs),batch(nxt),np.array([[.4,0.]]),np.array([r]),np.array([term or trunc]),[info])
        obs=nxt
    monkeypatch.setattr(type(env.solver),"solve",lambda *a,**kw:pytest.fail("Optimizer sampling must not solve equilibria"))
    data=replay.sample(32)
    assert set(data.observations)=={"observation","achieved_goal","desired_goal"}
    torch.testing.assert_close(data.jacobian[:,0,0],data.joints[:,0],rtol=0,atol=0)
    torch.testing.assert_close(data.observations["observation"][:,2].double()*.2,data.joints[:,0],rtol=0,atol=1e-8)
    assert (data.observations["desired_goal"][:,2]<.19-1e-4).any()
    assert len(data.jacobian)==len(data.actions)==32
    with pytest.raises(ValueError,match="VecNormalize"):
        replay.sample(2,env=object())
    vec.close()


def model_args(env):
    return dict(policy=PaperMlpPolicy,env=env,seed=15,learning_starts=4,buffer_size=100,batch_size=8,
                train_freq=1,gradient_steps=1,device="cpu",learning_rate=.0005,
                policy_kwargs={"net_arch":[16,16],"features_extractor_class":EquilibriumStateExtractor,
                               "features_extractor_kwargs":{"length_scale":.2}},
                replay_buffer_class=JacobianHerReplayBuffer,
                replay_buffer_kwargs={"copy_info_dict":True,"n_sampled_goal":4,"goal_selection_strategy":"future"})


def test_zero_weight_exactly_reproduces_sb3_ddpg():
    torch.set_num_threads(1)
    base=DDPG(**model_args(make_env(True)))
    base.learn(12)
    zero=JacobianDDPG(**model_args(make_env(True)),physics_lengths=[.2],physics_weight=0,physics_final_weight=0)
    zero.learn(12)
    assert zero.physics_update_count==0
    for key,value in base.policy.state_dict().items():
        torch.testing.assert_close(value,zero.policy.state_dict()[key],rtol=0,atol=0)
    base.get_env().close();zero.get_env().close()


def test_physics_updates_save_load_and_annealing(tmp_path):
    torch.set_num_threads(1)
    model=JacobianDDPG(**model_args(make_env(True)),physics_lengths=[.2],physics_weight=.2,
                       physics_final_weight=.02,physics_anneal_steps=100)
    assert model.current_physics_weight==.2
    model.learn(12)
    assert model.physics_update_count==8
    assert model.last_physics_metrics["jacobian_loss"]>=0
    assert model.last_physics_metrics["extra_equilibrium_calls"]==0
    obs,_=model.get_env().envs[0].reset(seed=19)
    original=model.predict(obs,deterministic=True)[0]
    path=tmp_path/"physics.zip";model.save(path)
    loaded=JacobianDDPG.load(path,env=make_env(True),device="cpu")
    assert loaded.physics_lengths==[.2]
    assert loaded.physics_weight==.2 and loaded.physics_update_count==8
    np.testing.assert_array_equal(original,loaded.predict(obs,deterministic=True)[0])
    loaded.num_timesteps=101
    assert loaded.current_physics_weight==pytest.approx(.02)
    model.get_env().close();loaded.get_env().close()
