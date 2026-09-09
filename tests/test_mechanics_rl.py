"""Executed-action replay, goal relabelling, API behavior and actual learning."""
import numpy as np
import pytest

from ctr_reach_envs.mechanics import TubeParameters
from ctr_reach_envs.mechanics.rl_env import EquilibriumReachEnv, GuidedRolloutWrapper


def make_env(guided=False, horizon=4):
    tube = TubeParameters(.2, 0., .001, .002, 50e9, 23e9, 0.)
    env = EquilibriumReachEnv(tubes=[tube], max_episode_steps=horizon)
    return GuidedRolloutWrapper(env) if guided else env


def reset_options(z=.18):
    return {"joints": [-.03, 0.], "goal": [0., 0., z]}


def batch(obs):
    return {k:v[None].copy() for k,v in obs.items()}


def test_plant_transition_independent_of_goal_and_contains_branch_state():
    a, b = make_env(), make_env()
    obs, _ = a.reset(seed=4, options=reset_options(.19))
    b.reset(seed=4, options=reset_options(.15))
    aa = a.step([.5, .1]); bb = b.step([.5, .1])
    assert a.observation_space.contains(obs)
    np.testing.assert_array_equal(aa[0]["observation"], bb[0]["observation"])
    np.testing.assert_array_equal(aa[0]["achieved_goal"], bb[0]["achieved_goal"])
    np.testing.assert_allclose(aa[0]["observation"][3:4],
        a.length*a.tracker.state.equilibrium.base_torsional_strain)
    np.testing.assert_allclose(aa[4]["applied_delta"], [.0005,.005])


def test_guided_action_replays_through_plant_with_a_different_goal():
    guided, plant = make_env(True), make_env()
    guided.reset(seed=8, options=reset_options(.18))
    plant.reset(seed=8, options=reset_options(.15))
    got = guided.step([-1.,0.])
    replay = plant.step(got[4]["executed_action"])
    assert got[4]["action_replaced"]
    assert got[4]["action_source"] == "jacobian_dls"
    assert got[4]["numerical_decrease_verified"]
    assert got[4]["executed_action"][0] > 0
    np.testing.assert_allclose(guided.unwrapped.tracker.state.equilibrium.tip,
                               plant.tracker.state.equilibrium.tip, atol=1e-10, rtol=0)
    np.testing.assert_array_equal(got[0]["achieved_goal"], replay[0]["achieved_goal"])


def test_three_tube_executed_action_replays_on_the_same_branch():
    guided=GuidedRolloutWrapper(EquilibriumReachEnv())
    plant=EquilibriumReachEnv()
    q=[-.1,-.05,-.02,.4,-.7,1.2]
    tip=plant.solver.solve(q).tip
    guided.reset(options={"joints":q,"goal":tip+[.01,0.,0.]})
    initial=guided.unwrapped.tracker.state
    plant.reset(options={"joints":q,"initial_torsion":initial.equilibrium.base_torsional_strain,
                         "goal":tip+[0.,.01,0.]})
    # A zero proposal forces a genuine multi-tube Jacobian fallback.
    obs,reward,term,trunc,info=guided.step(np.zeros(6))
    assert info["action_source"]=="jacobian_dls"
    assert info["numerical_decrease_verified"]
    plant.step(info["executed_action"])
    a=guided.unwrapped.tracker.state.equilibrium
    b=plant.tracker.state.equilibrium
    np.testing.assert_allclose(a.tip,b.tip,atol=2e-7,rtol=0)
    np.testing.assert_allclose(a.base_torsional_strain*plant.length,
                               b.base_torsional_strain*plant.length,atol=1e-5,rtol=0)


def test_timeout_is_not_termination_and_invalid_actions_do_not_commit():
    env = make_env(horizon=1)
    env.reset(seed=8, options=reset_options(.18))
    before = env.tracker.state.equilibrium.joints.copy()
    with pytest.raises(ValueError):
        env.step([np.nan,0.])
    np.testing.assert_array_equal(before, env.tracker.state.equilibrium.joints)
    obs, reward, term, trunc, _ = env.step([0.,0.])
    assert not term and trunc and reward == -1
    with pytest.raises(RuntimeError):
        env.step([0.,0.])


def test_seeded_reset_and_costs_cover_witnesses_and_initialization():
    env = make_env()
    first, info = env.reset(seed=5)
    second, again = env.reset(seed=5)
    for k in first:
        np.testing.assert_array_equal(first[k],second[k])
    assert info["reset_costs"]["equilibrium_calls"] >= 13
    assert info["reset_costs"]["sensitivity_calls"] >= 10
    assert env.reset_attempts == 2
    assert env.costs["equilibrium_calls"] == info["reset_costs"]["equilibrium_calls"]+again["reset_costs"]["equilibrium_calls"]


def test_rewards_are_vectorized_and_use_recorded_tolerance():
    env = make_env()
    achieved=np.array([[0.,0.,.17],[0.,0.,.17]])
    goals=np.array([[0.,0.,.1715],[0.,0.,.1715]])
    infos=[{"position_tolerance":.002},{"position_tolerance":.001}]
    np.testing.assert_array_equal(env.compute_reward(achieved,goals,infos),[0.,-1.])
    np.testing.assert_array_equal(env.compute_terminated(achieved,goals,infos),[True,False])


def test_gymnasium_and_sb3_environment_contract():
    from gymnasium.utils.env_checker import check_env
    check_env(make_env(),skip_render_check=True)
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env as sb3_check
    sb3_check(make_env(True))


@pytest.mark.parametrize("original_success", [False,True])
def test_her_keeps_executed_action_and_relabels_success_at_timeout(monkeypatch, original_success):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv
    from ctr_reach_envs.mechanics.rl_replay import ExecutedActionHerReplayBuffer
    env=make_env(True)
    obs,_=env.reset(seed=8, options=reset_options(.1715 if original_success else .18))
    nxt,reward,term,trunc,info=env.step([-1.,0.])
    vec=DummyVecEnv([lambda:make_env(True)])
    buf=ExecutedActionHerReplayBuffer(20,vec.observation_space,vec.action_space,env=vec,
                                     copy_info_dict=True,device="cpu")
    info["TimeLimit.truncated"]=not original_success
    buf.add(batch(obs),batch(nxt),np.array([[-1.,0.]]),np.array([reward]),np.array([True]),[info])
    np.testing.assert_array_equal(buf.actions[0,0],info["executed_action"])
    assert buf.actions[0,0,0]>0
    goal = np.array([[0.,0.,.19]],dtype=np.float32) if original_success else nxt["achieved_goal"][None].copy()
    monkeypatch.setattr(buf,"_sample_goals",lambda *a:goal)
    old_features=buf.observations["observation"].copy()
    sample=buf._get_virtual_samples(np.array([0]),np.array([0]))
    assert sample.dones.item()==int(not original_success)
    assert sample.rewards.item()==(-1 if original_success else 0)
    np.testing.assert_array_equal(sample.actions.numpy()[0], info["executed_action"])
    np.testing.assert_array_equal(buf.observations["observation"],old_features)
    np.testing.assert_array_equal(buf.observations["desired_goal"][0,0],obs["desired_goal"])
    np.testing.assert_array_equal(buf.infos[0,0]["proposed_action"],[-1.,0.])
    vec.close()


def test_missing_executed_action_fails_before_replay_mutation():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv
    from ctr_reach_envs.mechanics.rl_replay import ExecutedActionHerReplayBuffer
    vec=DummyVecEnv([make_env])
    obs=vec.reset()
    buf=ExecutedActionHerReplayBuffer(20,vec.observation_space,vec.action_space,env=vec,
                                     copy_info_dict=True,device="cpu")
    with pytest.raises(ValueError,match="Missing executed"):
        buf.add(obs,obs,np.zeros((1,2)),np.array([-1]),np.array([True]),[{}])
    assert buf.pos==0
    vec.close()


def test_ddpg_updates_actor_and_critic_and_save_load_preserves_predictions(tmp_path):
    sb3=pytest.importorskip("stable_baselines3")
    import torch
    from ctr_reach_envs.mechanics.rl_replay import ExecutedActionHerReplayBuffer
    from ctr_reach_envs.mechanics.rl_policy import EquilibriumStateExtractor
    from ctr_reach_envs.paper_policy import PaperMlpPolicy
    torch.set_num_threads(1)
    env=make_env(True)
    model=sb3.DDPG(PaperMlpPolicy,env,seed=5,learning_starts=4,
        buffer_size=100,batch_size=8,train_freq=1,gradient_steps=1,
        policy_kwargs={"net_arch":[16,16],"features_extractor_class":EquilibriumStateExtractor,
                       "features_extractor_kwargs":{"length_scale":.2}},device="cpu",
        replay_buffer_class=ExecutedActionHerReplayBuffer,
        replay_buffer_kwargs={"copy_info_dict":True,"n_sampled_goal":4,"goal_selection_strategy":"future"})
    actor=[p.detach().clone() for p in model.actor.parameters()]
    critic=[p.detach().clone() for p in model.critic.parameters()]
    model.learn(total_timesteps=12)
    assert model._n_updates>0
    assert any(not torch.equal(a,p) for a,p in zip(actor,model.actor.parameters()))
    assert any(not torch.equal(a,p) for a,p in zip(critic,model.critic.parameters()))
    assert all(torch.isfinite(p).all() for p in model.policy.parameters())
    buf=model.replay_buffer
    for i in range(buf.pos):
        np.testing.assert_array_equal(buf.actions[i,0],buf.infos[i,0]["executed_action"])
        terminal=buf.dones[i,0]
        if terminal:
            # SB3 must store the terminal observation, not the automatic reset.
            np.testing.assert_allclose(buf.next_observations["observation"][i,0,:3],
                [np.cos(buf.infos[i,0]["q_after"][1]),np.sin(buf.infos[i,0]["q_after"][1]),
                 buf.infos[i,0]["q_after"][0]/.2],atol=1e-6)
    obs,_=env.reset(seed=25)
    prediction=model.predict(obs,deterministic=True)[0]
    path=tmp_path/"model.zip"
    model.save(path)
    loaded=sb3.DDPG.load(path,env=make_env(True),device="cpu")
    np.testing.assert_array_equal(prediction,loaded.predict(obs,deterministic=True)[0])
    env.close(); loaded.get_env().close()
