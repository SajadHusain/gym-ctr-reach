"""Paper state and legacy DDPG network structure on the modern SB3 backend."""

import numpy as np
import torch as th
from torch import nn
from stable_baselines3 import DDPG
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import MultiInputPolicy


class PaperStateExtractor(BaseFeaturesExtractor):
    """Equation (6): nine joint features, achieved-minus-desired, tolerance.

    Goal error is reconstructed on every forward pass, including HER samples.
    Storing error inside the goal-independent observation would make it stale
    when HER replaces desired_goal.
    """

    def __init__(self, observation_space):
        if observation_space["observation"].shape != (10,):
            raise ValueError("The paper system-0 profile requires ten base observation features")
        super().__init__(observation_space, features_dim=13)

    def forward(self, observations):
        state = observations["observation"]
        error = observations["achieved_goal"] - observations["desired_goal"]
        return th.cat((state[:, :9], error, state[:, 9:10]), dim=1)


def initialize_legacy_mlp(module):
    """Match TF Dense defaults for hidden layers and small output weights."""
    layers = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
    for layer in layers:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
    nn.init.uniform_(layers[-1].weight, -0.003, 0.003)


class LateActionQ(nn.Module):
    """Feed state to hidden layer 1; then concatenate the action for layer 2."""

    def __init__(self, features_dim, action_dim, net_arch, activation_fn):
        super().__init__()
        if len(net_arch) < 2:
            raise ValueError("Late-action critic needs at least two hidden layers")
        self.features_dim = features_dim
        self.state_layer = nn.Sequential(nn.Linear(features_dim, net_arch[0]), activation_fn())
        layers = []
        input_dim = net_arch[0] + action_dim
        for width in net_arch[1:]:
            layers.extend([nn.Linear(input_dim, width), activation_fn()])
            input_dim = width
        layers.append(nn.Linear(input_dim, 1))
        self.value_layers = nn.Sequential(*layers)
        initialize_legacy_mlp(self)

    def forward(self, features_and_actions):
        state = features_and_actions[:, :self.features_dim]
        action = features_and_actions[:, self.features_dim:]
        hidden = self.state_layer(state)
        return self.value_layers(th.cat((hidden, action), dim=1))


class PaperCritic(ContinuousCritic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Called with named arguments by TD3Policy.make_critic.
        self.q_networks = []
        for index in range(self.n_critics):
            network = LateActionQ(
                kwargs["features_dim"], int(np.prod(self.action_space.shape)),
                kwargs["net_arch"], kwargs.get("activation_fn", nn.ReLU),
            )
            self.add_module(f"qf{index}", network)
            self.q_networks.append(network)


class PaperMlpPolicy(MultiInputPolicy):
    def make_actor(self, features_extractor=None):
        actor = super().make_actor(features_extractor)
        initialize_legacy_mlp(actor.mu)
        return actor

    def make_critic(self, features_extractor=None):
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return PaperCritic(**critic_kwargs).to(self.device)


class PaperDDPG(DDPG):
    """DDPG with legacy uniform-action exploration in addition to Gaussian noise."""

    def __init__(self, *args, random_exploration=0.294, **kwargs):
        if not 0.0 <= random_exploration <= 1.0:
            raise ValueError("random_exploration must be between zero and one")
        self.random_exploration = float(random_exploration)
        super().__init__(*args, **kwargs)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        action, buffer_action = super()._sample_action(learning_starts, action_noise, n_envs)
        if self.num_timesteps >= learning_starts:
            random_mask = np.random.random(n_envs) < self.random_exploration
            for index in np.flatnonzero(random_mask):
                action[index] = self.action_space.sample()
                # The replay action must be exactly the executed normalized action.
                buffer_action[index] = self.policy.scale_action(action[index])
        return action, buffer_action

    def train(self, gradient_steps, batch_size=100):
        # HER cannot sample unfinished episodes. Unlike a uniform warmup phase,
        # this guard retains the paper's policy/noise/random mixture from step 1.
        if np.count_nonzero(self.replay_buffer.ep_length) < batch_size:
            return
        super().train(gradient_steps, batch_size)
