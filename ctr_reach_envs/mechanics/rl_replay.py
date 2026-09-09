"""HER whose action labels describe the executed physical transition."""
from copy import deepcopy
import numpy as np
from ctr_reach_envs.her_replay_buffer import GoalTerminationHerReplayBuffer


class ExecutedActionHerReplayBuffer(GoalTerminationHerReplayBuffer):
    """Replace proposed actions before storing either real or virtual samples.

    Requires normalized [-1, 1] plant actions and copied transition information.
    Relabelling never reruns a goal-dependent controller or modifies plant state.
    Inherit the repository's reward AND termination relabelling fix.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not np.all(self.action_space.low == -1) or not np.all(self.action_space.high == 1):
            raise ValueError("Executed-action HER requires [-1, 1] action coordinates")

    def add(self, obs, next_obs, action, reward, done, infos):
        if len(infos) != self.n_envs:
            raise ValueError("One executed-action record per environment is required")
        executed = []
        for index, info in enumerate(infos):
            if "executed_action" not in info or "proposed_action" not in info:
                raise ValueError("Missing executed/proposed action: use the equilibrium RL environment")
            value = np.asarray(info["executed_action"], dtype=np.float32)
            proposal = np.asarray(info["proposed_action"], dtype=np.float32)
            if value.shape != self.action_space.shape or not np.all(np.isfinite(value)) or np.any(abs(value)>1):
                raise ValueError("Invalid executed action")
            if proposal.shape != value.shape or not np.allclose(proposal, action[index], rtol=0, atol=1e-6):
                raise ValueError("Collector action and logged proposal differ")
            executed.append(value)
        # Validate the whole vector batch before modifying the replay buffer.
        super().add(obs, next_obs, np.stack(executed), reward, done, deepcopy(infos))
