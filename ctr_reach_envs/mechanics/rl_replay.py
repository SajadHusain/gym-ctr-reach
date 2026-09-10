"""HER with an explicit action convention for the critic and its targets."""
from copy import deepcopy
import numpy as np
from ctr_reach_envs.her_replay_buffer import GoalTerminationHerReplayBuffer


class ExecutedActionHerReplayBuffer(GoalTerminationHerReplayBuffer):
    """Store plant proposals or legacy executed actions consistently.

    ``proposal`` treats goal-independent projection/backtracking/holds as part
    of the plant transition. It matches an actor/target critic queried with raw
    proposals, including when a nonzero proposal produces a held transition.

    ``executed`` retains the earlier convention for explicit legacy comparisons
    and checkpoint compatibility. It does not resolve the raw-actor/filtered-
    action mismatch. The historical class name is kept for saved SB3 models.
    """
    def __init__(self, *args, action_semantics="executed", **kwargs):
        if action_semantics not in ("proposal", "executed"):
            raise ValueError("action_semantics must be proposal or executed")
        self.action_semantics = action_semantics
        super().__init__(*args, **kwargs)
        if not np.all(self.action_space.low == -1) or not np.all(self.action_space.high == 1):
            raise ValueError("Executed-action HER requires [-1, 1] action coordinates")

    def add(self, obs, next_obs, action, reward, done, infos):
        if len(infos) != self.n_envs:
            raise ValueError("One executed-action record per environment is required")
        replay_actions = []
        for index, info in enumerate(infos):
            if "executed_action" not in info or "proposed_action" not in info:
                raise ValueError("Missing executed/proposed action: use the equilibrium RL environment")
            value = np.asarray(info["executed_action"], dtype=np.float32)
            proposal = np.asarray(info["proposed_action"], dtype=np.float32)
            if value.shape != self.action_space.shape or not np.all(np.isfinite(value)) or np.any(abs(value)>1):
                raise ValueError("Invalid executed action")
            if proposal.shape != value.shape or not np.allclose(proposal, action[index], rtol=0, atol=1e-6):
                raise ValueError("Collector action and logged proposal differ")
            if not np.all(np.isfinite(proposal)) or np.any(abs(proposal)>1):
                raise ValueError("Invalid proposed action")
            if self.action_semantics == "proposal" and info.get("action_source") != "plant":
                raise ValueError("Proposal-action HER requires the goal-independent plant; goal-dependent guidance is unsupported")
            replay_actions.append(proposal if self.action_semantics == "proposal" else value)
        # Validate the whole vector batch before modifying the replay buffer.
        super().add(obs, next_obs, np.stack(replay_actions), reward, done, deepcopy(infos))
