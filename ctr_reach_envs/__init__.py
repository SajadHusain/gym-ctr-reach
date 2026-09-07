"""Gymnasium registration for the CTR reaching environment."""

from gymnasium.envs.registration import register, registry

from ctr_reach_envs.config import default_env_kwargs


ENV_ID = "CTR-Reach-v1"

if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="ctr_reach_envs.envs:CtrReachEnv",
        kwargs=default_env_kwargs(evaluation=False),
    )

__all__ = ["ENV_ID", "default_env_kwargs"]

