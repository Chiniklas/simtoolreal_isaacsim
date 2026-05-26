"""Gym registration for the SimToolReal KUKA-SHARPA Isaac Lab task."""

import gymnasium as gym

from . import agents
from .simtoolreal_sharpa_env import SimToolRealSharpaEnv
from .simtoolreal_sharpa_env_cfg import SimToolRealSharpaEnvCfg


gym.register(
    id="simtoolreal_sharpa",
    entry_point="simtoolreal_lab.tasks.simtoolreal_sharpa.simtoolreal_sharpa_env:SimToolRealSharpaEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SimToolRealSharpaEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sapo_cfg.yaml",
    },
)

gym.register(
    id="simtoolreal_sharpa_pretrain_like",
    entry_point="simtoolreal_lab.tasks.simtoolreal_sharpa.simtoolreal_sharpa_env:SimToolRealSharpaEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SimToolRealSharpaEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sapo_pretrain_like_cfg.yaml",
    },
)
