"""Gym registration for the SHARPA nut-screw pick-place Isaac Lab task."""

import gymnasium as gym

from . import agents
from .sharpa_nutscrew_pick_place_env import SharpaNutscrewPickPlaceEnv
from .sharpa_nutscrew_pick_place_env_cfg import SharpaNutscrewPickPlaceEnvCfg


gym.register(
    id="sharpa_nutscrew_pick_place",
    entry_point="simtoolreal_lab.tasks.sharpa_nutscrew_pick_place.sharpa_nutscrew_pick_place_env:SharpaNutscrewPickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SharpaNutscrewPickPlaceEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sapo_cfg.yaml",
    },
)

gym.register(
    id="sharpa_nutscrew_pick_place_pretrain_like",
    entry_point="simtoolreal_lab.tasks.sharpa_nutscrew_pick_place.sharpa_nutscrew_pick_place_env:SharpaNutscrewPickPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SharpaNutscrewPickPlaceEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sapo_pretrain_like_cfg.yaml",
    },
)
