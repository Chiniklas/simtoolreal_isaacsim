"""Gym registration for the SHARPA nut-screw pick-place-screw Isaac Lab task."""

import gymnasium as gym

from . import agents
from .sharpa_nutscrew_pick_place_screw_env import SharpaNutscrewPickPlaceScrewEnv
from .sharpa_nutscrew_pick_place_screw_env_cfg import SharpaNutscrewPickPlaceScrewEnvCfg


gym.register(
    id="sharpa_nutscrew_pick_place_screw",
    entry_point="simtoolreal_lab.tasks.sharpa_nutscrew_pick_place_screw.sharpa_nutscrew_pick_place_screw_env:SharpaNutscrewPickPlaceScrewEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SharpaNutscrewPickPlaceScrewEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sapo_cfg.yaml",
    },
)

gym.register(
    id="sharpa_nutscrew_pick_place_screw_pretrain_like",
    entry_point="simtoolreal_lab.tasks.sharpa_nutscrew_pick_place_screw.sharpa_nutscrew_pick_place_screw_env:SharpaNutscrewPickPlaceScrewEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SharpaNutscrewPickPlaceScrewEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sapo_pretrain_like_cfg.yaml",
    },
)
