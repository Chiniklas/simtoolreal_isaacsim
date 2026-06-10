"""Gym registration for the SHARPA nut-screw Forge-style Isaac Lab task."""

import gymnasium as gym

from . import agents
from .sharpa_nutscrew_forge_env import SharpaNutscrewForgeEnv
from .sharpa_nutscrew_forge_env_cfg import SharpaNutscrewForgeEnvCfg


gym.register(
    id="sharpa_nutscrew_forge",
    entry_point="simtoolreal_lab.tasks.sharpa_nutscrew_forge.sharpa_nutscrew_forge_env:SharpaNutscrewForgeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SharpaNutscrewForgeEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_sapo_cfg.yaml",
    },
)

