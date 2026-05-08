"""Small RL-Games player wrapper for no-ROS sim2sim deployment."""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Optional

import numpy as np
import torch
import yaml
from gym import spaces
from omegaconf import OmegaConf


VENDORED_RL_GAMES = pathlib.Path(__file__).resolve().parents[2] / "rl_games"
if VENDORED_RL_GAMES.is_dir():
    sys.path.insert(0, str(VENDORED_RL_GAMES))


def _register_omegaconf_resolvers() -> None:
    resolvers = {
        "eq": lambda x, y: str(x).lower() == str(y).lower(),
        "contains": lambda x, y: str(x).lower() in str(y).lower(),
        "if": lambda pred, a, b: a if pred else b,
        "resolve_default": lambda default, arg: default if arg == "" else arg,
        "eval": lambda x: eval(x),
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)


def read_cfg(config_path: str | pathlib.Path, device: Optional[str] = None) -> dict:
    _register_omegaconf_resolvers()
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    cfg = OmegaConf.to_container(OmegaConf.create(raw_cfg), resolve=True)
    if device is not None:
        if "train" in cfg:
            train_cfg = cfg["train"]["params"]["config"]
            train_cfg["device"] = device
            train_cfg["device_name"] = device
        else:
            cfg["rl_device"] = device
            cfg["sim_device"] = device
    return cfg


class RlPlayer:
    """RL-Games player facade with the 140-observation / 29-action policy contract."""

    def __init__(
        self,
        num_observations: int,
        num_actions: int,
        config_path: str | pathlib.Path,
        checkpoint_path: Optional[str | pathlib.Path],
        device: str,
        num_envs: int = 1,
    ) -> None:
        from rl_games.torch_runner import Runner

        self.num_observations = num_observations
        self.num_actions = num_actions
        self.device = device
        self.num_envs = num_envs
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_observations,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_actions,), dtype=np.float32)
        self.set_env_state = lambda *args, **kwargs: None
        self.cfg = read_cfg(config_path=config_path, device=device)

        from rl_games.common import env_configurations

        env_configurations.register(
            "rlgpu", {"env_creator": lambda **kwargs: self, "vecenv_type": "RLGPU"}
        )

        config = self.cfg["train"] if "train" in self.cfg else {"params": self.cfg["params"]}
        config["params"]["config"]["num_actors"] = num_envs
        if checkpoint_path is not None:
            config["params"]["load_checkpoint"] = True
            config["params"]["load_path"] = str(checkpoint_path)

        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        runner = Runner()
        runner.load(config)
        self.player = runner.create_player()
        self.player.init_rnn()
        self.player.has_batch_dimension = True
        self.player.batch_size = num_envs
        if checkpoint_path is not None:
            self.player.restore(str(checkpoint_path))

    def get_normalized_action(self, obs: torch.Tensor, deterministic_actions: bool = True) -> torch.Tensor:
        batch_size = obs.shape[0]
        if obs.shape != (batch_size, self.num_observations):
            raise ValueError(f"Expected obs shape {(batch_size, self.num_observations)}, got {obs.shape}.")

        # SAPO / mixed-exploration checkpoints expect the exploration-coefficient token.
        if getattr(self.player, "intr_reward_coef_embd", None) is not None:
            token = self.player.intr_reward_coef_embd
            if token.shape[0] != batch_size:
                token = token[:1].repeat(batch_size, 1)
            obs = torch.cat([obs, token.to(obs.device)], dim=1)
        else:
            obs = torch.cat([obs, 50.0 + torch.zeros((batch_size, 1), device=obs.device)], dim=1)

        action = self.player.get_action(obs=obs, is_deterministic=deterministic_actions)
        return action.reshape(batch_size, self.num_actions)

    def reset(self) -> None:
        self.player.reset()

