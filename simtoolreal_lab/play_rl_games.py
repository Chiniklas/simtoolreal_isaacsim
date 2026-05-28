"""Play Isaac Lab RL-Games checkpoints with the vendored reference RL-Games fork."""

from __future__ import annotations

import argparse
import importlib
import math
import pickle
import pathlib
import sys
import yaml

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play an Isaac Lab RL-Games checkpoint.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--task", type=str, default="simtoolreal_sharpa", help="Gym task id.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.")
parser.add_argument("--object", type=str, default=None, help="Object or asset name to replay with.")
parser.add_argument("--debug_keypoints", action="store_true", default=False, help="Visualize object and goal keypoints.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

VENDORED_RL_GAMES = pathlib.Path(__file__).resolve().parent / "rl_games"
if VENDORED_RL_GAMES.is_dir():
    sys.path.insert(0, str(VENDORED_RL_GAMES))

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.algos_torch import torch_ext
from rl_games.torch_runner import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import simtoolreal_lab.tasks.sharpa_nutscrew_pick_place_screw.gym_setup  # noqa: F401
import simtoolreal_lab.tasks.simtoolreal_sharpa.gym_setup  # noqa: F401


class SimToolRealRlGamesVecEnvWrapper(RlGamesVecEnvWrapper):
    def get_env_state(self):
        if hasattr(self.unwrapped, "get_env_state"):
            return self.unwrapped.get_env_state()
        return None

    def set_env_state(self, env_state):
        if hasattr(self.unwrapped, "set_env_state"):
            self.unwrapped.set_env_state(env_state)


class SimToolRealRlGamesGpuEnv(RlGamesGpuEnv):
    def get_env_state(self):
        if hasattr(self.env, "get_env_state"):
            return self.env.get_env_state()
        return None

    def set_env_state(self, env_state):
        if hasattr(self.env, "set_env_state"):
            self.env.set_env_state(env_state)


def _apply_object_selection(env_cfg) -> None:
    cfg_module = importlib.import_module(env_cfg.__class__.__module__)
    cfg_module.apply_object_selection(env_cfg)


def _checkpoint_params_dir(checkpoint_path: str | pathlib.Path) -> pathlib.Path | None:
    checkpoint_path = pathlib.Path(checkpoint_path).resolve()
    for parent in checkpoint_path.parents:
        params_dir = parent / "params"
        if (params_dir / "agent.yaml").is_file():
            return params_dir
    return None


def _checkpoint_coef_id_count(checkpoint_path: str) -> int | None:
    checkpoint = torch_ext.load_checkpoint(checkpoint_path)
    if 0 in checkpoint:
        checkpoint = checkpoint[0]
    model_state = checkpoint.get("model", {})
    for name in ("a2c_network.extra_params", "a2c_network.sigma"):
        weight = model_state.get(name)
        if weight is not None and weight.ndim >= 2:
            return int(weight.shape[0])
    return None


def _load_replay_env_cfg(task_name: str, checkpoint_path: str):
    params_dir = _checkpoint_params_dir(checkpoint_path)
    env_pickle_path = params_dir / "env.pkl" if params_dir is not None else None
    if env_pickle_path is not None and env_pickle_path.is_file():
        print(f"[INFO]: Loading environment config from: {env_pickle_path}")
        with open(env_pickle_path, "rb") as f:
            return pickle.load(f)
    return parse_env_cfg(
        task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )


def _load_replay_agent_cfg(task_name: str, checkpoint_path: str) -> dict:
    params_dir = _checkpoint_params_dir(checkpoint_path)
    agent_yaml_path = params_dir / "agent.yaml" if params_dir is not None else None
    if agent_yaml_path is not None and agent_yaml_path.is_file():
        print(f"[INFO]: Loading agent config from: {agent_yaml_path}")
        with open(agent_yaml_path, "r") as f:
            return yaml.safe_load(f)
    return load_cfg_from_registry(task_name, "rl_games_cfg_entry_point")


def _player_obs(obs: torch.Tensor | dict[str, torch.Tensor], player: BasePlayer) -> torch.Tensor:
    """Convert Isaac Lab RL-Games observations to the tensor expected by RL-Games players."""
    if isinstance(obs, dict):
        obs = obs["obs"]
    if obs.dim() == 3 and obs.shape[0] == 1:
        obs = obs.squeeze(0)
    if player.intr_reward_coef_embd is not None:
        obs = torch.cat([obs, player.intr_reward_coef_embd], dim=1)
    return obs


def _restore_policy_only(player: BasePlayer, checkpoint_path: str) -> None:
    """Restore network weights without replaying reference IsaacGym env state."""
    checkpoint = torch_ext.load_checkpoint(checkpoint_path)
    if 0 in checkpoint:
        checkpoint = checkpoint[0]
    player.model.load_state_dict(checkpoint["model"])
    if player.normalize_input and "running_mean_std" in checkpoint:
        player.model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
    player.loaded_checkpoint = checkpoint_path


def _checkpoint_success_tolerance(checkpoint_path: str) -> float | None:
    """Return the saved environment success tolerance when present in a checkpoint."""
    checkpoint = torch_ext.load_checkpoint(checkpoint_path)
    if 0 in checkpoint:
        checkpoint = checkpoint[0]
    env_state = checkpoint.get("env_state") or {}
    success_tolerance = env_state.get("success_tolerance")
    if success_tolerance is None:
        return None
    return float(success_tolerance)


def main():
    resume_path = retrieve_file_path(args_cli.checkpoint)
    env_cfg = _load_replay_env_cfg(args_cli.task, resume_path)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False
    checkpoint_success_tolerance = _checkpoint_success_tolerance(resume_path)
    if checkpoint_success_tolerance is not None:
        env_cfg.success_tolerance = checkpoint_success_tolerance
    if args_cli.object is not None:
        env_cfg.object_name = args_cli.object
    env_cfg.debug_keypoints = args_cli.debug_keypoints
    _apply_object_selection(env_cfg)
    agent_cfg = _load_replay_agent_cfg(args_cli.task, resume_path)
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device
    coef_id_count = _checkpoint_coef_id_count(resume_path)
    if coef_id_count is not None:
        agent_cfg["params"]["config"].setdefault("player", {})["coef_id_count"] = coef_id_count

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SimToolRealRlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    agent_cfg["params"]["config"]["num_actors"] = env.num_envs
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: SimToolRealRlGamesGpuEnv(config_name, num_actors, **kwargs))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    runner = Runner()
    runner.load(agent_cfg)
    player: BasePlayer = runner.create_player()
    _restore_policy_only(player, resume_path)
    player.reset()
    player.has_batch_dimension = True
    player.batch_size = env.num_envs

    obs = env.reset()
    with torch.inference_mode():
        while simulation_app.is_running():
            action = player.get_action(_player_obs(obs, player), is_deterministic=True)
            obs, _, done, _ = env.step(action)
            if done.any():
                obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
