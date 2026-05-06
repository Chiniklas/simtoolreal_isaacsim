"""Play Isaac Lab RL-Games checkpoints with the vendored reference RL-Games fork."""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play an Isaac Lab RL-Games checkpoint.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Gym task id.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.")
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
from rl_games.torch_runner import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import simtoolreal_lab.tasks.simtoolreal_sharpa.gym_setup  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    resume_path = retrieve_file_path(args_cli.checkpoint)
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    runner = Runner()
    runner.load(agent_cfg)
    player: BasePlayer = runner.create_player()
    player.restore(resume_path)
    player.reset()

    obs = env.reset()
    with torch.inference_mode():
        while simulation_app.is_running():
            action = player.get_action(obs, is_deterministic=True)
            obs, _, done, _ = env.step(action)
            if done.any():
                obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
