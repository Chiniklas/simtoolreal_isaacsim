"""Play Forge checkpoints with the installed upstream RL-Games package."""

from __future__ import annotations

import argparse
import inspect
import math
import os
import random
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a vanilla RL-Games Forge checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record a replay video.")
parser.add_argument("--video_length", type=int, default=200, help="Recorded video length in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Fabric.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Forge Gym task id.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Registered vanilla PPO configuration."
)
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--real-time", action="store_true", default=False, help="Throttle replay to simulation time.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import simtoolreal_lab.tasks.sharpa_forgeUltra.gym_setup  # noqa: F401


def _wrap_env(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups):
    wrapper_params = inspect.signature(RlGamesVecEnvWrapper).parameters
    if "obs_groups" in wrapper_params:
        return RlGamesVecEnvWrapper(
            env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups
        )
    return RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Replay a Forge policy using upstream RL-Games."""

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]

    resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = _wrap_env(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register(
        "rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env}
    )

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    print(f"[INFO] Using upstream rl_games from: {sys.modules['rl_games'].__file__}")
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)
            if agent.is_rnn and agent.states is not None:
                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
                if done_indices.numel() > 0:
                    for state in agent.states:
                        state[:, done_indices, :] = 0.0
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
