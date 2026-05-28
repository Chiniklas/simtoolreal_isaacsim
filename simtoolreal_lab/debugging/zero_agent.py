"""Script to run a SimToolReal SHARPA environment with a zero-action agent.

Adapted from IsaacLab's scripts/environments/zero_agent.py. The two project-specific
additions are the gym_setup import (so the simtoolreal_sharpa task is registered)
and the post-CLI apply_object_selection call (so --object overrides actually land).
"""

from __future__ import annotations

import argparse
import importlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Zero-action agent for simtoolreal_sharpa.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="simtoolreal_sharpa", help="Gym task id.")
parser.add_argument("--object", type=str, default=None, help="Optional DexToolBench object name (e.g. claw_hammer).")
parser.add_argument("--max_steps", type=int, default=None, help="Stop after this many environment steps.")
parser.add_argument("--log_every", type=int, default=60, help="Print summary every N steps (0 to disable).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import simtoolreal_lab.tasks.sharpa_nutscrew_pick_place_screw.gym_setup  # noqa: F401
import simtoolreal_lab.tasks.simtoolreal_sharpa.gym_setup  # noqa: F401


def apply_object_selection(env_cfg) -> None:
    cfg_module = importlib.import_module(env_cfg.__class__.__module__)
    cfg_module.apply_object_selection(env_cfg)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.object is not None:
        env_cfg.object_name = args_cli.object
    apply_object_selection(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space:      {env.action_space}")
    print(f"[INFO]: object_name = {getattr(env_cfg, 'object_name', '<not set>')}")
    print(f"[INFO]: num_envs    = {env.unwrapped.num_envs}")

    env.reset()
    step = 0
    while simulation_app.is_running():
        if args_cli.max_steps is not None and step >= args_cli.max_steps:
            break
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            obs, rewards, terminated, truncated, _ = env.step(actions)
        step += 1
        if args_cli.log_every and step % args_cli.log_every == 0:
            r = rewards.detach().to(torch.float32)
            done_count = int((terminated | truncated).sum().item())
            print(
                f"[step {step:>5d}] reward mean={r.mean().item():+.4f} "
                f"min={r.min().item():+.4f} max={r.max().item():+.4f} "
                f"done_count={done_count}/{env.unwrapped.num_envs}",
                flush=True,
            )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
