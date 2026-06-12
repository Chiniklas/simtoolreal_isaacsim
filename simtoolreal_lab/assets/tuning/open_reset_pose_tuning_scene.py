"""Open the SHARPA nut-screw scene at the training reset pose.

This is intentionally an Isaac Lab env launcher, not a static USD authoring
script: the arm/finger reset pose is config-driven and only becomes real after
the task is instantiated and reset.
"""

from __future__ import annotations

import argparse
import importlib
import pathlib
import sys

from isaaclab.app import AppLauncher


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_TASK = "sharpa_nutscrew_forge"
DEFAULT_AGENT_CFG = PROJECT_ROOT / "simtoolreal_lab/tasks/sharpa_nutscrew_forge/agents/rl_games_sapo_cfg.yaml"


parser = argparse.ArgumentParser(description="Open the SHARPA reset-pose tuning scene.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Gym task id.")
parser.add_argument(
    "--agent_cfg",
    "--agent-cfg",
    dest="agent_cfg",
    type=pathlib.Path,
    default=DEFAULT_AGENT_CFG,
    help="Agent YAML whose env_cfg overrides define the tuning reset pose.",
)
parser.add_argument("--object", type=str, default=None, help="Optional object override.")
parser.add_argument("--debug_keypoints", action="store_true", default=False, help="Visualize keypoints.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import simtoolreal_lab.tasks.sharpa_nutscrew_forge.gym_setup  # noqa: F401
import simtoolreal_lab.tasks.sharpa_nutscrew_pick_place_screw.gym_setup  # noqa: F401


def _load_agent_cfg(path: pathlib.Path) -> dict:
    path = path.expanduser().resolve()
    print(f"[INFO]: Loading agent cfg from: {path}")
    with path.open("r", encoding="utf-8") as cfg_file:
        loaded_cfg = yaml.safe_load(cfg_file)
    if not isinstance(loaded_cfg, dict):
        raise ValueError(f"Agent cfg must be a YAML mapping: {path}")
    return loaded_cfg


def _apply_object_selection(env_cfg) -> None:
    cfg_module = importlib.import_module(env_cfg.__class__.__module__)
    cfg_module.apply_object_selection(env_cfg)


def _set_cfg_value(cfg, key_path: str, value) -> None:
    target = cfg
    keys = key_path.split(".")
    for key in keys[:-1]:
        if not hasattr(target, key):
            raise AttributeError(f"Unknown env cfg key '{key_path}': missing '{key}'.")
        target = getattr(target, key)
    final_key = keys[-1]
    if not hasattr(target, final_key):
        raise AttributeError(f"Unknown env cfg key '{key_path}': missing '{final_key}'.")
    setattr(target, final_key, value)


def _apply_agent_env_cfg(env_cfg, agent_cfg: dict) -> None:
    env_overrides = agent_cfg.get("env_cfg", {})
    for key_path, value in env_overrides.items():
        _set_cfg_value(env_cfg, key_path, value)

    if "sim_dt" in env_overrides:
        env_cfg.sim.dt = env_cfg.sim_dt
    if "decimation" in env_overrides:
        env_cfg.sim.render_interval = env_cfg.decimation


def main() -> None:
    agent_cfg = _load_agent_cfg(args_cli.agent_cfg)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    _apply_agent_env_cfg(env_cfg, agent_cfg)
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.object is not None:
        env_cfg.object_name = args_cli.object
    if args_cli.debug_keypoints:
        env_cfg.debug_keypoints = True
    _apply_object_selection(env_cfg)

    print(f"[INFO]: active_fingers = {getattr(env_cfg, 'active_fingers', '<not set>')}")
    print(f"[INFO]: reset overrides = {getattr(env_cfg, 'reset_joint_pos_overrides', {})}")
    print(f"[INFO]: table pos = {env_cfg.table_cfg.init_state.pos}")
    print(f"[INFO]: object start pose = {getattr(env_cfg, 'object_start_pose', None)}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()
    print("[INFO]: Scene reset. Tune joints in Physics Inspector, then copy values back into the YAML.")

    while simulation_app.is_running():
        env.sim.render()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
