"""Train Forge tasks with the installed upstream RL-Games vanilla PPO."""

from __future__ import annotations

import argparse
import inspect
import logging
import math
import os
import random
import re
import sys
import time
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train a Forge task with vanilla RL-Games PPO.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Recorded video length in steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Steps between recorded videos.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Forge Gym task id.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Registered vanilla PPO configuration."
)
parser.add_argument("--seed", type=int, default=None, help="Environment and agent seed.")
parser.add_argument("--distributed", action="store_true", default=False, help="Enable multi-GPU training.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to resume.")
parser.add_argument("--sigma", type=str, default=None, help="Initial policy standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="Maximum RL-Games epochs.")
parser.add_argument("--run_name", type=str, default=None, help="Experiment label included in the run directory.")
parser.add_argument(
    "--debug_rewards",
    action="store_true",
    default=False,
    help="Print the Forge reward/contact breakdown once per epoch.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver, IsaacAlgoObserver
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
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import simtoolreal_lab.tasks.sharpa_forgeUltra.gym_setup  # noqa: F401

logger = logging.getLogger(__name__)


class ObserverGroup(AlgoObserver):
    """Forward RL-Games observer hooks across Isaac Lab 2.2 and 2.3."""

    def __init__(self, observers):
        super().__init__()
        self.observers = observers

    def _call(self, method, *args):
        for observer in self.observers:
            callback = getattr(observer, method, None)
            if callback is not None:
                callback(*args)

    def before_init(self, base_name, config, experiment_name):
        self._call("before_init", base_name, config, experiment_name)

    def after_init(self, algo):
        self._call("after_init", algo)

    def process_infos(self, infos, done_indices):
        self._call("process_infos", infos, done_indices)

    def after_steps(self):
        self._call("after_steps")

    def after_clear_stats(self):
        self._call("after_clear_stats")

    def after_print_stats(self, frame, epoch_num, total_time):
        self._call("after_print_stats", frame, epoch_num, total_time)


def _wrap_env(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups):
    wrapper_params = inspect.signature(RlGamesVecEnvWrapper).parameters
    if "obs_groups" in wrapper_params:
        return RlGamesVecEnvWrapper(
            env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups
        )
    return RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)


def _run_directory(task_name: str, run_name: str | None) -> str:
    task_id = task_name.split(":")[-1]
    match = re.search(r"Isaac-Forge-([A-Za-z]+)", task_id)
    task_short = (
        match.group(1).lower()
        if match
        else re.sub(r"[^A-Za-z0-9]+", "-", task_id).strip("-").lower()
    )
    family = "KukaForge" if "Kuka" in task_id else "Forge"
    label = run_name if run_name else "run"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{family}-{task_short}-{label}-{timestamp}"


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train a Forge policy using the installed upstream RL-Games package."""

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError("Distributed training requires a CUDA device.")

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations
        if args_cli.max_iterations is not None
        else agent_cfg["params"]["config"]["max_epochs"]
    )

    resume_path = None
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    env_cfg.seed = agent_cfg["params"]["seed"]
    config_name = agent_cfg["params"]["config"]["name"]
    log_root_path = os.path.abspath(os.path.join("logs", "rl_games", config_name))
    log_dir = _run_directory(args_cli.task, args_cli.run_name)
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    env_cfg.log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Logging experiment in directory: {env_cfg.log_dir}")

    dump_yaml(os.path.join(env_cfg.log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(env_cfg.log_dir, "params", "agent.yaml"), agent_cfg)

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
            "video_folder": os.path.join(env_cfg.log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
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
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    observers = [IsaacAlgoObserver()]
    if args_cli.debug_rewards:
        from simtoolreal_lab.tasks.sharpa_forgeUltra.forge_kuka_debug import DebugRewardObserver

        observers.append(DebugRewardObserver())

    print(f"[INFO] Using upstream rl_games from: {sys.modules['rl_games'].__file__}")
    runner = Runner(ObserverGroup(observers) if len(observers) > 1 else observers[0])
    runner.load(agent_cfg)
    runner.reset()

    start_time = time.time()
    run_args = {"train": True, "play": False, "sigma": train_sigma}
    if resume_path is not None:
        run_args["checkpoint"] = resume_path
    runner.run(run_args)
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
