"""Train Isaac Lab tasks with the vendored reference RL-Games fork."""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys
import time
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an Isaac Lab task with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--task", type=str, default=None, help="Gym task id.")
parser.add_argument("--seed", type=int, default=None, help="Environment/agent seed.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run with multiple GPUs or nodes.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="Initial policy standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="Maximum RL-Games epochs.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
TASK_OUTPUT_ROOT = pathlib.Path(__file__).resolve().parent / "tasks" / "simtoolreal_sharpa"
if not any(arg.startswith("hydra.run.dir=") for arg in hydra_args):
    hydra_args.append(f"hydra.run.dir={TASK_OUTPUT_ROOT / 'outputs'}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

VENDORED_RL_GAMES = pathlib.Path(__file__).resolve().parent / "rl_games"
if VENDORED_RL_GAMES.is_dir():
    sys.path.insert(0, str(VENDORED_RL_GAMES))

import gymnasium as gym
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import simtoolreal_lab.tasks.simtoolreal_sharpa.gym_setup  # noqa: F401
from simtoolreal_lab.tasks.simtoolreal_sharpa.simtoolreal_sharpa_env_cfg import apply_object_selection


class SimToolRealRlGamesVecEnvWrapper(RlGamesVecEnvWrapper):
    def set_train_info(self, env_frames, *args, **kwargs):
        if hasattr(self.unwrapped, "set_train_info"):
            self.unwrapped.set_train_info(env_frames, *args, **kwargs)

    def get_env_state(self):
        if hasattr(self.unwrapped, "get_env_state"):
            return self.unwrapped.get_env_state()
        return None

    def set_env_state(self, env_state):
        if hasattr(self.unwrapped, "set_env_state"):
            self.unwrapped.set_env_state(env_state)


class SimToolRealRlGamesGpuEnv(RlGamesGpuEnv):
    def set_train_info(self, env_frames, *args, **kwargs):
        if hasattr(self.env, "set_train_info"):
            self.env.set_train_info(env_frames, *args, **kwargs)

    def get_env_state(self):
        if hasattr(self.env, "get_env_state"):
            return self.env.get_env_state()
        return None

    def set_env_state(self, env_state):
        if hasattr(self.env, "set_env_state"):
            self.env.set_env_state(env_state)


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RL-Games."""

    if args_cli.seed == -1:
        args_cli.seed = int(time.time())

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    apply_object_selection(env_cfg)
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    expl_type = agent_cfg["params"]["config"].get("expl_type", "none")
    continuous_space_cfg = agent_cfg["params"]["network"]["space"]["continuous"]
    if not str(expl_type).startswith("mixed_expl") and continuous_space_cfg.get("fixed_sigma") == "coef_cond":
        print("[INFO]: Setting fixed_sigma='fixed' because coef_cond requires mixed_expl coefficient IDs.")
        continuous_space_cfg["fixed_sigma"] = "fixed"

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
    log_root_path = str((TASK_OUTPUT_ROOT / "logs").resolve())
    # The reference SAPO RL-Games fork parses the leading token as policy_idx.
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", f"0_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = SimToolRealRlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: SimToolRealRlGamesGpuEnv(config_name, num_actors, **kwargs))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    batch_size = env.unwrapped.num_envs * agent_cfg["params"]["config"]["horizon_length"]
    if agent_cfg["params"]["config"]["minibatch_size"] > batch_size:
        print(f"[INFO]: Setting minibatch_size={batch_size} for batch_size={batch_size}.")
        agent_cfg["params"]["config"]["minibatch_size"] = batch_size
    central_value_config = agent_cfg["params"]["config"].get("central_value_config")
    if central_value_config is not None and central_value_config.get("minibatch_size", batch_size) > batch_size:
        minibatch_size = min(agent_cfg["params"]["config"]["minibatch_size"], batch_size)
        print(
            f"[INFO]: Setting central_value_config.minibatch_size={minibatch_size} "
            f"for batch_size={batch_size}."
        )
        central_value_config["minibatch_size"] = minibatch_size

    print(f"[INFO] Using rl_games from: {sys.modules['rl_games'].__file__}")
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
