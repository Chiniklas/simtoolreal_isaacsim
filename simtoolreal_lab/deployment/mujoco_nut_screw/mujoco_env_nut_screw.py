"""No-ROS MuJoCo sim2sim runner for the SHARPA nut-screw scene."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from simtoolreal_lab.deployment.mujoco.mujoco_env_no_ros import (
    DEFAULT_OBS_LIST,
    DEFAULT_POLICY_DIR,
    DEFAULT_TARGET_VOLUME_MAXS,
    DEFAULT_TARGET_VOLUME_MINS,
    N_ACT,
    N_OBS,
    MujocoEnvNoRos,
    MujocoMp4Recorder,
    _make_video_camera,
    _normalize_cli_flag_aliases,
    _policy_obs_list,
    _wait_for_enter_to_start,
)
from simtoolreal_lab.deployment.mujoco_nut_screw.mujoco_sim import (
    NUTSCREW_ASSET_DIR,
    NUT_SCREW_SCENE_PATH,
    MujocoNutScrewSim,
    MujocoNutScrewSimConfig,
    nut_screw_object_scales,
)
from simtoolreal_lab.deployment.mujoco_nut_screw.policy_player import RlPlayer


@dataclass
class MujocoNutScrewEnvArgs:
    config_path: Path = DEFAULT_POLICY_DIR / "config.yaml"
    checkpoint_path: Path = DEFAULT_POLICY_DIR / "model.pth"
    scene_xml_path: Path = NUT_SCREW_SCENE_PATH
    asset_root: Path = NUTSCREW_ASSET_DIR
    family: str = "M12"
    screw: str = "M12X30"
    nut: str = "M12_nut"
    enable_viewer: bool = True
    max_steps: int | None = None
    sim_hz: float = 600.0
    control_hz: float = 60.0
    randomize_goal: bool = False
    target_volume_mins: tuple[float, float, float] = DEFAULT_TARGET_VOLUME_MINS
    target_volume_maxs: tuple[float, float, float] = DEFAULT_TARGET_VOLUME_MAXS
    randomize_goal_rotation: bool = True
    reset_when_dropped: bool = True
    drop_reset_height: float | None = None
    seed: int | None = None
    visualize_keypoints: bool = False
    visualize_grasp_bounding_box: bool = False
    press_enter_to_execute: bool = False
    record_video: bool = False
    video_path: Path = Path("mujoco_nut_screw_rollout.mp4")
    video_fps: float = 30.0
    video_width: int = 1280
    video_height: int = 720
    video_camera: str = "side_table"


def _default_name_if_blank(value: str, fallback: str) -> str:
    return fallback if value == "" else value


def main() -> None:
    _normalize_cli_flag_aliases()
    args = tyro.cli(MujocoNutScrewEnvArgs)
    family = args.family.upper()
    nut_name = _default_name_if_blank(args.nut, f"{family}_nut")

    if not args.config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config_path}")
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim = MujocoNutScrewSim(
        MujocoNutScrewSimConfig(
            enable_viewer=args.enable_viewer,
            sim_dt=1.0 / args.sim_hz,
            scene_xml_path=args.scene_xml_path,
            family=family,
            nut_name=nut_name,
            asset_root=args.asset_root,
        )
    )
    policy = RlPlayer(
        num_observations=N_OBS,
        num_actions=N_ACT,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=device,
    )
    obs_list = _policy_obs_list(policy)
    if obs_list != DEFAULT_OBS_LIST:
        print(f"[MuJoCo] Using policy obs list from config: {obs_list}", flush=True)
    object_scales = nut_screw_object_scales(args.asset_root, family, nut_name)
    env = MujocoEnvNoRos(
        sim=sim,
        object_scales=object_scales,
        hand_moving_average=0.1,
        arm_moving_average=0.1,
        hand_dof_speed_scale=1.5,
        control_dt=1.0 / args.control_hz,
        device=device,
        obs_list=obs_list,
        visualize_keypoints=args.visualize_keypoints,
        visualize_grasp_bounding_box=args.visualize_grasp_bounding_box,
        randomize_goal=args.randomize_goal,
        target_volume_mins=args.target_volume_mins,
        target_volume_maxs=args.target_volume_maxs,
        randomize_goal_rotation=args.randomize_goal_rotation,
        reset_when_dropped=args.reset_when_dropped,
        drop_reset_height=args.drop_reset_height,
        seed=args.seed,
    )
    env.reset()
    policy.reset()

    if args.press_enter_to_execute and not _wait_for_enter_to_start(
        sim, args.visualize_keypoints, args.visualize_grasp_bounding_box, object_scales
    ):
        return

    recorder = None
    if args.record_video:
        recorder = MujocoMp4Recorder(
            sim=sim,
            path=args.video_path,
            fps=args.video_fps,
            width=args.video_width,
            height=args.video_height,
            camera=_make_video_camera(sim, args.video_camera),
        )
        recorder.capture_until(sim.mj_data.time)

    try:
        step = 0
        while sim._continue_running() and (args.max_steps is None or step < args.max_steps):
            start = time.time()
            obs = env.compute_observation()
            action = policy.get_normalized_action(obs, deterministic_actions=True)
            env.step(action)
            if env.should_reset_after_drop():
                print("[MuJoCo] Nut dropped after lift; resetting nut-screw scene.", flush=True)
                env.reset()
                policy.reset()
            if recorder is not None:
                recorder.capture_until(sim.mj_data.time)
            elapsed = time.time() - start
            sleep_dt = env.control_dt - elapsed
            if sleep_dt > 0:
                time.sleep(sleep_dt)
            else:
                print(
                    f"Control loop too slow: target={args.control_hz:.1f}Hz actual={1.0 / elapsed:.1f}Hz"
                )
            step += 1
    finally:
        if recorder is not None:
            recorder.close()


if __name__ == "__main__":
    main()
