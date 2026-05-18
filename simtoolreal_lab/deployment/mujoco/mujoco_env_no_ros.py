"""No-ROS MuJoCo sim2sim runner for a SimToolReal pretrained policy."""

from __future__ import annotations

import select
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from simtoolreal_lab.deployment.mujoco.mujoco_sim import (
    DEXTOOLBENCH_OBJECT_SCALES,
    MUJOCO_REPLAY_SCENE_PATH,
    SIMTOOLREAL_LAB_DIR,
    MujocoSim,
    MujocoSimConfig,
)
from simtoolreal_lab.deployment.mujoco.policy_player import RlPlayer


N_OBS = 140
N_ACT = 29
DEFAULT_POLICY_DIR = SIMTOOLREAL_LAB_DIR / "pretrained_policy"
DEFAULT_OBS_LIST = [
    "joint_pos",
    "joint_vel",
    "prev_action_targets",
    "palm_pos",
    "palm_rot",
    "object_rot",
    "fingertip_pos_rel_palm",
    "keypoints_rel_palm",
    "keypoints_rel_goal",
    "object_scales",
]
Q_LOWER_LIMITS = np.array(
    [
        -2.9671,
        -2.0944,
        -2.9671,
        -2.0944,
        -2.9671,
        -2.0944,
        -3.0543,
        -0.1745,
        -0.3491,
        -0.5236,
        -0.3491,
        0.0000,
        -0.1745,
        -0.0349,
        0.0000,
        0.0000,
        -0.1745,
        -0.0349,
        0.0000,
        0.0000,
        -0.1745,
        -0.0349,
        0.0000,
        0.0000,
        0.0000,
        -0.1745,
        -0.0349,
        0.0000,
        0.0000,
    ]
)
Q_UPPER_LIMITS = np.array(
    [
        2.9671,
        2.0944,
        2.9671,
        2.0944,
        2.9671,
        2.0944,
        3.0543,
        1.9199,
        0.1309,
        1.3963,
        0.3491,
        1.7453,
        1.5708,
        0.0349,
        1.7453,
        1.3963,
        1.5708,
        0.0349,
        1.7453,
        1.3963,
        1.5708,
        0.0349,
        1.7453,
        1.3963,
        0.2618,
        1.5708,
        0.0349,
        1.7453,
        1.3963,
    ]
)
KEYPOINT_OFFSETS = np.array([[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]], dtype=np.float32)
FINGERTIP_BODY_NAMES = [
    "palmleft_index_DP",
    "palmleft_middle_DP",
    "palmleft_ring_DP",
    "palmleft_thumb_DP",
    "palmleft_pinky_DP",
]
PALM_OFFSET = np.array([-0.00, -0.02, 0.16])
FINGERTIP_OFFSETS = np.array(
    [
        [0.02, 0.002, 0.0],
        [0.02, 0.002, 0.0],
        [0.02, 0.002, 0.0],
        [0.02, 0.002, 0.0],
        [0.02, 0.002, 0.0],
    ]
)


def _quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return q[[1, 2, 3, 0]]


def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_w = q[:, 3]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0)[:, None]
    b = np.cross(q_vec, v, axis=-1) * q_w[:, None] * 2.0
    c = q_vec * (np.sum(q_vec * v, axis=-1)[:, None]) * 2.0
    return a + b + c


def _unscale(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return (2.0 * x - upper - lower) / (upper - lower)


def _scale(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def _compute_keypoints(pos: np.ndarray, quat_xyzw: np.ndarray, scales: np.ndarray) -> np.ndarray:
    offsets = KEYPOINT_OFFSETS[None] * 0.04 * 1.5 * 0.5 * scales[:, None]
    keypoints = np.zeros((pos.shape[0], 4, 3), dtype=np.float32)
    for i in range(4):
        keypoints[:, i] = pos + _quat_rotate_xyzw(quat_xyzw, offsets[:, i])
    return keypoints


def _compute_joint_pos_targets(
    actions: np.ndarray,
    prev_targets: np.ndarray,
    hand_moving_average: float,
    arm_moving_average: float,
    arm_dof_speed_scale: float,
    dt: float,
) -> np.ndarray:
    cur_targets = prev_targets.copy()
    cur_targets[:, 7:] = _scale(actions[:, 7:], Q_LOWER_LIMITS[7:], Q_UPPER_LIMITS[7:])
    cur_targets[:, 7:] = hand_moving_average * cur_targets[:, 7:] + (1.0 - hand_moving_average) * prev_targets[:, 7:]
    cur_targets[:, 7:] = np.clip(cur_targets[:, 7:], Q_LOWER_LIMITS[7:], Q_UPPER_LIMITS[7:])

    cur_targets[:, :7] = prev_targets[:, :7] + arm_dof_speed_scale * dt * actions[:, :7]
    cur_targets[:, :7] = np.clip(cur_targets[:, :7], Q_LOWER_LIMITS[:7], Q_UPPER_LIMITS[:7])
    cur_targets[:, :7] = arm_moving_average * cur_targets[:, :7] + (1.0 - arm_moving_average) * prev_targets[:, :7]
    return cur_targets


class MujocoEnvNoRos:
    def __init__(
        self,
        sim: MujocoSim,
        object_scales: np.ndarray,
        hand_moving_average: float,
        arm_moving_average: float,
        hand_dof_speed_scale: float,
        control_dt: float,
        device: str,
        obs_list: list[str],
    ):
        self.sim = sim
        self.object_scales = object_scales
        self.hand_moving_average = hand_moving_average
        self.arm_moving_average = arm_moving_average
        self.hand_dof_speed_scale = hand_dof_speed_scale
        self.control_dt = control_dt
        self.device = device
        self.obs_list = obs_list

    @property
    def sim_steps_per_control_step(self) -> int:
        return max(1, int(round(self.control_dt / self.sim.config.sim_dt)))

    def compute_observation(self) -> torch.Tensor:
        sim_state = self.sim.get_sim_state()
        object_pose_w = np.concatenate(
            [sim_state["object_pos"], sim_state["object_quat_wxyz"][[1, 2, 3, 0]]]
        )
        goal_object_pose_w = np.concatenate(
            [sim_state["goal_object_pos"], sim_state["goal_object_quat_wxyz"][[1, 2, 3, 0]]]
        )
        q = sim_state["joint_positions"][None]
        qd = sim_state["joint_velocities"][None]
        wrist_pos, wrist_quat_wxyz = self.sim.get_body_pose("link7")
        palm_quat_xyzw = _quat_wxyz_to_xyzw(wrist_quat_wxyz)[None]
        palm_pos = wrist_pos + _quat_rotate_xyzw(palm_quat_xyzw, PALM_OFFSET[None])[0]
        fingertip_pos_list = []
        for idx, name in enumerate(FINGERTIP_BODY_NAMES):
            tip_pos, tip_quat_wxyz = self.sim.get_body_pose(name)
            tip_quat_xyzw = _quat_wxyz_to_xyzw(tip_quat_wxyz)[None]
            fingertip_pos_list.append(
                tip_pos + _quat_rotate_xyzw(tip_quat_xyzw, FINGERTIP_OFFSETS[idx : idx + 1])[0]
            )
        fingertip_pos = np.stack(fingertip_pos_list, axis=0)
        object_keypoints = _compute_keypoints(
            object_pose_w[None, :3], object_pose_w[None, 3:7], self.object_scales[None]
        )
        goal_keypoints = _compute_keypoints(
            goal_object_pose_w[None, :3], goal_object_pose_w[None, 3:7], self.object_scales[None]
        )
        obs_dict = {
            "joint_pos": _unscale(q, Q_LOWER_LIMITS, Q_UPPER_LIMITS),
            "joint_vel": qd,
            "prev_action_targets": self.sim.robot_joint_pos_targets[None],
            "palm_pos": palm_pos[None],
            "palm_rot": palm_quat_xyzw,
            "object_rot": object_pose_w[None, 3:7],
            "fingertip_pos_rel_palm": (fingertip_pos[None] - palm_pos[None, None]).reshape(1, -1),
            "keypoints_rel_palm": (object_keypoints - palm_pos[None, None]).reshape(1, -1),
            "keypoints_rel_goal": (object_keypoints - goal_keypoints).reshape(1, -1),
            "object_scales": self.object_scales[None],
        }
        obs = np.concatenate([obs_dict[key] for key in self.obs_list], axis=-1)
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        if obs_tensor.shape != (1, N_OBS):
            raise RuntimeError(f"Expected observation shape {(1, N_OBS)}, got {obs_tensor.shape}.")
        return obs_tensor

    def step(self, action: torch.Tensor) -> None:
        joint_pos_targets = _compute_joint_pos_targets(
            actions=action.detach().cpu().numpy(),
            prev_targets=self.sim.robot_joint_pos_targets[None],
            hand_moving_average=self.hand_moving_average,
            arm_moving_average=self.arm_moving_average,
            arm_dof_speed_scale=self.hand_dof_speed_scale,
            dt=self.control_dt,
        )
        self.sim.set_robot_joint_pos_targets(joint_pos_targets[0])
        for _ in range(self.sim_steps_per_control_step):
            self.sim.sim_step()
            if self.sim.viewer is not None:
                self.sim.viewer.sync()


@dataclass
class MujocoEnvNoRosArgs:
    config_path: Path = DEFAULT_POLICY_DIR / "config.yaml"
    checkpoint_path: Path = DEFAULT_POLICY_DIR / "model.pth"
    object_name: str = "claw_hammer"
    enable_viewer: bool = True
    max_steps: int | None = None
    sim_hz: float = 600.0
    control_hz: float = 60.0
    show_robot_collision_overlay: bool = True
    press_enter_to_execute: bool = False
    record_video: bool = False
    video_path: Path = Path("mujoco_rollout.mp4")
    video_fps: float = 30.0
    video_width: int = 1280
    video_height: int = 720
    video_camera: str = "side_table"


def _object_scales(object_name: str) -> np.ndarray:
    if object_name in DEXTOOLBENCH_OBJECT_SCALES:
        return np.array(DEXTOOLBENCH_OBJECT_SCALES[object_name])
    if object_name == "cuboidal_mallet":
        return np.array([6.0, 0.75, 0.5])
    if object_name == "cuboidal_hammer":
        return np.array([6.25, 0.75, 0.5])
    if object_name.startswith("cuboid_"):
        return np.array(object_name.split("_")[1:], dtype=float)
    known = ", ".join(sorted(DEXTOOLBENCH_OBJECT_SCALES))
    raise ValueError(f"Unknown object '{object_name}'. Known DexToolBench objects: {known}")


def _policy_obs_list(policy: RlPlayer) -> list[str]:
    return policy.cfg.get("task", {}).get("env", {}).get("obsList", DEFAULT_OBS_LIST)


def _wait_for_enter_to_start(sim: MujocoSim) -> bool:
    print("Adjust the MuJoCo viewer, then press Enter to start the rollout (Ctrl-D to quit).", flush=True)
    if not sys.stdin.isatty():
        return sys.stdin.readline() != ""

    while sim._continue_running():
        readable, _, _ = select.select([sys.stdin], [], [], 0.05)
        if readable:
            return sys.stdin.readline() != ""
        if sim.viewer is not None:
            sim.viewer.sync()
    return False


def _mp4_path(path: Path) -> Path:
    if path.suffix.lower() == ".mp4":
        return path
    return path.with_suffix(".mp4")


def _make_video_camera(sim: MujocoSim, camera: str):
    if camera == "side_table":
        object_pos = sim.get_sim_state()["object_pos"]
        mj_camera = sim.mujoco.MjvCamera()
        mj_camera.lookat[:] = np.array([object_pos[0], object_pos[1], 0.55])
        mj_camera.distance = 1.0
        mj_camera.azimuth = 90.0
        mj_camera.elevation = -35.0
        return mj_camera

    try:
        return int(camera)
    except ValueError:
        return camera


class MujocoMp4Recorder:
    def __init__(
        self,
        sim: MujocoSim,
        path: Path,
        fps: float,
        width: int,
        height: int,
        camera: int | str,
    ):
        if fps <= 0.0:
            raise ValueError(f"video_fps must be positive, got {fps}.")
        if width <= 0 or height <= 0:
            raise ValueError(f"video_width and video_height must be positive, got {width}x{height}.")

        try:
            import imageio.v2 as imageio
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Video recording requires imageio and imageio-ffmpeg in the active environment."
            ) from exc

        self.sim = sim
        self.path = _mp4_path(path)
        self.fps = fps
        self.camera = camera
        self.frame_period = 1.0 / fps
        self.next_frame_time = 0.0
        self.frame_count = 0
        self.renderer = None
        self.writer = None
        sim.mj_model.vis.global_.offwidth = max(sim.mj_model.vis.global_.offwidth, width)
        sim.mj_model.vis.global_.offheight = max(sim.mj_model.vis.global_.offheight, height)
        self.renderer = sim.mujoco.Renderer(sim.mj_model, height=height, width=width)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(
            self.path,
            fps=fps,
            codec="libx264",
            macro_block_size=1,
            quality=8,
        )

    def capture_until(self, sim_time: float) -> None:
        while self.next_frame_time <= sim_time + 1.0e-9:
            self.renderer.update_scene(self.sim.mj_data, camera=self.camera)
            self.writer.append_data(self.renderer.render())
            self.frame_count += 1
            self.next_frame_time += self.frame_period

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        if self.renderer is not None:
            self.renderer.close()
        print(f"Wrote {self.frame_count} video frames to {self.path}", flush=True)


def main() -> None:
    args = tyro.cli(MujocoEnvNoRosArgs)
    if not args.config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config_path}")
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    if not MUJOCO_REPLAY_SCENE_PATH.exists():
        raise FileNotFoundError(f"MuJoCo replay scene not found: {MUJOCO_REPLAY_SCENE_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sim = MujocoSim(
        MujocoSimConfig(
            enable_viewer=args.enable_viewer,
            sim_dt=1.0 / args.sim_hz,
            object_name=args.object_name,
            object_start_pos=np.array([0.0, 0.0, 0.58]),
            object_start_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            goal_object_start_pos=np.array([0.0, 0.0, 0.78]),
            goal_object_start_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            show_robot_collision_overlay=args.show_robot_collision_overlay,
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
    env = MujocoEnvNoRos(
        sim=sim,
        object_scales=_object_scales(args.object_name),
        hand_moving_average=0.1,
        arm_moving_average=0.1,
        hand_dof_speed_scale=1.5,
        control_dt=1.0 / args.control_hz,
        device=device,
        obs_list=obs_list,
    )

    if args.press_enter_to_execute and not _wait_for_enter_to_start(sim):
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
