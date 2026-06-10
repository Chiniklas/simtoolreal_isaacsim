"""No-ROS MuJoCo sim2sim runner for a SimToolReal pretrained policy."""

from __future__ import annotations

import select
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

from simtoolreal_lab.deployment.mujoco.mujoco_sim import (
    DEXTOOLBENCH_OBJECT_SCALES,
    MUJOCO_REPLAY_SCENE_PATH,
    SIMTOOLREAL_LAB_DIR,
    TABLE_TOP_Z,
    INIT_JOINT_POS,
    JOINT_NAMES,
    MujocoSim,
    MujocoSimConfig,
)
from simtoolreal_lab.deployment.mujoco.policy_player import RlPlayer


N_OBS = 140
N_ACT = 29
HandMode = Literal["full", "tripod", "pinch"]
FINGER_ORDER = ("index", "middle", "ring", "thumb", "pinky")
HAND_MODE_ACTIVE_FINGERS: dict[str, tuple[str, ...]] = {
    "full": FINGER_ORDER,
    "tripod": ("thumb", "index", "middle"),
    "pinch": ("thumb", "index"),
}
FINGER_JOINT_INDICES = {
    "thumb": tuple(range(7, 12)),
    "index": tuple(range(12, 16)),
    "middle": tuple(range(16, 20)),
    "ring": tuple(range(20, 24)),
    "pinky": tuple(range(24, 29)),
}
FINGERTIP_INDEX_BY_FINGER = {finger: idx for idx, finger in enumerate(FINGER_ORDER)}
ISAAC_TO_MUJOCO_JOINT_NAMES = {
    "iiwa14_joint_1": "joint1",
    "iiwa14_joint_2": "joint2",
    "iiwa14_joint_3": "joint3",
    "iiwa14_joint_4": "joint4",
    "iiwa14_joint_5": "joint5",
    "iiwa14_joint_6": "joint6",
    "iiwa14_joint_7": "joint7",
    "left_1_thumb_CMC_FE": "palmleft_thumb_CMC_FE",
    "left_thumb_CMC_AA": "palmleft_thumb_CMC_AA",
    "left_thumb_MCP_FE": "palmleft_thumb_MCP_FE",
    "left_thumb_MCP_AA": "palmleft_thumb_MCP_AA",
    "left_thumb_IP": "palmleft_thumb_IP",
    "left_2_index_MCP_FE": "palmleft_index_MCP_FE",
    "left_index_MCP_AA": "palmleft_index_MCP_AA",
    "left_index_PIP": "palmleft_index_PIP",
    "left_index_DIP": "palmleft_index_DIP",
    "left_3_middle_MCP_FE": "palmleft_middle_MCP_FE",
    "left_middle_MCP_AA": "palmleft_middle_MCP_AA",
    "left_middle_PIP": "palmleft_middle_PIP",
    "left_middle_DIP": "palmleft_middle_DIP",
    "left_4_ring_MCP_FE": "palmleft_ring_MCP_FE",
    "left_ring_MCP_AA": "palmleft_ring_MCP_AA",
    "left_ring_PIP": "palmleft_ring_PIP",
    "left_ring_DIP": "palmleft_ring_DIP",
    "left_5_pinky_CMC": "palmleft_pinky_CMC",
    "left_pinky_MCP_FE": "palmleft_pinky_MCP_FE",
    "left_pinky_MCP_AA": "palmleft_pinky_MCP_AA",
    "left_pinky_PIP": "palmleft_pinky_PIP",
    "left_pinky_DIP": "palmleft_pinky_DIP",
}
TRIPOD_RESET_JOINT_POS_OVERRIDES = {
    "iiwa14_joint_1": 2.617994,
    "iiwa14_joint_2": -0.785398,
    "iiwa14_joint_3": -1.527163,
    "iiwa14_joint_4": 1.396263,
    "iiwa14_joint_5": -0.872665,
    "iiwa14_joint_6": 0.174533,
    "iiwa14_joint_7": 0.022829,
    "left_1_thumb_CMC_FE": 1.9199,
    "left_thumb_CMC_AA": -0.3491,
    "left_thumb_MCP_FE": 0.3473,
    "left_thumb_MCP_AA": 0.3107,
    "left_thumb_IP": 0.2513,
    "left_2_index_MCP_FE": 0.0,
    "left_index_MCP_AA": 0.0,
    "left_index_PIP": 0.0,
    "left_index_DIP": 0.3229,
    "left_3_middle_MCP_FE": 0.0,
    "left_middle_MCP_AA": 0.0,
    "left_middle_PIP": 0.0,
    "left_middle_DIP": 0.3229,
    "left_4_ring_MCP_FE": 1.4,
    "left_ring_PIP": 1.5,
    "left_ring_DIP": 1.2,
    "left_5_pinky_CMC": 0.26,
    "left_pinky_MCP_FE": 1.4,
    "left_pinky_PIP": 1.5,
    "left_pinky_DIP": 1.2,
}
PINCH_RESET_JOINT_POS_OVERRIDES = {
    **TRIPOD_RESET_JOINT_POS_OVERRIDES,
    "left_3_middle_MCP_FE": 1.4,
    "left_middle_PIP": 1.5,
    "left_middle_DIP": 1.2,
}
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
GRASP_BOUNDING_BOX_OFFSETS = np.array(
    [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
    ],
    dtype=np.float32,
)
GRASP_BOUNDING_BOX_EDGES = (
    (0, 1),
    (0, 2),
    (0, 4),
    (1, 3),
    (1, 5),
    (2, 3),
    (2, 6),
    (3, 7),
    (4, 5),
    (4, 6),
    (5, 7),
    (6, 7),
)
KEYPOINT_MARKER_RADIUS = 0.012
GRASP_BOUNDING_BOX_LINE_RADIUS = 0.003
OBJECT_KEYPOINT_MARKER_RGBA = np.array([1.0, 0.0, 0.75, 1.0], dtype=np.float32)
GOAL_KEYPOINT_MARKER_RGBA = np.array([0.0, 1.0, 0.2, 1.0], dtype=np.float32)
OBJECT_GRASP_BOUNDING_BOX_RGBA = np.array([0.1, 0.45, 1.0, 1.0], dtype=np.float32)
GOAL_GRASP_BOUNDING_BOX_RGBA = np.array([1.0, 0.15, 0.85, 1.0], dtype=np.float32)
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
# Gentle replay default: table top is z=0.53 and the object starts at z=0.58.
# Keep randomized goals local and just above the tabletop unless explicitly overridden.
DEFAULT_TARGET_VOLUME_MINS = (-0.12, -0.05, 0.62)
DEFAULT_TARGET_VOLUME_MAXS = (0.12, 0.12, 0.72)
DROP_RESET_ARMING_LIFT = 0.04
TABLE_FALL_RESET_MARGIN = 0.03


def _quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return q[[1, 2, 3, 0]]


def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_w = q[:, 3]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0)[:, None]
    b = np.cross(q_vec, v, axis=-1) * q_w[:, None] * 2.0
    c = q_vec * (np.sum(q_vec * v, axis=-1)[:, None]) * 2.0
    return a + b + c


def _sample_random_quat_wxyz(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = rng.random(3)
    sqrt_u1 = np.sqrt(u1)
    sqrt_one_minus_u1 = np.sqrt(1.0 - u1)
    return np.array(
        [
            sqrt_one_minus_u1 * np.sin(2.0 * np.pi * u2),
            sqrt_one_minus_u1 * np.cos(2.0 * np.pi * u2),
            sqrt_u1 * np.sin(2.0 * np.pi * u3),
            sqrt_u1 * np.cos(2.0 * np.pi * u3),
        ],
        dtype=np.float64,
    )[[3, 0, 1, 2]]


def _unscale(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return (2.0 * x - upper - lower) / (upper - lower)


def _scale(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def _normalize_hand_mode(hand_mode: str) -> str:
    normalized = hand_mode.lower().replace("_", "-")
    if normalized not in HAND_MODE_ACTIVE_FINGERS:
        valid = ", ".join(HAND_MODE_ACTIVE_FINGERS)
        raise ValueError(f"Unknown hand mode '{hand_mode}'. Valid modes: {valid}.")
    return normalized


def _normalize_active_fingers(active_fingers: tuple[str, ...]) -> tuple[str, ...]:
    active = set(active_fingers)
    return tuple(finger for finger in FINGER_ORDER if finger in active)


def _active_joint_indices_for_fingers(active_fingers: tuple[str, ...]) -> np.ndarray:
    active = set(active_fingers)
    joint_indices = list(range(7))
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        if finger in active:
            joint_indices.extend(FINGER_JOINT_INDICES[finger])
    return np.asarray(joint_indices, dtype=np.int64)


def _active_fingertip_indices_for_fingers(active_fingers: tuple[str, ...]) -> np.ndarray:
    return np.asarray([FINGERTIP_INDEX_BY_FINGER[finger] for finger in _normalize_active_fingers(active_fingers)])


def _policy_obs_dim(num_actions: int, num_fingertips: int) -> int:
    return 3 * num_actions + 3 * num_fingertips + 38


def hand_mode_policy_dims(hand_mode: str) -> tuple[int, int]:
    active_fingers = HAND_MODE_ACTIVE_FINGERS[_normalize_hand_mode(hand_mode)]
    num_actions = int(_active_joint_indices_for_fingers(active_fingers).shape[0])
    num_fingertips = int(_active_fingertip_indices_for_fingers(active_fingers).shape[0])
    return _policy_obs_dim(num_actions, num_fingertips), num_actions


def _mode_reset_joint_pos(hand_mode: str) -> np.ndarray:
    q = INIT_JOINT_POS.copy()
    if hand_mode == "full":
        return q
    overrides = PINCH_RESET_JOINT_POS_OVERRIDES if hand_mode == "pinch" else TRIPOD_RESET_JOINT_POS_OVERRIDES
    for isaac_name, joint_pos in overrides.items():
        mujoco_name = ISAAC_TO_MUJOCO_JOINT_NAMES[isaac_name]
        q[JOINT_NAMES.index(mujoco_name)] = float(joint_pos)
    return q


def reset_joint_pos_from_isaac_overrides(hand_mode: str, overrides: dict[str, float] | None) -> np.ndarray:
    q = _mode_reset_joint_pos(hand_mode)
    if not overrides:
        return q
    for isaac_name, joint_pos in overrides.items():
        mujoco_name = ISAAC_TO_MUJOCO_JOINT_NAMES.get(isaac_name)
        if mujoco_name is None:
            continue
        q[JOINT_NAMES.index(mujoco_name)] = float(joint_pos)
    return q


def _compute_keypoints(pos: np.ndarray, quat_xyzw: np.ndarray, scales: np.ndarray) -> np.ndarray:
    offsets = KEYPOINT_OFFSETS[None] * 0.04 * 1.5 * 0.5 * scales[:, None]
    keypoints = np.zeros((pos.shape[0], 4, 3), dtype=np.float32)
    for i in range(4):
        keypoints[:, i] = pos + _quat_rotate_xyzw(quat_xyzw, offsets[:, i])
    return keypoints


def _compute_grasp_bounding_box_corners(
    pos: np.ndarray, quat_xyzw: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    offsets = GRASP_BOUNDING_BOX_OFFSETS[None] * 0.04 * 1.5 * 0.5 * scales[:, None]
    corners = np.zeros((pos.shape[0], 8, 3), dtype=np.float32)
    for i in range(8):
        corners[:, i] = pos + _quat_rotate_xyzw(quat_xyzw, offsets[:, i])
    return corners


def _compute_object_and_goal_keypoints(
    sim: MujocoSim, object_scales: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    sim_state = sim.get_sim_state()
    object_quat_xyzw = sim_state["object_quat_wxyz"][[1, 2, 3, 0]]
    goal_quat_xyzw = sim_state["goal_object_quat_wxyz"][[1, 2, 3, 0]]
    object_keypoints = _compute_keypoints(
        sim_state["object_pos"][None], object_quat_xyzw[None], object_scales[None]
    )[0]
    goal_keypoints = _compute_keypoints(
        sim_state["goal_object_pos"][None], goal_quat_xyzw[None], object_scales[None]
    )[0]
    return object_keypoints, goal_keypoints


def _compute_object_and_goal_grasp_bounding_boxes(
    sim: MujocoSim, object_scales: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    sim_state = sim.get_sim_state()
    object_quat_xyzw = sim_state["object_quat_wxyz"][[1, 2, 3, 0]]
    goal_quat_xyzw = sim_state["goal_object_quat_wxyz"][[1, 2, 3, 0]]
    object_corners = _compute_grasp_bounding_box_corners(
        sim_state["object_pos"][None], object_quat_xyzw[None], object_scales[None]
    )[0]
    goal_corners = _compute_grasp_bounding_box_corners(
        sim_state["goal_object_pos"][None], goal_quat_xyzw[None], object_scales[None]
    )[0]
    return object_corners, goal_corners


def _draw_keypoint_markers(sim: MujocoSim, object_scales: np.ndarray) -> None:
    _draw_debug_markers(
        sim,
        object_scales,
        visualize_keypoints=True,
        visualize_grasp_bounding_box=False,
    )


def _draw_debug_markers(
    sim: MujocoSim,
    object_scales: np.ndarray,
    visualize_keypoints: bool,
    visualize_grasp_bounding_box: bool,
) -> None:
    if sim.viewer is None:
        return

    with sim.viewer.lock():
        user_scn = sim.viewer.user_scn
        max_geoms = len(user_scn.geoms)
        user_scn.ngeom = 0

        if visualize_keypoints:
            object_keypoints, goal_keypoints = _compute_object_and_goal_keypoints(sim, object_scales)
            marker_positions = (object_keypoints, goal_keypoints)
            marker_colors = (OBJECT_KEYPOINT_MARKER_RGBA, GOAL_KEYPOINT_MARKER_RGBA)
            for keypoints, rgba in zip(marker_positions, marker_colors):
                for pos in keypoints:
                    if user_scn.ngeom >= max_geoms:
                        return
                    geom = user_scn.geoms[user_scn.ngeom]
                    sim.mujoco.mjv_initGeom(
                        geom,
                        sim.mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.full(3, KEYPOINT_MARKER_RADIUS, dtype=np.float64),
                        pos.astype(np.float64),
                        np.eye(3, dtype=np.float64).reshape(-1),
                        rgba.astype(np.float32),
                    )
                    user_scn.ngeom += 1

        if visualize_grasp_bounding_box:
            object_corners, goal_corners = _compute_object_and_goal_grasp_bounding_boxes(sim, object_scales)
            for corners, rgba in (
                (object_corners, OBJECT_GRASP_BOUNDING_BOX_RGBA),
                (goal_corners, GOAL_GRASP_BOUNDING_BOX_RGBA),
            ):
                for pos in corners:
                    if user_scn.ngeom >= max_geoms:
                        return
                    geom = user_scn.geoms[user_scn.ngeom]
                    sim.mujoco.mjv_initGeom(
                        geom,
                        sim.mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.full(3, KEYPOINT_MARKER_RADIUS * 0.7, dtype=np.float64),
                        pos.astype(np.float64),
                        np.eye(3, dtype=np.float64).reshape(-1),
                        rgba.astype(np.float32),
                    )
                    user_scn.ngeom += 1
                for start_idx, end_idx in GRASP_BOUNDING_BOX_EDGES:
                    if user_scn.ngeom >= max_geoms:
                        return
                    geom = user_scn.geoms[user_scn.ngeom]
                    start = corners[start_idx].astype(np.float64)
                    end = corners[end_idx].astype(np.float64)
                    sim.mujoco.mjv_initGeom(
                        geom,
                        sim.mujoco.mjtGeom.mjGEOM_CAPSULE,
                        np.zeros(3, dtype=np.float64),
                        np.zeros(3, dtype=np.float64),
                        np.eye(3, dtype=np.float64).reshape(-1),
                        rgba.astype(np.float32),
                    )
                    sim.mujoco.mjv_connector(
                        geom,
                        sim.mujoco.mjtGeom.mjGEOM_CAPSULE,
                        GRASP_BOUNDING_BOX_LINE_RADIUS,
                        start,
                        end,
                    )
                    user_scn.ngeom += 1


def _normalize_cli_flag_aliases() -> None:
    aliases = {
        "--visualize_grasp_bounding_box": "--visualize-grasp-bounding-box",
        "--visualize_grasp_bouding_box": "--visualize-grasp-bounding-box",
        "--visualize-grasp-bouding-box": "--visualize-grasp-bounding-box",
    }
    sys.argv = [aliases.get(arg, arg) for arg in sys.argv]


def _compute_joint_pos_targets(
    actions: np.ndarray,
    prev_targets: np.ndarray,
    hand_moving_average: float,
    arm_moving_average: float,
    arm_dof_speed_scale: float,
    dt: float,
    active_joint_indices: np.ndarray | None = None,
) -> np.ndarray:
    cur_targets = prev_targets.copy()
    active_joint_indices = np.arange(N_ACT, dtype=np.int64) if active_joint_indices is None else active_joint_indices
    if actions.shape[1] != active_joint_indices.shape[0]:
        raise ValueError(
            f"Expected action width {active_joint_indices.shape[0]} for active joints, got {actions.shape[1]}."
        )

    arm_mask = active_joint_indices < 7
    arm_indices = active_joint_indices[arm_mask]
    hand_indices = active_joint_indices[~arm_mask]
    n_arm = arm_indices.shape[0]

    if hand_indices.size:
        hand_actions = actions[:, n_arm:]
        cur_targets[:, hand_indices] = _scale(hand_actions, Q_LOWER_LIMITS[hand_indices], Q_UPPER_LIMITS[hand_indices])
        cur_targets[:, hand_indices] = (
            hand_moving_average * cur_targets[:, hand_indices]
            + (1.0 - hand_moving_average) * prev_targets[:, hand_indices]
        )
        cur_targets[:, hand_indices] = np.clip(
            cur_targets[:, hand_indices], Q_LOWER_LIMITS[hand_indices], Q_UPPER_LIMITS[hand_indices]
        )

    if arm_indices.size:
        cur_targets[:, arm_indices] = prev_targets[:, arm_indices] + arm_dof_speed_scale * dt * actions[:, :n_arm]
        cur_targets[:, arm_indices] = np.clip(
            cur_targets[:, arm_indices], Q_LOWER_LIMITS[arm_indices], Q_UPPER_LIMITS[arm_indices]
        )
        cur_targets[:, arm_indices] = (
            arm_moving_average * cur_targets[:, arm_indices]
            + (1.0 - arm_moving_average) * prev_targets[:, arm_indices]
        )
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
        visualize_keypoints: bool,
        visualize_grasp_bounding_box: bool,
        randomize_goal: bool,
        target_volume_mins: tuple[float, float, float],
        target_volume_maxs: tuple[float, float, float],
        randomize_goal_rotation: bool,
        reset_when_dropped: bool,
        drop_reset_height: float | None,
        seed: int | None,
        hand_mode: HandMode = "full",
        reset_joint_pos: np.ndarray | None = None,
    ):
        self.sim = sim
        self.object_scales = object_scales
        self.hand_mode = _normalize_hand_mode(hand_mode)
        active_fingers = HAND_MODE_ACTIVE_FINGERS[self.hand_mode]
        self.active_joint_indices = _active_joint_indices_for_fingers(active_fingers)
        self.active_fingertip_indices = _active_fingertip_indices_for_fingers(active_fingers)
        self.num_actions = int(self.active_joint_indices.shape[0])
        self.num_observations = _policy_obs_dim(self.num_actions, int(self.active_fingertip_indices.shape[0]))
        self.reset_joint_pos = (
            _mode_reset_joint_pos(self.hand_mode) if reset_joint_pos is None else np.asarray(reset_joint_pos, dtype=np.float64)
        )
        self.hand_moving_average = hand_moving_average
        self.arm_moving_average = arm_moving_average
        self.hand_dof_speed_scale = hand_dof_speed_scale
        self.control_dt = control_dt
        self.device = device
        self.obs_list = obs_list
        self.visualize_keypoints = visualize_keypoints
        self.visualize_grasp_bounding_box = visualize_grasp_bounding_box
        self.randomize_goal = randomize_goal
        self.target_volume_mins = np.array(target_volume_mins, dtype=np.float64)
        self.target_volume_maxs = np.array(target_volume_maxs, dtype=np.float64)
        self.randomize_goal_rotation = randomize_goal_rotation
        self.reset_when_dropped = reset_when_dropped
        self.rng = np.random.default_rng(seed)
        self.object_init_z = float(sim.config.object_start_pos[2])
        self.drop_reset_height = self.object_init_z if drop_reset_height is None else float(drop_reset_height)
        self.max_object_z_since_reset = self.object_init_z
        self.lifted_object = False
        print(
            f"[MuJoCo] hand_mode={self.hand_mode} obs={self.num_observations} actions={self.num_actions} "
            f"active_fingertips={self.active_fingertip_indices.shape[0]}",
            flush=True,
        )

    @property
    def sim_steps_per_control_step(self) -> int:
        return max(1, int(round(self.control_dt / self.sim.config.sim_dt)))

    def reset(self) -> None:
        goal_pos = self.sim.config.goal_object_start_pos
        goal_quat = self.sim.config.goal_object_start_quat_wxyz
        if self.randomize_goal:
            goal_pos = self.rng.uniform(self.target_volume_mins, self.target_volume_maxs)
            if self.randomize_goal_rotation:
                goal_quat = _sample_random_quat_wxyz(self.rng)
            print(
                "[MuJoCo] New randomized goal: "
                f"pos={np.round(goal_pos, 4).tolist()} quat_wxyz={np.round(goal_quat, 4).tolist()}",
                flush=True,
            )
        self.sim.reset_scene(goal_object_pos=goal_pos, goal_object_quat_wxyz=goal_quat)
        self.sim.set_robot_joint_pos_targets(self.reset_joint_pos)
        self.sim.set_robot_joint_positions(self.reset_joint_pos)
        self.object_init_z = float(self.sim.config.object_start_pos[2])
        self.drop_reset_height = self.object_init_z if self.drop_reset_height is None else self.drop_reset_height
        self.max_object_z_since_reset = self.object_init_z
        self.lifted_object = False

    def should_reset_after_drop(self) -> bool:
        if not self.reset_when_dropped:
            return False
        object_z = float(self.sim.get_sim_state()["object_pos"][2])
        self.max_object_z_since_reset = max(self.max_object_z_since_reset, object_z)
        armed_by_lift = self.max_object_z_since_reset > self.object_init_z + DROP_RESET_ARMING_LIFT
        dropped_after_lift = armed_by_lift and object_z < self.drop_reset_height
        fell_below_table = object_z < TABLE_TOP_Z - TABLE_FALL_RESET_MARGIN
        self.lifted_object = self.lifted_object or armed_by_lift
        return dropped_after_lift or fell_below_table

    def palm_pos(self) -> np.ndarray:
        wrist_pos, wrist_quat_wxyz = self.sim.get_body_pose("link7")
        palm_quat_xyzw = _quat_wxyz_to_xyzw(wrist_quat_wxyz)[None]
        return wrist_pos + _quat_rotate_xyzw(palm_quat_xyzw, PALM_OFFSET[None])[0]

    def should_reset_after_hand_drift(
        self,
        max_palm_object_distance: float | None,
        min_palm_z: float | None,
    ) -> tuple[bool, str]:
        palm_pos = self.palm_pos()
        object_pos = self.sim.get_sim_state()["object_pos"]
        if max_palm_object_distance is not None:
            distance = float(np.linalg.norm(palm_pos - object_pos))
            if distance > max_palm_object_distance:
                return True, f"palm-object distance {distance:.3f} > {max_palm_object_distance:.3f}"
        if min_palm_z is not None and float(palm_pos[2]) < min_palm_z:
            return True, f"palm z {float(palm_pos[2]):.3f} < {min_palm_z:.3f}"
        return False, ""

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
        fingertip_pos = fingertip_pos[self.active_fingertip_indices]
        object_keypoints = _compute_keypoints(
            object_pose_w[None, :3], object_pose_w[None, 3:7], self.object_scales[None]
        )
        goal_keypoints = _compute_keypoints(
            goal_object_pose_w[None, :3], goal_object_pose_w[None, 3:7], self.object_scales[None]
        )
        active_indices = self.active_joint_indices
        obs_dict = {
            "joint_pos": _unscale(q[:, active_indices], Q_LOWER_LIMITS[active_indices], Q_UPPER_LIMITS[active_indices]),
            "joint_vel": qd[:, active_indices],
            "prev_action_targets": self.sim.robot_joint_pos_targets[active_indices][None],
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
        if obs_tensor.shape != (1, self.num_observations):
            raise RuntimeError(f"Expected observation shape {(1, self.num_observations)}, got {obs_tensor.shape}.")
        return obs_tensor

    def step(self, action: torch.Tensor) -> None:
        joint_pos_targets = _compute_joint_pos_targets(
            actions=action.detach().cpu().numpy(),
            prev_targets=self.sim.robot_joint_pos_targets[None],
            hand_moving_average=self.hand_moving_average,
            arm_moving_average=self.arm_moving_average,
            arm_dof_speed_scale=self.hand_dof_speed_scale,
            dt=self.control_dt,
            active_joint_indices=self.active_joint_indices,
        )
        self.sim.set_robot_joint_pos_targets(joint_pos_targets[0])
        for _ in range(self.sim_steps_per_control_step):
            self.sim.sim_step()
            if self.sim.viewer is not None:
                if self.visualize_keypoints or self.visualize_grasp_bounding_box:
                    _draw_debug_markers(
                        self.sim,
                        self.object_scales,
                        self.visualize_keypoints,
                        self.visualize_grasp_bounding_box,
                    )
                self.sim.viewer.sync()


@dataclass
class MujocoEnvNoRosArgs:
    config_path: Path = DEFAULT_POLICY_DIR / "config.yaml"
    checkpoint_path: Path = DEFAULT_POLICY_DIR / "model.pth"
    object_name: str = "claw_hammer"
    hand_mode: HandMode = "full"
    enable_viewer: bool = True
    max_steps: int | None = None
    sim_hz: float = 600.0
    control_hz: float = 60.0
    show_robot_collision_overlay: bool = True
    use_proxy_object_collision: bool = True
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


def _wait_for_enter_to_start(
    sim: MujocoSim,
    visualize_keypoints: bool,
    visualize_grasp_bounding_box: bool,
    object_scales: np.ndarray,
) -> bool:
    print("Adjust the MuJoCo viewer, then press Enter to start the rollout (Ctrl-D to quit).", flush=True)
    if not sys.stdin.isatty():
        return sys.stdin.readline() != ""

    while sim._continue_running():
        readable, _, _ = select.select([sys.stdin], [], [], 0.05)
        if readable:
            return sys.stdin.readline() != ""
        if sim.viewer is not None:
            if visualize_keypoints or visualize_grasp_bounding_box:
                _draw_debug_markers(
                    sim,
                    object_scales,
                    visualize_keypoints,
                    visualize_grasp_bounding_box,
                )
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
    _normalize_cli_flag_aliases()
    args = tyro.cli(MujocoEnvNoRosArgs)
    if not args.config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config_path}")
    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    if not MUJOCO_REPLAY_SCENE_PATH.exists():
        raise FileNotFoundError(f"MuJoCo replay scene not found: {MUJOCO_REPLAY_SCENE_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_observations, num_actions = hand_mode_policy_dims(args.hand_mode)
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
            use_proxy_object_collision=args.use_proxy_object_collision,
        )
    )
    policy = RlPlayer(
        num_observations=num_observations,
        num_actions=num_actions,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=device,
    )
    obs_list = _policy_obs_list(policy)
    object_scales = _object_scales(args.object_name)
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
        hand_mode=args.hand_mode,
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
                print("[MuJoCo] Object dropped after lift; resetting replay scene.", flush=True)
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
