"""Observation/action helpers for the SHARPA nut-screw pick-place-screw task."""

from __future__ import annotations

import torch

from simtoolreal_lab.assets.kuka_sharpa import KUKA_SHARPA_JOINT_NAMES

FINGER_NAMES = ("index", "middle", "ring", "thumb", "pinky")
DEFAULT_ACTIVE_FINGERS = FINGER_NAMES


def normalize_active_fingers(active_fingers: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    active_set = set(active_fingers)
    unknown = sorted(active_set - set(FINGER_NAMES))
    if unknown:
        raise ValueError(f"Unknown active fingers {unknown}. Valid fingers: {FINGER_NAMES}")
    if not active_set:
        raise ValueError("At least one active finger is required.")
    return tuple(finger for finger in FINGER_NAMES if finger in active_set)


def masked_joint_names(active_fingers: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    active = set(normalize_active_fingers(active_fingers))
    return tuple(
        joint_name
        for joint_name in KUKA_SHARPA_JOINT_NAMES
        if joint_name.startswith("iiwa14_joint_") or any(f"_{finger}_" in joint_name for finger in active)
    )


DEFAULT_ACTUATED_JOINT_NAMES = masked_joint_names(DEFAULT_ACTIVE_FINGERS)


def build_observation_name_to_names(
    joint_names: tuple[str, ...] | list[str] = DEFAULT_ACTUATED_JOINT_NAMES,
    active_fingers: tuple[str, ...] | list[str] = DEFAULT_ACTIVE_FINGERS,
) -> dict[str, list[str]]:
    active_fingers = normalize_active_fingers(active_fingers)
    return {
        "joint_pos": [f"{name}_q" for name in joint_names],
        "joint_vel": [f"{name}_qd" for name in joint_names],
        "prev_action_targets": [f"{name}_prev_action_target" for name in joint_names],
        "palm_pos": [f"palm_center_pos_{axis}" for axis in "xyz"],
        "palm_rot": [f"palm_rot_{axis}" for axis in "xyzw"],
        "object_rot": [f"object_rot_{axis}" for axis in "xyzw"],
        "fingertip_pos_rel_palm": [
            f"fingertip_rel_pos_{finger}_{axis}" for finger in active_fingers for axis in "xyz"
        ],
        "keypoints_rel_palm": [f"keypoints_rel_palm_{idx}_{axis}" for idx in range(4) for axis in "xyz"],
        "keypoints_rel_goal": [f"keypoints_rel_goal_{idx}_{axis}" for idx in range(4) for axis in "xyz"],
        "object_scales": [f"object_scales_{axis}" for axis in "xyz"],
    }


OBS_NAME_TO_NAMES = build_observation_name_to_names()
OBS_LIST = [
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
OBS_NAMES = sum((OBS_NAME_TO_NAMES[name] for name in OBS_LIST), [])
N_OBS = len(OBS_NAMES)
assert len(OBS_NAMES) == N_OBS


def scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def unscale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return (2.0 * x - upper - lower) / (upper - lower)


def compute_joint_pos_targets(
    actions: torch.Tensor,
    prev_targets: torch.Tensor,
    lower_limits: torch.Tensor,
    upper_limits: torch.Tensor,
    hand_moving_average: float,
    arm_moving_average: float,
    arm_dof_speed_scale: float,
    dt: float,
) -> torch.Tensor:
    """Match the reference SHARPA action transform in torch.

    Arm actions are incremental velocity-like commands; hand actions are
    absolute normalized joint targets with a moving average.
    """

    cur_targets = prev_targets.clone()
    cur_targets[:, 7:] = scale(actions[:, 7:], lower_limits[:, 7:], upper_limits[:, 7:])
    cur_targets[:, 7:] = hand_moving_average * cur_targets[:, 7:] + (1.0 - hand_moving_average) * prev_targets[:, 7:]
    cur_targets[:, 7:] = torch.clamp(cur_targets[:, 7:], lower_limits[:, 7:], upper_limits[:, 7:])

    cur_targets[:, :7] = prev_targets[:, :7] + arm_dof_speed_scale * dt * actions[:, :7]
    cur_targets[:, :7] = torch.clamp(cur_targets[:, :7], lower_limits[:, :7], upper_limits[:, :7])
    cur_targets[:, :7] = arm_moving_average * cur_targets[:, :7] + (1.0 - arm_moving_average) * prev_targets[:, :7]
    return cur_targets
