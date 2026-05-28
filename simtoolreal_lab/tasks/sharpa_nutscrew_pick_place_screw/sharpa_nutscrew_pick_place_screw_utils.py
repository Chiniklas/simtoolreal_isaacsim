"""Observation/action helpers for the SHARPA nut-screw pick-place-screw task."""

from __future__ import annotations

import torch

from simtoolreal_lab.assets.kuka_sharpa import KUKA_SHARPA_JOINT_NAMES

OBS_NAME_TO_NAMES = {
    "joint_pos": [f"{name}_q" for name in KUKA_SHARPA_JOINT_NAMES],
    "joint_vel": [f"{name}_qd" for name in KUKA_SHARPA_JOINT_NAMES],
    "prev_action_targets": [f"{name}_prev_action_target" for name in KUKA_SHARPA_JOINT_NAMES],
    "palm_pos": [f"palm_center_pos_{axis}" for axis in "xyz"],
    "palm_rot": [f"palm_rot_{axis}" for axis in "xyzw"],
    "object_rot": [f"object_rot_{axis}" for axis in "xyzw"],
    "fingertip_pos_rel_palm": [
        f"fingertip_rel_pos_{finger}_{axis}"
        for finger in ["index", "middle", "ring", "thumb", "pinky"]
        for axis in "xyz"
    ],
    "keypoints_rel_palm": [f"keypoints_rel_palm_{idx}_{axis}" for idx in range(4) for axis in "xyz"],
    "keypoints_rel_goal": [f"keypoints_rel_goal_{idx}_{axis}" for idx in range(4) for axis in "xyz"],
    "object_scales": [f"object_scales_{axis}" for axis in "xyz"],
}
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
N_OBS = 140
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
