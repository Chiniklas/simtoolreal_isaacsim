"""Isaac Lab DirectRLEnv port of the reference SimToolReal KUKA-SHARPA task."""

from __future__ import annotations

import math
from collections.abc import Sequence
import re

import torch
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul, sample_uniform, saturate

from .simtoolreal_sharpa_env_cfg import DEXTOOLBENCH_OBJECT_SCALES, SimToolRealSharpaEnvCfg
from .simtoolreal_sharpa_utils import compute_joint_pos_targets, unscale


def quat_wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., 1:], q[..., 0:1]), dim=-1)


class SimToolRealSharpaEnv(DirectRLEnv):
    """First Isaac Lab implementation pass for the SimToolReal KUKA-SHARPA task.

    This keeps the reference action count and 140-value observation layout while
    using Isaac Lab-native articulation/rigid-object state access.
    """

    cfg: SimToolRealSharpaEnvCfg

    def __init__(self, cfg: SimToolRealSharpaEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._apply_object_mass()

        self.num_robot_dofs = self.robot.num_joints
        self.actuated_dof_indices = [self.robot.joint_names.index(name) for name in cfg.actuated_joint_names]
        self.num_actions = len(self.actuated_dof_indices)
        self.cfg.num_actions = self.num_actions

        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.prev_action_targets = torch.zeros_like(self.actions)
        self.dof_pos_targets = torch.zeros((self.num_envs, self.num_robot_dofs), device=self.device)

        self.palm_body_idx = self._resolve_body_index(self.cfg.palm_body_name)
        self.fingertip_body_indices = [self._resolve_body_index(name) for name in self.cfg.fingertip_body_names]
        self.palm_offset = torch.tensor(self.cfg.palm_offset, device=self.device)
        self.fingertip_offsets = torch.tensor(self.cfg.fingertip_offsets, device=self.device)

        joint_pos_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        self.robot_dof_lower_limits = joint_pos_limits[..., 0][:, self.actuated_dof_indices]
        self.robot_dof_upper_limits = joint_pos_limits[..., 1][:, self.actuated_dof_indices]

        self.robot_start_joint_pos = self.robot.data.default_joint_pos.clone()
        self.robot_start_joint_vel = torch.zeros_like(self.robot_start_joint_pos)
        self.dof_pos_targets[:] = self.robot_start_joint_pos
        self.prev_action_targets[:] = self.robot_start_joint_pos[:, self.actuated_dof_indices]

        self.object_goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_goal_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self.object_goal_rot[:, 0] = 1.0
        if getattr(self.cfg, "object_name", "") == "multi_dextoolbench":
            per_object_scales = torch.tensor(
                [DEXTOOLBENCH_OBJECT_SCALES[name] for name in self.cfg.multi_object_names], device=self.device
            )
            object_ids = torch.arange(self.num_envs, device=self.device) % per_object_scales.shape[0]
            self.object_scales = per_object_scales[object_ids]
        else:
            self.object_scales = torch.tensor(self.cfg.object_scales, device=self.device).repeat(self.num_envs, 1)
        self.object_init_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_last_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.successes = torch.zeros(self.num_envs, device=self.device)
        self.consecutive_successes = torch.zeros(self.num_envs, device=self.device)
        self.near_goal_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.lifted_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        num_fingertips = len(self.fingertip_body_indices)
        self.finger_rew_coeffs = torch.ones((self.num_envs, num_fingertips), device=self.device)
        self.closest_fingertip_dist = -torch.ones((self.num_envs, num_fingertips), device=self.device)
        self.furthest_hand_dist = -torch.ones(self.num_envs, device=self.device)
        self.closest_keypoint_max_dist = -torch.ones(self.num_envs, device=self.device)

        self.keypoint_offsets = self._make_keypoint_offsets()
        self.keypoint_debug_draw = self._make_keypoint_debug_draw() if self.cfg.debug_keypoints else None
        self._all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._compute_intermediate_values()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.goal_object = RigidObject(self.cfg.goal_object_cfg)
        self.table_contact_sensor = None
        if self.cfg.with_table_force_sensor:
            self.table_contact_sensor = ContactSensor(self.cfg.table_contact_sensor)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        if self.scene.cfg.replicate_physics:
            self.scene.clone_environments(copy_from_source=False)
        self._disable_goal_object_collisions()
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["goal_object"] = self.goal_object
        if self.table_contact_sensor is not None:
            self.scene.sensors["table_contact_sensor"] = self.table_contact_sensor
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _disable_goal_object_collisions(self) -> None:
        stage = sim_utils.get_current_stage()
        goal_paths = [str(prim.GetPath()) for prim in stage.Traverse() if prim.GetName() == "goal_object"]
        for goal_path in goal_paths:
            sim_utils.make_uninstanceable(goal_path, stage)

        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if not any(prim_path == goal_path or prim_path.startswith(f"{goal_path}/") for goal_path in goal_paths):
                continue
            collision_api = UsdPhysics.CollisionAPI(prim)
            if collision_api:
                collision_api.CreateCollisionEnabledAttr(False)

    def _apply_object_mass(self) -> None:
        self._apply_mass_to_rigid_object(self.object)
        self._apply_mass_to_rigid_object(self.goal_object)

    def _apply_mass_to_rigid_object(self, rigid_object: RigidObject) -> None:
        masses = rigid_object.root_physx_view.get_masses().clone()
        inertias = rigid_object.root_physx_view.get_inertias().clone()
        old_masses = masses.clone()
        per_body_mass = self.cfg.object_mass / rigid_object.num_bodies
        masses[:] = per_body_mass

        ratios = torch.where(old_masses > 0.0, masses / old_masses, torch.ones_like(masses))
        inertia_ratios = ratios
        while inertia_ratios.dim() < inertias.dim():
            inertia_ratios = inertia_ratios.unsqueeze(-1)
        inertias[:] = inertias * inertia_ratios

        env_ids = torch.arange(rigid_object.num_instances, device="cpu", dtype=torch.long)
        rigid_object.root_physx_view.set_masses(masses, env_ids)
        rigid_object.root_physx_view.set_inertias(inertias, env_ids)
        rigid_object.data.default_mass = masses.clone()
        rigid_object.data.default_inertia = inertias.clone()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions.copy_(self.actions)
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self._compute_action_targets()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(
            self.dof_pos_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def _resolve_body_index(self, body_name: str) -> int:
        matches, matched_names = self.robot.find_bodies(body_name)
        if matches:
            return matches[0]

        matches, matched_names = self.robot.find_bodies(f".*{re.escape(body_name)}")
        if len(matches) == 1:
            return matches[0]

        available_names = ", ".join(self.robot.body_names)
        if len(matches) > 1:
            raise ValueError(
                f"Body pattern '{body_name}' matched multiple bodies {matched_names}. "
                f"Available bodies: {available_names}"
            )
        raise ValueError(f"Body '{body_name}' not found. Available bodies: {available_names}")

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        obs = self._compute_reference_observations()
        obs = torch.clamp(obs, -self.cfg.clamp_abs_observations, self.cfg.clamp_abs_observations)
        observations = {"policy": obs}
        if self.cfg.asymmetric_obs:
            observations["critic"] = obs
        return observations

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        lifting_reward, lift_bonus_reward, lifted_object = self._lifting_reward()
        fingertip_delta_reward, hand_delta_penalty = self._distance_delta_rewards(lifted_object)
        keypoint_reward = self._keypoint_reward(lifted_object)

        keypoint_success_tolerance = self.cfg.success_tolerance * self.cfg.keypoint_scale
        near_goal = self.keypoints_max_dist <= keypoint_success_tolerance
        if self.cfg.force_consecutive_near_goal_steps:
            self.near_goal_steps = (self.near_goal_steps + near_goal.long()) * near_goal.long()
        else:
            self.near_goal_steps += near_goal.long()

        reached_goal = self.near_goal_steps >= self.cfg.success_steps
        self.successes += reached_goal.float()
        self.consecutive_successes.copy_(self.successes)

        object_lin_vel_penalty = -torch.sum(torch.square(self.object_vel[:, 0:3]), dim=-1)
        object_ang_vel_penalty = -torch.sum(torch.square(self.object_vel[:, 3:6]), dim=-1)
        object_lin_vel_penalty *= self.cfg.object_lin_vel_penalty_scale
        object_ang_vel_penalty *= self.cfg.object_ang_vel_penalty_scale

        arm_action_penalty, hand_action_penalty = self._action_penalties()

        reach_bonus = near_goal.float() * (self.cfg.reach_goal_bonus / self.cfg.success_steps)
        if self.cfg.force_consecutive_near_goal_steps:
            reach_bonus = reached_goal.float() * self.cfg.reach_goal_bonus

        reward = (
            fingertip_delta_reward
            + hand_delta_penalty
            + lifting_reward
            + lift_bonus_reward
            + keypoint_reward
            + reach_bonus
            + arm_action_penalty
            + hand_action_penalty
            + object_lin_vel_penalty
            + object_ang_vel_penalty
        )
        self.object_last_pos.copy_(self.object_pos)

        success_env_ids = reached_goal.nonzero(as_tuple=False).squeeze(-1)
        if success_env_ids.numel() > 0:
            self._reset_goals(success_env_ids, is_first_goal=False)
            self.near_goal_steps[success_env_ids] = 0
            self.closest_keypoint_max_dist[success_env_ids] = -1.0
            if self.cfg.max_consecutive_successes > 0:
                self.episode_length_buf[success_env_ids] = 0

        reward_terms = {
            "fingertip_delta_reward": fingertip_delta_reward.mean(),
            "hand_delta_penalty": hand_delta_penalty.mean(),
            "lifting_reward": lifting_reward.mean(),
            "lift_bonus_reward": lift_bonus_reward.mean(),
            "keypoint_reward": keypoint_reward.mean(),
            "reach_bonus": reach_bonus.mean(),
            "arm_action_penalty": arm_action_penalty.mean(),
            "hand_action_penalty": hand_action_penalty.mean(),
            "object_lin_vel_penalty": object_lin_vel_penalty.mean(),
            "object_ang_vel_penalty": object_ang_vel_penalty.mean(),
            "total_reward": reward.mean(),
        }
        self.extras["log"] = {
            "keypoints_max_dist": self.keypoints_max_dist.mean(),
            "object_lift": (0.05 + self.object_pos[:, 2] - self.object_init_pos[:, 2]).mean(),
            "success_rate": reached_goal.float().mean(),
            **reward_terms,
        }
        self.extras["reward_terms"] = reward_terms
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        object_fell = self.object_pos[:, 2] < self.cfg.object_z_low_reset_threshold
        hand_far_from_object = self.curr_fingertip_distances.max(dim=-1).values > self.cfg.hand_far_from_object_threshold
        if self.cfg.reset_when_dropped:
            dropped = (self.object_pos[:, 2] < self.object_init_pos[:, 2]) & self.lifted_object
        else:
            dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.table_contact_sensor is not None:
            table_force_too_high = self.table_contact_force_norm > self.cfg.table_force_threshold
        else:
            table_force_too_high = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        too_many_successes = self.consecutive_successes >= self.cfg.max_consecutive_successes
        terminated = object_fell | too_many_successes | hand_far_from_object | dropped | table_force_too_high
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._all_env_ids
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)

        num_ids = env_ids.shape[0]
        object_state = self.object.data.default_root_state[env_ids].clone()
        xy_noise = torch.stack(
            (
                sample_uniform(-self.cfg.reset_position_noise_x, self.cfg.reset_position_noise_x, (num_ids,), self.device),
                sample_uniform(-self.cfg.reset_position_noise_y, self.cfg.reset_position_noise_y, (num_ids,), self.device),
            ),
            dim=-1,
        )
        z_noise = sample_uniform(-self.cfg.reset_position_noise_z, self.cfg.reset_position_noise_z, (num_ids,), self.device)
        object_state[:, 0] = xy_noise[:, 0]
        object_state[:, 1] = xy_noise[:, 1]
        object_state[:, 2] = self.cfg.table_cfg.init_state.pos[2] + self.cfg.table_object_z_offset + z_noise
        object_state[:, 0:3] += self.scene.env_origins[env_ids]
        object_state[:, 3:7] = self._sample_object_quat(num_ids)
        if self.cfg.object_start_pose is not None:
            object_start_pose = torch.tensor(self.cfg.object_start_pose, device=self.device, dtype=object_state.dtype)
            object_state[:, 0:3] = object_start_pose[0:3] + self.scene.env_origins[env_ids]
            object_state[:, 3:7] = object_start_pose[[6, 3, 4, 5]]
        object_state[:, 7:13] = 0.0
        self.object.write_root_state_to_sim(object_state, env_ids)

        dof_pos = self.robot_start_joint_pos[env_ids].clone()
        dof_vel = self.robot_start_joint_vel[env_ids].clone()
        arm_noise = sample_uniform(
            -self.cfg.reset_dof_pos_noise_arm,
            self.cfg.reset_dof_pos_noise_arm,
            (num_ids, 7),
            self.device,
        )
        hand_noise = sample_uniform(
            -self.cfg.reset_dof_pos_noise_fingers,
            self.cfg.reset_dof_pos_noise_fingers,
            (num_ids, self.num_actions - 7),
            self.device,
        )
        dof_pos[:, self.actuated_dof_indices[:7]] += arm_noise
        dof_pos[:, self.actuated_dof_indices[7:]] += hand_noise
        dof_pos[:, self.actuated_dof_indices] = saturate(
            dof_pos[:, self.actuated_dof_indices],
            self.robot_dof_lower_limits[env_ids],
            self.robot_dof_upper_limits[env_ids],
        )
        dof_vel[:, self.actuated_dof_indices] = sample_uniform(
            -self.cfg.reset_dof_vel_noise,
            self.cfg.reset_dof_vel_noise,
            (num_ids, self.num_actions),
            self.device,
        )
        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(dof_pos[:, self.actuated_dof_indices], env_ids=env_ids, joint_ids=self.actuated_dof_indices)

        self.dof_pos_targets[env_ids] = dof_pos
        self.prev_action_targets[env_ids] = dof_pos[:, self.actuated_dof_indices]
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.near_goal_steps[env_ids] = 0
        self.lifted_object[env_ids] = False
        self.closest_fingertip_dist[env_ids] = -1.0
        self.furthest_hand_dist[env_ids] = -1.0
        self.closest_keypoint_max_dist[env_ids] = -1.0
        self.successes[env_ids] = 0.0
        self.consecutive_successes[env_ids] = 0.0

        self.object_init_pos[env_ids] = object_state[:, 0:3] - self.scene.env_origins[env_ids]
        self._reset_goals(env_ids, is_first_goal=True)
        self._compute_intermediate_values()
        self.object_init_pos[env_ids] = self.object_pos[env_ids]
        self.object_last_pos[env_ids] = self.object_pos[env_ids]

    def _compute_action_targets(self) -> None:
        if self.cfg.use_relative_control:
            speed = self.cfg.dof_speed_scale * self.step_dt
            targets = self.prev_action_targets + speed * self.actions
            targets = saturate(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        else:
            targets = compute_joint_pos_targets(
                actions=self.actions,
                prev_targets=self.prev_action_targets,
                lower_limits=self.robot_dof_lower_limits,
                upper_limits=self.robot_dof_upper_limits,
                hand_moving_average=self.cfg.hand_moving_average,
                arm_moving_average=self.cfg.arm_moving_average,
                arm_dof_speed_scale=self.cfg.dof_speed_scale,
                dt=self.step_dt,
            )
        self.prev_action_targets.copy_(targets)
        self.dof_pos_targets[:, self.actuated_dof_indices] = targets

    def _compute_intermediate_values(self) -> None:
        self.robot_dof_pos = self.robot.data.joint_pos[:, self.actuated_dof_indices]
        self.robot_dof_vel = self.robot.data.joint_vel[:, self.actuated_dof_indices]
        self.palm_rot = self.robot.data.body_quat_w[:, self.palm_body_idx]
        palm_offset_w = quat_apply(self.palm_rot, self.palm_offset.unsqueeze(0).expand(self.num_envs, -1))
        self.palm_pos = self.robot.data.body_pos_w[:, self.palm_body_idx] + palm_offset_w - self.scene.env_origins
        fingertip_rot = self.robot.data.body_quat_w[:, self.fingertip_body_indices]
        fingertip_offset_w = quat_apply(
            fingertip_rot.reshape(-1, 4),
            self.fingertip_offsets.unsqueeze(0).expand(self.num_envs, -1, -1).reshape(-1, 3),
        ).reshape(self.num_envs, -1, 3)
        self.fingertip_pos = (
            self.robot.data.body_pos_w[:, self.fingertip_body_indices]
            + fingertip_offset_w
            - self.scene.env_origins[:, None, :]
        )
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_vel = torch.cat((self.object.data.root_lin_vel_w, self.object.data.root_ang_vel_w), dim=-1)
        self.object_keypoints = self._compute_keypoints(self.object_pos, self.object_rot, self.object_scales)
        self.object_goal_pos = self.goal_object.data.root_pos_w - self.scene.env_origins
        self.object_goal_rot = self.goal_object.data.root_quat_w
        self.goal_keypoints = self._compute_keypoints(self.object_goal_pos, self.object_goal_rot, self.object_scales)
        self.fingertip_pos_rel_object = self.fingertip_pos - self.object_pos[:, None, :]
        self.curr_fingertip_distances = torch.norm(self.fingertip_pos_rel_object, dim=-1)
        self.closest_fingertip_dist = torch.where(
            self.closest_fingertip_dist < 0.0, self.curr_fingertip_distances, self.closest_fingertip_dist
        )
        self.furthest_hand_dist = torch.where(
            self.furthest_hand_dist < 0.0, self.curr_fingertip_distances[:, 0], self.furthest_hand_dist
        )
        self.keypoint_distances = torch.norm(self.object_keypoints - self.goal_keypoints, dim=-1)
        self.keypoints_max_dist = self.keypoint_distances.max(dim=-1).values
        self.closest_keypoint_max_dist = torch.where(
            self.closest_keypoint_max_dist < 0.0, self.keypoints_max_dist, self.closest_keypoint_max_dist
        )
        self._visualize_keypoints()
        if self.table_contact_sensor is not None:
            self.table_contact_force_norm = self.table_contact_sensor.data.net_forces_w.norm(dim=-1).max(dim=-1).values
        else:
            self.table_contact_force_norm = torch.zeros(self.num_envs, device=self.device)

    def _compute_reference_observations(self) -> torch.Tensor:
        fingertip_pos_rel_palm = (self.fingertip_pos - self.palm_pos[:, None, :]).reshape(self.num_envs, -1)
        keypoints_rel_palm = (self.object_keypoints - self.palm_pos[:, None, :]).reshape(self.num_envs, -1)
        keypoints_rel_goal = (self.object_keypoints - self.goal_keypoints).reshape(self.num_envs, -1)
        obs = torch.cat(
            (
                unscale(self.robot_dof_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits),
                self.robot_dof_vel,
                self.prev_action_targets,
                self.palm_pos,
                quat_wxyz_to_xyzw(self.palm_rot),
                quat_wxyz_to_xyzw(self.object_rot),
                fingertip_pos_rel_palm,
                keypoints_rel_palm,
                keypoints_rel_goal,
                self.object_scales,
            ),
            dim=-1,
        )
        if obs.shape[-1] != self.cfg.num_observations:
            raise RuntimeError(f"Expected {self.cfg.num_observations} observations, got {obs.shape[-1]}.")
        return obs

    def _reset_goals(self, env_ids: torch.Tensor, is_first_goal: bool) -> None:
        num_ids = env_ids.shape[0]
        goal_state = self.goal_object.data.default_root_state[env_ids].clone()

        mins = torch.tensor(self.cfg.target_volume_mins, device=self.device)
        maxs = torch.tensor(self.cfg.target_volume_maxs, device=self.device)

        if (not is_first_goal) and self.cfg.goal_sampling_type == "delta":
            current_goal_pos = self.object_goal_pos[env_ids]
            current_goal_rot = self.object_goal_rot[env_ids]
            goal_pos = current_goal_pos + sample_uniform(
                -self.cfg.delta_goal_distance,
                self.cfg.delta_goal_distance,
                (num_ids, 3),
                self.device,
            )
            goal_pos = torch.clamp(goal_pos, mins, maxs)
            goal_rot = self._sample_delta_quat(current_goal_rot, self.cfg.delta_rotation_degrees)
        elif (not is_first_goal) and self.cfg.goal_sampling_type == "coin_flip":
            current_goal_pos = self.object_goal_pos[env_ids]
            current_goal_rot = self.object_goal_rot[env_ids]
            coin_flips = sample_uniform(0.0, 1.0, (num_ids, 1), self.device)
            translation_goal_pos = current_goal_pos + sample_uniform(
                -self.cfg.delta_goal_distance,
                self.cfg.delta_goal_distance,
                (num_ids, 3),
                self.device,
            )
            translation_goal_pos = torch.clamp(translation_goal_pos, mins, maxs)
            rotation_goal_rot = self._sample_delta_quat(current_goal_rot, self.cfg.delta_rotation_degrees)
            goal_pos = torch.where(coin_flips < 0.5, translation_goal_pos, current_goal_pos)
            goal_rot = torch.where(coin_flips < 0.5, current_goal_rot, rotation_goal_rot)
        else:
            goal_pos = sample_uniform(0.0, 1.0, (num_ids, 3), self.device) * (maxs - mins) + mins
            goal_rot = self._sample_random_quat(num_ids)
            min_z = self.object_init_pos[env_ids, 2:3] - 0.05 + self.cfg.lifting_bonus_threshold
            goal_pos[:, 2:3] = torch.max(min_z, goal_pos[:, 2:3])

        self.object_goal_pos[env_ids] = goal_pos
        self.object_goal_rot[env_ids] = goal_rot
        if self.cfg.goal_object_pose is not None:
            goal_object_pose = torch.tensor(self.cfg.goal_object_pose, device=self.device, dtype=goal_state.dtype)
            self.object_goal_pos[env_ids] = goal_object_pose[0:3]
            self.object_goal_rot[env_ids] = goal_object_pose[[6, 3, 4, 5]]
        goal_state[:, 0:3] = goal_pos + self.scene.env_origins[env_ids]
        goal_state[:, 3:7] = goal_rot
        if self.cfg.goal_object_pose is not None:
            goal_state[:, 0:3] = self.object_goal_pos[env_ids] + self.scene.env_origins[env_ids]
            goal_state[:, 3:7] = self.object_goal_rot[env_ids]
        goal_state[:, 7:13] = 0.0
        self.goal_object.write_root_state_to_sim(goal_state, env_ids)

    def _sample_object_quat(self, num_ids: int) -> torch.Tensor:
        quat = torch.zeros(num_ids, 4, device=self.device)
        quat[:, 0] = 1.0
        if self.cfg.randomize_object_rotation:
            angle = sample_uniform(-math.pi, math.pi, (num_ids,), self.device)
            z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(num_ids, 1)
            quat = quat_from_angle_axis(angle, z_axis)
        return quat

    def _sample_random_quat(self, num_ids: int) -> torch.Tensor:
        uvw = sample_uniform(0.0, 1.0, (num_ids, 3), self.device)
        q_w = torch.sqrt(1.0 - uvw[:, 0]) * torch.sin(2.0 * math.pi * uvw[:, 1])
        q_x = torch.sqrt(1.0 - uvw[:, 0]) * torch.cos(2.0 * math.pi * uvw[:, 1])
        q_y = torch.sqrt(uvw[:, 0]) * torch.sin(2.0 * math.pi * uvw[:, 2])
        q_z = torch.sqrt(uvw[:, 0]) * torch.cos(2.0 * math.pi * uvw[:, 2])
        return torch.stack((q_w, q_x, q_y, q_z), dim=-1)

    def _sample_delta_quat(self, quat_wxyz: torch.Tensor, delta_rotation_degrees: float) -> torch.Tensor:
        random_direction = sample_uniform(0.0, 1.0, (quat_wxyz.shape[0], 3), self.device)
        random_direction = random_direction / torch.norm(random_direction, dim=-1, keepdim=True).clamp_min(1e-6)
        delta_rotation_radians = math.radians(delta_rotation_degrees)
        angle = sample_uniform(-delta_rotation_radians, delta_rotation_radians, (quat_wxyz.shape[0],), self.device)
        delta_quat = quat_from_angle_axis(angle, random_direction)
        return quat_mul(quat_wxyz, delta_quat)

    def _make_keypoint_offsets(self) -> torch.Tensor:
        half = 0.5 * self.cfg.object_base_size * self.cfg.keypoint_scale
        offsets = torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]],
            device=self.device,
        )
        return offsets * half

    def _compute_keypoints(
        self, pos: torch.Tensor, quat_wxyz: torch.Tensor, object_scales: torch.Tensor
    ) -> torch.Tensor:
        if pos.dim() == 3 and pos.shape[1] == 1:
            pos = pos[:, 0, :]
        if quat_wxyz.dim() == 3 and quat_wxyz.shape[1] == 1:
            quat_wxyz = quat_wxyz[:, 0, :]
        if object_scales.dim() == 3 and object_scales.shape[1] == 1:
            object_scales = object_scales[:, 0, :]

        if quat_wxyz.shape[0] == 1 and pos.shape[0] > 1:
            quat_wxyz = quat_wxyz.expand(pos.shape[0], -1)
        elif quat_wxyz.shape[0] != pos.shape[0]:
            raise RuntimeError(
                f"Keypoint batch mismatch: pos shape={tuple(pos.shape)}, quat shape={tuple(quat_wxyz.shape)}"
            )
        if object_scales.shape[0] == 1 and pos.shape[0] > 1:
            object_scales = object_scales.expand(pos.shape[0], -1)
        elif object_scales.shape[0] != pos.shape[0]:
            raise RuntimeError(
                f"Keypoint scale batch mismatch: pos shape={tuple(pos.shape)}, "
                f"scale shape={tuple(object_scales.shape)}"
            )

        offsets = self.keypoint_offsets.unsqueeze(0).expand(pos.shape[0], -1, -1) * object_scales[:, None, :]
        quat = quat_wxyz.unsqueeze(1).expand(-1, offsets.shape[1], -1)
        rotated_offsets = quat_apply(quat.reshape(-1, 4), offsets.reshape(-1, 3)).reshape(pos.shape[0], -1, 3)
        return pos.unsqueeze(1) + rotated_offsets

    def _make_keypoint_debug_draw(self):
        try:
            import omni.kit.app

            ext_manager = omni.kit.app.get_app().get_extension_manager()
            ext_manager.set_extension_enabled_immediate("isaacsim.util.debug_draw", True)
            from isaacsim.util.debug_draw import _debug_draw

            return _debug_draw.acquire_debug_draw_interface()
        except Exception:
            from omni.debugdraw import acquire_debug_draw_interface

            return acquire_debug_draw_interface()

    def _visualize_keypoints(self) -> None:
        if getattr(self, "keypoint_debug_draw", None) is None:
            return

        object_keypoints_w = self.object_keypoints + self.scene.env_origins[:, None, :]
        goal_keypoints_w = self.goal_keypoints + self.scene.env_origins[:, None, :]
        object_points = [tuple(point) for point in object_keypoints_w.reshape(-1, 3).detach().cpu().tolist()]
        goal_points = [tuple(point) for point in goal_keypoints_w.reshape(-1, 3).detach().cpu().tolist()]
        point_size = max(1.0, self.cfg.debug_keypoint_radius * 1000.0)

        self.keypoint_debug_draw.clear_points()
        self.keypoint_debug_draw.draw_points(
            object_points + goal_points,
            [(0.1, 0.45, 1.0, 1.0)] * len(object_points) + [(1.0, 0.15, 0.85, 1.0)] * len(goal_points),
            [point_size] * (len(object_points) + len(goal_points)),
        )

    def _distance_delta_rewards(self, lifted_object: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fingertip_deltas_closest = self.closest_fingertip_dist - self.curr_fingertip_distances
        self.closest_fingertip_dist = torch.minimum(self.closest_fingertip_dist, self.curr_fingertip_distances)

        hand_deltas_furthest = self.furthest_hand_dist - self.curr_fingertip_distances[:, 0]
        self.furthest_hand_dist = torch.maximum(self.furthest_hand_dist, self.curr_fingertip_distances[:, 0])

        fingertip_deltas = torch.clamp(fingertip_deltas_closest, 0.0, 10.0) * self.finger_rew_coeffs
        fingertip_delta_reward = torch.sum(fingertip_deltas, dim=-1) * (~lifted_object)

        hand_delta_penalty = torch.clamp(hand_deltas_furthest, -10.0, 0.0) * (~lifted_object)
        hand_delta_penalty = hand_delta_penalty * self.fingertip_offsets.shape[0]

        fingertip_delta_reward = fingertip_delta_reward * self.cfg.distance_delta_rew_scale
        hand_delta_penalty = hand_delta_penalty * 0.0
        return fingertip_delta_reward, hand_delta_penalty

    def _lifting_reward(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_lift = 0.05 + self.object_pos[:, 2] - self.object_init_pos[:, 2]
        lifting_reward = torch.clamp(z_lift, 0.0, 0.5)

        lifted_object = (z_lift > self.cfg.lifting_bonus_threshold) | self.lifted_object
        just_lifted = lifted_object & (~self.lifted_object)
        lift_bonus_reward = self.cfg.lifting_bonus * just_lifted.float()
        lifting_reward = lifting_reward * (~lifted_object)

        self.lifted_object = lifted_object
        lifting_reward = lifting_reward * self.cfg.lifting_rew_scale
        return lifting_reward, lift_bonus_reward, lifted_object

    def _keypoint_reward(self, lifted_object: torch.Tensor) -> torch.Tensor:
        keypoint_deltas = self.closest_keypoint_max_dist - self.keypoints_max_dist
        self.closest_keypoint_max_dist = torch.minimum(self.closest_keypoint_max_dist, self.keypoints_max_dist)
        keypoint_deltas = torch.clamp(keypoint_deltas, 0.0, 100.0)
        return keypoint_deltas * lifted_object * self.cfg.keypoint_rew_scale

    def _action_penalties(self) -> tuple[torch.Tensor, torch.Tensor]:
        kuka_actions_penalty = (
            torch.sum(torch.abs(self.robot_dof_vel[:, :7]), dim=-1) * self.cfg.kuka_actions_penalty_scale
        )
        hand_actions_penalty = (
            torch.sum(torch.abs(self.robot_dof_vel[:, 7:]), dim=-1) * self.cfg.hand_actions_penalty_scale
        )
        return -kuka_actions_penalty, -hand_actions_penalty
