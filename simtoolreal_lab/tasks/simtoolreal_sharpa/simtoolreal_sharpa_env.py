"""Isaac Lab DirectRLEnv port of the reference SimToolReal KUKA-SHARPA task."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul, sample_uniform, saturate

from .simtoolreal_sharpa_env_cfg import SimToolRealSharpaEnvCfg
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

        self.palm_body_idx = self.robot.body_names.index(self.cfg.palm_body_name)
        self.fingertip_body_indices = [self.robot.body_names.index(name) for name in self.cfg.fingertip_body_names]

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
        self.object_scales = torch.tensor(self.cfg.object_scales, device=self.device).repeat(self.num_envs, 1)
        self.object_last_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.success_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.successes = torch.zeros(self.num_envs, device=self.device)
        self.consecutive_successes = torch.zeros(self.num_envs, device=self.device)

        self.keypoint_offsets = self._make_keypoint_offsets()
        self._all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._compute_intermediate_values()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.table = RigidObject(self.cfg.table_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["object"] = self.object
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _apply_object_mass(self) -> None:
        masses = self.object.root_physx_view.get_masses().clone()
        inertias = self.object.root_physx_view.get_inertias().clone()
        old_masses = masses.clone()
        per_body_mass = self.cfg.object_mass / self.object.num_bodies
        masses[:] = per_body_mass

        ratios = torch.where(old_masses > 0.0, masses / old_masses, torch.ones_like(masses))
        inertias[:] = inertias * ratios.unsqueeze(-1)

        env_ids = torch.arange(self.object.num_instances, device="cpu", dtype=torch.long)
        self.object.root_physx_view.set_masses(masses, env_ids)
        self.object.root_physx_view.set_inertias(inertias, env_ids)
        self.object.data.default_mass = masses.clone()
        self.object.data.default_inertia = inertias.clone()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions.copy_(self.actions)
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self._compute_action_targets()

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(
            self.dof_pos_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

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

        dist_to_goal = torch.norm(self.object_pos - self.object_goal_pos, dim=-1)
        prev_dist_to_goal = torch.norm(self.object_last_pos - self.object_goal_pos, dim=-1)
        distance_delta_reward = self.cfg.distance_delta_rew_scale * (prev_dist_to_goal - dist_to_goal)

        object_lift = self.object_pos[:, 2] - self.cfg.table_top_z
        lifted_object = object_lift > self.cfg.lifting_bonus_threshold
        lifting_reward = self.cfg.lifting_rew_scale * torch.clamp(object_lift, min=0.0)
        lifting_reward = lifting_reward + self.cfg.lifting_bonus * lifted_object.float()

        keypoint_dist = torch.norm(self.object_keypoints - self.goal_keypoints, dim=-1).mean(dim=-1)
        keypoint_reward = self.cfg.keypoint_rew_scale * torch.exp(-10.0 * keypoint_dist)

        arm_action_penalty = torch.sum(self.actions[:, :7] ** 2, dim=-1) * self.cfg.kuka_actions_penalty_scale
        hand_action_penalty = torch.sum(self.actions[:, 7:] ** 2, dim=-1) * self.cfg.hand_actions_penalty_scale

        is_success = dist_to_goal < self.cfg.success_tolerance
        self.success_steps = torch.where(is_success, self.success_steps + 1, torch.zeros_like(self.success_steps))
        reached_goal = self.success_steps >= self.cfg.success_steps
        reach_bonus = self.cfg.reach_goal_bonus * reached_goal.float()
        self.successes += reached_goal.float()
        self.consecutive_successes = torch.where(reached_goal, self.consecutive_successes + 1.0, self.consecutive_successes)

        reward = (
            distance_delta_reward
            + lifting_reward
            + keypoint_reward
            + reach_bonus
            - arm_action_penalty
            - hand_action_penalty
        )
        self.object_last_pos.copy_(self.object_pos)
        self.extras["log"] = {
            "dist_to_goal": dist_to_goal.mean(),
            "object_lift": object_lift.mean(),
            "success_rate": is_success.float().mean(),
        }
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        object_fell = self.object_pos[:, 2] < self.cfg.fall_distance
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        too_many_successes = self.consecutive_successes >= self.cfg.max_consecutive_successes
        return object_fell | too_many_successes, time_out

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
        self.success_steps[env_ids] = 0
        self.consecutive_successes[env_ids] = 0.0

        self._reset_goals(env_ids, object_state)
        self._compute_intermediate_values()
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
        self.palm_pos = self.robot.data.body_pos_w[:, self.palm_body_idx] - self.scene.env_origins
        self.palm_rot = self.robot.data.body_quat_w[:, self.palm_body_idx]
        self.fingertip_pos = self.robot.data.body_pos_w[:, self.fingertip_body_indices] - self.scene.env_origins[:, None, :]
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_vel = torch.cat((self.object.data.root_lin_vel_w, self.object.data.root_ang_vel_w), dim=-1)
        self.object_keypoints = self._compute_keypoints(self.object_pos, self.object_rot)
        self.goal_keypoints = self._compute_keypoints(self.object_goal_pos, self.object_goal_rot)

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

    def _reset_goals(self, env_ids: torch.Tensor, object_state_w: torch.Tensor) -> None:
        object_pos = object_state_w[:, 0:3] - self.scene.env_origins[env_ids]
        if self.cfg.goal_sampling_type == "delta":
            theta = sample_uniform(-math.pi, math.pi, (env_ids.shape[0],), self.device)
            delta = torch.stack((torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)), dim=-1)
            goal_pos = object_pos + self.cfg.delta_goal_distance * delta
            goal_pos[:, 2] += self.cfg.lifting_bonus_threshold
        else:
            mins = torch.tensor(self.cfg.target_volume_mins, device=self.device)
            maxs = torch.tensor(self.cfg.target_volume_maxs, device=self.device)
            goal_pos = sample_uniform(0.0, 1.0, (env_ids.shape[0], 3), self.device) * (maxs - mins) + mins

        mins = torch.tensor(self.cfg.target_volume_mins, device=self.device)
        maxs = torch.tensor(self.cfg.target_volume_maxs, device=self.device)
        self.object_goal_pos[env_ids] = torch.clamp(goal_pos, mins, maxs)

        angle = math.radians(self.cfg.delta_rotation_degrees)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(env_ids.shape[0], 1)
        delta_rot = quat_from_angle_axis(torch.full((env_ids.shape[0],), angle, device=self.device), z_axis)
        self.object_goal_rot[env_ids] = quat_mul(delta_rot, object_state_w[:, 3:7])

    def _sample_object_quat(self, num_ids: int) -> torch.Tensor:
        quat = torch.zeros(num_ids, 4, device=self.device)
        quat[:, 0] = 1.0
        if self.cfg.randomize_object_rotation:
            angle = sample_uniform(-math.pi, math.pi, (num_ids,), self.device)
            z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(num_ids, 1)
            quat = quat_from_angle_axis(angle, z_axis)
        return quat

    def _make_keypoint_offsets(self) -> torch.Tensor:
        half = 0.5 * self.cfg.object_base_size * self.cfg.keypoint_scale
        offsets = torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]],
            device=self.device,
        )
        return offsets * half

    def _compute_keypoints(self, pos: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
        offsets = self.keypoint_offsets.unsqueeze(0).expand(pos.shape[0], -1, -1)
        quat = quat_wxyz[:, None, :].expand(-1, offsets.shape[1], -1)
        return pos[:, None, :] + quat_apply(quat, offsets)
