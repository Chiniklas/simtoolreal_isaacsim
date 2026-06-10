"""SHARPA nut-screw env with Forge-style staged grip reward shaping."""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply

from simtoolreal_lab.tasks.sharpa_nutscrew_pick_place_screw.sharpa_nutscrew_pick_place_screw_env import (
    SharpaNutscrewPickPlaceScrewEnv,
)

from .sharpa_nutscrew_forge_env_cfg import SharpaNutscrewForgeEnvCfg


class SharpaNutscrewForgeEnv(SharpaNutscrewPickPlaceScrewEnv):
    """Reuse the simplified nut-on-screw simulation with ForgeUltra-style rewards."""

    cfg: SharpaNutscrewForgeEnvCfg

    def __init__(self, cfg: SharpaNutscrewForgeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.proper_grip = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        self.frame_since_restart += 1
        self._update_success_tolerance_curriculum()

        lifting_reward, lift_bonus_reward, lifted_object = self._lifting_reward()
        fingertip_delta_reward, hand_delta_penalty = self._distance_delta_rewards(lifted_object)
        forge_reward, forge_terms, forge_metrics, proper_grip = self._forge_grip_reward()

        contact_gate = proper_grip
        if not getattr(self.cfg, "gate_keypoint_on_proper_grip", False):
            contact_gate = self.nut_contact
        if not getattr(self.cfg, "gate_keypoint_on_contact", False):
            contact_gate = torch.ones_like(self.nut_contact)

        keypoint_reward = self._keypoint_reward(lifted_object, contact_gate)

        keypoints_max_dist = self._reward_keypoints_max_dist()
        keypoint_success_tolerance = self.success_tolerance * self.cfg.keypoint_scale
        near_goal = keypoints_max_dist <= keypoint_success_tolerance
        if getattr(self.cfg, "gate_keypoint_on_contact", False):
            near_goal = near_goal & contact_gate
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
            + forge_reward
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
            self.closest_keypoint_max_dist_fixed_size[success_env_ids] = -1.0
            if self.cfg.max_consecutive_successes > 0:
                self.episode_length_buf[success_env_ids] = 0

        reward_terms = {}
        if self.cfg.distance_delta_rew_scale != 0.0:
            reward_terms["fingertip_delta_reward"] = fingertip_delta_reward.mean()
        if self.cfg.lifting_rew_scale != 0.0:
            reward_terms["lifting_reward"] = lifting_reward.mean()
        if self.cfg.lifting_bonus != 0.0:
            reward_terms["lift_bonus_reward"] = lift_bonus_reward.mean()
        if self.cfg.keypoint_rew_scale != 0.0:
            reward_terms["keypoint_reward"] = keypoint_reward.mean()
        if self.cfg.reach_goal_bonus != 0.0:
            reward_terms["reach_bonus"] = reach_bonus.mean()
        reward_terms.update(forge_terms)
        if self.cfg.kuka_actions_penalty_scale != 0.0:
            reward_terms["arm_action_penalty"] = arm_action_penalty.mean()
        if self.cfg.hand_actions_penalty_scale != 0.0:
            reward_terms["hand_action_penalty"] = hand_action_penalty.mean()
        if self.cfg.object_lin_vel_penalty_scale != 0.0:
            reward_terms["object_lin_vel_penalty"] = object_lin_vel_penalty.mean()
        if self.cfg.object_ang_vel_penalty_scale != 0.0:
            reward_terms["object_ang_vel_penalty"] = object_ang_vel_penalty.mean()
        reward_terms["total_reward"] = reward.mean()

        task_metrics = {
            "keypoints_max_dist": keypoints_max_dist.mean(),
            "success_rate": reached_goal.float().mean(),
            "success_tolerance": self.success_tolerance,
            **forge_metrics,
        }
        if getattr(self.cfg, "use_finger_contact_sensor", False):
            task_metrics["nut_contact_rate"] = self.nut_contact.float().mean()
            task_metrics["nut_contact_force"] = self.finger_contact_forces.sum(dim=-1).mean()
            task_metrics["nut_contact_steps"] = self.contact_steps.mean()
            for finger_idx, finger in enumerate(self.cfg.active_fingers):
                task_metrics[f"{finger}_contact_rate"] = self.finger_nut_contacts[:, finger_idx].float().mean()
                task_metrics[f"{finger}_contact_force"] = self.finger_contact_forces[:, finger_idx].mean()
        if getattr(self.cfg, "screwing_phase", False):
            task_metrics["nut_thread_angle"] = self.nut_thread_angle.mean()
            if self.max_nut_thread_angle > 0.0:
                task_metrics["nut_thread_progress"] = (self.nut_thread_angle / self.max_nut_thread_angle).mean()

        self.extras["log"] = {
            **task_metrics,
            **reward_terms,
        }
        self.extras["reward_terms"] = reward_terms
        return reward

    def _forge_grip_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        if not getattr(self.cfg, "forge_reward_shaping", True):
            zero = torch.zeros(self.num_envs, device=self.device)
            return zero, {}, {}, self.nut_contact

        finger_pos = self.fingertip_pos
        active_fingers = tuple(self.cfg.active_fingers)
        nut_radius = float(getattr(self.cfg, "forge_nut_radius", 0.011))
        surface_dist = (torch.norm(finger_pos - self.object_pos[:, None, :], dim=-1) - nut_radius).clamp(min=0.0)

        contact_context = surface_dist < float(getattr(self.cfg, "contact_context_dist", 0.035))
        if getattr(self.cfg, "use_finger_contact_sensor", False):
            finger_touch = self.finger_nut_contacts & contact_context
        else:
            finger_touch = contact_context

        required_count = int(getattr(self.cfg, "required_finger_contact_count", 1))
        required_count = max(1, min(required_count, len(active_fingers)))
        in_contact = finger_touch.sum(dim=-1) >= required_count
        for finger in tuple(getattr(self.cfg, "required_contact_fingers", ())):
            if finger in active_fingers:
                in_contact = in_contact & finger_touch[:, active_fingers.index(finger)]
        contact_f = in_contact.float()

        self.contact_steps = torch.where(in_contact, self.contact_steps + 1.0, torch.zeros_like(self.contact_steps))
        sustained = self.cfg.sustained_contact_bonus * self.contact_steps.clamp(max=float(self.cfg.sustained_contact_cap))

        # Forge uses -scale * (thumb_surface_dist + index_surface_dist). For tripod,
        # use the average active-finger distance and multiply by two to keep the
        # magnitude close to the original two-finger shaping.
        contact_distance_rew = -self.cfg.contact_shaping_scale * surface_dist.mean(dim=-1) * 2.0
        contact_bonus_rew = self.cfg.contact_bonus * contact_f + sustained * contact_f
        contact_rew = contact_distance_rew + contact_bonus_rew

        thumb_idx = active_fingers.index("thumb") if "thumb" in active_fingers else 0
        other_indices = [idx for idx in range(len(active_fingers)) if idx != thumb_idx]
        if len(other_indices) == 0:
            other_indices = [thumb_idx]
        thumb_pos = finger_pos[:, thumb_idx]
        opposing_center = finger_pos[:, other_indices].mean(dim=1)

        z_unit = torch.zeros_like(thumb_pos)
        z_unit[:, 2] = 1.0
        nut_axis = torch.nn.functional.normalize(quat_apply(self.object_rot, z_unit), dim=-1, eps=1.0e-6)
        pinch_dir = torch.nn.functional.normalize(opposing_center - thumb_pos, dim=-1, eps=1.0e-6)
        align = (pinch_dir * nut_axis).sum(dim=-1)
        perp_ok = (align.abs() < self.cfg.pinch_perp_threshold).float()
        proper_grip = in_contact & (perp_ok > 0.5)
        self.proper_grip = proper_grip

        perp_rew = contact_f * (-self.cfg.pinch_perp_reward_scale * align.square() + self.cfg.pinch_perp_bonus * perp_ok)

        hand_obj_dist = torch.norm(finger_pos - self.object_pos[:, None, :], dim=-1).mean(dim=-1)
        hand_obj_rew = -self.cfg.hand_obj_reward_scale * hand_obj_dist

        ee_axis = quat_apply(self.palm_rot, z_unit)
        tilt_deg = torch.rad2deg(torch.arccos(ee_axis[:, 2].abs().clamp(max=1.0)))
        near = (surface_dist.min(dim=-1).values < self.cfg.ee_tilt_gate_dist).float()
        ee_tilt_rew = near * self.cfg.ee_tilt_reward_scale * torch.exp(
            -0.5 * ((tilt_deg - self.cfg.ee_tilt_target_deg) / self.cfg.ee_tilt_band_deg) ** 2
        )

        reward = contact_rew + perp_rew + hand_obj_rew + ee_tilt_rew
        reward_terms = {
            "forge_contact_reward": contact_rew.mean(),
            "forge_contact_distance_reward": contact_distance_rew.mean(),
            "forge_contact_bonus_reward": contact_bonus_rew.mean(),
            "forge_perp_reward": perp_rew.mean(),
            "forge_hand_object_reward": hand_obj_rew.mean(),
            "forge_ee_tilt_reward": ee_tilt_rew.mean(),
        }
        task_metrics = {
            "proper_grip_rate": proper_grip.float().mean(),
            "forge_in_contact_rate": contact_f.mean(),
            "forge_contact_steps": self.contact_steps.mean(),
            "forge_surface_dist": surface_dist.mean(),
            "forge_pinch_perp_align": align.abs().mean(),
            "forge_perp_ok": perp_ok.mean(),
            "forge_ee_tilt_deg": tilt_deg.mean(),
        }
        return reward, reward_terms, task_metrics, proper_grip

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._all_env_ids
        super()._reset_idx(env_ids)
        if hasattr(self, "proper_grip"):
            self.proper_grip[env_ids] = False
