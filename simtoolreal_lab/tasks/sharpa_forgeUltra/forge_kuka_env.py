# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""forge_kuka: environment.

A single flattened ``ForgeKukaEnv(DirectRLEnv)`` that merges the vendored
``FactoryEnv`` base and ``ForgeEnv`` subclass into one class, with the
Franka FR3 morphology replaced by the KUKA iiwa14 + left SHARPA hand.

Morphology summary (arm-EE control, frozen hand):
- EE reference frame = midpoint of ``left_thumb_fingertip`` and
  ``left_index_fingertip`` (analog of the Franka left/right-finger midpoint).
  Orientation/angular velocity are taken from the thumb tip.
- Arm joints (``iiwa14_joint_1..7``) are resolved at runtime; their PD gains
  are zeroed so the EE-impedance torque controls them.
- The 22 SHARPA finger joints are held at a fixed posture via position control.
- All hard-coded ``[:, 0:7]`` / ``[:, 7:9]`` Franka slices are replaced with
  the resolved ``arm_joint_ids`` / ``hand_joint_ids``.
"""

import numpy as np
import torch

import carb
import isaacsim.core.utils.torch as torch_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_from_matrix
from pxr import Usd, UsdPhysics

from . import forge_kuka_control, forge_kuka_utils
from .forge_kuka_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, ForgeKukaEnvCfg

# SHARPA fingertip bodies used to define the EE reference frame.
THUMB_TIP_BODY = "left_thumb_fingertip"
INDEX_TIP_BODY = "left_index_fingertip"
# run07: the fingertip frames carry NO collider — the convex-hull colliders live on
# the elastomer pads. Contact sensors must sit here to read net_forces_w on contact.
THUMB_PAD_BODY = "left_thumb_elastomer"
INDEX_PAD_BODY = "left_index_elastomer"
# Fallback body for the force/torque sensor (SHARPA has no "force_sensor" body).
ARM_FLANGE_BODY = "iiwa14_link_ee"

# Action-driven pinch joints (Stage B): the policy commands the thumb's opposition
# (CMC_AA) and flexion (MCP_FE) to pinch the nut against the (fixed) index finger.
# Action in [-1, 1] maps linearly into these joint ranges (radians).
THUMB_PINCH_JOINTS = ("left_thumb_CMC_AA", "left_thumb_MCP_FE")
THUMB_PINCH_RANGES = ((-0.349, 0.349), (-0.524, 1.396))  # CMC_AA, MCP_FE
# run08: actuate the index MCP_FE too, so the policy forms a real two-finger pinch
# (extended -> flexed inward onto the nut). Range may need tuning to the joint limit.
INDEX_PINCH_JOINTS = ("left_2_index_MCP_FE",)
INDEX_PINCH_RANGES = ((0.0, 1.4),)


class ForgeKukaEnv(DirectRLEnv):
    cfg: ForgeKukaEnvCfg

    def __init__(self, cfg: ForgeKukaEnvCfg, render_mode: str | None = None, **kwargs):
        # Update number of obs/states.
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order])
        cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.state_order])
        cfg.observation_space += cfg.action_space
        cfg.state_space += cfg.action_space
        self.cfg_task = cfg.task

        super().__init__(cfg, render_mode, **kwargs)

        forge_kuka_utils.set_body_inertias(self._robot, self.scene.num_envs)
        self._init_tensors()
        self._set_default_dynamics_parameters()

        # --- Forge additions ---
        # Success prediction.
        self.success_pred_scale = 0.0
        self.first_pred_success_tx = {}
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            self.first_pred_success_tx[thresh] = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # Flip quaternions.
        self.flip_quats = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)

        # Force sensor information. SHARPA has no dedicated "force_sensor" body,
        # so fall back to the arm flange (tracked in todos.md).
        if "force_sensor" in self._robot.body_names:
            self.force_sensor_body_idx = self._robot.body_names.index("force_sensor")
        else:
            self.force_sensor_body_idx = self._robot.body_names.index(ARM_FLANGE_BODY)
        self.force_sensor_smooth = torch.zeros((self.num_envs, 6), device=self.device)
        self.force_sensor_world_smooth = torch.zeros((self.num_envs, 6), device=self.device)

        # Set nominal dynamics parameters for randomization.
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.default_dead_zone = torch.tensor(self.cfg.ctrl.default_dead_zone, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.pos_threshold = self.default_pos_threshold.clone()
        self.rot_threshold = self.default_rot_threshold.clone()

    def _set_default_dynamics_parameters(self):
        """Set parameters defining dynamic interactions."""
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )

        # Set masses and frictions.
        held_friction = getattr(self.cfg, "held_asset_friction_override", None)
        if held_friction is None:
            held_friction = self.cfg_task.held_asset_cfg.friction
        forge_kuka_utils.set_friction(self._held_asset, held_friction, self.scene.num_envs)
        forge_kuka_utils.set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction, self.scene.num_envs)
        forge_kuka_utils.set_friction(self._robot, self.cfg_task.robot_cfg.friction, self.scene.num_envs)

    def _init_tensors(self):
        """Initialize tensors once."""
        # Control targets.
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.ema_factor = self.cfg.ctrl.ema_factor
        self.dead_zone_thresholds = None

        # Fixed asset.
        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        # --- Morphology: resolve arm / hand joints and EE bodies. ---
        arm_names = [f"iiwa14_joint_{i}" for i in range(1, 8)]
        self.arm_joint_ids, _ = self._robot.find_joints(arm_names, preserve_order=True)
        self.hand_joint_ids, self.hand_joint_names = self._robot.find_joints(["left_.*"])
        self.num_arm_dofs = len(self.arm_joint_ids)
        self.arm_joint_ids_t = torch.tensor(self.arm_joint_ids, device=self.device, dtype=torch.long)
        self.hand_joint_ids_t = torch.tensor(self.hand_joint_ids, device=self.device, dtype=torch.long)

        # EE reference frame bodies: thumb+index tips give the pinch-center position
        # (where the nut is); the flange gives the (finger-independent) orientation.
        self.thumb_body_idx = self._robot.body_names.index(THUMB_TIP_BODY)
        self.index_body_idx = self._robot.body_names.index(INDEX_TIP_BODY)
        # run10: elastomer-pad body indices (the actual colliding pads) for the
        # fingertip->nut-center approach reward.
        self.thumb_pad_idx = self._robot.body_names.index(THUMB_PAD_BODY)
        self.index_pad_idx = self._robot.body_names.index(INDEX_PAD_BODY)
        self.flange_body_idx = self._robot.body_names.index(ARM_FLANGE_BODY)
        # Action-driven pinch joint indices (Stage B). Resolved from joint names.
        # run08: thumb (2 DOF) and, if action_driven_index, the index MCP_FE (1 DOF).
        pinch_joint_names = list(THUMB_PINCH_JOINTS)
        self.pinch_ranges = list(THUMB_PINCH_RANGES)
        if getattr(self.cfg, "action_driven_index", False):
            pinch_joint_names += list(INDEX_PINCH_JOINTS)
            self.pinch_ranges += list(INDEX_PINCH_RANGES)
        self.pinch_joint_ids = [self._robot.joint_names.index(n) for n in pinch_joint_names]
        # Calibrated palm-anchored grasp frame (fixed offset from the flange).
        # Derived once at the first grasp (see _calibrate_grasp_frame); None until then.
        self.grasp_offset_pos = None
        self.grasp_offset_quat = None
        # Pinch-demo: stored EE pose to hold while the fingers ramp closed.
        self._demo_hold_pos = None
        self._demo_hold_quat = None
        # run03: consecutive-steps-in-contact counter (for the sustained-grip bonus).
        self.contact_steps = torch.zeros(self.num_envs, device=self.device)
        # run07: LIVE proper-grip state (recomputed each step in _get_rewards) —
        # True when the thumb is pressing the nut AND the grip is perpendicular.
        # Gates the nut-position reward (kp_* + curr_engaged); NOT latched, so the
        # policy can't tap-and-drift to keep the reward (the run06 loophole).
        self.proper_grip = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Hand postures. OPEN = the robot's default hand pose (fingers spread);
        # CLOSED = OPEN with the cfg pinch override applied. For grasp_on_reset,
        # reset toggles the actively-commanded posture OPEN -> CLOSED (place the nut
        # between open pads, then close onto it). Otherwise it stays CLOSED.
        self.open_hand_joint_pos = self._robot.data.default_joint_pos[:, self.hand_joint_ids].clone()
        self.closed_hand_joint_pos = self.open_hand_joint_pos.clone()
        override = getattr(self.cfg, "frozen_hand_joint_pos", None)
        if override:
            for jname, angle in override.items():
                if jname in self.hand_joint_names:
                    self.closed_hand_joint_pos[:, self.hand_joint_names.index(jname)] = angle
                else:
                    carb.log_warn(f"forge_kuka: frozen_hand_joint_pos override joint '{jname}' not in hand joints; ignored.")
        # The posture actively held each control step (generate_ctrl_signals).
        self.frozen_hand_joint_pos = self.closed_hand_joint_pos.clone()

        # Zero the arm-joint PD gains so the EE-impedance torque fully controls
        # the arm (the hand keeps its configured gains to hold the posture).
        zeros_arm = torch.zeros((self.num_envs, self.num_arm_dofs), device=self.device)
        self._robot.write_joint_stiffness_to_sim(zeros_arm, joint_ids=self.arm_joint_ids)
        self._robot.write_joint_damping_to_sim(zeros_arm, joint_ids=self.arm_joint_ids)

        # Optionally stiffen the hand actuators so the pinch can actually grip
        # (SHARPA reference finger stiffness ~4-13 is far too soft).
        hand_stiff = getattr(self.cfg, "frozen_hand_stiffness", None)
        if hand_stiff is not None:
            n_hand = len(self.hand_joint_ids)
            k = torch.full((self.num_envs, n_hand), float(hand_stiff), device=self.device)
            self._robot.write_joint_stiffness_to_sim(k, joint_ids=self.hand_joint_ids)
            self._robot.write_joint_damping_to_sim(0.1 * k, joint_ids=self.hand_joint_ids)

        # Optionally raise the hand-joint effort ceiling (reference ~0.2-3.3 Nm is
        # too small for the position target to close/clamp). Applied via the physx
        # view max-force tensor, mirroring set_friction/set_body_inertias.
        hand_effort = getattr(self.cfg, "frozen_hand_effort", None)
        if hand_effort is not None:
            forces = self._robot.root_physx_view.get_dof_max_forces()
            forces[:, self.hand_joint_ids] = float(hand_effort)
            self._robot.root_physx_view.set_dof_max_forces(forces, torch.arange(self.num_envs))

        # Tensors for finite-differencing.
        self.last_update_timestamp = 0.0  # Note: This is for finite differencing body velocities.
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        self.prev_joint_pos = torch.zeros((self.num_envs, self.num_arm_dofs), device=self.device)

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)
        if self.cfg_task.name == "gear_mesh":
            self._small_gear_asset = Articulation(self.cfg_task.small_gear_cfg)
            self._large_gear_asset = Articulation(self.cfg_task.large_gear_cfg)

        # Mirror the proven simtoolreal_sharpa setup: only do physics-replicated
        # cloning when replicate_physics is on. With replicate_physics=False the
        # regex spawn already created an independent robot in every env, so we
        # must NOT call clone_environments (it would break the KUKA articulation),
        # and we explicitly filter inter-env collisions instead.
        if self.scene.cfg.replicate_physics:
            self.scene.clone_environments(copy_from_source=False)
        if not self.scene.cfg.replicate_physics or self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset
        if self.cfg_task.name == "gear_mesh":
            self.scene.articulations["small_gear"] = self._small_gear_asset
            self.scene.articulations["large_gear"] = self._large_gear_asset

        # run07: contact sensors on the thumb/index ELASTOMER PADS, read via
        # net_forces_w (total contact force on the pad). The fingertip frames have no
        # collider, and filtered force_matrix_w is GPU-dead for the SDF nut — so the
        # old run03 filtered-fingertip approach read flat 0. Nut-specificity is
        # recovered by spatial context (contact_context_dist) in _get_rewards.
        # run11: TRUE finger<->nut contact via FILTERED force_matrix_w. Sensors sit on
        # the elastomer pads (which carry the convex-hull colliders) and FILTER to the
        # nut's rigid-body LINK. run03-10 read net_forces_w because filtered
        # force_matrix_w looked "GPU-dead" for the SDF nut — but that was a filter-TARGET
        # bug: filtering to the HeldAsset wrapper Xform or the collision mesh returns 0 +
        # a "GPU contact filter not supported" warning, while filtering to the rigid-body
        # LINK populates fine even for an SDF collider (verified in-sim). Filtered force
        # excludes finger-finger self-contact by construction — killing the run10 exploit.
        if getattr(self.cfg, "use_contact_sensor", False):
            nut_filter = [self._nut_body_filter_expr()]
            self._thumb_contact = ContactSensor(
                ContactSensorCfg(
                    prim_path=f"{self.cfg.robot.prim_path}/{THUMB_PAD_BODY}",
                    filter_prim_paths_expr=nut_filter,
                    debug_vis=True,
                )
            )
            self._index_contact = ContactSensor(
                ContactSensorCfg(
                    prim_path=f"{self.cfg.robot.prim_path}/{INDEX_PAD_BODY}",
                    filter_prim_paths_expr=nut_filter,
                    debug_vis=True,
                )
            )
            self.scene.sensors["thumb_contact"] = self._thumb_contact
            self.scene.sensors["index_contact"] = self._index_contact

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _nut_body_filter_expr(self):
        """Return the ContactSensor filter expr for the nut's RIGID-BODY link. The
        held asset spawns as /World/envs/env_*/HeldAsset (a wrapper Xform); the actual
        rigid body is a child (e.g. factory_nut_loose). Filtered force_matrix_w only
        populates when the filter targets that LINK — not the wrapper or the collision
        mesh (those return 0 + a "GPU contact filter not supported" warning). We
        discover the link by name so it works regardless of the held asset."""
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        held = stage.GetPrimAtPath("/World/envs/env_0/HeldAsset")
        if held.IsValid():
            link = next((p for p in Usd.PrimRange(held) if p.HasAPI(UsdPhysics.RigidBodyAPI)), None)
            if link is not None:
                return f"/World/envs/env_.*/HeldAsset/{link.GetName()}"
        carb.log_warn("forge_kuka: nut rigid-body link not found; filtering to HeldAsset wrapper (may be GPU-dead).")
        return "/World/envs/env_.*/HeldAsset"

    def _compute_intermediate_values(self, dt):
        """Get values computed from raw tensors. This includes adding noise."""
        self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        self.fixed_quat = self._fixed_asset.data.root_quat_w

        self.held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins
        self.held_quat = self._held_asset.data.root_quat_w

        thumb_pos = self._robot.data.body_pos_w[:, self.thumb_body_idx] - self.scene.env_origins
        index_pos = self._robot.data.body_pos_w[:, self.index_body_idx] - self.scene.env_origins
        flange_pos = self._robot.data.body_pos_w[:, self.flange_body_idx] - self.scene.env_origins
        flange_quat = self._robot.data.body_quat_w[:, self.flange_body_idx]

        if getattr(self.cfg, "grasp_on_reset", False):
            # --- Palm-anchored grasp frame (Franka-fingertip-style) ---
            # Origin at the thumb/index pad midpoint, z = approach (toward the bolt),
            # x = pinch-closing axis. Once calibrated it is a FIXED flange offset, so
            # it is finger-independent (no drift when the pinch closes) and the
            # threading axis is well-defined. Falls back to the live pad midpoint +
            # flange orientation until calibration (first grasp) completes.
            if self.grasp_offset_pos is not None:
                grasp_pos = flange_pos + torch_utils.quat_rotate(flange_quat, self.grasp_offset_pos)
                grasp_quat = torch_utils.quat_mul(flange_quat, self.grasp_offset_quat)
            else:
                grasp_pos = 0.5 * (thumb_pos + index_pos)
                grasp_quat = flange_quat

            # Optional fixed orientation offset (euler xyz, deg) to fine-tune coaxiality.
            off = self.cfg.grasp_frame_rot_offset_deg
            if any(o != 0.0 for o in off):
                roll = torch.full((self.num_envs,), float(np.deg2rad(off[0])), device=self.device)
                pitch = torch.full((self.num_envs,), float(np.deg2rad(off[1])), device=self.device)
                yaw = torch.full((self.num_envs,), float(np.deg2rad(off[2])), device=self.device)
                grasp_quat = torch_utils.quat_mul(grasp_quat, torch_utils.quat_from_euler_xyz(roll, pitch, yaw))

            self.fingertip_midpoint_pos = grasp_pos
            self.fingertip_midpoint_quat = grasp_quat

            # Velocities of the grasp point, rigidly attached to the flange.
            flange_linvel = self._robot.data.body_lin_vel_w[:, self.flange_body_idx]
            flange_angvel = self._robot.data.body_ang_vel_w[:, self.flange_body_idx]
            r_world = grasp_pos - flange_pos
            self.fingertip_midpoint_linvel = flange_linvel + torch.cross(flange_angvel, r_world, dim=-1)
            self.fingertip_midpoint_angvel = flange_angvel

            # Geometric Jacobian of the grasp point (flange Jacobian shifted by r):
            #   J_v_grasp = J_v_flange - skew(r) @ J_w_flange ;  J_w_grasp = J_w_flange
            jacobians = self._robot.root_physx_view.get_jacobians()
            flange_jac = jacobians[:, self.flange_body_idx - 1, 0:6, :][:, :, self.arm_joint_ids]  # (E,6,7)
            jac_v = flange_jac[:, 0:3, :]
            jac_w = flange_jac[:, 3:6, :]
            rx, ry, rz = r_world[:, 0], r_world[:, 1], r_world[:, 2]
            zeros = torch.zeros_like(rx)
            skew_r = torch.stack(
                [
                    torch.stack([zeros, -rz, ry], dim=-1),
                    torch.stack([rz, zeros, -rx], dim=-1),
                    torch.stack([-ry, rx, zeros], dim=-1),
                ],
                dim=1,
            )  # (E,3,3)
            jac_v_grasp = jac_v - torch.bmm(skew_r, jac_w)
            self.fingertip_midpoint_jacobian = torch.cat([jac_v_grasp, jac_w], dim=1)  # (E,6,7)
        else:
            # Original Franka-style frame: midpoint of the two fingertip bodies.
            self.fingertip_midpoint_pos = 0.5 * (thumb_pos + index_pos)
            self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.thumb_body_idx]
            thumb_linvel = self._robot.data.body_lin_vel_w[:, self.thumb_body_idx]
            index_linvel = self._robot.data.body_lin_vel_w[:, self.index_body_idx]
            self.fingertip_midpoint_linvel = 0.5 * (thumb_linvel + index_linvel)
            self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.thumb_body_idx]
            jacobians = self._robot.root_physx_view.get_jacobians()
            thumb_jacobian = jacobians[:, self.thumb_body_idx - 1, 0:6, :][:, :, self.arm_joint_ids]
            index_jacobian = jacobians[:, self.index_body_idx - 1, 0:6, :][:, :, self.arm_joint_ids]
            self.fingertip_midpoint_jacobian = (thumb_jacobian + index_jacobian) * 0.5

        mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()
        self.arm_mass_matrix = mass_matrix[:, self.arm_joint_ids, :][:, :, self.arm_joint_ids]

        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

        # Finite-differencing results in more reliable velocity estimates.
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

        # Add state differences if velocity isn't being added.
        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        joint_diff = self.joint_pos[:, self.arm_joint_ids] - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt
        self.prev_joint_pos = self.joint_pos[:, self.arm_joint_ids].clone()

        self.last_update_timestamp = self._robot._data._sim_timestamp

        # --- Forge: add noise to fingertip obs and compute force sensing. ---
        pos_noise_level, rot_noise_level_deg = self.cfg.obs_rand.fingertip_pos, self.cfg.obs_rand.fingertip_rot_deg
        fingertip_pos_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        fingertip_pos_noise = fingertip_pos_noise @ torch.diag(
            torch.tensor([pos_noise_level, pos_noise_level, pos_noise_level], dtype=torch.float32, device=self.device)
        )
        self.noisy_fingertip_pos = self.fingertip_midpoint_pos + fingertip_pos_noise

        rot_noise_axis = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        rot_noise_axis /= torch.linalg.norm(rot_noise_axis, dim=1, keepdim=True)
        rot_noise_angle = torch.randn((self.num_envs,), dtype=torch.float32, device=self.device) * np.deg2rad(
            rot_noise_level_deg
        )
        self.noisy_fingertip_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_from_angle_axis(rot_noise_angle, rot_noise_axis)
        )
        self.noisy_fingertip_quat[:, [0, 3]] = 0.0
        self.noisy_fingertip_quat = self.noisy_fingertip_quat * self.flip_quats.unsqueeze(-1)

        # Repeat finite differencing with noisy fingertip positions.
        self.ee_linvel_fd = (self.noisy_fingertip_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.noisy_fingertip_pos.clone()

        rot_diff_quat = torch_utils.quat_mul(
            self.noisy_fingertip_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.ee_angvel_fd[:, 0:2] = 0.0
        self.prev_fingertip_quat = self.noisy_fingertip_quat.clone()

        # Update and smooth force values.
        self.force_sensor_world = self._robot.root_physx_view.get_link_incoming_joint_force()[
            :, self.force_sensor_body_idx
        ]

        alpha = self.cfg.ft_smoothing_factor
        self.force_sensor_world_smooth = alpha * self.force_sensor_world + (1 - alpha) * self.force_sensor_world_smooth

        self.force_sensor_smooth = torch.zeros_like(self.force_sensor_world)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.force_sensor_smooth[:, :3], self.force_sensor_smooth[:, 3:6] = forge_kuka_utils.change_FT_frame(
            self.force_sensor_world_smooth[:, 0:3],
            self.force_sensor_world_smooth[:, 3:6],
            (identity_quat, torch.zeros((self.num_envs, 3), device=self.device)),
            (identity_quat, self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise),
        )

        # Compute noisy force values.
        force_noise = torch.randn((self.num_envs, 3), dtype=torch.float32, device=self.device)
        force_noise *= self.cfg.obs_rand.ft_force
        self.noisy_force = self.force_sensor_smooth[:, 0:3] + force_noise

    def _get_factory_obs_state_dict(self):
        """Populate dictionaries for the policy and critic."""
        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        prev_actions = self.actions.clone()

        obs_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - noisy_fixed_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.ee_linvel_fd,
            "ee_angvel": self.ee_angvel_fd,
            "prev_actions": prev_actions,
        }

        state_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - self.fixed_pos_obs_frame,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
            "joint_pos": self.joint_pos[:, self.arm_joint_ids],
            "held_pos": self.held_pos,
            "held_pos_rel_fixed": self.held_pos - self.fixed_pos_obs_frame,
            "held_quat": self.held_quat,
            "fixed_pos": self.fixed_pos,
            "fixed_quat": self.fixed_quat,
            "task_prop_gains": self.task_prop_gains,
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
            "prev_actions": prev_actions,
        }
        return obs_dict, state_dict

    def _get_observations(self):
        """Get actor/critic inputs, adding FORGE observations."""
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        prev_actions = self.actions.clone()
        prev_actions[:, 3:5] = 0.0

        obs_dict.update(
            {
                "fingertip_pos": self.noisy_fingertip_pos,
                "fingertip_pos_rel_fixed": self.noisy_fingertip_pos - noisy_fixed_pos,
                "fingertip_quat": self.noisy_fingertip_quat,
                "force_threshold": self.contact_penalty_thresholds[:, None],
                "ft_force": self.noisy_force,
                "prev_actions": prev_actions,
            }
        )

        state_dict.update(
            {
                "ema_factor": self.ema_factor,
                "ft_force": self.force_sensor_smooth[:, 0:3],
                "force_threshold": self.contact_penalty_thresholds[:, None],
                "prev_actions": prev_actions,
            }
        )

        obs_tensors = forge_kuka_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = forge_kuka_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}

    def _reset_buffers(self, env_ids):
        """Reset buffers (factory + forge metrics)."""
        self.ep_succeeded[env_ids] = 0
        self.ep_success_times[env_ids] = 0
        # Reset success pred metrics.
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            self.first_pred_success_tx[thresh][env_ids] = 0

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        self.actions = self.ema_factor * action.clone().to(self.device) + (1 - self.ema_factor) * self.actions

    def close_gripper_in_place(self):
        """Keep EE in current position (no gripper to close; hand stays frozen)."""
        actions = torch.zeros((self.num_envs, 6), device=self.device)

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3] * self.pos_threshold
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)

        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        # run11: forge hard-locks the hold orientation to roll=pi / pitch=0 (Franka
        # "face down"). Gate it behind free_ee_orientation so the grasp-close reset
        # HOLDS the current (manually-posed) orientation instead of yanking it down.
        if not getattr(self.cfg, "free_ee_orientation", False):
            target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(ctrl_target_fingertip_midpoint_quat), dim=1)
            target_euler_xyz[:, 0] = 3.14159
            target_euler_xyz[:, 1] = 0.0
            ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
                roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
            )

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )

    def _apply_action(self):
        """FORGE actions are defined as targets relative to the fixed asset."""
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Pinch demo: ignore the policy; hold the arm at the stored pose and ramp the
        # thumb+index from open -> closed over pinch_demo_ramp_steps so the pinch is
        # visible. Lets us watch whether the fingers actually grip the nut.
        if getattr(self.cfg, "pinch_demo", False) and self._demo_hold_pos is not None:
            t = (self.episode_length_buf.float() / float(self.cfg.pinch_demo_ramp_steps)).clamp(0.0, 1.0)
            t = t.unsqueeze(-1)
            self.frozen_hand_joint_pos = (
                self.open_hand_joint_pos + t * (self.closed_hand_joint_pos - self.open_hand_joint_pos)
            )
            self.generate_ctrl_signals(
                ctrl_target_fingertip_midpoint_pos=self._demo_hold_pos,
                ctrl_target_fingertip_midpoint_quat=self._demo_hold_quat,
                ctrl_target_gripper_dof_pos=0.0,
            )
            # Reward terms aren't meaningful in the demo, but _get_rewards reads these.
            self.delta_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self.delta_yaw = torch.zeros((self.num_envs,), device=self.device)
            return

        # Step (0): Scale actions to allowed range.
        pos_actions = self.actions[:, 0:3]
        pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device))

        rot_actions = self.actions[:, 3:6]
        rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg.ctrl.rot_action_bounds, device=self.device))

        # Step (1): Compute desired pose targets in EE frame.
        # (1.a) Position. Action frame is assumed to be the top of the bolt (noisy estimate).
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        ctrl_target_fingertip_preclipped_pos = fixed_pos_action_frame + pos_actions
        # (1.b) Enforce rotation action constraints.
        # forge hard-locks the hand "straight down" (roll/pitch = 0); only unlock
        # when the policy is meant to orient the hand (e.g. tilt to a side grip).
        if not getattr(self.cfg, "free_ee_orientation", False):
            rot_actions[:, 0:2] = 0.0

        # Assumes joint limit is in (+x, -y)-quadrant of world frame.
        rot_actions[:, 2] = np.deg2rad(-180.0) + np.deg2rad(270.0) * (rot_actions[:, 2] + 1.0) / 2.0  # Joint limit.
        # (1.c) Get desired orientation target.
        bolt_frame_quat = torch_utils.quat_from_euler_xyz(
            roll=rot_actions[:, 0], pitch=rot_actions[:, 1], yaw=rot_actions[:, 2]
        )

        # run11: EE "base" orientation = the zero-action (neutral) target. forge's
        # Franka default is [pi,0,0] = face DOWN; the pinch task overrides it to the
        # manual tilt so zero action holds the pinch instead of pulling the EE vertical.
        ee_base = getattr(self.cfg, "ee_base_orn_euler", [np.pi, 0.0, 0.0])
        rot_180_euler = torch.tensor(ee_base, device=self.device).repeat(self.num_envs, 1)
        quat_bolt_to_ee = torch_utils.quat_from_euler_xyz(
            roll=rot_180_euler[:, 0], pitch=rot_180_euler[:, 1], yaw=rot_180_euler[:, 2]
        )

        ctrl_target_fingertip_preclipped_quat = torch_utils.quat_mul(quat_bolt_to_ee, bolt_frame_quat)

        # Step (2): Clip targets if they are too far from current EE pose.
        # (2.a): Clip position targets.
        self.delta_pos = ctrl_target_fingertip_preclipped_pos - self.fingertip_midpoint_pos  # Used for action_penalty.
        pos_error_clipped = torch.clip(self.delta_pos, -self.pos_threshold, self.pos_threshold)
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_error_clipped

        # (2.b) Clip orientation targets. Use Euler angles. We assume we are near upright, so
        # clipping yaw will effectively cause slow motions. When we clip, we also need to make
        # sure we avoid the joint limit.

        # (2.b.i) Get current and desired Euler angles.
        curr_roll, curr_pitch, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        desired_roll, desired_pitch, desired_yaw = torch_utils.get_euler_xyz(ctrl_target_fingertip_preclipped_quat)
        desired_xyz = torch.stack([desired_roll, desired_pitch, desired_yaw], dim=1)

        # (2.b.ii) Correct the direction of motion to avoid joint limit.
        # Map yaws between [-125, 235] degrees
        # (so that angles appear on a continuous span uninterrupted by the joint limit)
        curr_yaw = forge_kuka_utils.wrap_yaw(curr_yaw)
        desired_yaw = forge_kuka_utils.wrap_yaw(desired_yaw)

        # (2.b.iii) Clip motion in the correct direction.
        self.delta_yaw = desired_yaw - curr_yaw  # Used later for action_penalty.
        clipped_yaw = torch.clip(self.delta_yaw, -self.rot_threshold[:, 2], self.rot_threshold[:, 2])
        desired_xyz[:, 2] = curr_yaw + clipped_yaw

        # (2.b.iv) Clip roll and pitch.
        desired_roll = torch.where(desired_roll < 0.0, desired_roll + 2 * torch.pi, desired_roll)
        desired_pitch = torch.where(desired_pitch < 0.0, desired_pitch + 2 * torch.pi, desired_pitch)

        delta_roll = desired_roll - curr_roll
        clipped_roll = torch.clip(delta_roll, -self.rot_threshold[:, 0], self.rot_threshold[:, 0])
        desired_xyz[:, 0] = curr_roll + clipped_roll

        curr_pitch = torch.where(curr_pitch > torch.pi, curr_pitch - 2 * torch.pi, curr_pitch)
        desired_pitch = torch.where(desired_pitch > torch.pi, desired_pitch - 2 * torch.pi, desired_pitch)

        delta_pitch = desired_pitch - curr_pitch
        clipped_pitch = torch.clip(delta_pitch, -self.rot_threshold[:, 1], self.rot_threshold[:, 1])
        desired_xyz[:, 1] = curr_pitch + clipped_pitch

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=desired_xyz[:, 0], pitch=desired_xyz[:, 1], yaw=desired_xyz[:, 2]
        )

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )

        # Rigid-attach (Stage A): kinematically lock the nut to the bolt axis,
        # coaxial, with height + yaw driven by the hand. Turning/lowering the EE
        # threads it. Sidesteps the physical 2-finger grasp; the nut cannot fly off.
        if getattr(self.cfg, "rigid_nut_follow", False):
            yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)[2]
            z0 = torch.zeros_like(yaw)
            nut_quat = torch_utils.quat_from_euler_xyz(z0, z0, yaw)
            nut_pos = torch.stack(
                [self.fixed_pos[:, 0], self.fixed_pos[:, 1], self.fingertip_midpoint_pos[:, 2]], dim=-1
            ) + self.scene.env_origins
            self._held_asset.write_root_pose_to_sim(torch.cat([nut_pos, nut_quat], dim=-1))
            self._held_asset.write_root_velocity_to_sim(torch.zeros((self.num_envs, 6), device=self.device))

    def generate_ctrl_signals(
        self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, ctrl_target_gripper_dof_pos
    ):
        """Compute arm DOF torques (EE impedance) and hold the SHARPA hand frozen."""
        arm_torque, self.applied_wrench = forge_kuka_control.compute_dof_torque(
            cfg=self.cfg,
            dof_pos_arm=self.joint_pos[:, self.arm_joint_ids],
            dof_vel_arm=self.joint_vel[:, self.arm_joint_ids],
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.fingertip_midpoint_linvel,
            fingertip_midpoint_angvel=self.fingertip_midpoint_angvel,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            num_arm_dofs=self.num_arm_dofs,
            device=self.device,
            dead_zone_thresholds=self.dead_zone_thresholds,
        )

        # Scatter arm torque into the full-width effort buffer; hand torque = 0.
        self.joint_torque = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.joint_torque[:, self.arm_joint_ids] = arm_torque

        # Hold the SHARPA hand at the frozen posture via PhysX's PD controller.
        self.ctrl_target_joint_pos[:, self.hand_joint_ids] = self.frozen_hand_joint_pos

        # Stage B: the policy drives the pinch joints — the thumb (CMC_AA + MCP_FE) and
        # (run08, if action_driven_index) the index MCP_FE for a two-finger pinch.
        # These are the LAST len(pinch_joint_ids) actions; other hand joints stay frozen.
        if getattr(self.cfg, "action_driven_fingers", False):
            n_pinch = len(self.pinch_joint_ids)
            for k, (jid, (lo, hi)) in enumerate(zip(self.pinch_joint_ids, self.pinch_ranges)):
                a = (self.actions[:, -n_pinch + k] + 1.0) * 0.5  # last n actions -> [0,1]
                self.ctrl_target_joint_pos[:, jid] = lo + a * (hi - lo)

        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def _get_dones(self):
        """Check which environments are terminated.

        run04: per-env early termination. Forge's reset steps physics globally, so
        historically all envs had to reset in sync (this returned all-true-or-all-
        false). _reset_idx now snapshots/restores the non-terminating envs around
        the global reset servo, so we can terminate envs independently here.
        `terminated` (true failures) and `time_out` (truncation) are returned
        separately so PPO bootstraps the timeout value but NOT the failure states.
        """
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminated = torch.zeros_like(time_out)
        if getattr(self.cfg, "terminate_on_explosion", False):
            # |vel| on ANY joint (arm + hand) over the limit => sim blew up.
            joint_speed = self.joint_vel.abs().amax(dim=1)
            exploded = joint_speed > self.cfg.max_joint_vel
            terminated = terminated | exploded
            self.extras["logs_term_explode"] = exploded.float().mean()
            self.extras["logs_joint_speed_max"] = joint_speed.max()
        if getattr(self.cfg, "terminate_on_nut_far", False):
            # nut-center to bolt-tip distance (both env-relative). The bolt is
            # fixed, so fixed_pos_obs_frame stays valid through the episode.
            nut_dist = torch.linalg.vector_norm(self.held_pos - self.fixed_pos_obs_frame, dim=-1)
            far = nut_dist > self.cfg.max_nut_bolt_dist
            terminated = terminated | far
            self.extras["logs_term_nutfar"] = far.float().mean()
            self.extras["logs_nut_bolt_dist"] = nut_dist.mean()
        if getattr(self.cfg, "terminate_on_hand_far", False):
            # thumb/index midpoint to nut-center (both env-relative, matching the
            # run03 contact-reward convention). Captures the hand wandering off the
            # nut — the nut-far guard above can't, since the bolt shaft pins the nut.
            thumb = self._robot.data.body_pos_w[:, self.thumb_body_idx] - self.scene.env_origins
            index = self._robot.data.body_pos_w[:, self.index_body_idx] - self.scene.env_origins
            hand_dist = torch.linalg.vector_norm(0.5 * (thumb + index) - self.held_pos, dim=-1)
            hand_far = hand_dist > self.cfg.max_hand_nut_dist
            terminated = terminated | hand_far
            self.extras["logs_term_handfar"] = hand_far.float().mean()
            self.extras["logs_hand_nut_dist"] = hand_dist.mean()

        return terminated, time_out

    def _get_curr_successes(self, success_threshold, check_rot=False):
        """Get success mask at current timestep."""
        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        held_base_pos, held_base_quat = forge_kuka_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        target_held_base_pos, target_held_base_quat = forge_kuka_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device,
        )

        xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
        z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]

        is_centered = torch.where(xy_dist < 0.0025, torch.ones_like(curr_successes), torch.zeros_like(curr_successes))
        # Height threshold to target
        fixed_cfg = self.cfg_task.fixed_asset_cfg
        if self.cfg_task.name == "peg_insert" or self.cfg_task.name == "gear_mesh":
            height_threshold = fixed_cfg.height * success_threshold
        elif self.cfg_task.name == "nut_thread":
            height_threshold = fixed_cfg.thread_pitch * success_threshold
        else:
            raise NotImplementedError("Task not implemented")
        is_close_or_below = torch.where(
            z_disp < height_threshold, torch.ones_like(curr_successes), torch.zeros_like(curr_successes)
        )
        curr_successes = torch.logical_and(is_centered, is_close_or_below)

        if check_rot:
            _, _, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
            curr_yaw = forge_kuka_utils.wrap_yaw(curr_yaw)
            is_rotated = curr_yaw < self.cfg_task.ee_success_yaw
            curr_successes = torch.logical_and(curr_successes, is_rotated)

        return curr_successes

    def _log_factory_metrics(self, rew_dict, rew_scales, curr_successes):
        """Keep track of episode statistics and log rewards."""
        # Only log episode success rates at the end of an episode.
        if torch.any(self.reset_buf):
            self.extras["successes"] = torch.count_nonzero(curr_successes) / self.num_envs

        # Get the time at which an episode first succeeds.
        first_success = torch.logical_and(curr_successes, torch.logical_not(self.ep_succeeded))
        self.ep_succeeded[curr_successes] = 1

        first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
        self.ep_success_times[first_success_ids] = self.episode_length_buf[first_success_ids]
        nonzero_success_ids = self.ep_success_times.nonzero(as_tuple=False).squeeze(-1)

        if len(nonzero_success_ids) > 0:  # Only log for successful episodes.
            success_times = self.ep_success_times[nonzero_success_ids].sum() / len(nonzero_success_ids)
            self.extras["success_times"] = success_times

        # Log the SIGNED contribution each term makes to the total reward
        # (raw_term * scale), not the raw magnitude — penalties read negative,
        # and a term whose scale is 0 correctly reads 0 (i.e. disabled).
        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = (rew * rew_scales[rew_name]).mean()

    def _get_factory_rewards(self):
        """Update rewards and compute success statistics (Factory base reward)."""
        # Get successful and failed envs at current timestep
        check_rot = self.cfg_task.name == "nut_thread"
        curr_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )

        rew_dict, rew_scales = self._get_factory_rew_dict(curr_successes)

        rew_buf = torch.zeros_like(rew_dict["kp_coarse"])
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]

        self.prev_actions = self.actions.clone()

        self._log_factory_metrics(rew_dict, rew_scales, curr_successes)
        return rew_buf

    def _get_factory_rew_dict(self, curr_successes):
        """Compute reward terms at current timestep."""
        rew_dict, rew_scales = {}, {}

        # Compute pos of keypoints on held asset, and fixed asset in world frame
        held_base_pos, held_base_quat = forge_kuka_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        target_held_base_pos, target_held_base_quat = forge_kuka_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device,
        )

        keypoints_held = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        keypoints_fixed = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
        offsets = forge_kuka_utils.get_keypoint_offsets(self.cfg_task.num_keypoints, self.device)
        keypoint_offsets = offsets * self.cfg_task.keypoint_scale
        for idx, keypoint_offset in enumerate(keypoint_offsets):
            keypoints_held[:, idx] = torch_utils.tf_combine(
                held_base_quat,
                held_base_pos,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]
            keypoints_fixed[:, idx] = torch_utils.tf_combine(
                target_held_base_quat,
                target_held_base_pos,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                keypoint_offset.repeat(self.num_envs, 1),
            )[1]
        keypoint_dist = torch.norm(keypoints_held - keypoints_fixed, p=2, dim=-1).mean(-1)

        a0, b0 = self.cfg_task.keypoint_coef_baseline
        a1, b1 = self.cfg_task.keypoint_coef_coarse
        a2, b2 = self.cfg_task.keypoint_coef_fine
        # Action penalties.
        action_penalty_ee = torch.norm(self.actions, p=2)
        action_grad_penalty = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)
        curr_engaged = self._get_curr_successes(success_threshold=self.cfg_task.engage_threshold, check_rot=False)

        rew_dict = {
            "kp_baseline": forge_kuka_utils.squashing_fn(keypoint_dist, a0, b0),
            "kp_coarse": forge_kuka_utils.squashing_fn(keypoint_dist, a1, b1),
            "kp_fine": forge_kuka_utils.squashing_fn(keypoint_dist, a2, b2),
            "action_penalty_ee": action_penalty_ee,
            "action_grad_penalty": action_grad_penalty,
            "curr_engaged": curr_engaged.float(),
            "curr_success": curr_successes.float(),
        }
        rew_scales = {
            "kp_baseline": 1.0,
            "kp_coarse": 1.0,
            "kp_fine": 1.0,
            "action_penalty_ee": -self.cfg_task.action_penalty_ee_scale,
            "action_grad_penalty": -self.cfg_task.action_grad_penalty_scale,
            "curr_engaged": 1.0,
            "curr_success": 1.0,
        }

        # run07: gate the nut-position reward on a LIVE proper grip (thumb pressing
        # the nut AND perpendicular, recomputed each step in _get_rewards). Until
        # then kp_* and curr_engaged pay 0 — otherwise the policy banks ~+1.4/step
        # for the reset-placed nut without gripping. Live (not latched) so it can't
        # tap-and-drift (the run06 loophole). curr_success / penalties stay ungated.
        # One-step delayed (proper_grip updated later this step) — harmless.
        if getattr(self.cfg, "gate_kp_on_pinch", False):
            gate = self.proper_grip.float()
            for _k in ("kp_baseline", "kp_coarse", "kp_fine", "curr_engaged"):
                rew_dict[_k] = rew_dict[_k] * gate

        return rew_dict, rew_scales

    def _get_rewards(self):
        """FORGE reward includes a contact penalty and success prediction error."""
        # Use same base rewards as Factory.
        rew_buf = self._get_factory_rewards()

        rew_dict, rew_scales = {}, {}
        # Calculate action penalty for the asset-relative action space.
        pos_error = torch.norm(self.delta_pos, p=2, dim=-1) / self.cfg.ctrl.pos_action_threshold[0]
        rot_error = torch.abs(self.delta_yaw) / self.cfg.ctrl.rot_action_threshold[0]
        # Contact penalty.
        contact_force = torch.norm(self.force_sensor_smooth[:, 0:3], p=2, dim=-1, keepdim=False)
        contact_penalty = torch.nn.functional.relu(contact_force - self.contact_penalty_thresholds)
        # Add success prediction rewards.
        check_rot = self.cfg_task.name == "nut_thread"
        true_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )
        policy_success_pred = (self.actions[:, 6] + 1) / 2  # rescale from [-1, 1] to [0, 1]
        success_pred_error = (true_successes.float() - policy_success_pred).abs()
        # Delay success prediction penalty until some successes have occurred.
        if true_successes.float().mean() >= self.cfg_task.delay_until_ratio:
            self.success_pred_scale = 1.0

        # Add new FORGE reward terms.
        rew_dict = {
            "action_penalty_asset": pos_error + rot_error,
            "contact_penalty": contact_penalty,
            "success_pred_error": success_pred_error,
        }
        rew_scales = {
            "action_penalty_asset": -self.cfg_task.action_penalty_asset_scale,
            "contact_penalty": -self.cfg_task.contact_penalty_scale,
            "success_pred_error": -self.success_pred_scale,
        }
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]

        # Stage-B grip reward: forge has NO term for holding the asset (its grasp is
        # hardcoded), so without this the policy never learns to pinch. Reward the
        # thumb+index fingertips being close to the nut, with a bonus when BOTH are
        # within contact distance (a held pinch). Geometry-based contact proxy.
        if getattr(self.cfg, "action_driven_fingers", False):
            thumb = self._robot.data.body_pos_w[:, self.thumb_body_idx] - self.scene.env_origins
            index = self._robot.data.body_pos_w[:, self.index_body_idx] - self.scene.env_origins
            # --- Stage 1: CONTACT. Shape on distance to the nut SURFACE (subtract
            # the nut radius), not its center — a real side grip sits ~radius from
            # the center, which the old center-distance proxy wrongly read as "off".
            nut_radius = 0.5 * self.cfg_task.held_asset_cfg.diameter
            ds_thumb = (torch.norm(thumb - self.held_pos, dim=-1) - nut_radius).clamp(min=0.0)
            ds_index = (torch.norm(index - self.held_pos, dim=-1) - nut_radius).clamp(min=0.0)

            if getattr(self.cfg, "use_contact_sensor", False):
                # run11: TRUE nut contact = FILTERED force_matrix_w (pad<->nut rigid-body
                # link) above threshold. Unlike run07's net_forces_w this is nut-exclusive
                # (finger-finger self-contact never involves the nut body), so the spatial
                # contact_context_dist gate is now redundant — kept as a conservative AND.
                f_thumb = self._thumb_contact.data.force_matrix_w.norm(dim=-1).reshape(self.num_envs, -1).sum(-1)
                f_index = self._index_contact.data.force_matrix_w.norm(dim=-1).reshape(self.num_envs, -1).sum(-1)
                thr = self.cfg.contact_force_threshold
                cdist = self.cfg.contact_context_dist
                thumb_touch = (f_thumb > thr) & (ds_thumb < cdist)
                index_touch = (f_index > thr) & (ds_index < cdist)
                self.extras["logs_contact_force"] = (f_thumb + f_index).mean()
            else:
                # Fallback proxy: surface distance under a small threshold.
                thumb_touch = ds_thumb < 0.01
                index_touch = ds_index < 0.01

            # run08: with the index actuated too, a proper grip is a TWO-finger pinch —
            # both pads pressing the nut. contact_f drives the bonus + perp reward.
            in_contact = thumb_touch & index_touch
            contact_f = in_contact.float()

            # --- Stage 2: PERPENDICULAR. grip line (thumb->index) perpendicular to the
            # nut axis. align = cos(angle): 0 = perpendicular (ideal), 1 = parallel.
            z_unit = torch.zeros_like(thumb)
            z_unit[:, 2] = 1.0
            nut_axis = torch.nn.functional.normalize(torch_utils.quat_apply(self.held_quat, z_unit), dim=-1)
            pinch_dir = torch.nn.functional.normalize(index - thumb, dim=-1)
            align = (pinch_dir * nut_axis).sum(-1)
            perp_ok = (align.abs() < self.cfg.pinch_perp_threshold).float()  # within tolerance band

            # run08: PROPER GRIP = BOTH fingers pinching the nut AND perpendicular. LIVE
            # (recomputed every step, NOT latched) so the policy can't tap-and-drift
            # like run06. Gates the nut-position reward (read in _get_factory_rew_dict).
            self.proper_grip = in_contact & (perp_ok > 0.5)

            # Sustained-grip bonus: reward staying in contact across consecutive steps.
            self.contact_steps = torch.where(in_contact, self.contact_steps + 1.0, torch.zeros_like(self.contact_steps))
            sustained = self.cfg.sustained_contact_bonus * self.contact_steps.clamp(max=self.cfg.sustained_contact_cap)
            contact_rew = (
                -self.cfg.contact_shaping_scale * (ds_thumb + ds_index)
                + self.cfg.contact_bonus * contact_f
                + sustained * contact_f
            )
            rew_buf = rew_buf + contact_rew

            # run10: hand->object approach reward — averaged ELASTOMER-PAD distance to
            # the nut CENTER (the actual colliding pads), driving the fingertips onto
            # the nut. Pairs with the EE-tilt reward (keeps the contact perpendicular).
            thumb_pad = self._robot.data.body_pos_w[:, self.thumb_pad_idx] - self.scene.env_origins
            index_pad = self._robot.data.body_pos_w[:, self.index_pad_idx] - self.scene.env_origins
            hand_obj_dist = 0.5 * (
                torch.norm(thumb_pad - self.held_pos, dim=-1) + torch.norm(index_pad - self.held_pos, dim=-1)
            )
            hand_obj_rew = -self.cfg.hand_obj_reward_scale * hand_obj_dist
            rew_buf = rew_buf + hand_obj_rew

            # perp reward, gated on REAL contact (now live, not the dead sensor) — this
            # is the gradient that pulls the hand toward a perpendicular side-press.
            perp_rew = contact_f * (
                -self.cfg.pinch_perp_reward_scale * align**2 + self.cfg.pinch_perp_bonus * perp_ok
            )
            rew_buf = rew_buf + perp_rew

            # run09: EE-TILT orientation reward — bias the hand tilt toward ~50 deg from
            # vertical (the band where a clean side-pinch forms), Gaussian over the band.
            # Gated on the hand being NEAR the nut so it guides the final-approach
            # orientation and can't be banked from afar. tilt = angle between the EE
            # z-axis and world vertical. NOTE: verify logs_ee_tilt_deg reads ~50 at a
            # good pinch; if the EE frame's z-axis isn't the approach axis, adjust here.
            ee_axis = torch_utils.quat_apply(self.fingertip_midpoint_quat, z_unit)
            tilt_deg = torch.rad2deg(torch.arccos(ee_axis[:, 2].abs().clamp(max=1.0)))
            near = (torch.minimum(ds_thumb, ds_index) < self.cfg.ee_tilt_gate_dist).float()
            ee_tilt_rew = near * self.cfg.ee_tilt_reward_scale * torch.exp(
                -0.5 * ((tilt_deg - self.cfg.ee_tilt_target_deg) / self.cfg.ee_tilt_band_deg) ** 2
            )
            rew_buf = rew_buf + ee_tilt_rew

            # --- Stage 3: TURN — the existing forge keypoint/success reward (above)
            # is left on; the policy naturally threads only once it grips correctly. ---
            self.extras["logs_rew_contact"] = contact_rew.mean()
            self.extras["logs_in_contact"] = contact_f.mean()
            self.extras["logs_contact_steps"] = self.contact_steps.mean()
            self.extras["logs_rew_perp"] = perp_rew.mean()
            self.extras["logs_pinch_perp_align"] = align.abs().mean()  # 0=perp (good), 1=parallel
            self.extras["logs_perp_ok"] = perp_ok.mean()
            self.extras["logs_rew_ee_tilt"] = ee_tilt_rew.mean()
            self.extras["logs_ee_tilt_deg"] = tilt_deg.mean()
            self.extras["logs_rew_hand_obj"] = hand_obj_rew.mean()
            self.extras["logs_hand_obj_dist"] = hand_obj_dist.mean()

        self._log_forge_metrics(rew_dict, rew_scales, policy_success_pred)
        self.extras["logs_rew_total"] = rew_buf.mean()
        self.extras["logs_ep_len"] = self.episode_length_buf.float().mean()
        return rew_buf

    # Per-env state that the reset servo rewrites full-width (or that needs episode
    # continuity); snapshotted/restored for non-terminating envs on a partial reset.
    _RESET_KEEP_ATTRS = (
        "actions", "prev_actions", "ema_factor",
        "task_prop_gains", "task_deriv_gains", "pos_threshold", "rot_threshold",
        "contact_penalty_thresholds", "dead_zone_thresholds",
        "force_sensor_world_smooth", "force_sensor_smooth", "flip_quats",
        "prev_joint_pos", "prev_fingertip_pos", "prev_fingertip_quat",
        "ee_angvel_fd", "ee_linvel_fd", "contact_steps", "proper_grip",
        "init_fixed_pos_obs_noise", "fixed_pos_obs_frame",
    )

    def _reset_idx(self, env_ids):
        """run04: supports PARTIAL resets (some envs early-terminated, others still
        running). The Factory reset below steps physics globally with gravity off
        and a grasp servo, which would corrupt the still-running envs. So on a
        partial reset we snapshot those 'keep' envs first and restore them after the
        servo. On a full reset (the common timeout case) this is a no-op and the
        original path runs unchanged."""
        snap = self._snapshot_keep_envs(env_ids) if env_ids.shape[0] < self.num_envs else None

        # run03: reset the sustained-contact counter for the resetting envs.
        if hasattr(self, "contact_steps"):
            self.contact_steps[env_ids] = 0.0
        # run07: clear the live proper-grip state for the resetting envs.
        if hasattr(self, "proper_grip"):
            self.proper_grip[env_ids] = False
        # Open the hand before placing the nut (closed again during the grasp loop).
        if getattr(self.cfg, "grasp_on_reset", False):
            self.frozen_hand_joint_pos = self.open_hand_joint_pos.clone()

        super()._reset_idx(env_ids)

        # --- Factory reset ---
        self._set_assets_to_default_pose(env_ids)
        self._set_robot_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        self.step_sim_no_action()
        self.randomize_initial_state(env_ids)

        # --- Forge reset additions ---
        # Compute initial action for correct EMA computation.
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        pos_actions = self.fingertip_midpoint_pos - fixed_pos_action_frame
        pos_action_bounds = torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device)
        pos_actions = pos_actions @ torch.diag(1.0 / pos_action_bounds)
        self.actions[:, 0:3] = self.prev_actions[:, 0:3] = pos_actions

        # Relative yaw to bolt.
        # run11: un-rotate by the configurable EE base (its conjugate), not a hardcoded
        # -pi roll, so the initial yaw action is computed correctly for a tilted base.
        ee_base = getattr(self.cfg, "ee_base_orn_euler", [np.pi, 0.0, 0.0])
        base_euler = torch.tensor(ee_base, device=self.device).repeat(self.num_envs, 1)
        unrot_quat = torch_utils.quat_conjugate(
            torch_utils.quat_from_euler_xyz(roll=base_euler[:, 0], pitch=base_euler[:, 1], yaw=base_euler[:, 2])
        )

        fingertip_quat_rel_bolt = torch_utils.quat_mul(unrot_quat, self.fingertip_midpoint_quat)
        fingertip_yaw_bolt = torch_utils.get_euler_xyz(fingertip_quat_rel_bolt)[-1]
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt > torch.pi / 2, fingertip_yaw_bolt - 2 * torch.pi, fingertip_yaw_bolt
        )
        fingertip_yaw_bolt = torch.where(
            fingertip_yaw_bolt < -torch.pi, fingertip_yaw_bolt + 2 * torch.pi, fingertip_yaw_bolt
        )

        yaw_action = (fingertip_yaw_bolt + np.deg2rad(180.0)) / np.deg2rad(270.0) * 2.0 - 1.0
        self.actions[:, 5] = self.prev_actions[:, 5] = yaw_action
        self.actions[:, 6] = self.prev_actions[:, 6] = -1.0
        # run09: start the actuated finger flexion (MCP_FE = actions[8:]) OPEN so the
        # fingers don't begin pre-curled and over-flex; the policy closes from open.
        # (CMC_AA at actions[7] stays neutral.)
        if getattr(self.cfg, "action_driven_fingers", False):
            self.actions[:, 8:] = self.prev_actions[:, 8:] = -1.0
        # run10: start the EE pre-tilted toward a side pinch (roll action) so it does
        # not have to slowly rotate there. Calibrated: roll ~ -0.7 -> ~30 deg tilt.
        # VERIFY ee_tilt_deg ~30 at reset; flip the sign if it tilts away from the nut.
        if getattr(self.cfg, "free_ee_orientation", False):
            self.actions[:, 3] = self.prev_actions[:, 3] = self.cfg.ee_init_roll_action

        # EMA randomization.
        ema_rand = torch.rand((self.num_envs, 1), dtype=torch.float32, device=self.device)
        ema_lower, ema_upper = self.cfg.ctrl.ema_factor_range
        self.ema_factor = ema_lower + ema_rand * (ema_upper - ema_lower)

        # Set initial gains for the episode.
        prop_gains = self.default_gains.clone()
        self.pos_threshold = self.default_pos_threshold.clone()
        self.rot_threshold = self.default_rot_threshold.clone()
        prop_gains = forge_kuka_utils.get_random_prop_gains(
            prop_gains, self.cfg.ctrl.task_prop_gains_noise_level, self.num_envs, self.device
        )
        self.pos_threshold = forge_kuka_utils.get_random_prop_gains(
            self.pos_threshold, self.cfg.ctrl.pos_threshold_noise_level, self.num_envs, self.device
        )
        self.rot_threshold = forge_kuka_utils.get_random_prop_gains(
            self.rot_threshold, self.cfg.ctrl.rot_threshold_noise_level, self.num_envs, self.device
        )
        self.task_prop_gains = prop_gains
        self.task_deriv_gains = forge_kuka_utils.get_deriv_gains(prop_gains)

        contact_rand = torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
        contact_lower, contact_upper = self.cfg.task.contact_penalty_threshold_range
        self.contact_penalty_thresholds = contact_lower + contact_rand * (contact_upper - contact_lower)

        self.dead_zone_thresholds = (
            torch.rand((self.num_envs, 6), dtype=torch.float32, device=self.device) * self.default_dead_zone
        )

        self.force_sensor_world_smooth[:, :] = 0.0

        self.flip_quats = torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        rand_flips = torch.rand(self.num_envs) > 0.5
        self.flip_quats[rand_flips] = -1.0

        # run04: undo the global servo's effect on the still-running envs.
        if snap is not None:
            self._restore_keep_envs(snap)

    def _snapshot_keep_envs(self, reset_ids):
        """Snapshot the full per-env state of the envs NOT being reset, so the
        global reset servo (which steps all envs with gravity off) can be undone for
        them. Returns None-able dict consumed by _restore_keep_envs."""
        keep = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        keep[reset_ids] = False
        keep_ids = keep.nonzero(as_tuple=False).squeeze(-1)
        snap = {
            "ids": keep_ids,
            "joint_pos": self._robot.data.joint_pos[keep_ids].clone(),
            "joint_vel": self._robot.data.joint_vel[keep_ids].clone(),
            "held_pose": self._held_asset.data.root_state_w[keep_ids, 0:7].clone(),
            "held_vel": self._held_asset.data.root_state_w[keep_ids, 7:13].clone(),
            "book": {
                name: getattr(self, name)[keep_ids].clone()
                for name in self._RESET_KEEP_ATTRS
                if hasattr(self, name) and torch.is_tensor(getattr(self, name))
            },
        }
        return snap

    def _restore_keep_envs(self, snap):
        """Write the snapshotted keep-env state back into sim and the data buffers,
        erasing the global servo's drift on the still-running envs. write_*_to_sim
        updates both PhysX and the .data buffers, so no extra sim.step is needed."""
        keep_ids = snap["ids"]
        self._robot.write_joint_state_to_sim(snap["joint_pos"], snap["joint_vel"], env_ids=keep_ids)
        self._held_asset.write_root_pose_to_sim(snap["held_pose"], env_ids=keep_ids)
        self._held_asset.write_root_velocity_to_sim(snap["held_vel"], env_ids=keep_ids)
        for name, val in snap["book"].items():
            getattr(self, name)[keep_ids] = val
        self.scene.write_data_to_sim()
        self._compute_intermediate_values(dt=self.physics_dt)

    def _set_assets_to_default_pose(self, env_ids):
        """Move assets to default pose before randomization."""
        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_state[:, 0:3] += self.scene.env_origins[env_ids]
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

    def set_pos_inverse_kinematics(
        self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, env_ids
    ):
        """Set robot arm joint position using DLS IK."""
        ik_time = 0.0
        while ik_time < 0.25:
            # Compute error to target.
            pos_error, axis_angle_error = forge_kuka_control.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # Solve DLS problem.
            delta_dof_pos = forge_kuka_control.get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=self.fingertip_midpoint_jacobian[env_ids],
                device=self.device,
            )
            self.joint_pos[env_ids[:, None], self.arm_joint_ids_t] += delta_dof_pos[:, 0 : self.num_arm_dofs]
            self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_pos[env_ids,])

            self.ctrl_target_joint_pos[env_ids[:, None], self.arm_joint_ids_t] = self.joint_pos[
                env_ids[:, None], self.arm_joint_ids_t
            ]
            # Update dof state.
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            # Simulate and update tensors.
            self.step_sim_no_action()
            ik_time += self.physics_dt

        return pos_error, axis_angle_error

    def get_handheld_asset_relative_pose(self):
        """Get default relative pose between held asset and fingertip."""
        if self.cfg_task.name == "peg_insert":
            held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
            held_asset_relative_pos[:, 2] = self.cfg_task.held_asset_cfg.height
            held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length
        elif self.cfg_task.name == "gear_mesh":
            held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
            gear_base_offset = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset
            held_asset_relative_pos[:, 0] += gear_base_offset[0]
            held_asset_relative_pos[:, 2] += gear_base_offset[2]
            held_asset_relative_pos[:, 2] += self.cfg_task.held_asset_cfg.height / 2.0 * 1.1
        elif self.cfg_task.name == "nut_thread":
            held_asset_relative_pos = forge_kuka_utils.get_held_base_pos_local(
                self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
            )
        else:
            raise NotImplementedError("Task not implemented")

        held_asset_relative_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        if self.cfg_task.name == "nut_thread":
            # Rotate along z-axis of frame for default position.
            initial_rot_deg = self.cfg_task.held_asset_rot_init
            rot_yaw_euler = torch.tensor([0.0, 0.0, initial_rot_deg * np.pi / 180.0], device=self.device).repeat(
                self.num_envs, 1
            )
            held_asset_relative_quat = torch_utils.quat_from_euler_xyz(
                roll=rot_yaw_euler[:, 0], pitch=rot_yaw_euler[:, 1], yaw=rot_yaw_euler[:, 2]
            )

        return held_asset_relative_pos, held_asset_relative_quat

    def _set_robot_to_default_pose(self, joints, env_ids):
        """Return the arm to a default joint position; hold the hand frozen."""
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        # Freeze the SHARPA hand at its default posture.
        joint_pos[:, self.hand_joint_ids] = self.frozen_hand_joint_pos[env_ids]
        # Set the arm to the requested reset pose.
        joint_pos[:, self.arm_joint_ids] = torch.tensor(joints, device=self.device)[None, :]
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        self.step_sim_no_action()

    def step_sim_no_action(self):
        """Step the simulation without an action. Used for resets only.

        This method should only be called during resets when all environments
        reset at the same time.
        """
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)

    def _calibrate_grasp_frame(self):
        """Derive the palm-anchored grasp frame and store it as a fixed flange offset.

        With the pinch closed, build a frame at the thumb/index pad midpoint whose
        z-axis is the approach direction (flange -> pads) and x-axis is the pinch
        closing direction (thumb -> index). Express it as a constant transform from
        the flange so the control frame is rigid / finger-independent thereafter.
        """
        flange_pos = self._robot.data.body_pos_w[:, self.flange_body_idx]
        flange_quat = self._robot.data.body_quat_w[:, self.flange_body_idx]
        thumb = self._robot.data.body_pos_w[:, self.thumb_body_idx]
        index = self._robot.data.body_pos_w[:, self.index_body_idx]
        grasp_origin = 0.5 * (thumb + index)

        z_axis = torch.nn.functional.normalize(grasp_origin - flange_pos, dim=-1)  # approach
        x_axis = torch.nn.functional.normalize(index - thumb, dim=-1)  # pinch closing axis
        y_axis = torch.nn.functional.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1)
        x_axis = torch.cross(y_axis, z_axis, dim=-1)  # re-orthogonalize
        rot = torch.stack([x_axis, y_axis, z_axis], dim=-1)  # (E,3,3), columns = axes
        grasp_quat_w = quat_from_matrix(rot)

        self.grasp_offset_pos = torch_utils.quat_rotate_inverse(flange_quat, grasp_origin - flange_pos)
        self.grasp_offset_quat = torch_utils.quat_mul(torch_utils.quat_conjugate(flange_quat), grasp_quat_w)
        carb.log_info("forge_kuka: calibrated palm-anchored grasp frame.")

    def randomize_initial_state(self, env_ids):
        """Randomize initial state and perform any episode-level randomization."""
        # Disable gravity.
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        # NOTE: palm-anchored grasp-frame calibration is disabled (introduced an IK
        # divergence; convention/sign bug in the hand-rolled frame math). Falls back
        # to the flange-orientation + live pad-midpoint frame. Revisit deliberately.

        # (1.) Randomize fixed asset pose.
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        # (1.a.) Position
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
        fixed_asset_init_pos_rand = torch.tensor(
            self.cfg_task.fixed_asset_init_pos_noise, dtype=torch.float32, device=self.device
        )
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]
        # (1.b.) Orientation
        fixed_orn_init_yaw = np.deg2rad(self.cfg_task.fixed_asset_init_orn_deg)
        fixed_orn_yaw_range = np.deg2rad(self.cfg_task.fixed_asset_init_orn_range_deg)
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        fixed_orn_euler[:, 0:2] = 0.0  # Only change yaw.
        fixed_orn_quat = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0], fixed_orn_euler[:, 1], fixed_orn_euler[:, 2]
        )
        fixed_state[:, 3:7] = fixed_orn_quat
        # (1.c.) Velocity
        fixed_state[:, 7:] = 0.0  # vel
        # (1.d.) Update values.
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

        # (1.e.) Noisy position observation.
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        # run04: scope to env_ids — fixed_asset_pos_noise has len(env_ids) rows, so a
        # bare [:] assignment would shape-mismatch on a partial reset.
        self.init_fixed_pos_obs_noise[env_ids] = fixed_asset_pos_noise

        self.step_sim_no_action()

        # Compute the frame on the bolt that would be used as observation: fixed_pos_obs_frame
        # For example, the tip of the bolt can be used as the observation frame
        fixed_tip_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height
        if self.cfg_task.name == "gear_mesh":
            fixed_tip_pos_local[:, 0] = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset[0]

        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat,
            self.fixed_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            fixed_tip_pos_local,
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        # (2) Move arm to a location above the fixed asset.
        # Fixed-pose mode: skip the random IK servo entirely. The arm stays at the
        # manually-set reset_joints (placed by _set_robot_to_default_pose), so the
        # hand spawns at a single, hand-tuned pose every reset. Tune reset_joints on
        # the livestream until the hand holds the nut coaxially over the bolt.
        if getattr(self.cfg, "fixed_arm_pose_reset", False):
            pass  # arm already at reset_joints; nut is placed relative to it below
        else:
            # (a) get position vector to target
            bad_envs = env_ids.clone()
            ik_attempt = 0

            hand_down_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
            while True:
                n_bad = bad_envs.shape[0]

                above_fixed_pos = fixed_tip_pos.clone()
                # run11: full (x,y,z) offset from the bolt tip (was z-only / coaxial)
                # so the IK servo can target an off-axis pinch pose. x,y default 0 for
                # the other tasks, so this is a no-op there.
                above_fixed_pos += torch.tensor(self.cfg_task.hand_init_pos, device=self.device)

                rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
                above_fixed_pos_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
                hand_init_pos_rand = torch.tensor(self.cfg_task.hand_init_pos_noise, device=self.device)
                above_fixed_pos_rand = above_fixed_pos_rand @ torch.diag(hand_init_pos_rand)
                above_fixed_pos[bad_envs] += above_fixed_pos_rand

                # (b) get random orientation facing down
                hand_down_euler = (
                    torch.tensor(self.cfg_task.hand_init_orn, device=self.device).unsqueeze(0).repeat(n_bad, 1)
                )

                rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
                above_fixed_orn_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
                hand_init_orn_rand = torch.tensor(self.cfg_task.hand_init_orn_noise, device=self.device)
                above_fixed_orn_noise = above_fixed_orn_noise @ torch.diag(hand_init_orn_rand)
                hand_down_euler += above_fixed_orn_noise
                hand_down_quat[bad_envs, :] = torch_utils.quat_from_euler_xyz(
                    roll=hand_down_euler[:, 0], pitch=hand_down_euler[:, 1], yaw=hand_down_euler[:, 2]
                )

                # (c) iterative IK Method
                pos_error, aa_error = self.set_pos_inverse_kinematics(
                    ctrl_target_fingertip_midpoint_pos=above_fixed_pos,
                    ctrl_target_fingertip_midpoint_quat=hand_down_quat,
                    env_ids=bad_envs,
                )
                pos_error = torch.linalg.norm(pos_error, dim=1) > 1e-3
                angle_error = torch.norm(aa_error, dim=1) > 1e-3
                any_error = torch.logical_or(pos_error, angle_error)
                bad_envs = bad_envs[any_error.nonzero(as_tuple=False).squeeze(-1)]

                # Check IK succeeded for all envs, otherwise try again for those envs
                if bad_envs.shape[0] == 0:
                    break

                # Reachability is not tuned to the iiwa14 this iteration; cap the
                # retries so an unreachable sampled pose can't hang the reset
                # (tracked in todos.md). Proceed best-effort after the cap.
                if ik_attempt >= self.cfg.ctrl.max_ik_attempts:
                    carb.log_warn(
                        f"forge_kuka: IK did not converge for {bad_envs.shape[0]} env(s) after "
                        f"{ik_attempt} attempts; proceeding best-effort (workspace untuned)."
                    )
                    break

                self._set_robot_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=bad_envs)

                ik_attempt += 1

        self.step_sim_no_action()

        # Add flanking gears after servo (so arm doesn't move them).
        if self.cfg_task.name == "gear_mesh" and self.cfg_task.add_flanking_gears:
            small_gear_state = self._small_gear_asset.data.default_root_state.clone()[env_ids]
            small_gear_state[:, 0:7] = fixed_state[:, 0:7]
            small_gear_state[:, 7:] = 0.0  # vel
            self._small_gear_asset.write_root_pose_to_sim(small_gear_state[:, 0:7], env_ids=env_ids)
            self._small_gear_asset.write_root_velocity_to_sim(small_gear_state[:, 7:], env_ids=env_ids)
            self._small_gear_asset.reset()

            large_gear_state = self._large_gear_asset.data.default_root_state.clone()[env_ids]
            large_gear_state[:, 0:7] = fixed_state[:, 0:7]
            large_gear_state[:, 7:] = 0.0  # vel
            self._large_gear_asset.write_root_pose_to_sim(large_gear_state[:, 0:7], env_ids=env_ids)
            self._large_gear_asset.write_root_velocity_to_sim(large_gear_state[:, 7:], env_ids=env_ids)
            self._large_gear_asset.reset()

        # (3) Randomize asset-in-gripper location.
        # flip gripper z orientation
        flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
            q1=self.fingertip_midpoint_quat,
            t1=self.fingertip_midpoint_pos,
            q2=flip_z_quat,
            t2=torch.zeros((self.num_envs, 3), device=self.device),
        )

        # get default gripper in asset transform
        held_asset_relative_pos, held_asset_relative_quat = self.get_handheld_asset_relative_pose()
        asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(
            held_asset_relative_quat, held_asset_relative_pos
        )

        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=fingertip_flipped_quat, t1=fingertip_flipped_pos, q2=asset_in_hand_quat, t2=asset_in_hand_pos
        )

        # Add asset in hand randomization
        rand_sample = torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
        held_asset_pos_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
        if self.cfg_task.name == "gear_mesh":
            held_asset_pos_noise[:, 2] = -rand_sample[:, 2]  # [-1, 0]

        held_asset_pos_noise_level = torch.tensor(self.cfg_task.held_asset_pos_noise, device=self.device)
        held_asset_pos_noise = held_asset_pos_noise @ torch.diag(held_asset_pos_noise_level)
        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=translated_held_asset_quat,
            t1=translated_held_asset_pos,
            q2=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            t2=held_asset_pos_noise,
        )

        held_state = self._held_asset.data.default_root_state.clone()
        held_state[:, 0:3] = translated_held_asset_pos + self.scene.env_origins
        held_state[:, 3:7] = translated_held_asset_quat
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
        self._held_asset.reset()

        # Pinch variant: place the nut ON the bolt (coaxial, hole around the bolt
        # tip) rather than free in the gripper. The bolt shaft constrains the nut
        # laterally so it cannot fly off no matter how marginal the pinch is; the
        # fingers then only need to grip enough to TURN it down the thread. The hand
        # servos so the pinch center sits at the bolt tip, around the nut.
        if getattr(self.cfg, "grasp_on_reset", False):
            held_state = self._held_asset.data.default_root_state.clone()
            held_state[:, 0:3] = self.fixed_pos_obs_frame + self.scene.env_origins  # bolt tip
            # Lower the nut onto the shaft so the bolt passes through the hole and
            # actually constrains it (otherwise it sits at the tip and floats off).
            held_state[:, 2] -= 0.012
            held_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
            held_state[:, 7:] = 0.0
            self._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
            self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
            self._held_asset.reset()

        # Settle the EE in place. The SHARPA hand stays frozen (no gripper to
        # close); generate_ctrl_signals re-commands the frozen hand posture.
        # Set gains to use for quick resets.
        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.task_prop_gains = reset_task_prop_gains
        self.task_deriv_gains = forge_kuka_utils.get_deriv_gains(
            reset_task_prop_gains, self.cfg.ctrl.reset_rot_deriv_scale
        )

        self.step_sim_no_action()

        # Close the pinch onto the nut: switch the held posture OPEN -> CLOSED, then
        # let the grasp loop drive the fingers shut. Because the nut was placed
        # between the open pads, the fingers stop at contact instead of ejecting it.
        if getattr(self.cfg, "grasp_on_reset", False):
            self.frozen_hand_joint_pos = self.closed_hand_joint_pos.clone()

        # run11: when spawning at a fixed arm pose, PIN the arm joints through the
        # grasp-close so the finger-nut contact reaction can't drift the (gain-zeroed,
        # compliant) arm off the manual pinch config. Capture the intended arm pose and
        # re-write the arm joint state each substep.
        pin_arm = getattr(self.cfg, "fixed_arm_pose_reset", False)
        if pin_arm:
            arm_hold_pos = torch.tensor(self.cfg.ctrl.reset_joints, device=self.device).repeat(self.num_envs, 1)
            arm_hold_vel = torch.zeros_like(arm_hold_pos)

        grasp_time = 0.0
        while grasp_time < 0.25:
            self.close_gripper_in_place()
            self.step_sim_no_action()
            if pin_arm:
                self._robot.write_joint_state_to_sim(arm_hold_pos, arm_hold_vel, joint_ids=self.arm_joint_ids)
            grasp_time += self.sim.get_physics_dt()

        # Pinch demo: the grasp loop just closed the fingers, so the pad midpoint is
        # the true grasp center. Drop the nut there, reopen the fingers, and store
        # the EE pose to hold. The episode (_apply_action) then ramps the fingers
        # closed again -- visibly -- so we can watch them pinch the nut.
        if getattr(self.cfg, "pinch_demo", False):
            grasp_center = 0.5 * (
                self._robot.data.body_pos_w[:, self.thumb_body_idx]
                + self._robot.data.body_pos_w[:, self.index_body_idx]
            )
            held_state = self._held_asset.data.default_root_state.clone()
            held_state[:, 0:3] = grasp_center
            held_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
            held_state[:, 7:] = 0.0
            self._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
            self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
            self._held_asset.reset()
            self.frozen_hand_joint_pos = self.open_hand_joint_pos.clone()
            self.step_sim_no_action()
            self._demo_hold_pos = self.fingertip_midpoint_pos.clone()
            self._demo_hold_quat = self.fingertip_midpoint_quat.clone()

        self.prev_joint_pos = self.joint_pos[:, self.arm_joint_ids].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)

        # Zero initial velocity.
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        # Set initial gains for the episode.
        self.task_prop_gains = self.default_gains
        self.task_deriv_gains = forge_kuka_utils.get_deriv_gains(self.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))

    def _log_forge_metrics(self, rew_dict, rew_scales, policy_success_pred):
        """Log metrics to evaluate success prediction performance."""
        # Signed contribution (raw_term * scale), matching _log_factory_metrics.
        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = (rew * rew_scales[rew_name]).mean()

        for thresh, first_success_tx in self.first_pred_success_tx.items():
            curr_predicted_success = policy_success_pred > thresh
            first_success_idxs = torch.logical_and(curr_predicted_success, first_success_tx == 0)

            first_success_tx[:] = torch.where(first_success_idxs, self.episode_length_buf, first_success_tx)

            # Only log at the end.
            if torch.any(self.reset_buf):
                # Log prediction delay.
                delay_ids = torch.logical_and(self.ep_success_times != 0, first_success_tx != 0)
                delay_times = (first_success_tx[delay_ids] - self.ep_success_times[delay_ids]).sum() / delay_ids.sum()
                if delay_ids.sum().item() > 0:
                    self.extras[f"early_term_delay_all/{thresh}"] = delay_times

                correct_delay_ids = torch.logical_and(delay_ids, first_success_tx > self.ep_success_times)
                correct_delay_times = (
                    first_success_tx[correct_delay_ids] - self.ep_success_times[correct_delay_ids]
                ).sum() / correct_delay_ids.sum()
                if correct_delay_ids.sum().item() > 0:
                    self.extras[f"early_term_delay_correct/{thresh}"] = correct_delay_times.item()

                # Log early-term success rate (for all episodes we have "stopped", did we succeed?).
                pred_success_idxs = first_success_tx != 0  # Episodes which we have predicted success.

                true_success_preds = torch.logical_and(
                    self.ep_success_times[pred_success_idxs] > 0,  # Success has actually occurred.
                    self.ep_success_times[pred_success_idxs]
                    < first_success_tx[pred_success_idxs],  # Success occurred before we predicted it.
                )

                num_pred_success = pred_success_idxs.sum().item()
                et_prec = true_success_preds.sum() / num_pred_success
                if num_pred_success > 0:
                    self.extras[f"early_term_precision/{thresh}"] = et_prec

                true_success_idxs = self.ep_success_times > 0
                num_true_success = true_success_idxs.sum().item()
                et_recall = true_success_preds.sum() / num_true_success
                if num_true_success > 0:
                    self.extras[f"early_term_recall/{thresh}"] = et_recall
