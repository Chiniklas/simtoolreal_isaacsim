# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""forge_kuka: environment configuration.

Merged from the vendored ``factory_env_cfg`` and ``forge_env_cfg`` modules
into one flat configclass hierarchy, with the Franka FR3 robot replaced by
the KUKA iiwa14 + left SHARPA hand.

Morphology notes (see the design spec for full rationale):
- The arm is 7-DoF, exactly like the Franka arm, so the action / obs / state
  dimensions are unchanged and the rl_games recipes carry over verbatim.
- ``reset_joints`` and ``default_dof_pos_tensor`` are the iiwa14 arm poses
  (the SHARPA fingers are frozen, so they are not part of the EE controller).
- The robot is built from ``KUKA_SHARPA_CFG``; the env zeroes the arm-joint
  PD gains at runtime so the EE-impedance torque fully controls the arm,
  while the hand keeps its reference gains to hold the frozen posture.
"""

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from simtoolreal_lab.assets.kuka_sharpa_forge import KUKA_SHARPA_CFG

from .forge_kuka_events import randomize_dead_zone
from .forge_kuka_tasks_cfg import (
    ForgeKukaGearMesh,
    ForgeKukaNutThread,
    ForgeKukaNutThreadDeterministic,
    ForgeKukaPegInsert,
    ForgeTask,
)

# iiwa14 "ready" arm pose (7-DoF). This is an empirically-validated config in
# which the EE (thumb+index fingertip midpoint) hovers above the factory fixed
# asset at ~(0.6, 0, 0.1) with the hand pointing down — taken from a converged
# IK solution in the factory layout. Starting resets from here (vs the KUKA
# standalone pose, which sat joint-1 at -90deg) puts every env's reset IK in the
# correct solution basin, so it converges reliably instead of ~1/4 failing.
KUKA_ARM_RESET_JOINTS = [0.73, 2.01, 1.15, 1.92, -0.47, 1.76, 1.72]

# Nut-thread needs a different EE target yaw (hand_init_orn yaw ~1.83 vs peg's 0),
# so the wrist must rotate ~105 deg further. The peg pose pins joint_7 at 1.72,
# near the A7 limit (+-3.05), leaving no room -> IK saturates and fails. Start
# joint_7 mid-range (0.0) so the DLS IK has headroom to reach the nut-thread
# orientation. (Per-task; the peg/gear pose above is untouched.) Tune in-sim.
# run11: user-tuned pinch pose read off the Physics Inspector (degrees -> radians):
# j1..7 = 64.6, 85.6, -64.5, -81.1, -11.0, 47.8, 150.8 deg. With the IK servo active
# (fixed_arm_pose_reset=False) this is the SEED for the servo, which then drives the
# EE to hand_init_pos/hand_init_orn before the impedance controller takes over.
KUKA_ARM_RESET_JOINTS_NUTTHREAD = [1.1275, 1.4940, -1.1257, -1.4155, -0.1920, 0.8343, 2.6320]

# Robot base pose in the (per-env) factory frame. The Franka forge task mounts
# its base at the world origin facing +x, with the table at (0.55, 0, 0) and the
# fixed asset at ~(0.6, 0, 0.05). KUKA_SHARPA_CFG ships a standalone-scene base
# pose of (0, 0.8, 0), which is wrong for this layout, so we override it here.
# This is the primary knob for step 1 (per-env placement) — tune in-sim.
KUKA_BASE_POS = (0.0, 0.0, 0.0)
KUKA_BASE_ROT = (1.0, 0.0, 0.0, 0.0)

# Stage-A "pre-pinched" frozen-hand posture (radians, per joint). Thumb is brought
# across and flexed to oppose the index; index flexes toward the thumb to form a
# pinch holding the nut; middle/ring/pinky are curled out of the way. Joints not
# listed default to 0 (open). FIRST-GUESS values from the URDF joint limits —
# calibrate in-sim so the thumb/index pads actually pinch the 24 mm M16 nut.
# Closed thumb+index pinch that holds the M16 nut. Values are the user's
# hand-tuned Isaac-Sim joint pose (read off the joint panel in degrees, converted
# to radians here). Opposition is via thumb_MCP_AA (its max ~20 deg) — earlier
# guesses used the wrong AA joints at out-of-limit values that got clamped, so the
# thumb never opposed. Middle/ring/pinky are curled out of the way.
# run11: user-tuned pinch posture read off the Physics Inspector (deg -> rad).
# Thumb + index + middle are the values the user scrubbed to; ring/pinky/middle_DIP
# were off-screen in the inspector panel so they keep the prior curl (out of the way).
PINCH_HAND_POSTURE = {
    "left_1_thumb_CMC_FE": 1.9199,  # 110.0 deg
    "left_thumb_CMC_AA": -0.3491,   # -20.0 deg
    "left_thumb_MCP_FE": 0.3473,    # 19.9 deg
    "left_thumb_MCP_AA": 0.3107,    # 17.8 deg
    "left_thumb_IP": 0.2513,        # 14.4 deg
    "left_2_index_MCP_FE": 0.0,     # 0.0 deg (actuated; runtime value set by policy)
    "left_index_MCP_AA": 0.0,       # 0.0 deg
    "left_index_PIP": 0.0,          # 0.0 deg
    "left_index_DIP": 0.3229,       # 18.5 deg
    # middle / ring / pinky: curled out of the way (not part of the thumb+index pinch)
    "left_3_middle_MCP_FE": 1.4, "left_middle_PIP": 1.5, "left_middle_DIP": 1.2,
    "left_4_ring_MCP_FE": 1.4, "left_ring_PIP": 1.5, "left_ring_DIP": 1.2,
    "left_5_pinky_CMC": 0.26, "left_pinky_MCP_FE": 1.4, "left_pinky_PIP": 1.5, "left_pinky_DIP": 1.2,
}

OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "force_threshold": 1,
    "ft_force": 3,
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
    "force_threshold": 1,
    "ft_force": 3,
}


@configclass
class ObsRandCfg:
    fixed_asset_pos = [0.001, 0.001, 0.001]
    # Forge additions.
    fingertip_pos = 0.00025
    fingertip_rot_deg = 0.1
    ft_force = 1.0


@configclass
class CtrlCfg:
    ema_factor = 0.2

    pos_action_bounds = [0.05, 0.05, 0.05]
    rot_action_bounds = [1.0, 1.0, 1.0]

    pos_action_threshold = [0.02, 0.02, 0.02]
    rot_action_threshold = [0.097, 0.097, 0.097]

    # iiwa14 arm reset pose (7-DoF). The SHARPA hand is frozen separately.
    reset_joints = KUKA_ARM_RESET_JOINTS
    reset_task_prop_gains = [300, 300, 300, 20, 20, 20]
    reset_rot_deriv_scale = 10.0
    default_task_prop_gains = [565.0, 565.0, 565.0, 28.0, 28.0, 28.0]

    # Null space parameters (iiwa14 arm, 7-DoF).
    default_dof_pos_tensor = KUKA_ARM_RESET_JOINTS
    kp_null = 10.0
    kd_null = 6.3246

    # Forge controller randomization.
    ema_factor_range = [0.025, 0.1]
    task_prop_gains_noise_level = [0.41, 0.41, 0.41, 0.41, 0.41, 0.41]
    pos_threshold_noise_level = [0.25, 0.25, 0.25]
    rot_threshold_noise_level = [0.29, 0.29, 0.29]
    default_dead_zone = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0]

    # Cap on the reset IK servo loop. The iiwa14 reach is not tuned to the
    # factory workspace this iteration, so bound the retries to avoid a hang
    # if a sampled grasp pose is unreachable (tracked in todos.md).
    max_ik_attempts = 10


@configclass
class NutThreadCtrlCfg(CtrlCfg):
    """Control cfg for the nut-thread tasks: same as the base, but the arm reset
    pose / null-space target give the wrist yaw headroom for the nut-thread EE
    orientation (see KUKA_ARM_RESET_JOINTS_NUTTHREAD)."""

    reset_joints = KUKA_ARM_RESET_JOINTS_NUTTHREAD
    default_dof_pos_tensor = KUKA_ARM_RESET_JOINTS_NUTTHREAD


@configclass
class EventCfg:
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "mass_distribution_params": (-0.005, 0.005),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    held_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    fixed_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("fixed_asset"),
            "static_friction_range": (0.25, 1.25),  # TODO: Set these values based on asset type.
            "dynamic_friction_range": (0.25, 0.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 128,
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    dead_zone_thresholds = EventTerm(
        func=randomize_dead_zone,
        mode="interval",
        interval_range_s=(2.0, 2.0),  # (0.25, 0.25)
    )


@configclass
class ForgeKukaEnvCfg(DirectRLEnvCfg):
    decimation = 8
    # 6-DoF EE action + 1 success-prediction action (Forge). Arm is 7-DoF so
    # the EE controller dimensions match the Franka forge task exactly.
    action_space = 7
    # num_*: will be overwritten to correspond to obs_order, state_order.
    observation_space = 21
    state_space = 72

    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "ft_force",
        "force_threshold",
    ]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
        "task_prop_gains",
        "ema_factor",
        "ft_force",
        "pos_threshold",
        "rot_threshold",
        "force_threshold",
    ]

    task_name: str = "peg_insert"  # peg_insert, gear_mesh, nut_thread
    task: ForgeTask = ForgeTask()
    obs_rand: ObsRandCfg = ObsRandCfg()
    ctrl: CtrlCfg = CtrlCfg()
    events: EventCfg = EventCfg()

    ft_smoothing_factor: float = 0.25

    # Frozen-hand posture override (joint_name -> radians). None => use the robot's
    # default hand pose (open). Set to a closed-pinch dict (e.g. PINCH_HAND_POSTURE)
    # for the Stage-A pre-pinched nut-thread variant. The listed joints are held at
    # these angles via position control; unlisted hand joints stay at default.
    frozen_hand_joint_pos: dict | None = None

    # Hand actuator stiffness override (None => keep KUKA_SHARPA_CFG reference
    # gains, ~4-13, far too soft to grip). Set higher (Franka-faithful clamp) so
    # the pinch actually holds the nut. Damping is derived as 0.1*stiffness.
    frozen_hand_stiffness: float | None = None

    # Fixed orientation offset (euler xyz, degrees) applied to the flange-anchored
    # grasp control frame, to fine-tune so the threading axis is exactly coaxial
    # with the bolt. (0,0,0) = use the flange orientation directly.
    grasp_frame_rot_offset_deg: list = [0.0, 0.0, 0.0]

    # If True, the last two actions drive the thumb pinch (CMC_AA + MCP_FE); the
    # policy must grip the nut, not just move the arm (Stage B). Requires
    # action_space to include the two extra finger actions.
    action_driven_fingers: bool = False

    # run08: also actuate the index MCP_FE (1 extra action) so the policy forms a real
    # two-finger pinch instead of pressing the thumb against a frozen index. Requires
    # action_space to include the extra index action.
    action_driven_index: bool = False

    # Weight on the pinch-line perpendicularity reward: penalize the thumb<->index
    # grip line being non-perpendicular to the nut's axis (so the arm holds the hand
    # so the fingers grip the nut's SIDES, threading-compatible). 0 disables it.
    pinch_perp_reward_scale: float = 1.0

    # Perpendicularity tolerance: |cos(angle between grip line and nut axis)| below
    # this counts as "perpendicular enough" (~within 20 deg of perpendicular,
    # i.e. ±20% of the 90 deg target) and earns pinch_perp_bonus — a band that
    # actively pulls the hand into a side grip instead of only penalizing.
    pinch_perp_threshold: float = 0.30
    pinch_perp_bonus: float = 0.5

    # If True, the policy controls the FULL EE orientation (roll/pitch unlocked), so
    # it can tilt the hand to a perpendicular side grip. If False, forge's hard
    # "hand straight down" constraint applies (roll/pitch forced to 0).
    free_ee_orientation: bool = False
    # run11: the EE neutral (zero-action) orientation, euler xyz. forge's Franka
    # default [pi,0,0] faces the hand straight DOWN; override per task to set a tilted
    # neutral so the controller rests at a pinch pose instead of pulling vertical.
    # yaw stays 0 here (the yaw action owns wrist rotation for threading).
    ee_base_orn_euler: list = [3.14159, 0.0, 0.0]

    # --- contact sensing (run07: net_forces_w on the elastomer pads) ---
    # If True, attach ContactSensors to the thumb/index ELASTOMER PADS (which carry
    # the colliders) and read net_forces_w (total contact force on the pad). NOT
    # filtered (force_matrix_w is GPU-dead for the SDF nut); nut-specificity comes
    # from contact_context_dist below.
    use_contact_sensor: bool = False
    contact_force_threshold: float = 0.5  # N; pad net force above this = "in contact"
    # run11: contact sensing reads FILTERED force_matrix_w (pad<->nut rigid-body link),
    # which is nut-exclusive and excludes finger-finger self-contact. No proxy/asset
    # change needed — the SDF nut supports filtered contact when the filter targets the
    # rigid-body LINK (not the wrapper/mesh). See forge_kuka_env._nut_body_filter_expr.
    # Reward weights for the contact stage.
    contact_shaping_scale: float = 2.0    # -scale * (surface dist of each tip to nut)
    contact_bonus: float = 1.0            # bonus when BOTH tips register nut contact
    sustained_contact_bonus: float = 0.02  # per-step bonus * consecutive-contact steps (capped)
    sustained_contact_cap: int = 50       # cap on the consecutive-step multiplier

    # run07: gate the nut-position reward (kp_baseline/coarse/fine + curr_engaged)
    # on a LIVE proper grip — those terms pay 0 unless the thumb is pressing the nut
    # AND the grip is perpendicular, recomputed every step (NOT latched). Without
    # this the policy banks ~+1.4/step for the reset-placed nut without gripping.
    gate_kp_on_pinch: bool = False
    # A "proper grip" = thumb pad net_forces_w > contact_force_threshold AND the
    # fingertip within contact_context_dist of the nut surface (spatial filter, since
    # net force isn't nut-exclusive) AND |pinch_align| < pinch_perp_threshold.
    contact_context_dist: float = 0.035  # m; max fingertip->nut surface dist to credit net force as nut contact (run08: 0.02->0.035)

    # run09: EE-tilt orientation reward — reward the hand tilt (angle of the EE z-axis
    # from world vertical) toward ee_tilt_target_deg (Gaussian, ~full reward over
    # 45-55 deg), gated on a fingertip being within ee_tilt_gate_dist of the nut.
    # Additive to the perp reward. 0 scale disables it.
    ee_tilt_reward_scale: float = 0.5
    ee_tilt_target_deg: float = 30.0  # run10: 50 was unreachable (hand maxes ~33 deg via the roll action)
    ee_tilt_band_deg: float = 7.0
    ee_tilt_gate_dist: float = 0.05  # m; only reward tilt when a fingertip is this close to the nut
    ee_init_roll_action: float = -0.7  # run10: reset roll action -> ~30 deg start tilt (calibrated)

    # run10: hand->object approach reward — averaged elastomer-pad distance to the nut
    # CENTER (drives the fingertips onto the nut). 0 disables it.
    hand_obj_reward_scale: float = 1.0

    # Pinch-demo mode: no bolt interaction. Nut spawns at the grasp-plane center;
    # fingers start open and ramp closed over pinch_demo_ramp_steps DURING the
    # (rendered) episode while the arm is held still -> you can watch the pinch and
    # see whether the fingers actually grip the nut.
    pinch_demo: bool = False
    pinch_demo_ramp_steps: int = 60

    # If True, kinematically lock the held nut to the bolt axis (coaxial), with its
    # height + yaw driven by the hand each step. "Rigidly attaches" the nut so it
    # can't fly off and the policy threads it by lowering + rotating the EE
    # (Stage-A turn training; bypasses the physical 2-finger grasp).
    rigid_nut_follow: bool = False

    # If True, skip the random IK servo at reset and leave the arm at the fixed
    # reset_joints pose (the hand spawns identically every reset, hand-tuned to hold
    # the nut coaxially over the bolt). The nut is still placed between the fingers.
    fixed_arm_pose_reset: bool = False

    # If True, run Franka's open->close grasp at reset: start the hand OPEN, place
    # the nut between the open pads, then close to frozen_hand_joint_pos during the
    # grasp-settle loop (contact stops the fingers -> no interpenetration/ejection).
    # If False, the hand is simply held at frozen_hand_joint_pos the whole time.
    grasp_on_reset: bool = False

    # Override the hand-joint effort (max force) limit. The SHARPA finger reference
    # limits are tiny (~0.2-3.3 Nm), so a stiff position target instantly saturates
    # them and the fingers neither fully close nor generate grip force. Raise it so
    # the pinch can reach the commanded pose and clamp (Franka gripper is ~40-87 Nm).
    frozen_hand_effort: float | None = None

    # Override the held-asset (nut) friction. factory's NutM16 uses 0.01 (tuned for
    # the Franka's 7500-stiffness parallel jaw + huge clamp force); a 2-fingertip
    # pinch can't generate that force, so a near-frictionless nut can't be held.
    # Raise it for the pinch variant. None => keep the task's configured friction.
    held_asset_friction_override: float | None = None

    # run04: per-env early-termination guards. forge resets all envs in sync, so
    # _reset_idx snapshots the non-terminating envs and restores them after the
    # (global, physics-stepping) reset servo runs — letting tripped envs reset
    # independently. See _get_dones / _snapshot_keep_envs / _restore_keep_envs.
    # run04: the ONLY active early-termination is nut-off-the-bolt. The hand is left
    # free to explore (no hand-far guard), and the explosion guard is off too.
    terminate_on_nut_far: bool = True     # nut drifted off the bolt (task unrecoverable)
    max_nut_bolt_dist: float = 0.05       # m; nut-center to bolt-tip distance above this -> terminate
    terminate_on_explosion: bool = False  # disabled: hand free to explore (sim-instability sentinel; re-enable if needed)
    max_joint_vel: float = 50.0           # rad/s (only used if terminate_on_explosion)
    terminate_on_hand_far: bool = False   # disabled: hand free to explore / not engaging is allowed
    max_hand_nut_dist: float = 0.10       # m (only used if terminate_on_hand_far)

    episode_length_s = 10.0  # Probably need to override.
    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,  # Important to avoid interpenetration.
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=1,  # Important for stable simulation.
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # replicate_physics=False: the 29-DoF KUKA-SHARPA articulation does not
    # survive clone_environments(copy_from_source=False) physics replication in
    # every env (only some robots appear). The maintained simtoolreal_sharpa task
    # runs this same asset with replicate_physics=False for the same reason. Each
    # env becomes an independent USD copy (slower at scale; fine for inspection,
    # revisit for large-scale training).
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, replicate_physics=False)

    # KUKA iiwa14 + left SHARPA hand. Arm PD gains are zeroed at runtime in the
    # env so the EE-impedance torque controls the arm; the hand keeps its
    # configured gains to hold the frozen posture. The base pose is overridden
    # from the asset's standalone-scene default to the factory layout (see
    # KUKA_BASE_POS above); ``.replace`` deep-copies so the global cfg is intact.
    robot = KUKA_SHARPA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot.init_state.pos = KUKA_BASE_POS
    robot.init_state.rot = KUKA_BASE_ROT


@configclass
class ForgeKukaTaskPegInsertCfg(ForgeKukaEnvCfg):
    task_name = "peg_insert"
    task = ForgeKukaPegInsert()
    episode_length_s = 10.0


@configclass
class ForgeKukaTaskGearMeshCfg(ForgeKukaEnvCfg):
    task_name = "gear_mesh"
    task = ForgeKukaGearMesh()
    episode_length_s = 20.0


@configclass
class ForgeKukaTaskNutThreadCfg(ForgeKukaEnvCfg):
    task_name = "nut_thread"
    task = ForgeKukaNutThread()
    ctrl: CtrlCfg = NutThreadCtrlCfg()  # wrist-yaw headroom for the nut-thread orientation
    episode_length_s = 30.0


@configclass
class ForgeKukaPinchNutThreadCfg(ForgeKukaTaskNutThreadCfg):
    """Stage-A nut-thread: nut pre-pinched in a closed thumb+index grip, others
    curled away. Same arm-EE threading control as the baseline; the only change
    is the frozen-hand posture (open -> closed pinch) so the hand statically holds
    the teleported nut while we validate the threading reward + control. No finger
    actuation in the action space yet (that is Stage B)."""

    task = ForgeKukaNutThreadDeterministic()  # no nut/bolt spawn randomization
    rigid_nut_follow = True  # nut locked to bolt axis, threaded by the EE (Stage-A turn training)
    # Deterministic IK servo (no randomness now) places the coaxial grasp frame
    # directly above the bolt -> nut spawns on top of the bolt, in the fingers.
    fixed_arm_pose_reset = False
    frozen_hand_joint_pos = PINCH_HAND_POSTURE
    frozen_hand_stiffness = None   # run04: revert the Kp=100/Kd=10 override -> native SHARPA finger gains (Kp~5-13, Kd~0.04-0.4) so the thumb DOFs are responsive, not over-damped
    frozen_hand_effort = 20.0      # lift the tiny finger effort ceiling so the pinch can close+clamp
    grasp_on_reset = True  # open -> place nut -> close onto it (avoids ejection)
    held_asset_friction_override = 1.0  # a 0.01-friction nut can't be fingertip-pinched


@configclass
class ForgeKukaPinchThreadCfg(ForgeKukaPinchNutThreadCfg):
    """Stage-B: policy controls the arm EE (6) + thumb pinch (CMC_AA, MCP_FE) and
    must physically pinch the nut on the bolt and turn it down the thread. Nut is
    physical on the bolt (constrained by the shaft), not kinematically locked."""

    action_space = 10  # 6 EE + success-pred(6) + thumb CMC_AA(7),MCP_FE(8) + index MCP_FE(9)
    # run11: spawn arm directly at the manual pinch config (KUKA_ARM_RESET_JOINTS_NUTTHREAD),
    # NO IK servo -> no hand_init_orn down-facing constraint. The grasp-close hold keeps
    # the manual orientation (close_gripper_in_place gated on free_ee_orientation).
    fixed_arm_pose_reset = True
    action_driven_fingers = True
    action_driven_index = True  # run08: two-finger pinch (thumb + index)
    rigid_nut_follow = False  # nut is physical on the bolt; the policy must grip to turn it
    free_ee_orientation = True  # unlock EE roll/pitch so the hand can tilt to a side grip
    ee_base_orn_euler = [1.95, 0.82, 0.0]  # run11: neutral = manual pinch tilt (roll 112deg, pitch 47deg), not face-down
    use_contact_sensor = True   # run07: net_forces_w on the elastomer pads (real contact)
    gate_kp_on_pinch = True     # run07: kp_* + curr_engaged pay only during a live proper grip


@configclass
class ForgeKukaPinchDemoCfg(ForgeKukaPinchNutThreadCfg):
    """Pinch demo: no bolt threading. Nut at the grasp-plane center, arm held, and
    the fingers ramp closed during the rendered episode so the pinch is visible."""

    pinch_demo = True
    fixed_arm_pose_reset = True  # arm stays put; we only watch the fingers
    rigid_nut_follow = False  # nut is free, so we can see whether the fingers hold it
