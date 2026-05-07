"""Isaac Lab asset config for the reference KUKA iiwa14 + left SHARPA hand."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

MODULE_PATH = os.path.dirname(__file__)
URDF_PATH = os.path.join(
    MODULE_PATH,
    "urdf",
    "kuka_sharpa_description",
    "iiwa14_left_sharpa_adjusted_restricted.urdf",
)

# Keep this order aligned with reference.isaacgymenvs.utils.observation_action_utils_sharpa.
KUKA_SHARPA_JOINT_NAMES = [
    "iiwa14_joint_1",
    "iiwa14_joint_2",
    "iiwa14_joint_3",
    "iiwa14_joint_4",
    "iiwa14_joint_5",
    "iiwa14_joint_6",
    "iiwa14_joint_7",
    "left_1_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "left_2_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_3_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_4_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_5_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
]

_ARM_JOINT_RE = "iiwa14_joint_[1-7]"
_HAND_JOINT_RE = "left_.*"

KUKA_SHARPA_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=URDF_PATH,
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=False,
        self_collision=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=500.0,
            max_depenetration_velocity=30.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness={
                    _ARM_JOINT_RE: 300.0,
                    _HAND_JOINT_RE: 3.0,
                },
                damping={
                    _ARM_JOINT_RE: 30.0,
                    _HAND_JOINT_RE: 0.1,
                },
            ),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.8, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "iiwa14_joint_1": -1.571,
            "iiwa14_joint_2": 1.571,
            "iiwa14_joint_3": 0.0,
            "iiwa14_joint_4": 1.376,
            "iiwa14_joint_5": 0.0,
            "iiwa14_joint_6": 1.485,
            "iiwa14_joint_7": 1.308,
            _HAND_JOINT_RE: 0.0,
        },
    ),
    actuators={
        "kuka_sharpa": ImplicitActuatorCfg(
            joint_names_expr=KUKA_SHARPA_JOINT_NAMES,
            effort_limit_sim={
                _ARM_JOINT_RE: 300.0,
                _HAND_JOINT_RE: 1.5,
            },
            velocity_limit_sim={
                _ARM_JOINT_RE: 10.0,
                _HAND_JOINT_RE: 15.0,
            },
            stiffness={
                _ARM_JOINT_RE: 300.0,
                _HAND_JOINT_RE: 3.0,
            },
            damping={
                _ARM_JOINT_RE: 30.0,
                _HAND_JOINT_RE: 0.1,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration for the reference KUKA iiwa14 + SHARPA robot."""
