"""Configuration for the SHARPA nut-screw pick-place-screw Isaac Lab task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from simtoolreal_lab.assets.kuka_sharpa import KUKA_SHARPA_CFG, KUKA_SHARPA_JOINT_NAMES


SIMTOOLREAL_LAB_DIR = Path(__file__).resolve().parents[2]
NUTSCREW_ASSET_DIRS = (SIMTOOLREAL_LAB_DIR / "assets" / "M6", SIMTOOLREAL_LAB_DIR / "assets" / "M10")
GENERATED_NUTSCREW_ASSET_DIR = SIMTOOLREAL_LAB_DIR / "assets" / "nutscrew_generated"
NUTSCREW_TABLE_SIZE = (0.475, 0.4, 0.3)
NUTSCREW_TABLE_TOP_Z = 0.53
NUTSCREW_TABLE_POS = (
    0.0,
    0.0,
    NUTSCREW_TABLE_TOP_Z - 0.5 * NUTSCREW_TABLE_SIZE[2],
)
FAMILY_PITCHES = {
    "M6": 0.001,
    "M8": 0.00125,
    "M10": 0.0015,
    "M12": 0.00175,
    "M16": 0.002,
    "M20": 0.0025,
}

FINGER_NAMES = ("index", "middle", "ring", "thumb", "pinky")
DEFAULT_ACTIVE_FINGERS = FINGER_NAMES
FINGERTIP_BODY_NAMES_BY_FINGER = {
    "thumb": "left_thumb_DP",
    "index": "left_index_DP",
    "middle": "left_middle_DP",
    "ring": "left_ring_DP",
    "pinky": "left_pinky_DP",
}
FINGERTIP_OFFSET_BY_FINGER = {
    finger: (0.02, 0.002, 0.0) for finger in FINGER_NAMES
}
FINGER_CONTACT_BODY_NAMES = {
    "thumb": (
        "left_thumb_CMC_VL",
        "left_thumb_MC",
        "left_thumb_MCP_VL",
        "left_thumb_PP",
        "left_thumb_DP",
    ),
    "index": (
        "left_index_MCP_VL",
        "left_index_PP",
        "left_index_MP",
        "left_index_DP",
    ),
    "middle": (
        "left_middle_MCP_VL",
        "left_middle_PP",
        "left_middle_MP",
        "left_middle_DP",
    ),
    "ring": (
        "left_ring_MCP_VL",
        "left_ring_PP",
        "left_ring_MP",
        "left_ring_DP",
    ),
    "pinky": (
        "left_pinky_MC",
        "left_pinky_MCP_VL",
        "left_pinky_PP",
        "left_pinky_MP",
        "left_pinky_DP",
    ),
}


def _normalize_active_fingers(active_fingers: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    active_set = set(active_fingers)
    unknown = sorted(active_set - set(FINGER_NAMES))
    if unknown:
        raise ValueError(f"Unknown active fingers {unknown}. Valid fingers: {FINGER_NAMES}")
    if not active_set:
        raise ValueError("At least one active finger is required.")
    return tuple(finger for finger in FINGER_NAMES if finger in active_set)


def _joint_is_arm(joint_name: str) -> bool:
    return joint_name.startswith("iiwa14_joint_")


def _joint_is_finger(joint_name: str, finger: str) -> bool:
    return f"_{finger}_" in joint_name


def _finger_masked_joint_names(active_fingers: tuple[str, ...] | list[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    active = set(_normalize_active_fingers(active_fingers))
    actuated_joint_names = tuple(
        joint_name
        for joint_name in KUKA_SHARPA_JOINT_NAMES
        if _joint_is_arm(joint_name) or any(_joint_is_finger(joint_name, finger) for finger in active)
    )
    inactive_joint_names = tuple(
        joint_name
        for joint_name in KUKA_SHARPA_JOINT_NAMES
        if (not _joint_is_arm(joint_name))
        and not any(_joint_is_finger(joint_name, finger) for finger in active)
    )
    return actuated_joint_names, inactive_joint_names


def _fingertip_body_names(active_fingers: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(FINGERTIP_BODY_NAMES_BY_FINGER[finger] for finger in _normalize_active_fingers(active_fingers))


def _fingertip_offsets(active_fingers: tuple[str, ...] | list[str]) -> tuple[tuple[float, float, float], ...]:
    return tuple(FINGERTIP_OFFSET_BY_FINGER[finger] for finger in _normalize_active_fingers(active_fingers))


def _table_contact_filter_paths(active_fingers: tuple[str, ...] | list[str]) -> list[str]:
    return [
        f"/World/envs/env_.*/Robot/kuka_sharpa/{body_name}"
        for finger in _normalize_active_fingers(active_fingers)
        for body_name in FINGER_CONTACT_BODY_NAMES[finger]
    ]


def _policy_obs_dim(num_actions: int, num_fingertips: int) -> int:
    return 3 * num_actions + 3 * num_fingertips + 38


def _critic_state_dim(num_actions: int, num_fingertips: int) -> int:
    return 3 * num_actions + 4 * num_fingertips + 55


def apply_finger_mask(cfg) -> None:
    active_fingers = _normalize_active_fingers(getattr(cfg, "active_fingers", DEFAULT_ACTIVE_FINGERS))
    actuated_joint_names, inactive_joint_names = _finger_masked_joint_names(active_fingers)
    fingertip_body_names = _fingertip_body_names(active_fingers)

    cfg.active_fingers = active_fingers
    cfg.actuated_joint_names = actuated_joint_names
    cfg.inactive_joint_names = inactive_joint_names
    cfg.fingertip_body_names = fingertip_body_names
    cfg.fingertip_offsets = _fingertip_offsets(active_fingers)

    cfg.num_actions = len(actuated_joint_names)
    cfg.action_space = cfg.num_actions
    cfg.num_observations = _policy_obs_dim(cfg.num_actions, len(fingertip_body_names))
    cfg.observation_space = cfg.num_observations
    cfg.num_states = _critic_state_dim(cfg.num_actions, len(fingertip_body_names))
    cfg.state_space = cfg.num_states

    if hasattr(cfg, "table_contact_sensor"):
        cfg.table_contact_sensor.filter_prim_paths_expr = _table_contact_filter_paths(active_fingers)


DEFAULT_ACTUATED_JOINT_NAMES, DEFAULT_INACTIVE_JOINT_NAMES = _finger_masked_joint_names(DEFAULT_ACTIVE_FINGERS)
DEFAULT_FINGERTIP_BODY_NAMES = _fingertip_body_names(DEFAULT_ACTIVE_FINGERS)
DEFAULT_FINGERTIP_OFFSETS = _fingertip_offsets(DEFAULT_ACTIVE_FINGERS)
DEFAULT_NUM_ACTIONS = len(DEFAULT_ACTUATED_JOINT_NAMES)
DEFAULT_NUM_FINGERTIPS = len(DEFAULT_FINGERTIP_BODY_NAMES)
DEFAULT_NUM_OBSERVATIONS = _policy_obs_dim(DEFAULT_NUM_ACTIONS, DEFAULT_NUM_FINGERTIPS)
DEFAULT_NUM_STATES = _critic_state_dim(DEFAULT_NUM_ACTIONS, DEFAULT_NUM_FINGERTIPS)


@dataclass(frozen=True)
class MeshBounds:
    mins: tuple[float, float, float]
    maxs: tuple[float, float, float]


@dataclass(frozen=True)
class NutScrewSpawnPose:
    screw_pos: tuple[float, float, float]
    nut_pos: tuple[float, float, float]
    screw_top_z: float
    nut_bottom_z: float
    nut_top_z: float
    thread_length: float


def discover_nutscrew_usd_paths() -> dict[str, Path]:
    usd_paths: dict[str, Path] = {}
    for asset_dir in NUTSCREW_ASSET_DIRS:
        if not asset_dir.exists():
            continue
        for usd_path in sorted(asset_dir.glob("*.usd")):
            usd_paths[usd_path.stem] = usd_path
    if not usd_paths and not GENERATED_NUTSCREW_ASSET_DIR.exists():
        searched = ", ".join(str(path) for path in NUTSCREW_ASSET_DIRS)
        raise FileNotFoundError(f"No nut-screw USD assets found in: {searched}")
    return usd_paths


NUTSCREW_USD_PATHS = discover_nutscrew_usd_paths()
NUTSCREW_OBJECT_SCALES = {asset_name: (1.0, 1.0, 1.0) for asset_name in NUTSCREW_USD_PATHS}


def generated_nutscrew_asset_path(family: str, name: str, suffix: str) -> Path:
    path = GENERATED_NUTSCREW_ASSET_DIR / family / f"{name}.{suffix}"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing generated nut-screw asset: {path}\n"
            "Generate it with:\n"
            "  python simtoolreal_lab/tasks/sharpa_nutscrew_pick_place_screw/tests/screw_generator.py "
            "--families M8 M10 M12 M16 M20 --formats usd obj stl --overwrite"
        )
    return path


def obj_bounds(mesh_path: Path) -> MeshBounds:
    vertices: list[tuple[float, float, float]] = []
    with mesh_path.open("r", encoding="utf-8", errors="ignore") as obj_file:
        for line in obj_file:
            if line.startswith("v "):
                x, y, z = (float(value) for value in line.split()[1:4])
                vertices.append((x, y, z))
    if not vertices:
        raise ValueError(f"OBJ has no vertices: {mesh_path}")

    mins = tuple(min(vertex[idx] for vertex in vertices) for idx in range(3))
    maxs = tuple(max(vertex[idx] for vertex in vertices) for idx in range(3))
    return MeshBounds(mins=mins, maxs=maxs)


def compute_nutscrew_spawn_pose(family: str, screw_name: str, nut_name: str, clearance: float) -> NutScrewSpawnPose:
    screw_bounds = obj_bounds(generated_nutscrew_asset_path(family, screw_name, "obj"))
    nut_bounds = obj_bounds(generated_nutscrew_asset_path(family, nut_name, "obj"))
    screw_origin_z = NUTSCREW_TABLE_TOP_Z - screw_bounds.mins[2]
    screw_top_z = screw_origin_z + screw_bounds.maxs[2]
    nut_center_z = screw_top_z - clearance - nut_bounds.maxs[2]
    return NutScrewSpawnPose(
        screw_pos=(0.0, 0.0, screw_origin_z),
        nut_pos=(0.0, 0.0, nut_center_z),
        screw_top_z=screw_top_z,
        nut_bottom_z=nut_center_z + nut_bounds.mins[2],
        nut_top_z=nut_center_z + nut_bounds.maxs[2],
        thread_length=max(screw_bounds.maxs[2], 0.0),
    )


def generated_nut_mesh_size(family: str, nut_name: str) -> tuple[float, float, float]:
    bounds = obj_bounds(generated_nutscrew_asset_path(family, nut_name, "obj"))
    return tuple(bounds.maxs[idx] - bounds.mins[idx] for idx in range(3))


def _object_rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=False,
        disable_gravity=False,
        enable_gyroscopic_forces=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=2,
        sleep_threshold=0.005,
        stabilization_threshold=0.0025,
        max_depenetration_velocity=100.0,
    )


def _screwing_phase_object_rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=False,
        disable_gravity=True,
        enable_gyroscopic_forces=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=2,
        sleep_threshold=0.005,
        stabilization_threshold=0.0025,
        max_depenetration_velocity=100.0,
    )


def _fixed_screw_rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=True,
        disable_gravity=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=2,
    )


def _goal_rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=False,
        disable_gravity=True,
        enable_gyroscopic_forces=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=2,
        sleep_threshold=0.005,
        stabilization_threshold=0.0025,
        max_depenetration_velocity=100.0,
    )


def _goal_collision_props() -> sim_utils.CollisionPropertiesCfg:
    return sim_utils.CollisionPropertiesCfg(collision_enabled=False)


def make_cube_object_cfg(mass: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=_object_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 0.9)),
            physics_material=RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.63), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_cube_goal_object_cfg(mass: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_object",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=_goal_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=_goal_collision_props(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            physics_material=RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.35, -0.06, 0.71), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_generated_nut_object_cfg(family: str, nut_name: str, mass: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(generated_nutscrew_asset_path(family, nut_name, "usd")),
            rigid_props=_screwing_phase_object_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.63), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_generated_nut_goal_object_cfg(family: str, nut_name: str, mass: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(generated_nutscrew_asset_path(family, nut_name, "usd")),
            rigid_props=_goal_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=_goal_collision_props(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.63), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_fixed_screw_cfg(family: str, screw_name: str, pose: NutScrewSpawnPose) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/fixed_screw",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(generated_nutscrew_asset_path(family, screw_name, "usd")),
            rigid_props=_fixed_screw_rigid_props(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pose.screw_pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


def get_nutscrew_usd_path(asset_name: str) -> Path:
    if asset_name not in NUTSCREW_USD_PATHS:
        known = ", ".join(sorted(NUTSCREW_USD_PATHS))
        raise ValueError(f"Unknown nut-screw asset '{asset_name}'. Known assets: {known}")
    return NUTSCREW_USD_PATHS[asset_name]


def make_nutscrew_object_cfg(asset_name: str, mass: float) -> RigidObjectCfg:
    usd_path = get_nutscrew_usd_path(asset_name)

    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            rigid_props=_object_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.63), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_nutscrew_goal_object_cfg(asset_name: str, mass: float) -> RigidObjectCfg:
    usd_path = get_nutscrew_usd_path(asset_name)

    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            rigid_props=_goal_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=_goal_collision_props(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.35, -0.06, 0.71), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_multi_nutscrew_object_cfg(asset_names: list[str], mass: float) -> RigidObjectCfg:
    usd_paths = [str(get_nutscrew_usd_path(asset_name)) for asset_name in asset_names]

    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=usd_paths,
            random_choice=False,
            rigid_props=_object_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.63), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_multi_nutscrew_goal_object_cfg(asset_names: list[str], mass: float) -> RigidObjectCfg:
    usd_paths = [str(get_nutscrew_usd_path(asset_name)) for asset_name in asset_names]

    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_object",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=usd_paths,
            random_choice=False,
            rigid_props=_goal_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=_goal_collision_props(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.35, -0.06, 0.71), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def configure_cube_object(cfg: "SharpaNutscrewPickPlaceScrewEnvCfg") -> None:
    cfg.object_name = "cube"
    cfg.object_cfg = make_cube_object_cfg(cfg.object_mass)
    cfg.goal_object_cfg = make_cube_goal_object_cfg(cfg.object_mass)
    cfg.object_scales = (1.0, 1.0, 1.0)
    cfg.scene.replicate_physics = True


def configure_nutscrew_object(cfg: "SharpaNutscrewPickPlaceScrewEnvCfg", asset_name: str) -> None:
    cfg.object_name = asset_name
    cfg.object_cfg = make_nutscrew_object_cfg(asset_name, cfg.object_mass)
    cfg.goal_object_cfg = make_nutscrew_goal_object_cfg(asset_name, cfg.object_mass)
    cfg.object_scales = NUTSCREW_OBJECT_SCALES[asset_name]


def configure_multi_nutscrew_objects(cfg: "SharpaNutscrewPickPlaceScrewEnvCfg", asset_names: list[str]) -> None:
    cfg.object_name = "multi_nutscrew"
    cfg.multi_asset_names = tuple(asset_names)
    cfg.object_cfg = make_multi_nutscrew_object_cfg(asset_names, cfg.object_mass)
    cfg.goal_object_cfg = make_multi_nutscrew_goal_object_cfg(asset_names, cfg.object_mass)
    cfg.object_scales = NUTSCREW_OBJECT_SCALES[asset_names[0]]


def apply_object_selection(cfg: "SharpaNutscrewPickPlaceScrewEnvCfg") -> None:
    if cfg.screwing_phase:
        configure_screwing_phase(cfg)
        return
    if cfg.object_name == "cube":
        configure_cube_object(cfg)
    elif cfg.object_name == "multi_nutscrew":
        cfg.scene.replicate_physics = False
        configure_multi_nutscrew_objects(cfg, list(cfg.multi_asset_names))
    elif cfg.object_name in NUTSCREW_USD_PATHS:
        configure_nutscrew_object(cfg, cfg.object_name)
    else:
        known = ", ".join(["cube", "multi_nutscrew", *sorted(NUTSCREW_USD_PATHS)])
        raise ValueError(f"Unknown object_name '{cfg.object_name}'. Known values: {known}")


def configure_screwing_phase(cfg: "SharpaNutscrewPickPlaceScrewEnvCfg") -> None:
    pose = compute_nutscrew_spawn_pose(
        cfg.screwing_family,
        cfg.screwing_screw_name,
        cfg.screwing_nut_name,
        cfg.screwing_clearance,
    )
    cfg.object_name = cfg.screwing_nut_name
    cfg.object_cfg = make_generated_nut_object_cfg(cfg.screwing_family, cfg.screwing_nut_name, cfg.object_mass)
    cfg.goal_object_cfg = make_generated_nut_goal_object_cfg(cfg.screwing_family, cfg.screwing_nut_name, cfg.object_mass)
    cfg.fixed_screw_cfg = make_fixed_screw_cfg(cfg.screwing_family, cfg.screwing_screw_name, pose)
    cfg.table_cfg.init_state.pos = NUTSCREW_TABLE_POS
    cfg.table_cfg.spawn.size = NUTSCREW_TABLE_SIZE
    cfg.object_start_pose = (*pose.nut_pos, 0.0, 0.0, 0.0, 1.0)
    cfg.goal_object_pose = None
    cfg.object_scales = (1.0, 1.0, 1.0)
    cfg.object_scale_noise_multiplier_range = (1.0, 1.0)
    cfg.debug_grasp_bounding_box_size = generated_nut_mesh_size(cfg.screwing_family, cfg.screwing_nut_name)
    cfg.debug_keypoint_size = cfg.debug_grasp_bounding_box_size
    cfg.randomize_object_rotation = False
    cfg.screwing_thread_length = pose.thread_length
    cfg.reset_position_noise_x = 0.0
    cfg.reset_position_noise_y = 0.0
    cfg.reset_position_noise_z = 0.0
    cfg.table_reset_z_range = 0.0
    cfg.object_z_low_reset_threshold = NUTSCREW_TABLE_TOP_Z - 0.05
    cfg.force_scale = 0.0


DEFAULT_SCREWING_FAMILY = "M20"
DEFAULT_SCREWING_SCREW_NAME = "M20X30"
DEFAULT_SCREWING_NUT_NAME = "M20_nut"
DEFAULT_SCREWING_CLEARANCE = 0.0
DEFAULT_SCREWING_POSE = compute_nutscrew_spawn_pose(
    DEFAULT_SCREWING_FAMILY,
    DEFAULT_SCREWING_SCREW_NAME,
    DEFAULT_SCREWING_NUT_NAME,
    DEFAULT_SCREWING_CLEARANCE,
)


@configclass
class SharpaNutscrewPickPlaceScrewEnvCfg(DirectRLEnvCfg):
    """Direct RL config mirroring the reference IsaacGym SimToolReal task."""

    # env timing
    sim_dt = 1.0 / 60.0
    decimation = 1
    episode_length_s = 10.0
    num_actions = DEFAULT_NUM_ACTIONS
    num_observations = DEFAULT_NUM_OBSERVATIONS
    num_states = DEFAULT_NUM_STATES
    observation_space = DEFAULT_NUM_OBSERVATIONS
    state_space = DEFAULT_NUM_STATES
    action_space = DEFAULT_NUM_ACTIONS
    asymmetric_obs = True

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=sim_dt,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_patch_count=4 * 5 * 2**15,
            gpu_collision_stack_size=2**29,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1536, env_spacing=1.2, replicate_physics=True)

    # assets
    screwing_phase = True
    screwing_family = DEFAULT_SCREWING_FAMILY
    screwing_screw_name = DEFAULT_SCREWING_SCREW_NAME
    screwing_nut_name = DEFAULT_SCREWING_NUT_NAME
    screwing_clearance = DEFAULT_SCREWING_CLEARANCE
    screwing_thread_length = DEFAULT_SCREWING_POSE.thread_length
    screwing_goal_angle_range_degrees = (60.0, 90.0)
    screwing_kinematic_constraint = True
    screwing_thread_angular_damping = 12.0
    screwing_thread_static_angular_velocity = 0.04
    screwing_thread_delta_deadband = 1.0e-4
    object_name = DEFAULT_SCREWING_NUT_NAME
    multi_asset_names = tuple(sorted(NUTSCREW_USD_PATHS))
    robot_cfg = KUKA_SHARPA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    active_fingers = DEFAULT_ACTIVE_FINGERS
    actuated_joint_names = DEFAULT_ACTUATED_JOINT_NAMES
    inactive_joint_names = DEFAULT_INACTIVE_JOINT_NAMES
    palm_body_name = "iiwa14_link_7"
    fingertip_body_names = DEFAULT_FINGERTIP_BODY_NAMES
    palm_offset = (0.0, -0.02, 0.16)
    fingertip_offsets = DEFAULT_FINGERTIP_OFFSETS

    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=NUTSCREW_TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.56, 0.35)),
            physics_material=RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=NUTSCREW_TABLE_POS, rot=(1.0, 0.0, 0.0, 0.0)),
    )
    table_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/table",
        debug_vis=False,
        filter_prim_paths_expr=_table_contact_filter_paths(DEFAULT_ACTIVE_FINGERS),
    )
    object_mass = 0.05
    object_cfg: RigidObjectCfg = make_generated_nut_object_cfg(DEFAULT_SCREWING_FAMILY, DEFAULT_SCREWING_NUT_NAME, object_mass)
    goal_object_cfg: RigidObjectCfg = make_generated_nut_goal_object_cfg(DEFAULT_SCREWING_FAMILY, DEFAULT_SCREWING_NUT_NAME, object_mass)
    fixed_screw_cfg: RigidObjectCfg = make_fixed_screw_cfg(
        DEFAULT_SCREWING_FAMILY,
        DEFAULT_SCREWING_SCREW_NAME,
        DEFAULT_SCREWING_POSE,
    )

    # reset/control
    clamp_abs_observations = 10.0
    use_relative_control = False
    dof_speed_scale = 1.5
    hand_moving_average = 0.1
    arm_moving_average = 0.1
    reset_position_noise_x = 0.0
    reset_position_noise_y = 0.0
    reset_position_noise_z = 0.0
    reset_dof_pos_noise_fingers = 0.1
    reset_dof_pos_noise_arm = 0.1
    reset_dof_vel_noise = 0.5
    reset_joint_pos_overrides: dict[str, float] | None = None
    randomize_object_rotation = False
    object_start_pose: tuple[float, float, float, float, float, float, float] | None = (
        *DEFAULT_SCREWING_POSE.nut_pos,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    goal_object_pose: tuple[float, float, float, float, float, float, float] | None = None
    debug_keypoints = False
    debug_grasp_bounding_box = False
    debug_keypoint_radius = 0.012
    debug_keypoint_size: tuple[float, float, float] | None = None
    debug_keypoint_padding = 0.001
    debug_grasp_bounding_box_size: tuple[float, float, float] | None = None
    debug_grasp_bounding_box_padding = 0.001
    debug_grasp_bounding_box_line_width = 3.0
    rotation_indicator_enabled = True
    rotation_indicator_length = 0.028
    rotation_indicator_width = 0.001
    rotation_indicator_z_offset = 0.007

    # reference task geometry
    table_top_z = 0.53
    table_reset_z_range = 0.0
    table_object_z_offset = 0.25
    object_base_size = 0.04
    object_scales = (1.0, 1.0, 1.0)
    object_scale_noise_multiplier_range = (1.0, 1.0)
    fixed_size_keypoint_reward = True
    fixed_size = (0.141, 0.03025, 0.0271)
    keypoint_scale = 1.5
    target_volume_mins = (-0.35, -0.1, 0.68)
    target_volume_maxs = (0.35, 0.2, 1.05)
    target_volume_region_scale = 1.0
    goal_sampling_type = "delta"
    delta_goal_distance = 0.1
    delta_rotation_degrees = 90.0

    # rewards/resets
    lifting_rew_scale = 20.0
    lifting_bonus = 300.0
    lifting_bonus_threshold = 0.15
    keypoint_rew_scale = 200.0
    distance_delta_rew_scale = 50.0
    reach_goal_bonus = 1000.0
    kuka_actions_penalty_scale = 0.03
    hand_actions_penalty_scale = 0.003
    fall_distance = 0.24
    fall_penalty = 0.0
    object_lin_vel_penalty_scale = 0.0
    object_ang_vel_penalty_scale = 0.0
    object_z_low_reset_threshold = NUTSCREW_TABLE_TOP_Z - 0.05
    hand_far_from_object_threshold = 1.5
    with_table_force_sensor = True
    table_force_threshold = 5.0
    reset_when_dropped = True
    success_tolerance = 0.075
    target_success_tolerance = 0.01
    tolerance_curriculum_increment = 0.9
    tolerance_curriculum_interval = 3000
    eval_success_tolerance = None
    success_steps = 10
    max_consecutive_successes = 50
    force_consecutive_near_goal_steps = False

    # sim2real/domain-randomization delays and observation noise
    use_obs_delay = True
    obs_delay_max = 3
    use_action_delay = True
    action_delay_max = 3
    use_object_state_delay_noise = True
    object_state_delay_max = 10
    object_state_xyz_noise_std = 0.01
    object_state_rotation_noise_degrees = 5.0
    joint_velocity_obs_noise_std = 0.01

    # object force/torque disturbances
    force_scale = 0.0
    force_prob_range = (0.001, 0.1)
    force_decay = 0.99
    force_decay_interval = 0.08
    force_only_when_lifted = True
    torque_scale = 0.0
    torque_prob_range = (0.001, 0.1)
    torque_decay = 0.99
    torque_decay_interval = 0.08
    torque_only_when_lifted = False

    def __post_init__(self):
        super().__post_init__()
        apply_object_selection(self)
        apply_finger_mask(self)
