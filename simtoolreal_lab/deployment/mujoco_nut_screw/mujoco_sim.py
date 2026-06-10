"""MuJoCo loader for the static KUKA iiwa14 + SHARPA nut-screw MJCF scene."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np

from simtoolreal_lab.deployment.mujoco.mujoco_sim import (
    INIT_JOINT_POS,
    JOINT_NAMES,
    N_JOINTS,
    SIMTOOLREAL_LAB_DIR,
    TABLE_POS,
    TABLE_TOP_Z,
    FrictionConfig,
    _maybe_import_viewer,
    _obj_bounds,
    _require_mujoco,
)


NUTSCREW_ASSET_DIR = SIMTOOLREAL_LAB_DIR / "assets/nutscrew_generated"
NUT_SCREW_SCENE_PATH = Path(__file__).resolve().parent / "scenes/iiwa_sharpa_nut_screw.xml"
DEFAULT_FAMILY = "M12"
DEFAULT_SCREW_NAME = "M12X30"
DEFAULT_NUT_NAME = "M12_nut"
DEFAULT_SCREWING_CLEARANCE = 0.0
# M12/M12X30 mesh-bounds result for the Isaac screwing-phase convention:
# nut top coincides with the screw tip.
DEFAULT_OBJECT_START_POS = np.array([0.0, 0.0, 0.567])
DEFAULT_GOAL_OBJECT_START_POS = DEFAULT_OBJECT_START_POS.copy()
FAMILY_PITCHES = {
    "M6": 0.001,
    "M8": 0.00125,
    "M10": 0.0015,
    "M12": 0.00175,
    "M16": 0.002,
    "M20": 0.0025,
}


@dataclass
class MujocoNutScrewSimConfig:
    enable_viewer: bool = True
    sim_dt: float = 1.0 / 600.0
    friction: FrictionConfig = field(default_factory=FrictionConfig)
    family: str = DEFAULT_FAMILY
    screw_name: str = DEFAULT_SCREW_NAME
    nut_name: str = DEFAULT_NUT_NAME
    screwing_clearance: float = DEFAULT_SCREWING_CLEARANCE
    asset_root: Path = NUTSCREW_ASSET_DIR
    scene_xml_path: Path = NUT_SCREW_SCENE_PATH
    object_start_pos: np.ndarray | None = None
    object_start_quat_wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    goal_object_start_pos: np.ndarray | None = None
    goal_object_start_quat_wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    show_robot_collision_overlay: bool = True
    screwing_kinematic_constraint: bool = True
    screwing_thread_length: float | None = None
    screwing_thread_angular_damping: float = 12.0
    screwing_thread_static_angular_velocity: float = 0.04
    screwing_thread_delta_deadband: float = 1.0e-4

    def __post_init__(self) -> None:
        spawn = nut_screw_spawn_pose(
            self.asset_root,
            self.family,
            self.screw_name,
            self.nut_name,
            self.screwing_clearance,
        )
        if self.object_start_pos is None:
            self.object_start_pos = spawn["nut_pos"].copy()
        if self.goal_object_start_pos is None:
            self.goal_object_start_pos = spawn["nut_pos"].copy()
        if self.screwing_thread_length is None:
            self.screwing_thread_length = float(spawn["thread_length"])

    @property
    def object_name(self) -> str:
        return self.nut_name

    @property
    def friction_array(self) -> np.ndarray:
        return np.array(
            [self.friction.sliding_friction, self.friction.torsional_friction, self.friction.rolling_friction]
        )


def _asset_path(asset_root: Path, family: str, name: str, suffix: str = ".obj") -> Path:
    path = asset_root.expanduser().resolve() / family / f"{name}{suffix}"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing nut-screw mesh: {path}\n"
            "Generate OBJ assets with:\n"
            "  python simtoolreal_lab/tasks/sharpa_nutscrew_pick_place_screw/tests/screw_generator.py "
            "--families M8 M10 M12 M16 M20 --formats usd obj stl --overwrite"
        )
    return path


def _object_scales_from_mesh(mesh_path: Path) -> np.ndarray:
    mins, maxs = _obj_bounds(mesh_path)
    # Training keypoints use a nominal 6cm full box when scale == 1.
    return np.maximum((maxs - mins) / 0.06, np.array([0.05, 0.05, 0.05]))


def nut_screw_object_scales(asset_root: Path, family: str, nut_name: str) -> np.ndarray:
    return _object_scales_from_mesh(_asset_path(asset_root, family, nut_name))


def nut_screw_spawn_pose(
    asset_root: Path,
    family: str,
    screw_name: str,
    nut_name: str,
    clearance: float = DEFAULT_SCREWING_CLEARANCE,
) -> dict[str, np.ndarray | float]:
    screw_mins, screw_maxs = _obj_bounds(_asset_path(asset_root, family, screw_name))
    nut_mins, nut_maxs = _obj_bounds(_asset_path(asset_root, family, nut_name))
    screw_origin_z = TABLE_TOP_Z - float(screw_mins[2])
    screw_top_z = screw_origin_z + float(screw_maxs[2])
    nut_center_z = screw_top_z - clearance - float(nut_maxs[2])
    nut_pos = np.array([0.0, 0.0, nut_center_z], dtype=float)
    return {
        "screw_pos": np.array([0.0, 0.0, screw_origin_z], dtype=float),
        "nut_pos": nut_pos,
        "screw_top_z": screw_top_z,
        "nut_bottom_z": nut_center_z + float(nut_mins[2]),
        "nut_top_z": nut_center_z + float(nut_maxs[2]),
        "thread_length": max(float(screw_maxs[2]), 0.0),
    }


class MujocoNutScrewSim:
    """MuJoCo replay scene backed by a concrete nut-screw MJCF file."""

    def __init__(self, config: MujocoNutScrewSimConfig):
        self.config = config
        self.mujoco = _require_mujoco()
        self.viewer = None
        self.nut_thread_angle = 0.0
        self.max_nut_thread_angle = self._max_nut_thread_angle()
        self._init_scene()
        self.set_robot_joint_pos_targets(INIT_JOINT_POS)
        self.set_robot_joint_positions(INIT_JOINT_POS)

    def _init_scene(self) -> None:
        scene_xml_path = self.config.scene_xml_path.expanduser().resolve()
        if not scene_xml_path.exists():
            raise FileNotFoundError(f"MuJoCo scene XML not found: {scene_xml_path}")
        self.mj_model = self.mujoco.MjModel.from_xml_path(str(scene_xml_path))
        self.mj_data = self.mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.config.sim_dt
        if self.config.enable_viewer:
            viewer_mod = _maybe_import_viewer()
            if viewer_mod is None:
                raise ModuleNotFoundError("mujoco.viewer is required when enable_viewer=True.")
            self.viewer = viewer_mod.launch_passive(self.mj_model, self.mj_data)
        self._validate()
        self._set_scene_initial_state()
        print(f"[MuJoCo] Loaded static nut-screw scene: {scene_xml_path}")

    def _validate(self) -> None:
        missing = [name for name in JOINT_NAMES if name not in self.joint_names]
        if missing:
            raise RuntimeError(f"MuJoCo model is missing policy joints: {missing}")
        for body_name in ("object", "goal_object", "fixed_screw"):
            if body_name not in self.body_names:
                raise RuntimeError(f"MuJoCo model is missing body '{body_name}'.")

    def _set_scene_initial_state(self) -> None:
        self.nut_thread_angle = 0.0
        self.set_robot_joint_positions(INIT_JOINT_POS)
        self.set_object_pose(self.config.object_start_pos, self.config.object_start_quat_wxyz)
        self.set_goal_object_pose(self.config.goal_object_start_pos, self.config.goal_object_start_quat_wxyz)
        self._apply_screwing_kinematic_constraint(integrate_thread_angle=False)
        self.mujoco.mj_forward(self.mj_model, self.mj_data)

    def _object_free_joint_addrs(self) -> tuple[int, int] | None:
        body_id = self.mj_model.body(name="object").id
        joint_id = int(self.mj_model.body_jntadr[body_id])
        if joint_id < 0:
            return None
        joint = self.mj_model.joint(joint_id)
        if joint.type[0] != self.mujoco.mjtJoint.mjJNT_FREE:
            return None
        return int(joint.qposadr[0]), int(joint.dofadr[0])

    def _max_nut_thread_angle(self) -> float:
        pitch = FAMILY_PITCHES.get(self.config.family, FAMILY_PITCHES["M12"])
        thread_length = float(self.config.screwing_thread_length or 0.0)
        if pitch <= 0.0 or thread_length <= 0.0:
            return 0.0
        return thread_length * (2.0 * math.pi) / pitch

    @staticmethod
    def _quat_from_z_angle(angle: float) -> np.ndarray:
        half_angle = 0.5 * angle
        return np.array([math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)], dtype=float)

    def _apply_screwing_kinematic_constraint(self, integrate_thread_angle: bool = True) -> None:
        if not self.config.screwing_kinematic_constraint:
            return
        addrs = self._object_free_joint_addrs()
        if addrs is None:
            return

        qpos_adr, qvel_adr = addrs
        pitch = FAMILY_PITCHES.get(self.config.family, FAMILY_PITCHES["M12"])
        angular_velocity = float(self.mj_data.qvel[qvel_adr + 5])

        angular_damping = float(self.config.screwing_thread_angular_damping)
        if angular_damping > 0.0:
            angular_velocity *= math.exp(-angular_damping * self.mj_model.opt.timestep)

        static_angular_velocity = float(self.config.screwing_thread_static_angular_velocity)
        if static_angular_velocity > 0.0 and abs(angular_velocity) < static_angular_velocity:
            angular_velocity = 0.0

        if integrate_thread_angle:
            delta_yaw = angular_velocity * self.mj_model.opt.timestep
            delta_deadband = float(self.config.screwing_thread_delta_deadband)
            if delta_deadband > 0.0 and abs(delta_yaw) < delta_deadband:
                delta_yaw = 0.0
            self.nut_thread_angle = float(
                np.clip(self.nut_thread_angle + delta_yaw, 0.0, self.max_nut_thread_angle)
            )

        at_thread_top = self.nut_thread_angle <= 1.0e-6 and angular_velocity < 0.0
        at_thread_end = self.nut_thread_angle >= self.max_nut_thread_angle - 1.0e-6 and angular_velocity > 0.0
        if at_thread_top or at_thread_end:
            angular_velocity = 0.0

        slide = -(pitch / (2.0 * math.pi)) * self.nut_thread_angle
        object_pos = np.asarray(self.config.object_start_pos, dtype=float).copy()
        object_pos[2] += slide
        object_quat = self._quat_from_z_angle(self.nut_thread_angle)

        self.mj_data.qpos[qpos_adr : qpos_adr + 3] = object_pos
        self.mj_data.qpos[qpos_adr + 3 : qpos_adr + 7] = object_quat
        self.mj_data.qvel[qvel_adr : qvel_adr + 2] = 0.0
        self.mj_data.qvel[qvel_adr + 2] = -(pitch / (2.0 * math.pi)) * angular_velocity
        self.mj_data.qvel[qvel_adr + 3 : qvel_adr + 5] = 0.0
        self.mj_data.qvel[qvel_adr + 5] = angular_velocity
        self.mujoco.mj_forward(self.mj_model, self.mj_data)

    def _set_free_body_pose(self, body_name: str, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
        body_id = self.mj_model.body(name=body_name).id
        joint_id = int(self.mj_model.body_jntadr[body_id])
        if joint_id < 0:
            return
        joint = self.mj_model.joint(joint_id)
        if joint.type[0] != self.mujoco.mjtJoint.mjJNT_FREE:
            return
        qpos_adr = int(joint.qposadr[0])
        qvel_adr = int(joint.dofadr[0])
        self.mj_data.qpos[qpos_adr : qpos_adr + 3] = pos
        self.mj_data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat_wxyz
        self.mj_data.qvel[qvel_adr : qvel_adr + 6] = 0.0

    def set_object_pose(self, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
        self._set_free_body_pose("object", pos, quat_wxyz)
        self.mujoco.mj_forward(self.mj_model, self.mj_data)

    def set_goal_object_pose(self, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
        goal_body_id = self.mj_model.body(name="goal_object").id
        mocap_id = int(self.mj_model.body_mocapid[goal_body_id])
        if mocap_id >= 0:
            self.mj_data.mocap_pos[mocap_id] = pos
            self.mj_data.mocap_quat[mocap_id] = quat_wxyz
        else:
            self._set_free_body_pose("goal_object", pos, quat_wxyz)
        self.mujoco.mj_forward(self.mj_model, self.mj_data)

    def reset_scene(
        self,
        object_pos: np.ndarray | None = None,
        object_quat_wxyz: np.ndarray | None = None,
        goal_object_pos: np.ndarray | None = None,
        goal_object_quat_wxyz: np.ndarray | None = None,
    ) -> None:
        self.mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.nut_thread_angle = 0.0
        self.set_robot_joint_pos_targets(INIT_JOINT_POS)
        self.set_robot_joint_positions(INIT_JOINT_POS)
        self.set_object_pose(
            self.config.object_start_pos if object_pos is None else object_pos,
            self.config.object_start_quat_wxyz if object_quat_wxyz is None else object_quat_wxyz,
        )
        self.set_goal_object_pose(
            self.config.goal_object_start_pos if goal_object_pos is None else goal_object_pos,
            self.config.goal_object_start_quat_wxyz
            if goal_object_quat_wxyz is None
            else goal_object_quat_wxyz,
        )
        self._apply_screwing_kinematic_constraint(integrate_thread_angle=False)

    def set_robot_joint_positions(self, q: np.ndarray) -> None:
        if q.shape != (N_JOINTS,):
            raise ValueError(f"Expected q shape {(N_JOINTS,)}, got {q.shape}.")
        for value, joint_name in zip(q, JOINT_NAMES):
            joint = self.mj_model.joint(name=joint_name)
            self.mj_data.qpos[joint.qposadr[0]] = value
            self.mj_data.qvel[joint.dofadr[0]] = 0.0
        self.mj_data.ctrl[:N_JOINTS] = q
        self.mujoco.mj_forward(self.mj_model, self.mj_data)

    def set_robot_joint_pos_targets(self, q_targets: np.ndarray) -> None:
        if q_targets.shape != (N_JOINTS,):
            raise ValueError(f"Expected q_targets shape {(N_JOINTS,)}, got {q_targets.shape}.")
        self.robot_joint_pos_targets = q_targets.copy()

    def get_body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        body_id = self.mj_model.body(name=body_name).id
        return self.mj_data.xpos[body_id].copy(), self.mj_data.xquat[body_id].copy()

    def get_sim_state(self) -> dict[str, np.ndarray]:
        self.mujoco.mj_forward(self.mj_model, self.mj_data)
        q = np.array([self.mj_data.qpos[self.mj_model.joint(name=name).qposadr[0]] for name in JOINT_NAMES])
        qd = np.array([self.mj_data.qvel[self.mj_model.joint(name=name).dofadr[0]] for name in JOINT_NAMES])
        object_pos, object_quat_wxyz = self.get_body_pose("object")
        goal_pos, goal_quat_wxyz = self.get_body_pose("goal_object")
        base_name = "base" if "base" in self.body_names else self.body_names[1]
        base_pos, base_quat_wxyz = self.get_body_pose(base_name)
        return {
            "table_pos": TABLE_POS.copy(),
            "table_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
            "object_pos": object_pos,
            "object_quat_wxyz": object_quat_wxyz,
            "goal_object_pos": goal_pos,
            "goal_object_quat_wxyz": goal_quat_wxyz,
            "robot_base_pos": base_pos,
            "robot_base_quat_wxyz": base_quat_wxyz,
            "joint_positions": q,
            "joint_velocities": qd,
        }

    def sim_step(self) -> None:
        self.mj_data.ctrl[:N_JOINTS] = self.robot_joint_pos_targets
        self.mujoco.mj_step(self.mj_model, self.mj_data)
        self._apply_screwing_kinematic_constraint()

    def run(self) -> None:
        loop_dts = []
        while self._continue_running():
            start = time.time()
            self.sim_step()
            if self.viewer is not None:
                self.viewer.sync()
            elapsed = time.time() - start
            sleep_dt = self.config.sim_dt - elapsed
            if sleep_dt > 0:
                time.sleep(sleep_dt)
                loop_dts.append(self.config.sim_dt)
            else:
                loop_dts.append(elapsed)
            if len(loop_dts) >= int(5.0 / self.config.sim_dt):
                fps = 1.0 / np.array(loop_dts)
                print(f"MuJoCo FPS mean={fps.mean():.1f} min={fps.min():.1f} max={fps.max():.1f}")
                loop_dts.clear()

    def _continue_running(self) -> bool:
        return self.viewer.is_running() if self.viewer is not None else True

    @property
    def joint_names(self) -> list[str]:
        return [self.mj_model.joint(i).name for i in range(self.mj_model.njnt)]

    @property
    def body_names(self) -> list[str]:
        return [self.mj_model.body(i).name for i in range(self.mj_model.nbody)]


def main() -> None:
    sim = MujocoNutScrewSim(MujocoNutScrewSimConfig(enable_viewer=True))
    sim.run()


if __name__ == "__main__":
    main()
