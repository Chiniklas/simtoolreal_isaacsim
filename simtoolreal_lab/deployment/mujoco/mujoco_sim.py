"""MuJoCo scene for KUKA iiwa14 + left SHARPA sim2sim policy playback."""

from __future__ import annotations

import xml.etree.ElementTree as ET
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


N_IIWA_JOINTS = 7
N_SHARPA_JOINTS = 22
N_JOINTS = N_IIWA_JOINTS + N_SHARPA_JOINTS
REPO_ROOT = Path(__file__).resolve().parents[3]
SIMTOOLREAL_LAB_DIR = Path(__file__).resolve().parents[2]
DEXTOOLBENCH_OBJECT_SCALES = {
    "mallet_hammer": (6.0, 0.75, 0.5),
    "claw_hammer": (2.5, 0.5625, 0.375),
    "long_screwdriver": (2.5, 0.75, 0.75),
    "short_screwdriver": (1.75, 0.875, 0.875),
    "handle_eraser": (2.25, 0.8, 0.25),
    "flat_eraser": (2.5, 0.7, 1.25),
    "flat_spatula": (5.0, 0.375, 0.1875),
    "spoon_spatula": (3.0, 0.5, 0.5),
    "sharpie_marker": (2.125, 0.55, 0.55),
    "staples_marker": (3.0, 0.45, 0.45),
    "red_brush": (2.5, 0.5, 0.375),
    "blue_brush": (3.0, 0.875, 0.5),
}

IIWA_INIT_JOINT_POS = np.array([-1.571, 1.39647, 0.0, 1.55053, 0.0, 1.485, 1.308])
SHARPA_INIT_JOINT_POS = np.zeros(N_SHARPA_JOINTS)
INIT_JOINT_POS = np.concatenate([IIWA_INIT_JOINT_POS, SHARPA_INIT_JOINT_POS])

JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "palmleft_thumb_CMC_FE",
    "palmleft_thumb_CMC_AA",
    "palmleft_thumb_MCP_FE",
    "palmleft_thumb_MCP_AA",
    "palmleft_thumb_IP",
    "palmleft_index_MCP_FE",
    "palmleft_index_MCP_AA",
    "palmleft_index_PIP",
    "palmleft_index_DIP",
    "palmleft_middle_MCP_FE",
    "palmleft_middle_MCP_AA",
    "palmleft_middle_PIP",
    "palmleft_middle_DIP",
    "palmleft_ring_MCP_FE",
    "palmleft_ring_MCP_AA",
    "palmleft_ring_PIP",
    "palmleft_ring_DIP",
    "palmleft_pinky_CMC",
    "palmleft_pinky_MCP_FE",
    "palmleft_pinky_MCP_AA",
    "palmleft_pinky_PIP",
    "palmleft_pinky_DIP",
]

ROBOT_URDF_PATH = (
    SIMTOOLREAL_LAB_DIR
    / "assets/kuka_sharpa/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
)
MUJOCO_REPLAY_SCENE_PATH = SIMTOOLREAL_LAB_DIR / "assets/mujoco_wasm/scenes/iiwa_sharpa.xml"
MUJOCO_REPLAY_MESH_DIR = MUJOCO_REPLAY_SCENE_PATH.parent / "meshes/iiwa_sharpa"
DEXTOOLBENCH_ASSET_DIR = SIMTOOLREAL_LAB_DIR / "assets/dextoolbench"
TABLE_POS = np.array([0.0, 0.0, 0.38])
TABLE_SIZE = np.array([0.475, 0.4, 0.3])
TABLE_TOP_Z = TABLE_POS[2] + TABLE_SIZE[2] / 2.0


@dataclass
class FrictionConfig:
    sliding_friction: float = 1.0
    torsional_friction: float = 0.005
    rolling_friction: float = 0.0001


@dataclass
class MujocoSimConfig:
    enable_viewer: bool = True
    sim_dt: float = 1.0 / 600.0
    friction: FrictionConfig = field(default_factory=FrictionConfig)
    object_name: str = "cuboidal_mallet"
    object_start_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.58]))
    object_start_quat_wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    goal_object_start_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.78]))
    goal_object_start_quat_wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    scene_xml_path: Path = MUJOCO_REPLAY_SCENE_PATH
    use_proxy_object_collision: bool = True
    show_robot_collision_overlay: bool = True

    @property
    def friction_array(self) -> np.ndarray:
        return np.array(
            [self.friction.sliding_friction, self.friction.torsional_friction, self.friction.rolling_friction]
        )


def _require_mujoco():
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MuJoCo is required for simtoolreal_lab.deployment.mujoco. "
            "Install it in the active environment, for example `python -m pip install mujoco`."
        ) from exc
    return mujoco


def _maybe_import_viewer():
    try:
        import mujoco.viewer
    except ModuleNotFoundError:
        return None
    return mujoco.viewer


def _dextoolbench_mesh_path(object_name: str) -> Path:
    if object_name not in DEXTOOLBENCH_OBJECT_SCALES:
        known = ", ".join(sorted(DEXTOOLBENCH_OBJECT_SCALES))
        raise ValueError(f"Unknown DexToolBench object '{object_name}'. Known objects: {known}")
    matches = sorted(DEXTOOLBENCH_ASSET_DIR.glob(f"*/*/{object_name}.obj"))
    if not matches:
        raise FileNotFoundError(f"Missing DexToolBench OBJ for '{object_name}' under {DEXTOOLBENCH_ASSET_DIR}")
    if len(matches) > 1:
        paths = ", ".join(str(path) for path in matches)
        raise RuntimeError(f"Ambiguous DexToolBench OBJ for '{object_name}': {paths}")
    return matches[0]


def _obj_bounds(mesh_path: Path, mesh_scale: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    scale = np.ones(3) if mesh_scale is None else np.asarray(mesh_scale, dtype=float)
    vertices = []
    with mesh_path.open("r", encoding="utf-8", errors="ignore") as obj_file:
        for line in obj_file:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
    if not vertices:
        raise ValueError(f"OBJ has no vertices: {mesh_path}")
    vertices = np.asarray(vertices, dtype=float) * scale
    return vertices.min(axis=0), vertices.max(axis=0)


def _set_or_append(parent: ET.Element, tag: str, attributes: dict[str, str]) -> ET.Element:
    name = attributes.get("name")
    if name is not None:
        for child in parent.findall(tag):
            if child.attrib.get("name") == name:
                child.attrib.update(attributes)
                return child
    child = ET.SubElement(parent, tag)
    child.attrib.update(attributes)
    return child


def _remove_child_tags(parent: ET.Element, tags: set[str]) -> None:
    for child in list(parent):
        if child.tag in tags:
            parent.remove(child)


def _format_vec(values: np.ndarray | tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)


def _add_proxy_object_collision(body: ET.Element, object_name: str, prefix: str) -> None:
    if object_name == "claw_hammer":
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{prefix}_handle_collision_proxy",
                "type": "box",
                "size": "0.10 0.015 0.01",
                "rgba": "0.5 0.5 0.5 0",
                "density": "400",
                "condim": "6",
                "friction": "1 0.005 0.0001",
                "group": "3",
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{prefix}_head_collision_proxy",
                "type": "capsule",
                "fromto": "0.10 -0.03 0 0.10 0.03 0",
                "size": "0.02",
                "rgba": "0.4 0.4 0.4 0",
                "density": "300",
                "condim": "6",
                "friction": "1 0.005 0.0001",
                "group": "3",
            },
        )
        return

    mesh_path = _dextoolbench_mesh_path(object_name)
    mins, maxs = _obj_bounds(mesh_path)
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{prefix}_collision_box_proxy",
            "type": "box",
            "pos": _format_vec(0.5 * (mins + maxs)),
            "size": _format_vec(np.maximum(0.5 * (maxs - mins), np.array([0.005, 0.005, 0.005]))),
            "rgba": "0.5 0.5 0.5 0",
            "density": "400",
            "condim": "6",
            "friction": "1 0.005 0.0001",
            "group": "3",
        },
    )


def _scene_xml_with_dextoolbench_object(scene_xml_path: Path, config: "MujocoSimConfig") -> str:
    tree = ET.parse(scene_xml_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str(MUJOCO_REPLAY_MESH_DIR.resolve()))

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    mesh_path = _dextoolbench_mesh_path(config.object_name)
    mesh_name = f"simtoolreal_{config.object_name}_visual_mesh"
    _set_or_append(
        asset,
        "mesh",
        {
            "name": mesh_name,
            "file": str(mesh_path.resolve()),
        },
    )

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"MuJoCo scene has no worldbody: {scene_xml_path}")
    object_body = worldbody.find("./body[@name='object']")
    goal_body = worldbody.find("./body[@name='goal_object']")
    if object_body is None:
        raise RuntimeError(f"MuJoCo scene has no object body: {scene_xml_path}")

    object_body.set("pos", _format_vec(config.object_start_pos))
    object_body.set("quat", _format_vec(config.object_start_quat_wxyz))
    _remove_child_tags(object_body, {"geom"})
    ET.SubElement(
        object_body,
        "geom",
        {
            "name": f"object_{config.object_name}_visual",
            "type": "mesh",
            "mesh": mesh_name,
            "rgba": "0.62 0.62 0.62 1",
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
            "group": "1",
        },
    )
    if config.use_proxy_object_collision:
        _add_proxy_object_collision(object_body, config.object_name, "object")
    else:
        ET.SubElement(
            object_body,
            "geom",
            {
                "name": f"object_{config.object_name}_mesh_collision",
                "type": "mesh",
                "mesh": mesh_name,
                "rgba": "0.62 0.62 0.62 1",
                "density": "400",
                "condim": "6",
                "friction": "1 0.005 0.0001",
                "group": "3",
            },
        )

    if goal_body is not None:
        goal_body.set("pos", _format_vec(config.goal_object_start_pos))
        goal_body.set("quat", _format_vec(config.goal_object_start_quat_wxyz))
        _remove_child_tags(goal_body, {"geom"})
        ET.SubElement(
            goal_body,
            "geom",
            {
                "name": f"goal_{config.object_name}_visual",
                "type": "mesh",
                "mesh": mesh_name,
                "rgba": "0 1 0 0.35",
                "contype": "0",
                "conaffinity": "0",
                "density": "0",
                "group": "1",
            },
        )

    return ET.tostring(root, encoding="unicode")


class MujocoSim:
    """MuJoCo replay scene based on the browser demo MJCF assets."""

    def __init__(self, config: MujocoSimConfig):
        self.config = config
        self.mujoco = _require_mujoco()
        self.viewer = None
        self._init_scene()
        self.set_robot_joint_pos_targets(INIT_JOINT_POS)
        self.set_robot_joint_positions(INIT_JOINT_POS)

    def _init_scene(self) -> None:
        if not self.config.scene_xml_path.exists():
            raise FileNotFoundError(f"MuJoCo scene XML not found: {self.config.scene_xml_path}")
        if self.config.object_name in DEXTOOLBENCH_OBJECT_SCALES:
            scene_xml = _scene_xml_with_dextoolbench_object(self.config.scene_xml_path, self.config)
            print(
                f"[MuJoCo] Visualizing DexToolBench object '{self.config.object_name}' "
                f"with {'proxy' if self.config.use_proxy_object_collision else 'mesh'} collision."
            )
            self.mj_model = self.mujoco.MjModel.from_xml_string(scene_xml)
        else:
            print(
                "[MuJoCo] The browser-demo MJCF scene uses its built-in hammer object; "
                f"--object-name {self.config.object_name!r} only changes policy object scale metadata."
            )
            self.mj_model = self.mujoco.MjModel.from_xml_path(str(self.config.scene_xml_path))
        self.mj_data = self.mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.config.sim_dt
        if self.config.enable_viewer:
            viewer_mod = _maybe_import_viewer()
            if viewer_mod is None:
                raise ModuleNotFoundError("mujoco.viewer is required when enable_viewer=True.")
            self.viewer = viewer_mod.launch_passive(self.mj_model, self.mj_data)

        self._validate()
        self._set_scene_initial_state()

    def _set_scene_initial_state(self) -> None:
        self.set_robot_joint_positions(INIT_JOINT_POS)
        self.set_object_pose(self.config.object_start_pos, self.config.object_start_quat_wxyz)
        self.set_goal_object_pose(self.config.goal_object_start_pos, self.config.goal_object_start_quat_wxyz)
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

    def _add_table(self, spec) -> None:
        table_body = spec.worldbody.add_body(name="table")
        table_body.pos = TABLE_POS.copy()
        table_geom = table_body.add_geom(name="table_geom")
        table_geom.type = self.mujoco.mjtGeom.mjGEOM_BOX
        table_geom.size = TABLE_SIZE / 2.0
        table_geom.rgba = np.array([1.0, 1.0, 1.0, 1.0])
        table_geom.friction = self.config.friction_array.copy()
        table_geom.condim = 6

    def _add_lighting(self, spec) -> None:
        key_light = spec.worldbody.add_light(name="key_light")
        key_light.pos = np.array([0.4, -0.7, 2.2])
        key_light.dir = np.array([-0.2, 0.3, -1.0])
        key_light.diffuse = np.array([0.9, 0.9, 0.85])
        key_light.specular = np.array([0.25, 0.25, 0.25])

        fill_light = spec.worldbody.add_light(name="fill_light")
        fill_light.pos = np.array([-0.8, 0.7, 1.4])
        fill_light.dir = np.array([0.4, -0.3, -1.0])
        fill_light.diffuse = np.array([0.35, 0.38, 0.42])
        fill_light.specular = np.array([0.05, 0.05, 0.05])

        camera = spec.worldbody.add_camera(name="overview")
        camera.pos = np.array([0.65, -1.1, 1.15])
        camera.quat = np.array([0.86, 0.36, 0.18, 0.32])

    def _add_robot_collision_overlay(self, spec) -> None:
        robot_geoms = list(spec.geoms)
        for idx, geom in enumerate(robot_geoms):
            if geom.contype == 0 and geom.conaffinity == 0:
                continue
            overlay = geom.parent.add_geom(name=f"robot_collision_overlay_{idx}")
            overlay.type = geom.type
            overlay.meshname = geom.meshname
            overlay.pos = geom.pos.copy()
            overlay.quat = geom.quat.copy()
            overlay.size = geom.size.copy()
            overlay.fromto = geom.fromto.copy()
            overlay.rgba = np.array([0.1, 0.45, 1.0, 0.28])
            overlay.group = 2
            overlay.contype = 0
            overlay.conaffinity = 4

    def _add_object(
        self,
        spec,
        name: str,
        color: np.ndarray,
        start_pos: np.ndarray,
        start_quat_wxyz: np.ndarray,
        disable_contacts: bool,
        movable: bool,
    ) -> None:
        body = spec.worldbody.add_body(name=name)
        body.pos = start_pos
        body.quat = start_quat_wxyz

        if movable:
            free_joint = body.add_joint(name=f"{name}_free_joint")
            free_joint.type = self.mujoco.mjtJoint.mjJNT_FREE

        geoms = []
        if self.config.object_name.startswith("cuboid"):
            geoms = self._add_cuboid_object(body, name, color)
        else:
            mesh_path, mesh_scale = _mesh_from_object_urdf(self.config.object_name)
            if name == "object":
                print(f"[MuJoCo] Loading object '{self.config.object_name}' from {mesh_path}")
            mesh_name = f"{name}_{self.config.object_name}_mesh"
            mesh = spec.add_mesh(name=mesh_name)
            mesh.file = str(mesh_path)
            mesh.scale = mesh_scale
            visual_geom = body.add_geom(name=f"{name}_{self.config.object_name}_visual_geom")
            visual_geom.type = self.mujoco.mjtGeom.mjGEOM_MESH
            visual_geom.meshname = mesh_name
            visual_geom.rgba = color
            visual_geom.contype = 0
            visual_geom.conaffinity = 4

            if self.config.use_proxy_object_collision:
                mins, maxs = _obj_bounds(mesh_path, mesh_scale)
                collision_geom = body.add_geom(name=f"{name}_{self.config.object_name}_collision_box")
                collision_geom.type = self.mujoco.mjtGeom.mjGEOM_BOX
                collision_geom.pos = 0.5 * (mins + maxs)
                collision_geom.size = np.maximum(0.5 * (maxs - mins), np.array([0.005, 0.005, 0.005]))
                collision_geom.rgba = np.array([color[0], color[1], color[2], 0.0])
                collision_geom.friction = self.config.friction_array.copy()
                collision_geom.density = 400.0
                geoms.append(collision_geom)
            else:
                visual_geom.contype = 1
                visual_geom.conaffinity = 1
                visual_geom.density = 400.0
                visual_geom.friction = self.config.friction_array.copy()
                geoms.append(visual_geom)

        if disable_contacts:
            for geom in geoms:
                geom.contype = 0
                geom.conaffinity = 0

    def _add_cuboid_object(self, body, name: str, color: np.ndarray) -> list:
        if self.config.object_name in {"cuboidal_mallet", "cuboidal_hammer"}:
            handle_length, handle_width, handle_thickness = 0.24, 0.03, 0.02
            head_width, head_length, head_thickness = 0.05, 0.08, 0.045
            if self.config.object_name == "cuboidal_hammer":
                handle_length, head_width, head_length, head_thickness = 0.25, 0.02, 0.11, 0.02

            handle = body.add_geom(name=f"{name}_handle_geom")
            handle.type = self.mujoco.mjtGeom.mjGEOM_BOX
            handle.size = np.array([handle_length / 2.0, handle_width / 2.0, handle_thickness / 2.0])
            handle.rgba = color
            handle.friction = self.config.friction_array.copy()
            handle.density = 400.0

            head = body.add_geom(name=f"{name}_head_geom")
            head.type = self.mujoco.mjtGeom.mjGEOM_BOX
            head.size = np.array([head_width / 2.0, head_length / 2.0, head_thickness / 2.0])
            head.pos = np.array([handle_length / 2.0 + head_width / 2.0, 0.0, 0.0])
            head.rgba = color
            head.friction = self.config.friction_array.copy()
            head.density = 400.0
            return [handle, head]

        scales = np.array(self.config.object_name.split("_")[1:], dtype=float)
        if scales.shape != (3,):
            raise ValueError(f"Invalid cuboid object name: {self.config.object_name}")
        geom = body.add_geom(name=f"{name}_object_geom")
        geom.type = self.mujoco.mjtGeom.mjGEOM_BOX
        geom.size = 0.5 * 0.04 * scales
        geom.rgba = color
        geom.friction = self.config.friction_array.copy()
        geom.density = 400.0
        return [geom]

    def _validate(self) -> None:
        missing = [name for name in JOINT_NAMES if name not in self.joint_names]
        if missing:
            raise RuntimeError(f"MuJoCo model is missing policy joints: {missing}")
        for body_name in ("object",):
            if body_name not in self.body_names:
                raise RuntimeError(f"MuJoCo model is missing body '{body_name}'.")

    def set_object_pose(self, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
        self._set_free_body_pose("object", pos, quat_wxyz)
        self.mujoco.mj_forward(self.mj_model, self.mj_data)

    def set_goal_object_pose(self, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
        if "goal_object" not in self.body_names:
            return
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
        if "goal_object" in self.body_names:
            goal_pos, goal_quat_wxyz = self.get_body_pose("goal_object")
        else:
            goal_pos = self.config.goal_object_start_pos.copy()
            goal_quat_wxyz = self.config.goal_object_start_quat_wxyz.copy()
        table_pos = TABLE_POS.copy()
        table_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        base_name = "iiwa14_link_0" if "iiwa14_link_0" in self.body_names else self.body_names[1]
        if "base" in self.body_names:
            base_name = "base"
        base_pos, base_quat_wxyz = self.get_body_pose(base_name)
        return {
            "table_pos": table_pos,
            "table_quat_wxyz": table_quat_wxyz,
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
    sim = MujocoSim(MujocoSimConfig(enable_viewer=True))
    sim.run()


if __name__ == "__main__":
    main()
