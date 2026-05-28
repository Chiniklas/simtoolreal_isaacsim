"""Standalone Isaac Lab nut/screw spawn scene.

This is the IsaacSim counterpart to
``simtoolreal_lab/deployment/mujoco_nut_screw/test_nut_screw_meshing.py``.
It intentionally builds a tiny scene only containing:

* a table,
* a screw with its head down on the table and thread along +Z,
* a nut placed concentrically at the top of the screw thread.

The generated screw mesh convention is the same one used by the MuJoCo test:
the screw origin is at the head bearing surface, the threaded shaft points
along local +Z, and the head extends along local -Z.  This script reads the OBJ
bounds for placement and spawns the matching USD files for visualization.

The nut motion is pure kinematic screw motion.  Each frame the script commands
nut yaw and sets vertical slide from the metric pitch:
``slide = -pitch / (2*pi) * yaw``.  Positive spin therefore drives the nut down
along the fixed screw.

Run from the repository root, for example:

.. code-block:: bash

    python simtoolreal_lab/tasks/sharpa_nutscrew_pick_place_screw/tests/test_nut_screw_meshing.py

If Isaac Lab is installed only in the project environment:

.. code-block:: bash

    conda run -n simtoolreal python simtoolreal_lab/tasks/sharpa_nutscrew_pick_place_screw/tests/test_nut_screw_meshing.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ASSET_ROOT = REPO_ROOT / "simtoolreal_lab/assets/nutscrew_generated"
TABLE_SIZE = np.array([0.24, 0.20, 0.04], dtype=float)
TABLE_TOP_Z = 0.53
TABLE_POS = np.array([0.0, 0.0, TABLE_TOP_Z - 0.5 * TABLE_SIZE[2]], dtype=float)
SIM_DT = 1.0 / 60.0
FAMILY_PITCHES = {
    "M6": 0.001,
    "M8": 0.00125,
    "M10": 0.0015,
    "M12": 0.00175,
    "M16": 0.002,
    "M20": 0.0025,
}


@dataclass(frozen=True)
class MeshBounds:
    mins: np.ndarray
    maxs: np.ndarray

    @property
    def size(self) -> np.ndarray:
        return self.maxs - self.mins


@dataclass(frozen=True)
class SpawnPose:
    screw_pos: np.ndarray
    nut_pos: np.ndarray
    screw_top_z: float
    nut_bottom_z: float


def _asset_path(asset_root: Path, family: str, name: str, suffix: str) -> Path:
    path = asset_root.expanduser().resolve() / family / f"{name}.{suffix}"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {suffix.upper()} mesh: {path}\n"
            "Generate assets with:\n"
            "  python simtoolreal_lab/tasks/sharpa_nutscrew_pick_place_screw/tests/screw_generator.py "
            "--families M8 M10 M12 M16 M20 --formats usd obj stl --overwrite"
        )
    return path


def _obj_bounds(mesh_path: Path) -> MeshBounds:
    vertices = []
    with mesh_path.open("r", encoding="utf-8", errors="ignore") as obj_file:
        for line in obj_file:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
    if not vertices:
        raise ValueError(f"OBJ has no vertices: {mesh_path}")

    vertices_np = np.asarray(vertices, dtype=float)
    return MeshBounds(mins=vertices_np.min(axis=0), maxs=vertices_np.max(axis=0))


def _compute_spawn_pose(screw_bounds: MeshBounds, nut_bounds: MeshBounds, clearance: float) -> SpawnPose:
    screw_origin_z = TABLE_TOP_Z - float(screw_bounds.mins[2])
    screw_top_z = screw_origin_z + float(screw_bounds.maxs[2])
    nut_center_z = screw_top_z + clearance - float(nut_bounds.mins[2])
    return SpawnPose(
        screw_pos=np.array([0.0, 0.0, screw_origin_z], dtype=float),
        nut_pos=np.array([0.0, 0.0, nut_center_z], dtype=float),
        screw_top_z=screw_top_z,
        nut_bottom_z=nut_center_z + float(nut_bounds.mins[2]),
    )


def _thread_slope(family: str) -> float:
    return -FAMILY_PITCHES.get(family, FAMILY_PITCHES["M12"]) / (2.0 * math.pi)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a simple Isaac Lab nut/screw scene.")
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--family", default="M12")
    parser.add_argument("--screw", default="M12X30")
    parser.add_argument("--nut", default="M12_nut")
    parser.add_argument("--clearance", type=float, default=0.002, help="Gap between screw tip and nut bottom, in meters.")
    parser.add_argument(
        "--spin-velocity",
        type=float,
        default=6.0,
        help="Commanded nut angular velocity around Z, rad/s. Positive spins the nut downward.",
    )
    parser.add_argument(
        "--cycle",
        action="store_true",
        help="Restart the nut at the thread top after it reaches the head-side end of the thread.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Run a finite number of steps. Default runs until the app closes.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _make_pose_ops(prim_path: str):
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing prim for kinematic pose update: {prim_path}")

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    orient_op = xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    return translate_op, orient_op


def _set_kinematic_pose(pose_ops, position: np.ndarray, yaw: float) -> None:
    from pxr import Gf

    half_yaw = 0.5 * yaw
    pose_ops[0].Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    pose_ops[1].Set(Gf.Quatd(float(math.cos(half_yaw)), Gf.Vec3d(0.0, 0.0, float(math.sin(half_yaw)))))


def _spawn_scene(args: argparse.Namespace, screw_usd: Path, nut_usd: Path, pose: SpawnPose) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
    from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

    table_cfg = sim_utils.CuboidCfg(
        size=tuple(TABLE_SIZE.tolist()),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.56, 0.35)),
        physics_material=RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
    )
    table_cfg.func(
        "/World/table",
        table_cfg,
        translation=tuple(TABLE_POS.tolist()),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )

    fixed_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
    screw_cfg = sim_utils.UsdFileCfg(
        usd_path=str(screw_usd),
        rigid_props=fixed_props,
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    screw_cfg.func(
        "/World/fixed_screw",
        screw_cfg,
        translation=tuple(pose.screw_pos.tolist()),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )

    nut_cfg = sim_utils.UsdFileCfg(
        usd_path=str(nut_usd),
        rigid_props=fixed_props,
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    nut_cfg.func(
        "/World/nut",
        nut_cfg,
        translation=tuple(pose.nut_pos.tolist()),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)


def main() -> None:
    args = _parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import isaaclab.sim as sim_utils

    screw_obj = _asset_path(args.asset_root, args.family, args.screw, "obj")
    nut_obj = _asset_path(args.asset_root, args.family, args.nut, "obj")
    screw_usd = _asset_path(args.asset_root, args.family, args.screw, "usd")
    nut_usd = _asset_path(args.asset_root, args.family, args.nut, "usd")

    screw_bounds = _obj_bounds(screw_obj)
    nut_bounds = _obj_bounds(nut_obj)
    pose = _compute_spawn_pose(screw_bounds, nut_bounds, args.clearance)
    thread_length = max(float(screw_bounds.maxs[2]), 0.0)
    slope = _thread_slope(args.family)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=SIM_DT))
    _spawn_scene(args, screw_usd, nut_usd, pose)
    nut_pose_ops = _make_pose_ops("/World/nut")
    _set_kinematic_pose(nut_pose_ops, pose.nut_pos, 0.0)
    if hasattr(sim, "set_camera_view"):
        sim.set_camera_view(eye=(0.16, -0.30, 0.72), target=(0.0, 0.0, TABLE_TOP_Z + 0.02))
    sim.reset()

    print(f"[INFO] Screw USD: {screw_usd}")
    print(f"[INFO] Nut USD:   {nut_usd}")
    print(f"[INFO] Screw pose: {np.round(pose.screw_pos, 5).tolist()} quat=[1.0, 0.0, 0.0, 0.0]")
    print(f"[INFO] Nut pose:   {np.round(pose.nut_pos, 5).tolist()} quat=[1.0, 0.0, 0.0, 0.0]")
    print(f"[INFO] Screw top z: {pose.screw_top_z:.4f} m")
    print(f"[INFO] Nut bottom z: {pose.nut_bottom_z:.4f} m")
    print(f"[INFO] Screw bbox size: {np.round(screw_bounds.size, 5).tolist()} m")
    print(f"[INFO] Nut bbox size:   {np.round(nut_bounds.size, 5).tolist()} m")
    print(f"[INFO] Analytical thread pitch: {FAMILY_PITCHES.get(args.family, FAMILY_PITCHES['M12']):.6f} m/rev")
    print(f"[INFO] Kinematic spin velocity: {args.spin_velocity:.3f} rad/s")

    step = 0
    max_steps = args.steps
    if max_steps is None and getattr(args, "headless", False):
        max_steps = 240

    while simulation_app.is_running() and (max_steps is None or step < max_steps):
        elapsed = step * SIM_DT
        commanded_yaw = args.spin_velocity * elapsed
        commanded_travel = -slope * commanded_yaw
        if args.cycle and thread_length > 0.0:
            travel = commanded_travel % thread_length
        else:
            travel = min(max(commanded_travel, 0.0), thread_length)
        yaw = travel / -slope if slope != 0.0 else 0.0
        slide = -travel
        nut_pos = pose.nut_pos + np.array([0.0, 0.0, slide], dtype=float)
        _set_kinematic_pose(nut_pose_ops, nut_pos, yaw)
        sim.step(render=True)
        step += 1

    simulation_app.close()


if __name__ == "__main__":
    main()
