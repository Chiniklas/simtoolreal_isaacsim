"""Simulate a nut and screw pair on a table in Isaac Sim.

Example:
    python simtoolreal_lab/tasks/sharpa_nutscrew_pick_place/tests/test_nut_screw_pair.py
    python simtoolreal_lab/tasks/sharpa_nutscrew_pick_place/tests/test_nut_screw_pair.py --family M12 --screw M12X30 --nut M12_nut
    python simtoolreal_lab/tasks/sharpa_nutscrew_pick_place/tests/test_nut_screw_pair.py --asset-root simtoolreal_lab/assets/nutscrew_generated
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Spawn a fixed screw and dynamic concentric nut on a table.")
parser.add_argument("--family", default="M6", help="Asset family to use for default names, for example M6, M10, or M12.")
parser.add_argument("--screw", default=None, help="Screw asset stem, for example M6X20, M10X25, or M12X30.")
parser.add_argument("--nut", default=None, help="Nut asset stem, for example M6_nut, M10_nut, or M12_nut.")
parser.add_argument(
    "--asset-root",
    type=Path,
    default=Path(__file__).resolve().parents[3] / "assets",
    help="Root folder containing metric family subfolders.",
)
parser.add_argument("--scale", type=float, default=1.0, help="Uniform USD scale.")
parser.add_argument("--clearance", type=float, default=0.002, help="Gap above the table after placement.")
parser.add_argument(
    "--nut-start-height",
    type=float,
    default=None,
    help="Nut center height above the screw/head interface. Default places the nut bottom near the screw thread tip.",
)
parser.add_argument("--nut-tip-offset", type=float, default=-0.003, help="Distance below the screw tip for the initial nut bottom. Negative starts above the tip.")
parser.add_argument("--nut-spin", type=float, default=25.0, help="Initial nut angular velocity around +Z in rad/s.")
parser.add_argument("--nut-mass", type=float, default=0.03, help="Nut mass in kg.")
parser.add_argument("--sdf-resolution", type=int, default=256, help="SDF collision resolution for threaded meshes.")
parser.add_argument("--table-height", type=float, default=0.3, help="Table thickness/height in meters.")
parser.add_argument("--table-top-z", type=float, default=0.53, help="Table top z position, matching the training env by default.")
parser.add_argument("--no-pause", action="store_true", help="Start simulation immediately without waiting for Enter.")
parser.add_argument("--steps", type=int, default=0, help="Number of sim steps to run. Use 0 to run until the viewer closes.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _asset_path(asset_name: str, asset_root: Path | None = None) -> Path:
    family = asset_name.split("_", 1)[0].split("X", 1)[0]
    root = args_cli.asset_root if asset_root is None else asset_root
    path = root.expanduser().resolve() / family / f"{asset_name}.usd"
    if not path.is_file():
        available = sorted(candidate.stem for candidate in path.parent.glob("*.usd")) if path.parent.is_dir() else []
        hint = f" Available in {path.parent}: {', '.join(available)}" if available else ""
        raise FileNotFoundError(f"Missing USD asset: {path}.{hint}")
    return path


SCREW_NAME = args_cli.screw or f"{args_cli.family}X20"
NUT_NAME = args_cli.nut or f"{args_cli.family}_nut"
SCREW_PATH = _asset_path(SCREW_NAME)
NUT_PATH = _asset_path(NUT_NAME)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane  # noqa: E402


def _spawn_usd(prim_path: str, usd_path: Path, pos: tuple[float, float, float]) -> None:
    cfg = sim_utils.UsdFileCfg(
        usd_path=str(usd_path),
        scale=(args_cli.scale, args_cli.scale, args_cli.scale),
    )
    cfg.func(prim_path, cfg, translation=pos, orientation=(1.0, 0.0, 0.0, 0.0))


def _set_translate(prim_path: str, pos: tuple[float, float, float]) -> None:
    prim = sim_utils.get_current_stage().GetPrimAtPath(prim_path)
    UsdGeom.XformCommonAPI(prim).SetTranslate(pos)


def _world_bounds(prim_path: str):
    stage = sim_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    aligned_box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    return aligned_box.GetMin(), aligned_box.GetMax()


def _place_bottom_on_z(prim_path: str, x: float, y: float, target_bottom_z: float) -> float:
    min_bound, _max_bound = _world_bounds(prim_path)
    min_z = min_bound[2]
    target_z = target_bottom_z - min_z
    _set_translate(prim_path, (x, y, target_z))
    return target_z


def _mesh_prims_under(prim_path: str):
    stage = sim_utils.get_current_stage()
    root = stage.GetPrimAtPath(prim_path)
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            yield prim


def _apply_collision(prim_path: str, approximation: str = "sdf") -> None:
    for mesh_prim in _mesh_prims_under(prim_path):
        if not mesh_prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(mesh_prim)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        mesh_collision.CreateApproximationAttr().Set(approximation)
        if approximation == "sdf":
            sdf_collision = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh_prim)
            sdf_collision.CreateSdfResolutionAttr().Set(args_cli.sdf_resolution)
            sdf_collision.CreateSdfMarginAttr().Set(0.001)


def _apply_fixed_body(prim_path: str) -> None:
    prim = sim_utils.get_current_stage().GetPrimAtPath(prim_path)
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_body.CreateRigidBodyEnabledAttr(True)
    rigid_body.CreateKinematicEnabledAttr(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim)


def _apply_dynamic_body(prim_path: str, mass: float, angular_velocity_z: float) -> None:
    prim = sim_utils.get_current_stage().GetPrimAtPath(prim_path)
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_body.CreateRigidBodyEnabledAttr(True)
    rigid_body.CreateKinematicEnabledAttr(False)
    rigid_body.CreateAngularVelocityAttr().Set((0.0, 0.0, angular_velocity_z))
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)
    physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_body.CreateDisableGravityAttr(False)
    physx_body.CreateSolverPositionIterationCountAttr(16)
    physx_body.CreateSolverVelocityIterationCountAttr(4)


def _print_bounds(label: str, prim_path: str) -> None:
    min_bound, max_bound = _world_bounds(prim_path)
    size = tuple(max_bound[i] - min_bound[i] for i in range(3))
    print(f"[INFO] {label} visual bbox size: ({size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f}) m")


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=1)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(0.32, -0.42, args_cli.table_top_z + 0.16), target=(0.0, 0.0, args_cli.table_top_z + 0.035))

    spawn_ground_plane("/World/ground", GroundPlaneCfg())
    table_center_z = args_cli.table_top_z - 0.5 * args_cli.table_height
    table_cfg = sim_utils.CuboidCfg(
        size=(0.475, 0.4, args_cli.table_height),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.56, 0.35)),
    )
    table_cfg.func("/World/table", table_cfg, translation=(0.0, 0.0, table_center_z))
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    screw_bottom_z = args_cli.table_top_z + args_cli.clearance
    _spawn_usd("/World/screw", SCREW_PATH, (0.0, 0.0, screw_bottom_z))
    _spawn_usd("/World/nut", NUT_PATH, (0.0, 0.0, args_cli.table_top_z))
    screw_origin_z = _place_bottom_on_z("/World/screw", 0.0, 0.0, screw_bottom_z)
    _screw_min, screw_max = _world_bounds("/World/screw")
    nut_start_height = args_cli.nut_start_height
    if nut_start_height is None:
        nut_min, nut_max = _world_bounds("/World/nut")
        nut_half_height = 0.5 * (nut_max[2] - nut_min[2])
        nut_start_height = max(0.0, screw_max[2] - screw_origin_z - args_cli.nut_tip_offset + nut_half_height)
    nut_start_z = screw_origin_z + nut_start_height
    _set_translate("/World/nut", (0.0, 0.0, nut_start_z))

    _apply_collision("/World/screw", approximation="sdf")
    _apply_collision("/World/nut", approximation="sdf")
    _apply_fixed_body("/World/screw")
    _apply_dynamic_body("/World/nut", args_cli.nut_mass, 0.0)

    sim.reset()
    sim.step()

    print(f"[INFO] Spawned screw: {SCREW_PATH}")
    print(f"[INFO] Spawned nut:   {NUT_PATH}")
    print(f"[INFO] Screw fixed on table, thread along +Z. Screw top z: {screw_max[2]:.4f} m")
    print(f"[INFO] Nut starts concentric with its bottom near the thread tip at z={nut_start_z:.4f} m with spin={args_cli.nut_spin:.2f} rad/s")
    _print_bounds("Screw", "/World/screw")
    _print_bounds("Nut", "/World/nut")
    print("[INFO] Close the Isaac Sim window to stop, or pass --steps N for a finite run.")
    if not args_cli.no_pause:
        input("[INFO] Press Enter to start gravity/spin simulation...")
    UsdPhysics.RigidBodyAPI(sim_utils.get_current_stage().GetPrimAtPath("/World/nut")).CreateAngularVelocityAttr().Set(
        (0.0, 0.0, args_cli.nut_spin)
    )

    step_count = 0
    max_steps = args_cli.steps if args_cli.steps > 0 else None
    print(f"[INFO] Viewer running: {simulation_app.is_running()}")
    while simulation_app.is_running():
        sim.step()
        step_count += 1
        if max_steps is not None and step_count >= max_steps:
            break


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
