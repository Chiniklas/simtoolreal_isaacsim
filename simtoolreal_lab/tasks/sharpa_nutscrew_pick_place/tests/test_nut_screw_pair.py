"""Visualize a nut and screw pair in Isaac Sim.

Example:
    python simtoolreal_lab/tasks/sharpa_nutscrew_pick_place/tests/test_nut_screw_pair.py
    python simtoolreal_lab/tasks/sharpa_nutscrew_pick_place/tests/test_nut_screw_pair.py --family M10 --screw M10X25 --nut M10_nut
    python simtoolreal_lab/tasks/sharpa_nutscrew_pick_place/tests/test_nut_screw_pair.py --asset-root simtoolreal_lab/assets/nutscrew_generated
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Spawn one nut and one screw USD asset for visual inspection.")
parser.add_argument("--family", choices=("M6", "M10"), default="M6", help="Asset family to use for default names.")
parser.add_argument("--screw", default=None, help="Screw asset stem, for example M6X20 or M10X25.")
parser.add_argument("--nut", default=None, help="Nut asset stem, for example M6_nut or M10_nut.")
parser.add_argument(
    "--asset-root",
    type=Path,
    default=Path(__file__).resolve().parents[3] / "assets",
    help="Root folder containing M6/M10 subfolders.",
)
parser.add_argument("--spacing", type=float, default=0.08, help="Distance between the nut and screw in meters.")
parser.add_argument("--z", type=float, default=0.02, help="Initial spawn height in meters.")
parser.add_argument("--scale", type=float, default=1.0, help="Uniform USD scale.")
parser.add_argument("--clearance", type=float, default=0.002, help="Gap above the ground after auto-placement.")
parser.add_argument("--no-auto-ground", action="store_true", help="Do not lift assets based on their world bounds.")
parser.add_argument("--steps", type=int, default=0, help="Number of sim steps to run. Use 0 to run until the viewer closes.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _asset_path(asset_name: str, asset_root: Path | None = None) -> Path:
    family = "M10" if asset_name.startswith("M10") else "M6"
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
from pxr import Usd, UsdGeom  # noqa: E402
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane  # noqa: E402


def _spawn_usd(prim_path: str, usd_path: Path, pos: tuple[float, float, float]) -> None:
    cfg = sim_utils.UsdFileCfg(
        usd_path=str(usd_path),
        scale=(args_cli.scale, args_cli.scale, args_cli.scale),
    )
    cfg.func(prim_path, cfg, translation=pos, orientation=(1.0, 0.0, 0.0, 0.0))


def _world_bounds(prim_path: str):
    stage = sim_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    aligned_box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    return aligned_box.GetMin(), aligned_box.GetMax()


def _place_bottom_on_ground(prim_path: str, x: float, y: float) -> None:
    min_bound, _max_bound = _world_bounds(prim_path)
    min_z = min_bound[2]
    prim = sim_utils.get_current_stage().GetPrimAtPath(prim_path)
    target_z = args_cli.clearance - min_z
    UsdGeom.XformCommonAPI(prim).SetTranslate((x, y, target_z))


def _print_bounds(label: str, prim_path: str) -> None:
    min_bound, max_bound = _world_bounds(prim_path)
    size = tuple(max_bound[i] - min_bound[i] for i in range(3))
    print(f"[INFO] {label} visual bbox size: ({size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f}) m")


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, render_interval=1)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(0.28, -0.36, 0.22), target=(0.0, 0.0, 0.02))

    spawn_ground_plane("/World/ground", GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    half_spacing = 0.5 * args_cli.spacing
    _spawn_usd("/World/screw", SCREW_PATH, (-half_spacing, 0.0, args_cli.z))
    _spawn_usd("/World/nut", NUT_PATH, (half_spacing, 0.0, args_cli.z))

    sim.reset()
    if not args_cli.no_auto_ground:
        _place_bottom_on_ground("/World/screw", -half_spacing, 0.0)
        _place_bottom_on_ground("/World/nut", half_spacing, 0.0)
        sim.step()

    print(f"[INFO] Spawned screw: {SCREW_PATH}")
    print(f"[INFO] Spawned nut:   {NUT_PATH}")
    _print_bounds("Screw", "/World/screw")
    _print_bounds("Nut", "/World/nut")
    print("[INFO] Close the Isaac Sim window to stop, or pass --steps N for a finite run.")

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
