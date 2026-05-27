"""Generate centered, Z-axis-aligned wrapper USDs for M6/M10 nut-screw assets.

The source M6/M10 USDs are left untouched. This script writes wrapper USDs that
reference each source asset, rotate its original screw/nut axis onto local Z,
and translate its visual bounds center onto the wrapper origin.

Run with Isaac Sim / Isaac Lab Python so the ``pxr`` module is available:

    python simtoolreal_lab/tasks/sharpa_nutscrew_pick_place/tests/reform_nutscrew_usds.py
    python simtoolreal_lab/tasks/sharpa_nutscrew_pick_place/tests/reform_nutscrew_usds.py --source-axis x --target-axis z
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher


ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets"


parser = argparse.ArgumentParser(description="Create recentered nut-screw USD wrappers.")
parser.add_argument(
    "--input-roots",
    nargs="+",
    type=Path,
    default=[ASSET_ROOT / "M6", ASSET_ROOT / "M10"],
    help="Folders containing source USDs.",
)
parser.add_argument(
    "--output-root",
    type=Path,
    default=ASSET_ROOT / "nutscrew_reformed",
    help="Folder where corrected wrapper USDs are written.",
)
parser.add_argument(
    "--source-axis",
    choices=("auto", "x", "y", "z"),
    default="auto",
    help="Axis used by the original assets for the rotational axis. Auto uses x for screws and y for nuts/washers.",
)
parser.add_argument(
    "--target-axis",
    choices=("x", "y", "z"),
    default="z",
    help="Desired canonical rotational axis in generated wrappers.",
)
parser.add_argument(
    "--meters-per-source-unit",
    type=float,
    default=0.001,
    help="Scale applied to source CAD units. Default treats source USD coordinates as millimeters.",
)
parser.add_argument(
    "--only",
    nargs="+",
    default=None,
    help="Optional asset stems to process, for example: --only M6X20 M6_nut",
)
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated USDs.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

print("[INFO] Launching Isaac Sim so USD/PXR APIs are available...", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Gf, Usd, UsdGeom  # noqa: E402

print("[INFO] Isaac Sim launched. Starting nut-screw USD reform batch.", flush=True)


AXES = {
    "x": Gf.Vec3d(1.0, 0.0, 0.0),
    "y": Gf.Vec3d(0.0, 1.0, 0.0),
    "z": Gf.Vec3d(0.0, 0.0, 1.0),
}


def _rotation_to_target_axis(source_axis: str, target_axis: str) -> Gf.Rotation:
    source = AXES[source_axis]
    target = AXES[target_axis]
    dot = max(-1.0, min(1.0, Gf.Dot(source, target)))
    if dot > 1.0 - 1.0e-8:
        return Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), 0.0)
    if dot < -1.0 + 1.0e-8:
        helper = AXES["y"] if source_axis != "y" else AXES["x"]
        axis = Gf.Cross(source, helper).GetNormalized()
        return Gf.Rotation(axis, 180.0)
    axis = Gf.Cross(source, target).GetNormalized()
    angle = math.degrees(math.acos(dot))
    return Gf.Rotation(axis, angle)


def _rotation_matrix(rotation: Gf.Rotation) -> Gf.Matrix4d:
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(rotation)
    return matrix


def _rotated_bounds(source_usd: Path, rotation: Gf.Rotation) -> tuple[Gf.Vec3d, Gf.Vec3d]:
    stage = Usd.Stage.Open(str(source_usd))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {source_usd}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    rot_matrix = _rotation_matrix(rotation)
    min_bound = Gf.Vec3d(float("inf"), float("inf"), float("inf"))
    max_bound = Gf.Vec3d(float("-inf"), float("-inf"), float("-inf"))
    point_count = 0

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if not points:
            continue
        local_to_world = xform_cache.GetLocalToWorldTransform(prim)
        for point in points:
            point_w = local_to_world.Transform(Gf.Vec3d(point))
            point_r = rot_matrix.Transform(point_w)
            min_bound = Gf.Vec3d(
                min(min_bound[0], point_r[0]),
                min(min_bound[1], point_r[1]),
                min(min_bound[2], point_r[2]),
            )
            max_bound = Gf.Vec3d(
                max(max_bound[0], point_r[0]),
                max(max_bound[1], point_r[1]),
                max(max_bound[2], point_r[2]),
            )
        point_count += len(points)

    if point_count == 0:
        raise RuntimeError(f"No mesh points found in USD asset: {source_usd}")
    print(f"[INFO] Read {point_count} mesh points from {source_usd.name}.", flush=True)
    return min_bound, max_bound


def _write_wrapper(source_usd: Path, output_usd: Path, rotation: Gf.Rotation, source_axis: str) -> None:
    if output_usd.exists() and not args.overwrite:
        print(f"[SKIP] Exists: {output_usd}", flush=True)
        return

    print(f"[INFO] Computing rotated bounds for {source_usd}...", flush=True)
    min_bound, max_bound = _rotated_bounds(source_usd, rotation)
    center = (min_bound + max_bound) * 0.5
    size = max_bound - min_bound

    output_usd.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_usd))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    root = UsdGeom.Xform.Define(stage, "/Object")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.XformCommonAPI(root).SetScale(
        (args.meters_per_source_unit, args.meters_per_source_unit, args.meters_per_source_unit)
    )

    centered = UsdGeom.Xform.Define(stage, "/Object/Centered")
    UsdGeom.XformCommonAPI(centered).SetTranslate((-center[0], -center[1], -center[2]))

    aligned = UsdGeom.Xform.Define(stage, "/Object/Centered/Aligned")
    UsdGeom.XformCommonAPI(aligned).SetRotate(
        rotation.Decompose(Gf.Vec3d(1.0, 0.0, 0.0), Gf.Vec3d(0.0, 1.0, 0.0), Gf.Vec3d(0.0, 0.0, 1.0)),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )

    geometry = UsdGeom.Xform.Define(stage, "/Object/Centered/Aligned/Geometry")
    geometry.GetPrim().GetReferences().AddReference(str(source_usd))

    stage.GetRootLayer().Save()
    scaled_size = size * args.meters_per_source_unit
    print(
        f"[OK] {source_usd.name} -> {output_usd} "
        f"axis={source_axis}->{args.target_axis} "
        f"source_bbox=({size[0]:.5f}, {size[1]:.5f}, {size[2]:.5f}) "
        f"scaled_bbox_m=({scaled_size[0]:.5f}, {scaled_size[1]:.5f}, {scaled_size[2]:.5f}) "
        f"center=({center[0]:.5f}, {center[1]:.5f}, {center[2]:.5f})",
        flush=True,
    )


def _source_axis_for_asset(source_usd: Path) -> str:
    if args.source_axis != "auto":
        return args.source_axis
    name = source_usd.stem.lower()
    if "washer" in name or "nut" in name:
        return "y"
    return "x"


def main() -> None:
    source_paths: list[Path] = []
    for input_root in args.input_roots:
        source_paths.extend(sorted(input_root.expanduser().resolve().glob("*.usd")))
    if args.only is not None:
        only = set(args.only)
        source_paths = [path for path in source_paths if path.stem in only]
    if not source_paths:
        roots = ", ".join(str(path) for path in args.input_roots)
        raise FileNotFoundError(f"No USD files found in: {roots}")

    print(f"[INFO] Found {len(source_paths)} USD asset(s) to reform.", flush=True)
    for source_usd in source_paths:
        source_axis = _source_axis_for_asset(source_usd)
        rotation = _rotation_to_target_axis(source_axis, args.target_axis)
        family = source_usd.parent.name
        output_usd = args.output_root.expanduser().resolve() / family / source_usd.name
        _write_wrapper(source_usd, output_usd, rotation, source_axis)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
