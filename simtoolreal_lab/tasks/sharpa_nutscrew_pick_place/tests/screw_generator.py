"""Procedural nut/screw USD generator for the SHARPA nut-screw task.

README
======

Why this exists
---------------
The original assets in ``simtoolreal_lab/assets/M6`` and ``M10`` appear to come
from CAD-style Maker-space models. They are visually detailed, authored in a CAD
unit scale that Isaac interprets poorly, and their origins/axes are inconvenient
for manipulation:

* sizes can appear 1000x too large unless scaled from millimeters to meters;
* origins are not consistently at the geometric center;
* the screw/nut rotational axis is not consistently local Z;
* detailed thread meshes are slow to load when repeated across many envs.

This script is a native Python replacement for the Maker-space OpenSCAD idea. It
does not require OpenSCAD or BOSL2. It writes simple ASCII USD mesh files that
Isaac Sim can load directly. The generated assets are intentionally low detail:
good enough for task development, visualization, grasping, and early RL tests,
but not meant to be manufacturing-grade CAD.

Coordinate and scale conventions
--------------------------------
All generated geometry follows these conventions:

* Units: meters.
* Origin: center of the complete generated mesh bounding box.
* Axis: local Z is the screw/nut/washer rotational axis.
* Screw orientation: shaft extends downward from the head before recentering.
  After recentering, the whole object is centered around the origin.
* Meshes are visual meshes only. If you need production physics, add simplified
  collision approximation in the task config or as a separate collision USD.

Default output
--------------
By default this writes a complete M6/M10 set matching the current asset names:

* ``M6X5.usd``, ``M6X8.usd``, ..., ``M6X70.usd``
* ``M6_nut.usd``, ``M6_nut_low.usd``, ``M6_nut_flange.usd``
* ``M6_domenut.usd``, ``M6_wingnut.usd``
* ``M6_washer.usd``, ``M6_washer_extended.usd``
* the same pattern for M10

Generated assets go to:

``simtoolreal_lab/assets/nutscrew_generated/M6``
``simtoolreal_lab/assets/nutscrew_generated/M10``

Typical usage
-------------
Generate the full default library:

.. code-block:: bash

    python simtoolreal_lab/tasks/sharpa_nutscrew_pick_place/tests/screw_generator.py --overwrite

Generate only M6:

.. code-block:: bash

    python simtoolreal_lab/tasks/sharpa_nutscrew_pick_place/tests/screw_generator.py --families M6 --overwrite

Generate lower-detail assets for faster loading:

.. code-block:: bash

    python simtoolreal_lab/tasks/sharpa_nutscrew_pick_place/tests/screw_generator.py --segments 24 --thread-steps-per-turn 5 --overwrite

Visualize the generated assets:

.. code-block:: bash

    python simtoolreal_lab/tasks/sharpa_nutscrew_pick_place/tests/test_nut_screw_pair.py \\
      --asset-root simtoolreal_lab/assets/nutscrew_generated \\
      --family M6 --screw M6X20 --nut M6_nut

Design notes
------------
The script implements a pragmatic subset of the OpenSCAD/BOSL2 generator:

* screws use a cylindrical head and a helical-looking sinusoidal thread surface;
* nuts use an annular hex-prism mesh with circular bore;
* flange nuts add a low circular flange;
* dome nuts add a simple hemispherical cap;
* wing nuts add two simple oval wing solids;
* washers are annular discs.

Thread geometry is deliberately approximate. The goal is stable, compact Isaac
assets with correct pose conventions, not exact ISO thread topology.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path


ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets"
DEFAULT_OUTPUT_ROOT = ASSET_ROOT / "nutscrew_generated"


@dataclass(frozen=True)
class MetricSpec:
    diameter_mm: float
    pitch_mm: float
    head_diameter_mm: float
    head_height_mm: float
    nut_width_mm: float
    nut_height_mm: float
    washer_id_mm: float
    washer_od_mm: float
    washer_height_mm: float
    washer_extended_od_mm: float
    washer_extended_height_mm: float


SPECS = {
    "M6": MetricSpec(
        diameter_mm=6.0,
        pitch_mm=1.0,
        head_diameter_mm=10.0,
        head_height_mm=6.0,
        nut_width_mm=10.0,
        nut_height_mm=5.0,
        washer_id_mm=6.4,
        washer_od_mm=12.0,
        washer_height_mm=1.6,
        washer_extended_od_mm=24.0,
        washer_extended_height_mm=2.0,
    ),
    "M10": MetricSpec(
        diameter_mm=10.0,
        pitch_mm=1.5,
        head_diameter_mm=16.0,
        head_height_mm=10.0,
        nut_width_mm=17.0,
        nut_height_mm=8.0,
        washer_id_mm=10.5,
        washer_od_mm=20.0,
        washer_height_mm=2.0,
        washer_extended_od_mm=40.0,
        washer_extended_height_mm=3.0,
    ),
}

SCREW_LENGTHS = {
    "M6": (5, 8, 10, 12, 15, 20, 25, 70),
    "M10": (5, 8, 10, 12, 15, 20, 25, 30),
}


def mm(value: float) -> float:
    return value * 0.001


def add_vertex(vertices: list[tuple[float, float, float]], x: float, y: float, z: float) -> int:
    vertices.append((x, y, z))
    return len(vertices) - 1


def append_mesh(
    vertices: list[tuple[float, float, float]],
    faces: list[list[int]],
    mesh_vertices: list[tuple[float, float, float]],
    mesh_faces: list[list[int]],
) -> None:
    offset = len(vertices)
    vertices.extend(mesh_vertices)
    faces.extend([[idx + offset for idx in face] for face in mesh_faces])


def recenter(vertices: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    min_xyz = [min(v[i] for v in vertices) for i in range(3)]
    max_xyz = [max(v[i] for v in vertices) for i in range(3)]
    center = [(min_xyz[i] + max_xyz[i]) * 0.5 for i in range(3)]
    return [(x - center[0], y - center[1], z - center[2]) for x, y, z in vertices]


def cylinder(radius: float, height: float, segments: int, z_min: float = 0.0) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    bottom = []
    top = []
    z_max = z_min + height
    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        bottom.append(add_vertex(vertices, x, y, z_min))
        top.append(add_vertex(vertices, x, y, z_max))
    bottom_center = add_vertex(vertices, 0.0, 0.0, z_min)
    top_center = add_vertex(vertices, 0.0, 0.0, z_max)
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([bottom[i], bottom[j], top[j], top[i]])
        faces.append([bottom_center, bottom[i], bottom[j]])
        faces.append([top_center, top[j], top[i]])
    return vertices, faces


def helical_threaded_shaft(
    diameter: float,
    pitch: float,
    length: float,
    segments: int,
    steps_per_turn: int,
    z_min: float = 0.0,
) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    turns = max(length / pitch, 1.0)
    z_steps = max(int(turns * steps_per_turn), 8)
    core_radius = diameter * 0.46
    thread_height = diameter * 0.065

    def thread_profile(theta: float, z: float) -> float:
        phase = (z / pitch - theta / (2.0 * math.pi)) % 1.0
        triangular = 1.0 - abs(phase * 2.0 - 1.0)
        return core_radius + thread_height * triangular

    rings: list[list[int]] = []
    for k in range(z_steps + 1):
        z = z_min + length * k / z_steps
        ring = []
        for i in range(segments):
            theta = 2.0 * math.pi * i / segments
            radius = thread_profile(theta, z)
            ring.append(add_vertex(vertices, radius * math.cos(theta), radius * math.sin(theta), z))
        rings.append(ring)

    for k in range(z_steps):
        for i in range(segments):
            j = (i + 1) % segments
            faces.append([rings[k][i], rings[k][j], rings[k + 1][j], rings[k + 1][i]])

    bottom_center = add_vertex(vertices, 0.0, 0.0, z_min)
    top_center = add_vertex(vertices, 0.0, 0.0, z_min + length)
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([bottom_center, rings[0][i], rings[0][j]])
        faces.append([top_center, rings[-1][j], rings[-1][i]])
    return vertices, faces


def hex_radius_for_angle(across_flats_radius: float, angle: float) -> float:
    sector = (angle + math.pi / 6.0) % (math.pi / 3.0) - math.pi / 6.0
    return across_flats_radius / max(math.cos(sector), 1.0e-6)


def annular_hex_prism(
    across_flats: float,
    inner_radius: float,
    height: float,
    segments: int,
    z_min: float = 0.0,
) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    outer_bottom = []
    outer_top = []
    inner_bottom = []
    inner_top = []
    apothem = across_flats * 0.5
    z_max = z_min + height
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        outer_radius = hex_radius_for_angle(apothem, theta)
        outer_bottom.append(add_vertex(vertices, outer_radius * math.cos(theta), outer_radius * math.sin(theta), z_min))
        outer_top.append(add_vertex(vertices, outer_radius * math.cos(theta), outer_radius * math.sin(theta), z_max))
        inner_bottom.append(add_vertex(vertices, inner_radius * math.cos(theta), inner_radius * math.sin(theta), z_min))
        inner_top.append(add_vertex(vertices, inner_radius * math.cos(theta), inner_radius * math.sin(theta), z_max))

    for i in range(segments):
        j = (i + 1) % segments
        faces.append([outer_bottom[i], outer_bottom[j], outer_top[j], outer_top[i]])
        faces.append([inner_bottom[j], inner_bottom[i], inner_top[i], inner_top[j]])
        faces.append([outer_top[i], outer_top[j], inner_top[j], inner_top[i]])
        faces.append([outer_bottom[j], outer_bottom[i], inner_bottom[i], inner_bottom[j]])
    return vertices, faces


def annular_disc(
    inner_radius: float,
    outer_radius: float,
    height: float,
    segments: int,
    z_min: float = 0.0,
) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    ob = []
    ot = []
    ib = []
    it = []
    z_max = z_min + height
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        ob.append(add_vertex(vertices, outer_radius * math.cos(theta), outer_radius * math.sin(theta), z_min))
        ot.append(add_vertex(vertices, outer_radius * math.cos(theta), outer_radius * math.sin(theta), z_max))
        ib.append(add_vertex(vertices, inner_radius * math.cos(theta), inner_radius * math.sin(theta), z_min))
        it.append(add_vertex(vertices, inner_radius * math.cos(theta), inner_radius * math.sin(theta), z_max))
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([ob[i], ob[j], ot[j], ot[i]])
        faces.append([ib[j], ib[i], it[i], it[j]])
        faces.append([ot[i], ot[j], it[j], it[i]])
        faces.append([ob[j], ob[i], ib[i], ib[j]])
    return vertices, faces


def dome_cap(radius: float, z_base: float, segments: int, rings: int) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    ring_ids = []
    for r in range(rings + 1):
        phi = 0.5 * math.pi * r / rings
        ring_radius = radius * math.cos(phi)
        z = z_base + radius * math.sin(phi)
        ring = []
        if r == rings:
            ring.append(add_vertex(vertices, 0.0, 0.0, z))
        else:
            for i in range(segments):
                theta = 2.0 * math.pi * i / segments
                ring.append(add_vertex(vertices, ring_radius * math.cos(theta), ring_radius * math.sin(theta), z))
        ring_ids.append(ring)
    for r in range(rings):
        if r == rings - 1:
            apex = ring_ids[r + 1][0]
            for i in range(segments):
                faces.append([ring_ids[r][i], ring_ids[r][(i + 1) % segments], apex])
        else:
            for i in range(segments):
                j = (i + 1) % segments
                faces.append([ring_ids[r][i], ring_ids[r][j], ring_ids[r + 1][j], ring_ids[r + 1][i]])
    return vertices, faces


def oval_wing(width: float, depth: float, height: float, x_center: float, segments: int) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    bottom = []
    top = []
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        x = x_center + 0.5 * width * math.cos(theta)
        y = 0.5 * depth * math.sin(theta)
        bottom.append(add_vertex(vertices, x, y, -0.5 * height))
        top.append(add_vertex(vertices, x, y, 0.5 * height))
    bc = add_vertex(vertices, x_center, 0.0, -0.5 * height)
    tc = add_vertex(vertices, x_center, 0.0, 0.5 * height)
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([bottom[i], bottom[j], top[j], top[i]])
        faces.append([bc, bottom[i], bottom[j]])
        faces.append([tc, top[j], top[i]])
    return vertices, faces


def screw_mesh(spec: MetricSpec, shaft_length_mm: float, segments: int, steps_per_turn: int) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    shaft_length = mm(shaft_length_mm)
    head_height = mm(spec.head_height_mm)
    shaft, shaft_faces = helical_threaded_shaft(
        diameter=mm(spec.diameter_mm),
        pitch=mm(spec.pitch_mm),
        length=shaft_length,
        segments=segments,
        steps_per_turn=steps_per_turn,
        z_min=0.0,
    )
    append_mesh(vertices, faces, shaft, shaft_faces)
    head, head_faces = cylinder(mm(spec.head_diameter_mm) * 0.5, head_height, segments, z_min=shaft_length)
    append_mesh(vertices, faces, head, head_faces)
    return recenter(vertices), faces


def nut_mesh(spec: MetricSpec, height_scale: float, segments: int) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    vertices, faces = annular_hex_prism(
        across_flats=mm(spec.nut_width_mm),
        inner_radius=mm(spec.diameter_mm) * 0.55,
        height=mm(spec.nut_height_mm) * height_scale,
        segments=segments,
    )
    return recenter(vertices), faces


def flange_nut_mesh(spec: MetricSpec, segments: int) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    vertices, faces = nut_mesh(spec, 1.0, segments)
    flange, flange_faces = annular_disc(
        inner_radius=mm(spec.diameter_mm) * 0.55,
        outer_radius=mm(spec.nut_width_mm) * 0.82,
        height=mm(spec.nut_height_mm) * 0.22,
        segments=segments,
        z_min=-mm(spec.nut_height_mm) * 0.5,
    )
    append_mesh(vertices, faces, flange, flange_faces)
    return recenter(vertices), faces


def dome_nut_mesh(spec: MetricSpec, segments: int) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    vertices, faces = nut_mesh(spec, 0.9, segments)
    top_z = max(v[2] for v in vertices)
    cap, cap_faces = dome_cap(mm(spec.nut_width_mm) * 0.45, top_z, segments, rings=8)
    append_mesh(vertices, faces, cap, cap_faces)
    return recenter(vertices), faces


def wing_nut_mesh(spec: MetricSpec, segments: int) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    vertices, faces = nut_mesh(spec, 0.8, segments)
    wing_width = mm(spec.nut_width_mm) * 1.4
    wing_depth = mm(spec.nut_width_mm) * 0.45
    wing_height = mm(spec.nut_height_mm) * 0.65
    x_offset = mm(spec.nut_width_mm) * 0.82
    left, left_faces = oval_wing(wing_width, wing_depth, wing_height, -x_offset, segments)
    right, right_faces = oval_wing(wing_width, wing_depth, wing_height, x_offset, segments)
    append_mesh(vertices, faces, left, left_faces)
    append_mesh(vertices, faces, right, right_faces)
    return recenter(vertices), faces


def washer_mesh(spec: MetricSpec, extended: bool, segments: int) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    od = spec.washer_extended_od_mm if extended else spec.washer_od_mm
    height = spec.washer_extended_height_mm if extended else spec.washer_height_mm
    vertices, faces = annular_disc(
        inner_radius=mm(spec.washer_id_mm) * 0.5,
        outer_radius=mm(od) * 0.5,
        height=mm(height),
        segments=segments,
    )
    return recenter(vertices), faces


def usd_array(values, tuple_size: int | None = None) -> str:
    if tuple_size is None:
        return "[" + ", ".join(str(v) for v in values) + "]"
    chunks = []
    for value in values:
        chunks.append("(" + ", ".join(f"{component:.8g}" for component in value) + ")")
    return "[" + ", ".join(chunks) + "]"


def write_usd(path: Path, vertices: list[tuple[float, float, float]], faces: list[list[int]], color: tuple[float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = [len(face) for face in faces]
    indices = [idx for face in faces for idx in face]
    with path.open("w", encoding="utf-8") as f:
        f.write("#usda 1.0\n")
        f.write("(\n    defaultPrim = \"Object\"\n    metersPerUnit = 1\n    upAxis = \"Z\"\n)\n\n")
        f.write("def Xform \"Object\"\n{\n")
        f.write("    def Mesh \"mesh\"\n    {\n")
        f.write("        uniform token subdivisionScheme = \"none\"\n")
        f.write(f"        point3f[] points = {usd_array(vertices, 3)}\n")
        f.write(f"        int[] faceVertexCounts = {usd_array(counts)}\n")
        f.write(f"        int[] faceVertexIndices = {usd_array(indices)}\n")
        f.write(f"        color3f[] primvars:displayColor = [({color[0]}, {color[1]}, {color[2]})] (\n")
        f.write("            interpolation = \"constant\"\n        )\n")
        f.write("    }\n}\n")


def generate_family(family: str, output_root: Path, segments: int, thread_steps_per_turn: int, overwrite: bool) -> None:
    spec = SPECS[family]
    family_root = output_root / family
    color = (0.82, 0.82, 0.8)

    for length in SCREW_LENGTHS[family]:
        path = family_root / f"{family}X{length}.usd"
        if path.exists() and not overwrite:
            print(f"[SKIP] {path}")
            continue
        vertices, faces = screw_mesh(spec, length, segments, thread_steps_per_turn)
        write_usd(path, vertices, faces, color)
        print(f"[OK] {path} vertices={len(vertices)} faces={len(faces)}")

    generators = {
        f"{family}_nut": lambda: nut_mesh(spec, 1.0, segments),
        f"{family}_nut_low": lambda: nut_mesh(spec, 0.55, segments),
        f"{family}_nut_flange": lambda: flange_nut_mesh(spec, segments),
        f"{family}_domenut": lambda: dome_nut_mesh(spec, segments),
        f"{family}_wingnut": lambda: wing_nut_mesh(spec, segments),
        f"{family}_washer": lambda: washer_mesh(spec, False, segments),
        f"{family}_washer_extended": lambda: washer_mesh(spec, True, segments),
    }
    for name, generator in generators.items():
        path = family_root / f"{name}.usd"
        if path.exists() and not overwrite:
            print(f"[SKIP] {path}")
            continue
        vertices, faces = generator()
        write_usd(path, vertices, faces, color)
        print(f"[OK] {path} vertices={len(vertices)} faces={len(faces)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate low-poly M6/M10 nut and screw USD assets.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root containing M6/M10 folders.")
    parser.add_argument("--families", nargs="+", choices=sorted(SPECS), default=sorted(SPECS), help="Metric families to generate.")
    parser.add_argument("--segments", type=int, default=36, help="Radial segment count for cylindrical/annular geometry.")
    parser.add_argument("--thread-steps-per-turn", type=int, default=8, help="Longitudinal thread resolution per pitch turn.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated USD files.")
    args = parser.parse_args()

    segments = max(12, int(args.segments))
    thread_steps_per_turn = max(3, int(args.thread_steps_per_turn))
    for family in args.families:
        generate_family(family, args.output_root.expanduser().resolve(), segments, thread_steps_per_turn, args.overwrite)


if __name__ == "__main__":
    main()
