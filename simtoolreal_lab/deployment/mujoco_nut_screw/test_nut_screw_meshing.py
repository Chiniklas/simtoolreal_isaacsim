"""Standalone MuJoCo nut/screw meshing test.

This script is intentionally independent from the iiwa+SHARPA deployment scene.
It creates a tiny MuJoCo world with only:

* a table,
* a fixed screw standing upright on the table,
* a constrained nut initialized concentrically at the very end of the screw
  thread.

The collision setup is intentionally fixed to proxy collision:

* detailed generated screw/nut meshes are visual-only;
* the screw has a non-contact shaft guide plus a head cylinder stop;
* the nut collision is six simple boxes arranged around the hole.

Proxy collision avoids unstable detailed mesh-vs-mesh thread contact. The thread
mechanics are instead represented analytically with MuJoCo joints:

* ``nut_spin_joint`` is a hinge around local/world Z;
* ``nut_thread_slide_joint`` is a slide along local/world Z;
* an equality constraint couples them as
  ``slide = -pitch / (2*pi) * spin``.
* the script commands a constant nut spin velocity and projects the slide/spin
  state onto the same equation after each step. This keeps the tiny pitch-ratio
  relation exact even when the soft equality solver/contact solver would
  otherwise drift.

This means the visual thread is used for appearance, while the screw relation is
stable and explicit. A positive commanded nut spin moves the nut downward along
the screw. The nut body has ``gravcomp="1"``, so gravity does not drive it; the
screw-down motion comes from the commanded rotational velocity.

The generated screw OBJ convention is:

* screw origin is at the head bearing surface;
* local +Z points along the threaded shaft;
* the head extends along -Z.

The script reads the actual OBJ bounds and computes:

* screw bottom on the table;
* screw top at the thread tip;
* nut center so the nut bottom starts just above that thread tip.

Run from the repo root:

.. code-block:: bash

    python simtoolreal_lab/deployment/mujoco_nut_screw/test_nut_screw_meshing.py

If MuJoCo is only installed in the project conda env:

.. code-block:: bash

    conda run -n simtoolreal python simtoolreal_lab/deployment/mujoco_nut_screw/test_nut_screw_meshing.py
"""

from __future__ import annotations

import argparse
import select
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ASSET_ROOT = REPO_ROOT / "simtoolreal_lab/assets/nutscrew_generated"
TABLE_SIZE = np.array([0.24, 0.20, 0.04], dtype=float)
TABLE_TOP_Z = 0.53
TABLE_POS = np.array([0.0, 0.0, TABLE_TOP_Z - 0.5 * TABLE_SIZE[2]], dtype=float)
FAMILY_DIAMETERS = {
    "M6": 0.006,
    "M8": 0.008,
    "M10": 0.010,
    "M12": 0.012,
    "M16": 0.016,
    "M20": 0.020,
}
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

    @property
    def half_extents(self) -> np.ndarray:
        return 0.5 * self.size


def _require_mujoco():
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MuJoCo is required. Install it in the active environment, "
            "or run with `conda run -n simtoolreal python ...`."
        ) from exc
    return mujoco


def _maybe_import_viewer():
    try:
        import mujoco.viewer
    except ModuleNotFoundError:
        return None
    return mujoco.viewer


def _wait_for_enter(viewer) -> bool:
    print("[INFO] Press Enter to start simulation (Ctrl-D to quit).", flush=True)
    if not sys.stdin.isatty():
        return sys.stdin.readline() != ""

    while viewer is None or viewer.is_running():
        readable, _, _ = select.select([sys.stdin], [], [], 0.05)
        if readable:
            return sys.stdin.readline() != ""
        if viewer is not None:
            viewer.sync()
    return False


def _format_vec(values: np.ndarray | tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)


def _asset_path(asset_root: Path, family: str, name: str) -> Path:
    path = asset_root.expanduser().resolve() / family / f"{name}.obj"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing OBJ mesh: {path}\n"
            "Generate it with:\n"
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


def _yaw_quat(theta: float) -> np.ndarray:
    return np.array([np.cos(0.5 * theta), 0.0, 0.0, np.sin(0.5 * theta)], dtype=float)


def _nut_collision_proxy_xml(bounds: MeshBounds, family: str, density: float) -> str:
    diameter = FAMILY_DIAMETERS.get(family, 0.012)
    outer_radius = max(abs(bounds.mins[0]), abs(bounds.maxs[0]), abs(bounds.mins[1]), abs(bounds.maxs[1]))
    inner_radius = diameter * 0.56
    radial_half = max(0.5 * (outer_radius - inner_radius), 0.001)
    tangential_half = max(outer_radius * 0.32, 0.001)
    center_radius = inner_radius + radial_half
    z_half = max(float(bounds.half_extents[2]), 0.001)
    geoms = []
    for idx in range(6):
        theta = idx * np.pi / 3.0
        pos = np.array([center_radius * np.cos(theta), center_radius * np.sin(theta), 0.0])
        geoms.append(
            "    "
            f'<geom name="nut_collision_proxy_{idx}" type="box" '
            f'pos="{_format_vec(pos)}" quat="{_format_vec(_yaw_quat(theta))}" '
            f'size="{_format_vec(np.array([radial_half, tangential_half, z_half]))}" '
            f'rgba="0 0.85 0 0.25" density="{density:.8g}" condim="6" friction="1 0.005 0.0001" group="3"/>'
        )
    return "\n".join(geoms)


def _build_xml(
    screw_path: Path,
    nut_path: Path,
    screw_bounds: MeshBounds,
    nut_bounds: MeshBounds,
    family: str,
    clearance: float,
    density: float,
) -> tuple[str, np.ndarray]:
    screw_origin_z = TABLE_TOP_Z - float(screw_bounds.mins[2])
    screw_top_z = screw_origin_z + float(screw_bounds.maxs[2])
    nut_center_z = screw_top_z + clearance - float(nut_bounds.mins[2])
    nut_start_pos = np.array([0.0, 0.0, nut_center_z], dtype=float)

    diameter = FAMILY_DIAMETERS.get(family, 0.012)
    shaft_radius = 0.5 * diameter
    shaft_length = max(float(screw_bounds.maxs[2]), 0.001)
    head_height = max(-float(screw_bounds.mins[2]), 0.001)
    head_radius = max(
        abs(screw_bounds.mins[0]),
        abs(screw_bounds.maxs[0]),
        abs(screw_bounds.mins[1]),
        abs(screw_bounds.maxs[1]),
    )
    pitch = FAMILY_PITCHES.get(family, FAMILY_PITCHES["M12"])
    thread_slope = -pitch / (2.0 * np.pi)
    slide_min = -max(shaft_length, 0.001)
    slide_max = max(clearance + 0.01, 0.01)
    screw_geoms = f"""
      <geom name="screw_visual" type="mesh" mesh="screw_mesh" rgba="0.62 0.62 0.62 1" contype="0" conaffinity="0" group="1"/>
      <geom name="screw_shaft_guide" type="cylinder" pos="0 0 {0.5 * shaft_length:.8g}" size="{shaft_radius:.8g} {0.5 * shaft_length:.8g}" rgba="0.2 0.4 1 0.12" contype="0" conaffinity="0" group="3"/>
      <geom name="screw_head_collision" type="cylinder" pos="0 0 {-0.5 * head_height:.8g}" size="{head_radius:.8g} {0.5 * head_height:.8g}" rgba="0.2 0.4 1 0.2" density="{density:.8g}" condim="6" friction="1 0.005 0.0001" group="3"/>
"""
    nut_geoms = f"""
      <geom name="nut_visual" type="mesh" mesh="nut_mesh" rgba="0.75 0.75 0.72 1" contype="0" conaffinity="0" group="1"/>
{_nut_collision_proxy_xml(nut_bounds, family, density)}
"""

    xml = f"""
<mujoco model="pure_nut_screw_meshing">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.001" integrator="implicitfast" cone="elliptic" impratio="10" gravity="0 0 -9.81"/>
  <visual>
    <global azimuth="-120" elevation="-20"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.7 0.7 0.7" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <mesh name="screw_mesh" file="{screw_path}"/>
    <mesh name="nut_mesh" file="{nut_path}"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.05" material="groundplane"/>
    <light name="top" pos="0 0 2" dir="0 0 -1" mode="trackcom"/>
    <camera name="closeup" pos="0.16 -0.30 0.72" xyaxes="0.88 0.47 0 -0.20 0.38 0.90"/>
    <body name="table" pos="{_format_vec(TABLE_POS)}">
      <geom name="table_geom" type="box" size="{_format_vec(0.5 * TABLE_SIZE)}" rgba="1 1 1 1" friction="1 0.005 0.0001" condim="6"/>
    </body>
    <body name="fixed_screw" pos="0 0 {screw_origin_z:.8g}">
{screw_geoms.rstrip()}
    </body>
    <body name="nut" pos="{_format_vec(nut_start_pos)}" gravcomp="1">
      <joint name="nut_spin_joint" type="hinge" axis="0 0 1" damping="0.0001"/>
      <joint name="nut_thread_slide_joint" type="slide" axis="0 0 1" range="{slide_min:.8g} {slide_max:.8g}" limited="true" damping="0.0001"/>
{nut_geoms.rstrip()}
    </body>
  </worldbody>
  <equality>
    <joint name="nut_thread_constraint" joint1="nut_thread_slide_joint" joint2="nut_spin_joint" polycoef="0 {thread_slope:.12g} 0 0 0" solref="0.002 1" solimp="0.95 0.99 0.001"/>
  </equality>
</mujoco>
"""
    return xml, nut_start_pos


def _thread_slope(family: str) -> float:
    return -FAMILY_PITCHES.get(family, FAMILY_PITCHES["M12"]) / (2.0 * np.pi)


def _project_thread_state(model, data, family: str) -> None:
    slope = _thread_slope(family)
    spin_joint = model.joint("nut_spin_joint")
    slide_joint = model.joint("nut_thread_slide_joint")
    spin_qadr = int(spin_joint.qposadr[0])
    slide_qadr = int(slide_joint.qposadr[0])
    spin_dadr = int(spin_joint.dofadr[0])
    slide_dadr = int(slide_joint.dofadr[0])
    data.qpos[spin_qadr] = data.qpos[slide_qadr] / slope
    data.qvel[spin_dadr] = data.qvel[slide_dadr] / slope


def _command_thread_velocity(model, data, family: str, spin_velocity: float) -> None:
    slope = _thread_slope(family)
    spin_joint = model.joint("nut_spin_joint")
    slide_joint = model.joint("nut_thread_slide_joint")
    data.qvel[int(spin_joint.dofadr[0])] = spin_velocity
    data.qvel[int(slide_joint.dofadr[0])] = slope * spin_velocity


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a pure nut/screw MuJoCo scene.")
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--family", default="M12")
    parser.add_argument("--screw", default="M12X30")
    parser.add_argument("--nut", default="M12_nut")
    parser.add_argument("--clearance", type=float, default=0.002, help="Gap between screw tip and nut bottom, in meters.")
    parser.add_argument("--density", type=float, default=7800.0, help="Uniform material density for proxy collision geoms, kg/m^3.")
    parser.add_argument(
        "--spin-velocity",
        type=float,
        default=6.0,
        help="Constant commanded nut angular velocity around Z, rad/s. Positive spins the nut downward.",
    )
    parser.add_argument("--steps", type=int, default=None, help="Run finite steps headlessly/viewer; omit to run until viewer closes.")
    parser.add_argument("--headless", action="store_true", help="Run without viewer.")
    parser.add_argument("--press-enter-to-start", action="store_true", help="Pause after initialization until Enter is pressed.")
    args = parser.parse_args()

    mujoco = _require_mujoco()
    screw_path = _asset_path(args.asset_root, args.family, args.screw)
    nut_path = _asset_path(args.asset_root, args.family, args.nut)
    screw_bounds = _obj_bounds(screw_path)
    nut_bounds = _obj_bounds(nut_path)
    xml, nut_start_pos = _build_xml(
        screw_path,
        nut_path,
        screw_bounds,
        nut_bounds,
        args.family,
        args.clearance,
        args.density,
    )

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    _command_thread_velocity(model, data, args.family, args.spin_velocity)
    mujoco.mj_forward(model, data)

    screw_origin_z = TABLE_TOP_Z - float(screw_bounds.mins[2])
    screw_top_z = screw_origin_z + float(screw_bounds.maxs[2])
    print(f"[INFO] Screw: {screw_path}")
    print(f"[INFO] Nut:   {nut_path}")
    print(f"[INFO] Screw top z: {screw_top_z:.4f} m")
    print(
        "[INFO] Nut initialized at thread end: "
        f"center={np.round(nut_start_pos, 5).tolist()} bottom_z={nut_start_pos[2] + nut_bounds.mins[2]:.4f} m"
    )
    print(f"[INFO] Screw bbox size: {np.round(screw_bounds.size, 5).tolist()} m")
    print(f"[INFO] Nut bbox size:   {np.round(nut_bounds.size, 5).tolist()} m")
    print(f"[INFO] Collision mode: proxy, density={args.density:.1f} kg/m^3")
    print(f"[INFO] Analytical thread pitch: {FAMILY_PITCHES.get(args.family, FAMILY_PITCHES['M12']):.6f} m/rev")
    print(f"[INFO] Nut gravcomp: 1.0, commanded spin velocity={args.spin_velocity:.3f} rad/s")
    print(f"[INFO] Nut mass from MuJoCo model: {float(model.body('nut').mass[0]):.6f} kg")

    viewer = None
    if not args.headless:
        viewer_mod = _maybe_import_viewer()
        if viewer_mod is None:
            raise ModuleNotFoundError("mujoco.viewer is required unless --headless is passed.")
        viewer = viewer_mod.launch_passive(model, data)
        print("[INFO] Close the MuJoCo viewer to stop.")

    if args.press_enter_to_start and not _wait_for_enter(viewer):
        if viewer is not None:
            viewer.close()
        return

    step = 0
    while args.steps is None or step < args.steps:
        if viewer is not None and not viewer.is_running():
            break
        start = time.time()
        _command_thread_velocity(model, data, args.family, args.spin_velocity)
        mujoco.mj_step(model, data)
        _project_thread_state(model, data, args.family)
        _command_thread_velocity(model, data, args.family, args.spin_velocity)
        mujoco.mj_forward(model, data)
        if viewer is not None:
            viewer.sync()
        sleep_dt = model.opt.timestep - (time.time() - start)
        if viewer is not None and sleep_dt > 0.0:
            time.sleep(sleep_dt)
        step += 1

    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
