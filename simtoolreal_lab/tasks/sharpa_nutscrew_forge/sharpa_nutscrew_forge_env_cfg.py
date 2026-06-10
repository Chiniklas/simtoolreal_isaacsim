"""Configuration for the SHARPA nut-screw task with Forge-style reward shaping."""

from __future__ import annotations

from isaaclab.utils import configclass

from simtoolreal_lab.tasks.sharpa_nutscrew_pick_place_screw.sharpa_nutscrew_pick_place_screw_env_cfg import *  # noqa: F403
from simtoolreal_lab.tasks.sharpa_nutscrew_pick_place_screw.sharpa_nutscrew_pick_place_screw_env_cfg import (
    SharpaNutscrewPickPlaceScrewEnvCfg,
    apply_finger_mask,
    apply_object_selection as _apply_base_object_selection,
)


@configclass
class SharpaNutscrewForgeEnvCfg(SharpaNutscrewPickPlaceScrewEnvCfg):
    """Screw-only SHARPA nut task using the previous simulation and Forge grip rewards."""

    # Use the simplified generated nut-on-screw setup from the previous task.
    screwing_phase = True

    # Forge-style staged grip shaping. The downstream keypoint/success reward is
    # gated on a live proper grip, so the policy must touch the nut before turn
    # progress pays.
    forge_reward_shaping = True
    gate_keypoint_on_proper_grip = True

    # Tripod contact definition: thumb must participate, and at least one of
    # index/middle must also touch.
    active_fingers = ("thumb", "index", "middle")
    use_finger_contact_sensor = True
    contact_force_threshold = 0.02
    require_all_finger_contacts = False
    required_finger_contact_count = 2
    required_contact_fingers = ("thumb",)
    gate_keypoint_on_contact = True

    # Approximate radial distance from nut center to the side/corner surface.
    # M12 nut width is ~19 mm across flats; a slightly larger effective radius
    # keeps side grips from being penalized as "far from center".
    forge_nut_radius = 0.011
    contact_context_dist = 0.035

    # Contact stage: surface-distance shaping plus small dense/sustained contact
    # bonus. These are lower than the ForgeUltra run03 defaults to avoid swamping
    # screw progress in the 60 Hz task.
    contact_shaping_scale = 2.0
    contact_bonus = 0.2
    sustained_contact_bonus = 0.002
    sustained_contact_cap = 50

    # Perpendicular side-pinch stage. The grip line is thumb to mean(index,middle);
    # align=0 is perpendicular to the nut axis.
    pinch_perp_reward_scale = 1.0
    pinch_perp_threshold = 0.30
    pinch_perp_bonus = 0.5

    # Hand-to-object and wrist-tilt shaping from ForgeUltra.
    hand_obj_reward_scale = 1.0
    ee_tilt_reward_scale = 0.5
    ee_tilt_target_deg = 30.0
    ee_tilt_band_deg = 7.0
    ee_tilt_gate_dist = 0.05


def apply_object_selection(cfg: SharpaNutscrewForgeEnvCfg) -> None:
    """Apply object and finger choices after YAML/CLI overrides."""

    _apply_base_object_selection(cfg)
    apply_finger_mask(cfg)
