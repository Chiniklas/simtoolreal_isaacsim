"""Smoke test for loading the SimToolReal KUKA-SHARPA Isaac Lab scene."""

from __future__ import annotations

import os

import pytest


isaaclab_app = pytest.importorskip("isaaclab.app", reason="Isaac Lab is required to load this scene.")
AppLauncher = isaaclab_app.AppLauncher
_SIM_APP = AppLauncher(headless=True).app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import simtoolreal_lab.tasks.simtoolreal_sharpa.gym_setup  # noqa: E402,F401
from simtoolreal_lab.tasks.simtoolreal_sharpa.simtoolreal_sharpa_env_cfg import (  # noqa: E402
    SimToolRealSharpaEnvCfg,
    configure_multi_dextoolbench_objects,
    configure_dextoolbench_object,
)


@pytest.fixture(scope="module")
def simulation_app():
    yield _SIM_APP
    _SIM_APP.close()


def test_simtoolreal_sharpa_scene_loads_and_steps(simulation_app):
    cfg = SimToolRealSharpaEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = os.environ.get("SIMTOOLREAL_TEST_DEVICE", "cuda:0")

    env = gym.make("simtoolreal_sharpa", cfg=cfg)
    try:
        unwrapped = env.unwrapped

        assert unwrapped.num_envs == 1
        assert set(unwrapped.scene.articulations) == {"robot"}
        assert set(unwrapped.scene.rigid_objects) == {"table", "object"}

        obs, _ = env.reset()
        assert obs["policy"].shape == (1, cfg.num_observations)
        assert obs["critic"].shape == (1, cfg.num_states)

        actions = torch.zeros((1, unwrapped.num_actions), device=unwrapped.device)
        obs, rewards, terminated, truncated, _ = env.step(actions)

        assert obs["policy"].shape == (1, cfg.num_observations)
        assert rewards.shape == (1,)
        assert terminated.shape == (1,)
        assert truncated.shape == (1,)
    finally:
        env.close()


def test_simtoolreal_sharpa_dextoolbench_object_loads(simulation_app):
    cfg = SimToolRealSharpaEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = os.environ.get("SIMTOOLREAL_TEST_DEVICE", "cuda:0")
    configure_dextoolbench_object(cfg, "claw_hammer")

    env = gym.make("simtoolreal_sharpa", cfg=cfg)
    try:
        unwrapped = env.unwrapped
        assert unwrapped.cfg.object_name == "claw_hammer"
        assert tuple(unwrapped.cfg.object_scales) == (2.5, 0.5625, 0.375)

        masses = unwrapped.object.root_physx_view.get_masses()
        assert torch.allclose(masses.sum(dim=1), torch.full((1,), cfg.object_mass, device=masses.device))

        obs, _ = env.reset()
        assert obs["policy"].shape == (1, cfg.num_observations)
    finally:
        env.close()


def test_simtoolreal_sharpa_multi_dextoolbench_object_count_matches_envs(simulation_app):
    cfg = SimToolRealSharpaEnvCfg()
    cfg.scene.num_envs = 4
    cfg.sim.device = os.environ.get("SIMTOOLREAL_TEST_DEVICE", "cuda:0")
    configure_multi_dextoolbench_objects(cfg, ["claw_hammer", "long_screwdriver"])
    cfg.scene.replicate_physics = False

    env = gym.make("simtoolreal_sharpa", cfg=cfg)
    try:
        unwrapped = env.unwrapped
        assert unwrapped.object.data.root_pos_w.shape[0] == cfg.scene.num_envs
        assert unwrapped.goal_object.data.root_pos_w.shape[0] == cfg.scene.num_envs
        stage = unwrapped.scene.stage
        object_child_names = [
            tuple(child.GetName() for child in stage.GetPrimAtPath(f"/World/envs/env_{env_id}/object").GetChildren())
            for env_id in range(cfg.scene.num_envs)
        ]
        goal_container_counts = [
            sum(
                1
                for prim in stage.Traverse()
                if str(prim.GetPath()).startswith(f"/World/envs/env_{env_id}/") and prim.GetName() == "goal_object"
            )
            for env_id in range(cfg.scene.num_envs)
        ]
        assert len(set(object_child_names)) > 1
        assert goal_container_counts == [1] * cfg.scene.num_envs

        obs, _ = env.reset()
        assert obs["policy"].shape == (cfg.scene.num_envs, cfg.num_observations)
    finally:
        env.close()


def test_simtoolreal_sharpa_keypoints_follow_object_pose(simulation_app):
    cfg = SimToolRealSharpaEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = os.environ.get("SIMTOOLREAL_TEST_DEVICE", "cuda:0")
    cfg.object_scale_noise_multiplier_range = (1.0, 1.0)
    configure_dextoolbench_object(cfg, "claw_hammer")

    env = gym.make("simtoolreal_sharpa", cfg=cfg)
    try:
        unwrapped = env.unwrapped
        device = unwrapped.device
        pos = torch.tensor([[0.1, -0.2, 0.7]], device=device)
        scales = torch.tensor([[2.5, 0.5625, 0.375]], device=device)
        base_offsets = torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]],
            device=device,
        ) * (cfg.object_base_size * cfg.keypoint_scale * 0.5)

        identity_quat_wxyz = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        keypoints = unwrapped._compute_keypoints(pos, identity_quat_wxyz, scales)
        assert torch.allclose(keypoints, pos[:, None, :] + base_offsets[None] * scales[:, None])

        z_90_quat_wxyz = torch.tensor([[2**-0.5, 0.0, 0.0, 2**-0.5]], device=device)
        keypoints = unwrapped._compute_keypoints(pos, z_90_quat_wxyz, scales)
        scaled_offsets = base_offsets * scales
        rotated_offsets = torch.stack((-scaled_offsets[:, 1], scaled_offsets[:, 0], scaled_offsets[:, 2]), dim=-1)
        assert torch.allclose(keypoints, pos[:, None, :] + rotated_offsets[None], atol=1.0e-6)
    finally:
        env.close()
