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
