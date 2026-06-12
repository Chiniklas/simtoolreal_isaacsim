"""Gym registration for the KUKA-SHARPA forge tasks."""

import gymnasium as gym

from . import agents


_TASK_PACKAGE = __package__
_ENV_ENTRY_POINT = f"{_TASK_PACKAGE}.forge_kuka_env:ForgeKukaEnv"


def _register(task_id: str, env_cfg_name: str, agent_cfg_name: str) -> None:
    gym.register(
        id=task_id,
        entry_point=_ENV_ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_TASK_PACKAGE}.forge_kuka_env_cfg:{env_cfg_name}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:{agent_cfg_name}",
        },
    )


_register("Isaac-Forge-PegInsert-Kuka-v0", "ForgeKukaTaskPegInsertCfg", "rl_games_ppo_cfg.yaml")
_register("Isaac-Forge-GearMesh-Kuka-v0", "ForgeKukaTaskGearMeshCfg", "rl_games_ppo_cfg.yaml")
_register(
    "Isaac-Forge-NutThread-Kuka-v0",
    "ForgeKukaTaskNutThreadCfg",
    "rl_games_ppo_cfg_nut_thread.yaml",
)
_register(
    "Isaac-Forge-NutThread-KukaPinch-v0",
    "ForgeKukaPinchNutThreadCfg",
    "rl_games_ppo_cfg_nut_thread.yaml",
)
_register(
    "Isaac-Forge-NutThread-KukaPinchThread-v0",
    "ForgeKukaPinchThreadCfg",
    "rl_games_ppo_cfg_nut_thread.yaml",
)
_register(
    "Isaac-Forge-NutThread-KukaPinchDemo-v0",
    "ForgeKukaPinchDemoCfg",
    "rl_games_ppo_cfg_nut_thread.yaml",
)
