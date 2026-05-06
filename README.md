# SimToolReal Isaac Lab

This repository contains an Isaac Sim / Isaac Lab implementation of SimToolReal
for the reference KUKA iiwa14 + left SHARPA hand setup.

The original IsaacGym SimToolReal codebase is kept under `reference/` for
comparison and parity work. The active Isaac Lab package is `simtoolreal_lab`,
and the registered Gym task is `simtoolreal_sharpa`.

## Repository Layout

```text
reference/
  Original IsaacGym SimToolReal implementation, DexToolBench tooling, baseline
  scripts, and the customized SAPO/RL-Games fork.

simtoolreal_lab/
  Active Isaac Sim / Isaac Lab package.

simtoolreal_lab/assets/kuka_sharpa/
  KUKA-SHARPA URDF, meshes, and Isaac Lab articulation config.

simtoolreal_lab/tasks/simtoolreal_sharpa/
  Isaac Lab DirectRLEnv implementation for SimToolReal.

simtoolreal_lab/rl_games/
  Customized RL-Games fork copied from the reference project.
```

## Setup

Start from an Isaac Lab environment, then clone it for this project:

```bash
conda create --name simtoolreal --clone env_isaaclab
conda activate simtoolreal
cd /home/chi-zhang/projects/simtoolreal_isaacsim
```

Install the local package:

```bash
./install.sh
```

This is equivalent to:

```bash
python -m pip install -e .
```

Isaac Sim and Isaac Lab provide the simulation stack. The customized RL-Games
fork is included in this repository, so there is no separate dependency file.

If assets are managed by Git LFS, materialize them:

```bash
git lfs install
git lfs pull
```

Optional pointer-file check:

```bash
rg -l "version https://git-lfs.github.com/spec/v1" simtoolreal_lab/assets reference/assets | wc -l
```

Expected output is `0`.

## RL-Games

The launcher scripts use the customized RL-Games fork at:

```text
simtoolreal_lab/rl_games
```

Verify that the project fork is selected:

```bash
conda run -n simtoolreal python -c \
"import sys, pathlib; sys.path.insert(0, str(pathlib.Path('simtoolreal_lab/rl_games').resolve())); import rl_games, rl_games.common.a2c_common as a; print(rl_games.__file__); print('mixed_expl' in open(a.__file__).read())"
```

The second printed value should be `True`.

## Smoke Test

Run one reset and one step:

```bash
conda run -n simtoolreal python -c \
"from isaaclab.app import AppLauncher; app = AppLauncher(headless=True).app; import torch, gymnasium as gym; import simtoolreal_lab.tasks.simtoolreal_sharpa.gym_setup; from simtoolreal_lab.tasks.simtoolreal_sharpa.simtoolreal_sharpa_env_cfg import SimToolRealSharpaEnvCfg; cfg = SimToolRealSharpaEnvCfg(); cfg.scene.num_envs = 1; cfg.sim.device = 'cuda:0'; env = gym.make('simtoolreal_sharpa', cfg=cfg); obs, _ = env.reset(); print(obs['policy'].shape); obs, rew, terminated, truncated, info = env.step(torch.zeros((1, cfg.num_actions), device=env.unwrapped.device)); print(obs['policy'].shape, rew.shape); env.close(); app.close()"
```

Expected shapes:

```text
torch.Size([1, 140])
torch.Size([1, 140]) torch.Size([1])
```

## Train

Run training from the repository root:

```bash
conda activate simtoolreal
cd /home/chi-zhang/projects/simtoolreal_isaacsim

python simtoolreal_lab/train_rl_games.py \
  --task simtoolreal_sharpa \
  --num_envs 64 \
  --headless \
  --max_iterations 10 \
  agent.params.config.minibatch_size=1024 \
  agent.params.config.horizon_length=16 \
  agent.params.config.mini_epochs=4 \
  agent.params.config.expl_type=none \
  agent.wandb_activate=False
```

Enable the customized SAPO / mixed-exploration path with:

```bash
python simtoolreal_lab/train_rl_games.py \
  --task simtoolreal_sharpa \
  --num_envs 4096 \
  --headless \
  agent.params.config.expl_type=mixed_expl \
  agent.params.config.use_others_experience=lf \
  agent.params.config.off_policy_ratio=1.0 \
  agent.params.config.expl_reward_type=entropy \
  agent.params.config.expl_coef_block_size=4096 \
  agent.wandb_activate=False
```

## Play

```bash
conda activate simtoolreal
cd /home/chi-zhang/projects/simtoolreal_isaacsim

python simtoolreal_lab/play_rl_games.py \
  --task simtoolreal_sharpa \
  --num_envs 8 \
  --headless \
  --checkpoint /path/to/checkpoint.pth
```

## Current Status

Implemented:

- KUKA-SHARPA Isaac Lab asset config.
- `simtoolreal_sharpa` Gym registration.
- DirectRLEnv reset/action/observation/reward/done skeleton.
- 29-action and 140-observation SHARPA policy contract.
- Customized RL-Games fork support.

Remaining parity work:

- DexToolBench object loading in Isaac Lab.
- Full reference reward and statistic breakdown.
- Observation/action delay queues.
- Object force and torque disturbances.
- Evaluation and video utilities matching `reference/`.

For the original IsaacGym documentation, see `reference/README.md`.
