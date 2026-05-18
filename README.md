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

Run training from the repository root. Keep SAPO training replay-compatible with
the pretrained baseline by preserving six exploration blocks:

```text
num_envs / agent.params.config.expl_coef_block_size = 6
```

The replay/player path is tuned for this six-block policy shape
(`a2c_network.sigma` is `6 x 29`, with matching learned `extra_params`), so do
not use one-block debug settings such as `--num_envs 16` with
`expl_coef_block_size=16` for checkpoints you plan to replay.

Small headless smoke train, still six-block compatible:

```bash
conda activate simtoolreal
cd /home/chi-zhang/projects/simtoolreal_isaacsim

python simtoolreal_lab/train_rl_games.py \
  --task simtoolreal_sharpa \
  --num_envs 60 \
  --headless \
  --max_iterations 10 \
  agent.params.config.expl_type=mixed_expl_learn_param \
  agent.params.config.use_others_experience=lf \
  agent.params.config.off_policy_ratio=1.0 \
  agent.params.config.expl_reward_type=entropy \
  agent.params.config.expl_coef_block_size=10 \
  env.object_scale_noise_multiplier_range='[0.9,1.1]' \
  env.force_consecutive_near_goal_steps=True \
  env.force_scale=20.0 \
  env.torque_scale=2.0 \
  agent.wandb_activate=False
```

Visual debug train, also six-block compatible:

```bash
python simtoolreal_lab/train_rl_games.py \
  --task simtoolreal_sharpa \
  --num_envs 18 \
  agent.params.config.expl_type=mixed_expl_learn_param \
  agent.params.config.use_others_experience=lf \
  agent.params.config.off_policy_ratio=1.0 \
  agent.params.config.expl_reward_type=entropy \
  agent.params.config.expl_coef_block_size=3 \
  env.object_scale_noise_multiplier_range='[0.9,1.1]' \
  env.force_consecutive_near_goal_steps=True \
  env.force_scale=20.0 \
  env.torque_scale=2.0 \
  agent.wandb_activate=False
```

Scaled SAPO / mixed-exploration training:

```bash
python simtoolreal_lab/train_rl_games.py \
  --task simtoolreal_sharpa \
  --num_envs 1536 \
  --headless \
  agent.params.config.expl_type=mixed_expl_learn_param \
  agent.params.config.use_others_experience=lf \
  agent.params.config.off_policy_ratio=1.0 \
  agent.params.config.expl_reward_type=entropy \
  agent.params.config.expl_coef_block_size=256 \
  env.object_scale_noise_multiplier_range='[0.9,1.1]' \
  env.force_consecutive_near_goal_steps=True \
  env.force_scale=20.0 \
  env.torque_scale=2.0 \
  agent.wandb_activate=False
```

The reference training launcher defaults to `24576` envs with
`expl_coef_block_size=4096`; the scaled command above uses `1536 / 256` to keep
the same six-block SAPO structure while fitting on single-GPU setups with
roughly 1000-ish environments.

Checkpoint behavior comes from the RL-Games agent config. The default
`save_frequency: 1000` writes `nn/model_<epoch>.pth` every 1000 epochs.
Independently, `last/model.pth` is refreshed every 3 epochs, and every new best
mean reward after `save_best_after: 100` updates `best/model.pth` directly.

Hydra output and RL-Games training logs are written inside the task directory:

```text
simtoolreal_lab/tasks/simtoolreal_sharpa/outputs/<date>/<time>/
simtoolreal_lab/tasks/simtoolreal_sharpa/logs/<experiment_name>/
```

## Reference Parity Gaps

The Isaac Lab training path has been aligned with the reference on SAPO
six-block policy shape, asymmetric critic state size/content, DexToolBench
object spawning/collision, table reset/contact-force handling, success-reset
semantics, goal-object behavior, fixed-size keypoint reward/success, and
checkpoint-config-aware replay.

Still open:

- Observation/action delay is not implemented yet. Reference flags include
  `useObsDelay` and `useActionDelay`.
- Object-state delay/noise is not implemented yet. Reference flag:
  `useObjectStateDelayNoise`.
- Joint velocity observation noise is not implemented yet. Reference setting:
  `jointVelocityObsNoiseStd`.

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

## MuJoCo Sim2Sim

The no-ROS MuJoCo sim2sim runner lives under `simtoolreal_lab/deployment/mujoco`.
It loads the pretrained RL-Games policy and the same browser-demo MuJoCo MJCF
asset bundle mirrored at
`simtoolreal_lab/assets/mujoco_wasm/scenes/iiwa_sharpa.xml`. The replay computes
the 140-value policy observation from MuJoCo state and applies the 29-action
joint target command through the scene's MJCF actuators. The bundled scene uses
its built-in hammer object body; `--object-name claw_hammer` selects the matching
policy object-scale metadata.

```bash
conda activate simtoolreal
cd /home/chi-zhang/projects/simtoolreal_isaacsim

python -m simtoolreal_lab.deployment.mujoco.mujoco_env_no_ros \
  --config-path simtoolreal_lab/pretrained_policy/config.yaml \
  --checkpoint-path simtoolreal_lab/pretrained_policy/model.pth \
  --object-name claw_hammer \
  --press-enter-to-execute \
  --record-video \
  --video-path outputs/mujoco_rollout.mp4 \
  --video-camera side_table
```

`--press-enter-to-execute` pauses once after the viewer opens so the scene can be
adjusted before the policy rollout and video recording begin.

For a headless smoke run:

```bash
python -m simtoolreal_lab.deployment.mujoco.mujoco_env_no_ros \
  --no-enable-viewer \
  --max-steps 1 \
  --object-name claw_hammer
```

This path requires the Python `mujoco` package in the active environment.

If it is not installed yet:

```bash
python -m pip install mujoco
```

Run with the MuJoCo viewer:

```bash
python -m simtoolreal_lab.deployment.mujoco.mujoco_env_no_ros \
  --object-name claw_hammer
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
- Final reward/reset/statistics parity review.
- Observation/action delay queues.
- Evaluation and video utilities matching `reference/`.

Reward parity review, simplified:

- [x] Use USD-based KUKA-SHARPA asset in training.
- [x] Use reference-style palm and fingertip frames with offsets.
- [x] Switch success check to keypoint-goal tolerance instead of object-center distance.
- [x] Switch pre-lift shaping to fingertip-to-object distance deltas.
- [x] Switch lift reward to reference-style threshold bonus plus post-threshold gating.
- [x] Switch keypoint reward to improvement-based shaping after lift.
- [x] Switch action penalties from commanded actions to joint-velocity penalties.
- [x] Add object linear/angular velocity penalty hooks with reference default scales of `0.0`.
- [x] Add reference-style object force/torque disturbances and object-scale noise.
- [x] Reset goal on success without fully resetting the environment.
- [x] Re-check reset logic against reference (`hand_far_from_object`, dropped-object hysteresis, table-force resets).
- [x] Match reference reset sampling for table height, object orientation, dropped threshold, and DOF noise.
- [x] Re-check goal sampling details against reference delta-goal behavior.
- [ ] Re-check logging/statistics breakdown against reference training curves.

For the original IsaacGym documentation, see `reference/README.md`.
