# SimToolReal Isaac Lab

This repository contains an Isaac Sim / Isaac Lab implementation of SimToolReal
for the reference KUKA iiwa14 + left SHARPA hand setup.

The original IsaacGym SimToolReal codebase is kept under `reference/` for
comparison and parity work. The active Isaac Lab package is `simtoolreal_lab`,
and the registered Gym task is `simtoolreal_sharpa`.

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

Scaled SAPO / mixed-exploration training:

```bash
python simtoolreal_lab/train_rl_games.py \
  --task simtoolreal_sharpa \
  --num_envs 1536 \
  --headless \
  agent.params.config.expl_coef_block_size=256 \
  agent.wandb_activate=False
```

The reference training launcher defaults to `24576` envs with
`expl_coef_block_size=4096`; the scaled command above uses `1536 / 256` to keep
the same six-block SAPO structure while fitting on a smaller single-GPU budget.

The reference SAPO settings live in
`simtoolreal_lab/tasks/simtoolreal_sharpa/agents/rl_games_sapo_cfg.yaml`. The
reference domain-randomization settings live in
`simtoolreal_lab/tasks/simtoolreal_sharpa/simtoolreal_sharpa_env_cfg.py`, so
they do not need to be repeated as terminal overrides. For normal
reference-style training, keep the command short and override domain
randomization only for ablations, debugging, or intentionally fixed-object
runs.

Checkpoint behavior comes from the RL-Games agent config. The default
`save_frequency: 1000` writes `nn/model_<epoch>.pth` every 1000 epochs.
Independently, `last/model.pth` is refreshed every 3 epochs, and every new best
mean reward after `save_best_after: 100` updates `best/model.pth` directly.

Hydra output and RL-Games training logs are written inside the task directory:

```text
simtoolreal_lab/tasks/simtoolreal_sharpa/outputs/<date>/<time>/
simtoolreal_lab/tasks/simtoolreal_sharpa/logs/<experiment_name>/
```

## Domain Randomization

The Isaac Lab training path has been aligned with the reference on SAPO
six-block policy shape, asymmetric critic state size/content, DexToolBench
object spawning/collision, table reset/contact-force handling, success-reset
semantics, goal-object behavior, fixed-size keypoint reward/success, success
tolerance curriculum, checkpoint-config-aware replay, and the active
sim2real/domain-randomization options used by the reference recipe.

Active reset and task randomization:

- `env.reset_position_noise_x=0.1`, `env.reset_position_noise_y=0.1`,
  `env.reset_position_noise_z=0.02`
- `env.randomize_object_rotation=True`
- `env.reset_dof_pos_noise_fingers=0.1`, `env.reset_dof_pos_noise_arm=0.1`,
  `env.reset_dof_vel_noise=0.5`
- `env.table_reset_z_range=0.01`
- `env.object_scale_noise_multiplier_range='[0.9,1.1]'` for reference-style
  multi-object training, or `'[1.0,1.0]'` for fixed-size single-object runs
- random delta goal sampling with `env.delta_goal_distance=0.1` and
  `env.delta_rotation_degrees=90.0`

Active external object disturbance randomization:

- `env.force_scale=20.0`, `env.force_prob_range='[0.001,0.1]'`,
  `env.force_only_when_lifted=True`
- `env.torque_scale=2.0`, `env.torque_prob_range='[0.001,0.1]'`,
  `env.torque_only_when_lifted=True`

Active sim2real delay/noise randomization:

- `env.use_obs_delay=True`, `env.obs_delay_max=3`
- `env.use_action_delay=True`, `env.action_delay_max=3`
- `env.use_object_state_delay_noise=True`, `env.object_state_delay_max=10`,
  `env.object_state_xyz_noise_std=0.01`,
  `env.object_state_rotation_noise_degrees=5.0`
- `env.joint_velocity_obs_noise_std=0.1`

The reference YAML also contains IsaacGym's generic `task.randomization_params`
block for gravity, action/observation noise, robot/object mass, stiffness,
damping, friction, armature, and restitution. That block is inactive in the
reference training recipe because `task.randomize=False`, so it is not part of
the matched active pipeline here.

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
