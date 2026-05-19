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

## Upstream assets (manual)

Some URDFs and meshes are not tracked in this repo (see `.gitignore`) and must
be copied in by hand from the upstream `tylerlum/simtoolreal` repo. Until this
is automated, the procedure is:

```bash
git clone --depth 1 --branch main https://github.com/tylerlum/simtoolreal.git /tmp/simtoolreal_upstream

# DexToolBench URDFs + meshes (~20 MB)
cp -r /tmp/simtoolreal_upstream/assets/urdf/dextoolbench \
      simtoolreal_lab/assets/dextoolbench

# Standalone left SHARPA hand description (~7.6 MB)
cp -r /tmp/simtoolreal_upstream/assets/urdf/left_sharpa_ha4 \
      simtoolreal_lab/assets/kuka_sharpa/left_sharpa_ha4

# Extra table variants
cp /tmp/simtoolreal_upstream/assets/urdf/table_narrow_bowl_plate.urdf \
   /tmp/simtoolreal_upstream/assets/urdf/table_narrow_cuttingboard.urdf \
   /tmp/simtoolreal_upstream/assets/urdf/table_narrow_nail.urdf \
   /tmp/simtoolreal_upstream/assets/urdf/table_narrow_screwdriver_hole.urdf \
   /tmp/simtoolreal_upstream/assets/urdf/table_narrow_whiteboard.urdf \
   simtoolreal_lab/assets/table/urdf/

rm -rf /tmp/simtoolreal_upstream
```

These paths are ignored by git, so the files stay local-only.

## Convert assets to USD

Isaac Lab loads `.usd` files, not URDF. The training launcher expects two USD
artifacts that must be generated locally from the URDFs above before the first
training run:

1. `simtoolreal_lab/assets/kuka_sharpa/kuka_sharpa.usd` — the robot
2. `simtoolreal_lab/assets/dextoolbench_usd/<object>/<object>.usd` — one per
   DexToolBench tool (12 total: `claw_hammer`, `mallet_hammer`,
   `long_screwdriver`, `short_screwdriver`, `handle_eraser`, `flat_eraser`,
   `flat_spatula`, `spoon_spatula`, `sharpie_marker`, `staples_marker`,
   `red_brush`, `blue_brush`)

Robot URDF → USD via IsaacLab's bundled converter:

```bash
python "$ISAACLAB_PATH/scripts/tools/convert_urdf.py" \
  simtoolreal_lab/assets/kuka_sharpa/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf \
  simtoolreal_lab/assets/kuka_sharpa/kuka_sharpa.usd \
  --headless
```

DexToolBench URDFs → USDs via the batch converter shipped in this repo:

```bash
python simtoolreal_lab/assets/batch_convert_urdf.py \
  simtoolreal_lab/assets/dextoolbench \
  simtoolreal_lab/assets/dextoolbench_usd \
  --headless
```

Each conversion boots Isaac Sim, so expect a few minutes total. The batch
converter walks `dextoolbench/<category>/<object>/<object>.urdf`, skips the
`environments/` subtree, and rewrites each URDF's `<robot name=...>` in place
to `object_<name>` (idempotent on re-runs).

Stage-composition warnings about `Unresolved reference prim path … fingertip/visuals`
during the first training run are cosmetic — physics is unaffected.

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


