# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project: SimToolReal (Isaac Lab fork)

This is a fork of the [Stanford SimToolReal](https://github.com/tylerlum/simtoolreal) paper code,
ported from **IsaacGym + Python 3.8** to **Isaac Lab 2.2.1 + a newer Python**. The original
release sat on `da1c321` ("SimToolReal Release: Feb 18, 2026"); the fork begins at `0d08a31`
("initial trial to transfer from isaacgym to isaacsim").

The task: a KUKA iiwa14 arm + left SHARPA five-finger hand learns dexterous **tool use** —
grasp a DexToolBench tool (hammer, screwdriver, eraser, spatula, marker, brush), lift it, and
reach a 6-DoF goal pose. Trained with PPO + SAPO mixed exploration.

## Repository layout

```
simtoolreal_lab/                     # active Isaac Lab package (pip-installed as simtoolreal_lab)
├── train_rl_games.py                # training launcher (Hydra + RL-Games)
├── play_rl_games.py                 # checkpoint replay in Isaac Lab
├── tasks/simtoolreal_sharpa/
│   ├── simtoolreal_sharpa_env.py        # DirectRLEnv: obs/reward/reset (887 lines, core file)
│   ├── simtoolreal_sharpa_env_cfg.py    # env config + object selection helpers
│   ├── simtoolreal_sharpa_utils.py      # joint-target/scale math (shared with mujoco)
│   ├── gym_setup.py                     # gym.register("simtoolreal_sharpa", ...)
│   ├── agents/rl_games_sapo_cfg.yaml    # PPO + SAPO/mixed-expl config
│   └── tests/test_load_scene.py
├── rl_games/                        # VENDORED RL-Games fork w/ SAPO mods — DO NOT pip-install
│   └── rl_games/{algos_torch,common}/   # mixed_expl, coef_cond, extra_params live here
├── deployment/mujoco/               # sim2sim MuJoCo replay (no ROS)
│   ├── mujoco_env_no_ros.py             # 140-obs/29-act policy ↔ MuJoCo state bridge
│   ├── mujoco_sim.py                    # MJCF scene wrapper
│   └── policy_player.py                 # RL-Games player facade for replay
└── assets/
    ├── kuka_sharpa/                     # robot URDF + ArticulationCfg
    │   ├── kuka_sharpa.py               # KUKA_SHARPA_CFG, KUKA_SHARPA_JOINT_NAMES, ref stiffness/damping
    │   └── urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf
    ├── dextoolbench_usd/                # 12 tool USDs (not git-tracked; download separately)
    └── mujoco_wasm/scenes/iiwa_sharpa.xml   # MuJoCo replay scene

docs/simtoolreal.pdf                 # paper
download_dextoolbench_data.py        # downloads tool data (zips from Stanford)
download_pretrained_policy.py        # fetches reference checkpoint
```

`reference/` (gitignored) is the original IsaacGym codebase kept locally for parity work.

## Setup

```bash
conda create --name simtoolreal --clone env_isaaclab   # base must already have Isaac Lab 2.2.1
conda activate simtoolreal
python -m pip install -e .
```

Heavy assets are Git LFS (`*.usd`, `*.obj`, `*.pth`). Run `git lfs install && git lfs pull`
after cloning. DexToolBench USDs go under `simtoolreal_lab/assets/dextoolbench_usd/`; the
pretrained policy lands in `simtoolreal_lab/pretrained_policy/`. Both directories are gitignored.

## Policy / training mechanics

Understand these before changing training, env, or replay code — they are tightly coupled.

### Observation/action contract (must stay aligned across Isaac Lab and MuJoCo)
- **Action**: 29-dim, `[-1, 1]`. First 7 = iiwa arm (incremental velocity-style), last 22 = SHARPA hand (absolute normalized + moving-average filter). See `compute_joint_pos_targets` in `simtoolreal_sharpa_utils.py` — it is duplicated in numpy at `mujoco_env_no_ros.py::_compute_joint_pos_targets`. Keep them identical.
- **Policy observation** (140-dim, see `_compute_reference_observations`):
  joint_pos (29) + joint_vel (29) + prev_action_targets (29) + palm_pos (3) + palm_rot xyzw (4) + object_rot xyzw (4) + fingertip_pos_rel_palm (5×3) + keypoints_rel_palm (4×3) + keypoints_rel_goal (4×3) + object_scales (3). Order is canonical — see `simtoolreal_sharpa_utils.OBS_NAMES`.
- **Critic state** (162-dim, `_compute_reference_states`): policy obs + palm 6-d vel + object 6-d vel + closest-keypoint dist + closest-fingertip dists + lifted flag + log(ep_len) + log(successes) + 0.01·reward. Asymmetric PPO consumes this through `central_value_config`.
- **Quaternion convention**: Isaac Lab is wxyz, the policy contract is xyzw. `quat_wxyz_to_xyzw` and `_quat_wxyz_to_xyzw` do the conversion at observation time. Do not change which side reorders.

### SAPO / mixed exploration (six-block policy)
- The vendored `rl_games` fork adds knobs: `expl_type=mixed_expl_learn_param`, `expl_coef_block_size`, `expl_reward_coef_scale`, `expl_reward_type=entropy`, `use_others_experience=lf`, `off_policy_ratio`.
- Mechanics (in `rl_games/common/a2c_common.py` ~line 339): the env count is split into `num_envs / expl_coef_block_size` *blocks*, each block gets a different intrinsic-reward coefficient (linspace 0.5→0.0). Block IDs are linearly spaced 50→0 and appended to the obs as the last column. `fixed_sigma=coef_cond` (in `network_builder.py`) makes the actor learn a per-block `sigma` and `extra_params` matrix — pretrained checkpoints have shape `6 × 29` for `a2c_network.sigma`.
- **Six-block invariant**: replay assumes `num_envs / expl_coef_block_size == 6`. Train with `1536 / 256`, `24576 / 4096`, etc. Do not train with `--num_envs 16 expl_coef_block_size=16` if you plan to replay — checkpoint shapes will mismatch. `play_rl_games.py::_checkpoint_coef_id_count` reads this dim from the checkpoint and threads it into the player.
- Replay obs has `intr_reward_coef_embd` (block-id column) concatenated before the network call (`policy_player.py::get_normalized_action`, `play_rl_games.py::_player_obs`). MuJoCo deployment uses a default token of `50.0` when the checkpoint has no SAPO state.

### Rewards (in `_get_rewards`)
fingertip distance-delta (pre-lift) + lifting reward + lift bonus (one-shot at z > `lifting_bonus_threshold=0.15`) + keypoint reward (4-corner fixed-size keypoints, post-lift) + reach-goal bonus + arm/hand action penalties. Goal is "near" when **all four** fixed-size keypoints are within `success_tolerance × keypoint_scale`. With `force_consecutive_near_goal_steps=True` the per-step counter resets on a miss; you only get the bonus when you hold the pose for `success_steps=10` consecutive steps. On success, the goal resamples (delta mode by default) and the episode continues until `max_consecutive_successes=50`.

### Object selection (in `_env_cfg.apply_object_selection`)
- `cube`: simple Isaac Lab cuboid spawn, replicate_physics on.
- `<tool_name>` (one of 12 DexToolBench keys): single USD per env, scales pulled from `DEXTOOLBENCH_OBJECT_SCALES`.
- `multi_dextoolbench`: `MultiUsdFileCfg` with `replicate_physics=False`; each env gets one tool round-robin. **Known issue**: `_reroot_multi_usd_rigid_bodies` re-applies `RigidBodyAPI` on parents but the loaded collision shapes are not always restored — see commit `8c6fd51` ("fixed multi-object loading issue, but the collision is not restored").

## Common commands

Run everything from the repo root.

```bash
# Training (single-GPU defaults: 1536 envs, six exploration blocks of size 256)
python simtoolreal_lab/train_rl_games.py \
  --task simtoolreal_sharpa --num_envs 1536 --headless \
  agent.params.config.expl_type=mixed_expl_learn_param \
  agent.params.config.use_others_experience=lf \
  agent.params.config.off_policy_ratio=1.0 \
  agent.params.config.expl_reward_type=entropy \
  agent.params.config.expl_coef_block_size=256 \
  env.object_scale_noise_multiplier_range='[0.9,1.1]' \
  env.force_consecutive_near_goal_steps=True \
  env.force_scale=20.0 env.torque_scale=2.0 \
  agent.wandb_activate=False

# Replay an Isaac Lab checkpoint
python simtoolreal_lab/play_rl_games.py \
  --task simtoolreal_sharpa --num_envs 8 --headless \
  --checkpoint <path-to>/nn/model_<epoch>.pth

# MuJoCo sim2sim replay (uses pretrained_policy by default)
python -m simtoolreal_lab.deployment.mujoco.mujoco_env_no_ros \
  --object-name claw_hammer --press-enter-to-execute

# Smoke test (loads the scene, steps one frame)
SIMTOOLREAL_TEST_DEVICE=cuda:0 pytest simtoolreal_lab/tasks/simtoolreal_sharpa/tests/test_load_scene.py
```

Checkpoints: `nn/model_<epoch>.pth` (every `save_frequency=1000`), `last/model.pth` (every 3 epochs), `best/model.pth` (after `save_best_after=100`). Hydra outputs land in `simtoolreal_lab/tasks/simtoolreal_sharpa/{outputs,logs}/`.

## Conventions / gotchas

- **Do not pip-install `rl-games` from upstream**. `train_rl_games.py` and `play_rl_games.py` prepend `simtoolreal_lab/rl_games` to `sys.path` so the vendored SAPO fork wins. Verify with the snippet in README §RL-Games (the `mixed_expl` string must appear in `a2c_common.py`).
- **`apply_object_selection` must be called twice** in train/play: once in `SimToolRealSharpaEnvCfg.__post_init__`, and again in the launcher after CLI overrides. Adding a new object means adding a key to `DEXTOOLBENCH_OBJECT_SCALES` *and* dropping a USD at `assets/dextoolbench_usd/<name>/<name>.usd`.
- **Joint-name ordering is load-bearing**. `KUKA_SHARPA_JOINT_NAMES` (29 names) must stay in the same order as the reference IsaacGym `observation_action_utils_sharpa`; the policy was trained against this index layout. The MuJoCo replay scene uses `palmleft_*` prefixes (`mujoco_sim.py::JOINT_NAMES`) — a separate name set that is *position-aligned* with the Isaac Lab list.
- **Asymmetric obs**: the env returns `{"policy": obs, "critic": state}`. The training pipeline's `RlGamesVecEnvWrapper` + `central_value_config` wire this up. Don't change the observation dict keys.
- **Goal collisions are explicitly disabled** in `_disable_goal_object_collisions`. The goal object is a kinematic visual marker — never enable its collisions.
- **Object mass override**: `_apply_object_mass` overrides per-body mass at startup so every tool weighs `cfg.object_mass=0.05 kg` regardless of the USD's authored mass. Inertias are rescaled proportionally.
- **Reference parity gaps** (from README, still TODO): observation/action delay, object-state delay/noise, joint-velocity observation noise. The reference IsaacGym flags are `useObsDelay`, `useActionDelay`, `useObjectStateDelayNoise`, `jointVelocityObsNoiseStd`.
- **Path in README is wrong** for this machine: it references `/home/chi-zhang/projects/simtoolreal_isaacsim`. The actual working directory is `/home/carsten.oertel/code/simtoolreal_isaacsim`.
