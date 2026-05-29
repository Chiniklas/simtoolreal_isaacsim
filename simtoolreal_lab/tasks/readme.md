# SimToolReal Task Notes

This folder currently contains the original `simtoolreal_sharpa` task and the
new screwing-focused task:

```text
sharpa_nutscrew_pick_place_screw
```

## `sharpa_nutscrew_pick_place_screw`

Current purpose: isolate the screwing/unscrewing phase for the SHARPA hand and
KUKA arm. The scene starts with a nut already engaged at the top of a fixed
screw thread, so training can focus on grasping the nut and turning it.

Default generated assets:

```text
family: M20
screw:  M20X30
nut:    M20_nut
```

The generated nut/screw assets come from:

```text
simtoolreal_lab/assets/nutscrew_generated
```

The helper generator lives at:

```bash
python simtoolreal_lab/tasks/sharpa_nutscrew_pick_place_screw/tests/screw_generator.py
```

## Scene Setup

- Robot: KUKA + SHARPA hand is included.
- Table: aligned with the previous task table, size `(0.475, 0.4, 0.3)`, center
  position `(0.0, 0.0, 0.38)`, top surface `z = 0.53`.
- Screw: fixed/kinematic, head down on the table, thread along `+Z`.
- Nut: initialized concentrically at the top of the screw thread.
- Goal nut: sampled along the screw kinematic direction, initially 60-90 degrees
  from the start pose and then 60-90 degrees beyond the current nut angle after
  each reached goal.
- Rotation indicators: blue arrow on the real nut, magenta arrow on the goal nut.

## Physics Model

This is a simple hybrid model, not true thread contact.

- The nut is a dynamic rigid body, so the robot hand can contact and rotate it.
- Nut gravity is disabled for the screwing-only phase.
- Nut visual mesh has a convex-hull collision approximation for hand contact.
- The screw remains fixed and is not used for detailed mesh thread contact.
- The env reads nut angular velocity around world `Z` and projects the nut back
  onto the analytical screw path:

```text
z = z_start - pitch / (2*pi) * yaw
```

Positive screw-axis spin screws the nut downward. Negative spin unscrews it
upward, clamped at the thread top. Non-concentric translation and roll/pitch are
removed by projection so pressing or lifting the nut does not count as screw
motion.

The screw constraint also applies simple damping/friction so contact impulses do
not make the nut spin forever:

```text
screwing_thread_angular_damping = 12.0
screwing_thread_static_angular_velocity = 0.04
screwing_thread_delta_deadband = 1.0e-4
```

## Reward And Goal

The reward structure is still inherited from the pick/place task. The main
change is goal generation: in screwing phase, each new goal advances along the
screw path rather than sampling from the generic target volume.

Success does not immediately terminate the episode. Reaching the goal increments
success counters and samples a new goal. Episode termination is still handled by
timeout/failure conditions.

## Termination

Current termination conditions:

- Nut/object z falls below `table_top_z - 0.05`.
- Hand/fingertips are farther than `hand_far_from_object_threshold` from the nut.
- SHARPA hand contact with the table exceeds `table_force_threshold = 5.0 N`.
- Dropped-after-lifted condition from the original task, if enabled.
- Too many consecutive successes.
- Timeout at the configured episode length.

## Visualization

Open the task in Isaac viewer with a zero-action agent:

```bash
conda run -n simtoolreal python simtoolreal_lab/debugging/zero_agent.py \
  --task sharpa_nutscrew_pick_place_screw \
  --num_envs 1
```

Show the four real-nut/goal-nut keypoints and the graspable bounding boxes:

```bash
conda run -n simtoolreal python simtoolreal_lab/debugging/zero_agent.py \
  --task sharpa_nutscrew_pick_place_screw \
  --num_envs 1 \
  --visualize_keypoints \
  --visualize_grasp_bounding_box
```

The same visualization flags are available for `random_agent.py` and
`train_rl_games.py`.

Headless smoke check:

```bash
conda run -n simtoolreal python simtoolreal_lab/debugging/zero_agent.py \
  --task sharpa_nutscrew_pick_place_screw \
  --num_envs 1 \
  --headless \
  --max_steps 2 \
  --log_every 1
```

## Training

Default six-block SAPO training shape, matching the previous task convention:

```bash
conda run -n simtoolreal python simtoolreal_lab/train_rl_games.py \
  --task sharpa_nutscrew_pick_place_screw \
  --num_envs 1536 \
  --headless \
  agent.params.config.expl_coef_block_size=256 \
  agent.wandb_activate=False
```

Visible tiny debug run that preserves six SAPO exploration blocks:

```bash
conda run -n simtoolreal python simtoolreal_lab/train_rl_games.py \
  --task sharpa_nutscrew_pick_place_screw \
  --num_envs 6 \
  --visualize_keypoints \
  --visualize_grasp_bounding_box \
  agent.params.config.minibatch_size=96 \
  agent.params.config.central_value_config.minibatch_size=96 \
  agent.params.config.expl_coef_block_size=1
```

Headless tiny debug run:

```bash
conda run -n simtoolreal python simtoolreal_lab/train_rl_games.py \
  --task sharpa_nutscrew_pick_place_screw \
  --num_envs 6 \
  --headless \
  agent.params.config.minibatch_size=96 \
  agent.params.config.central_value_config.minibatch_size=96 \
  agent.params.config.expl_coef_block_size=1
```

## Known Limitations

- Thread contact is not physically simulated.
- Screw/nut interaction is an analytical pitch projection.
- Nut collision currently uses a coarse convex-hull approximation for hand
  contact.
- The reward and termination logic are still mostly inherited from the original
  pick/place task and may need task-specific cleanup for pure screwing training.
