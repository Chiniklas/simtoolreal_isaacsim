# Sharpa Nutscrew Pick-Place-Screw Notes

## Current Reward Intent

This task has been adapted from grasping/reaching long-handle tools to hand nut
screwing. The nut starts already on the bolt, so lifting is not part of the task
objective.

- Lift reward is disabled in the screwing phase:
  - `lifting_rew_scale: 0.0`
  - `lifting_bonus: 0.0`
  - `lifting_bonus_threshold: -1.0`
- The negative lift threshold marks the nut as already past the old pick-place
  lift gate, so downstream keypoint/screwing rewards can still run.
- Keypoint/success reward is contact-gated for the tripod config.
- Valid tripod contact currently means:
  - thumb must contact the nut
  - at least 2 active tripod fingers must contact the nut
  - active tripod fingers are thumb, index, middle

## Contact Sensor Validation

The contact gate depends on filtered IsaacLab `ContactSensor` readings from the
finger distal links to the nut object. The local USD's elastomer pads are fixed
child links and do not carry contact reporter API, so the sensors attach to the
reporting `*_DP` bodies instead of `*_elastomer` bodies. Validate this before
trusting reward curves.

Watch these TensorBoard metrics:

- `nut_contact_rate`
- `nut_contact_force`
- `nut_contact_steps`
- `thumb_contact_rate`
- `index_contact_rate`
- `middle_contact_rate`
- `thumb_contact_force`
- `index_contact_force`
- `middle_contact_force`

Expected behavior:

- visible thumb+index contact should raise `thumb_contact_rate`,
  `index_contact_rate`, and `nut_contact_rate`
- visible thumb+middle contact should raise `thumb_contact_rate`,
  `middle_contact_rate`, and `nut_contact_rate`
- visible index+middle contact without thumb should not raise `nut_contact_rate`
- if all per-finger contact rates stay exactly zero during visible contact, the
  filter target or sensor body path is probably wrong
- if `nut_contact_force` is nonzero but all contact rates are zero, the contact
  force threshold may still be too high for the DP-body sensor setup
- if env construction fails with "could not find any bodies with contact reporter
  API", the sensor was probably attached to a non-reporting child link

Important reference lesson from ForgeUltra: filtered `force_matrix_w` only worked
when the filter targeted the nut rigid-body link, not a wrapper Xform or a
collision mesh.

## Reward Scale Validation

The contact bonus was adapted from ForgeUltra, then scaled down for this task.

Current tripod values:

- `contact_force_threshold: 0.02`
- `contact_bonus: 0.2`
- `sustained_contact_bonus: 0.002`
- `sustained_contact_cap: 50`

This gives valid contact about `0.202` reward on the first contact step and caps
at `0.3` per step after 50 consecutive contact steps.

Validate that contact does not dominate the task:

- `contact_bonus_reward` should help contact emerge, but should not dwarf
  `keypoint_reward`
- `nut_contact_rate` increasing without `nut_thread_progress` increasing means
  the policy may be learning to hold contact without turning
- if that happens, reduce `contact_bonus` / `sustained_contact_bonus` further or
  make the bonus curriculum-only

## Screwing Progress Validation

The main task signal should still be screw/keypoint progress, not just gripping.

Watch:

- `keypoint_reward`
- `reach_bonus`
- `keypoints_max_dist`
- `nut_thread_angle`
- `nut_thread_progress`
- `success_rate`
- `total_reward`

Expected healthy pattern:

- contact metrics rise first
- keypoint reward becomes nonzero only during valid thumb-involved tripod contact
- `nut_thread_progress` rises after stable contact appears
- success rate should not increase unless the contact gate is also active

Suspicious patterns:

- high `nut_contact_rate`, flat `nut_thread_progress`: contact reward too strong
  or turning action/control is ineffective
- flat `nut_contact_rate`, visible contact in replay: sensor/filter issue
- high keypoint/reach reward with low contact rate: contact gate is not actually
  gating the success path
- large action penalties and poor progress: hand/arm control may be thrashing

## Current Config Focus

Primary config under test:

- `agents/rl_games_sapo_tripod_cfg.yaml`

Key choices in that config:

- tripod fingers: thumb, index, middle
- contact gate: thumb required, at least 2 tripod fingers in contact
- lift reward disabled
- keypoint reward gated on valid contact
- reduced contact bonus scale
