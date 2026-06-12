# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sync-safe per-epoch reward/penalty/termination debug printout for forge_kuka.

The danger with printing reward terms during training is that ``.item()`` /
``.cpu()`` / ``print`` on a GPU tensor each forces a CPU<->GPU synchronization
that stalls the rollout. This module avoids that with a three-part pattern:

1. Accumulate each logged term as a running sum **on the GPU, every step** (a
   pure device op, no sync). This piggybacks on the scalar tensors the env
   already writes to ``extras`` (``logs_rew_*``, ``successes``, etc.).
2. Transfer **once per epoch**: stack all terms and do a single ``.cpu()`` — one
   cheap sync for the whole table, instead of one per term per step.
3. Hand the resulting CPU dict to a **background daemon thread** that does the
   string formatting and ``print``, so the stdout I/O never blocks training.

Use ``DebugRewardObserver`` as an rl_games ``AlgoObserver`` (per-epoch printout
during training). Use ``ThreadedTablePrinter`` directly for inference loops
(play / zero / random agents), where you can enqueue per episode.
"""

from __future__ import annotations

import queue
import threading

import torch

try:
    from rl_games.common.algo_observer import AlgoObserver
except Exception:  # rl_games not importable in this context (e.g. plain inference)
    AlgoObserver = object


_TABLE_WIDTH = 66

# Short display names for reward terms (key has the "logs_rew_" prefix stripped).
_REW_ALIAS = {
    "curr_engaged": "engaged",
    "curr_success": "success",
    "action_penalty_ee": "act_pen_ee",
    "action_penalty_asset": "act_pen_asset",
    "action_grad_penalty": "act_grad_pen",
    "contact_penalty": "contact_pen",
    "success_pred_error": "succ_pred_err",
}

# Non-reward metric keys surfaced in the CONTACT / GRIP block: (key, label, unit).
#   unit: "pct" -> x100 with %, "cm" -> x100 with cm, "raw" -> as-is.
_GRIP_KEYS = (
    ("logs_in_contact", "in_contact", "pct"),
    ("logs_perp_ok", "perp_ok", "pct"),
    ("logs_contact_steps", "contact_steps", "raw"),
    ("logs_contact_force", "contact_force", "raw"),
    ("logs_pinch_perp_align", "pinch_align", "raw"),
    ("logs_ee_tilt_deg", "ee_tilt_deg", "raw"),
    ("logs_nut_bolt_dist", "nut_bolt_dist", "cm"),
    ("logs_hand_nut_dist", "hand_nut_dist", "cm"),
)


def _two_col(cells):
    """Lay a list of preformatted cell strings into two columns."""
    half = _TABLE_WIDTH // 2
    lines = []
    for i in range(0, len(cells), 2):
        lines.append("".join(c.ljust(half) for c in cells[i : i + 2]).rstrip())
    return lines


def render_breakdown(title, label, terms):
    """Build the sectioned breakdown string from a flat {key: value} dict.

    Reward terms are the SIGNED contribution to the total (raw * scale), so
    penalties read negative and a disabled term (scale 0) reads 0. Unrecognised
    ``logs_*`` keys are still shown (OTHER block) so nothing silently disappears.
    """
    terms = dict(terms)
    total = terms.pop("logs_rew_total", None)
    ep_len = terms.pop("logs_ep_len", None)
    grip_lookup = {k: (lbl, unit) for k, lbl, unit in _GRIP_KEYS}

    rewards, grip, term_evt, early, success, other = {}, {}, {}, {}, {}, {}
    for k, v in terms.items():
        if k.startswith("logs_rew_"):
            name = k[len("logs_rew_") :]
            rewards[_REW_ALIAS.get(name, name)] = v
        elif k in grip_lookup:
            grip[k] = v
        elif k.startswith("logs_term_"):
            term_evt[k[len("logs_term_") :]] = v
        elif k.startswith("early_term_"):
            early[k[len("early_term_") :]] = v
        elif k in ("successes", "success_times"):
            success[k] = v
        elif k.startswith("logs_"):
            other[k[len("logs_") :]] = v
        else:
            other[k] = v

    bar = "=" * _TABLE_WIDTH
    rule = "-" * _TABLE_WIDTH
    head = f"  {title} │ {label}"
    if ep_len is not None:
        head += f" │ avg_ep_len {ep_len:.0f}"
    out = [bar, head, bar, " REWARDS  (signed contribution to total reward)"]
    if total is not None:
        out.append(f"  {'TOTAL':<16}{total:>+9.3f}")
    out += _two_col(
        [f"  {n:<16}{v:>+9.3f}" for n, v in sorted(rewards.items(), key=lambda kv: kv[1], reverse=True)]
    )

    if grip:
        out += [rule, " CONTACT / GRIP"]
        cells = []
        for k in grip_lookup:
            if k not in grip:
                continue
            lbl, unit = grip_lookup[k]
            v = grip[k]
            s = f"{v * 100:>6.1f}%" if unit == "pct" else f"{v * 100:>6.2f}cm" if unit == "cm" else f"{v:>7.3f}"
            cells.append(f"  {lbl:<16}{s}")
        out += _two_col(cells)

    if term_evt or early:
        out += [rule, " TERMINATIONS  (rate)"]
        if term_evt:
            out += _two_col([f"  {n:<16}{v * 100:>6.1f}%" for n, v in sorted(term_evt.items())])
        nz = [f"{n}={v:.2f}" for n, v in sorted(early.items()) if abs(v) > 1e-9]
        if nz:
            out.append("  early_term: " + "  ".join(nz))

    if success:
        out += [rule, " SUCCESS"]
        if "successes" in success:
            out.append(f"  successes    {success['successes'] * 100:>6.1f}%")
        if "success_times" in success:
            out.append(f"  success_time {success['success_times']:>7.1f} steps")

    if other:
        out += [rule, " OTHER"]
        out += _two_col([f"  {n:<16}{v:>+9.3f}" for n, v in sorted(other.items())])

    out.append(bar)
    return "\n".join(out)


class ThreadedTablePrinter:
    """Background-thread printer. ``enqueue`` is non-blocking; a daemon worker
    formats the table and prints it, so the caller (training loop) never waits on
    stdout."""

    def __init__(self, title: str = "forge_kuka"):
        self._title = title
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            label, terms = item
            print(render_breakdown(self._title, label, terms), flush=True)

    def enqueue(self, label: str, terms: dict[str, float]):
        """Hand a CPU dict of {term_name: value} to the printer thread (non-blocking)."""
        self._q.put((label, dict(terms)))


def _accumulate_scalar_infos(sums: dict, infos) -> bool:
    """Add every 0-dim tensor / float in ``infos`` to running GPU sums (no sync).

    Returns True if anything was accumulated.
    """
    if not isinstance(infos, dict):
        return False
    found = False
    for k, v in infos.items():
        if torch.is_tensor(v):
            if v.dim() == 0:
                sums[k] = sums.get(k, 0.0) + v
                found = True
        elif isinstance(v, (int, float)):
            sums[k] = sums.get(k, 0.0) + float(v)
            found = True
    return found


class DebugRewardObserver(AlgoObserver):
    """rl_games observer that prints the mean of every scalar term the env logged
    to ``extras`` (reward components, penalties, success/termination rates) once
    per epoch, on a background thread, with a single per-epoch CPU transfer."""

    def __init__(self, title: str = "forge_kuka"):
        super().__init__()
        self.printer = ThreadedTablePrinter(title)
        self._sums: dict = {}
        self._count: int = 0

    def after_init(self, algo):  # noqa: D401 - rl_games hook
        self.algo = algo

    def process_infos(self, infos, done_indices):
        # Called every env step with the env's `extras`. Accumulate on-GPU only.
        if _accumulate_scalar_infos(self._sums, infos):
            self._count += 1

    def after_print_stats(self, frame, epoch_num, total_time):
        # Called once per epoch. Pull each accumulated term to a CPU scalar, then
        # print off-thread. Done per-term (not torch.stack) because forge logs a mix
        # of GPU tensors and python floats (e.g. early_term_* via .item()), which
        # can't be stacked together. These transfers happen once per epoch, not per
        # step, so they don't gate the rollout.
        if self._count == 0 or not self._sums:
            return
        terms = {}
        for k, s in self._sums.items():
            val = s.detach().cpu().item() if torch.is_tensor(s) else float(s)
            terms[k] = val / self._count
        self.printer.enqueue(f"epoch {epoch_num} │ frame {frame}", terms)
        self._sums = {}
        self._count = 0
