"""
Guard against config.py and EXPERIMENTS.md drifting apart.

Both files describe the same 30 experiments: config.PRESETS is the
executable version, EXPERIMENTS.md is the presentation-facing narrative.
The README states outright that they must stay in sync, but nothing
enforced it, so a value changed in one place could silently disagree with
the other right up until someone noticed in the writeup.

This script makes that impossible to miss. For every preset it checks:

  1. the dict key matches the preset's own run_id,
  2. the run_id appears in an EXPERIMENTS.md table exactly once,
  3. every overridden numeric value (learning_rate, gamma, batch_size,
     the epsilon fields) appears in that run's table row.

Value matching is numeric, not textual, so "1e-6" in the table matches
1e-06 in the dict without caring about formatting. Run it locally or in
CI before a submission:

    python check_presets.py

Exit code 0 means they agree; 1 means they have drifted, with the
specific mismatches printed.
"""

from __future__ import annotations

import os
import re
import sys

from config import PRESETS, EXPERIMENTS_DIR

EXPERIMENTS_MD = os.path.join(os.path.dirname(EXPERIMENTS_DIR), "EXPERIMENTS.md")

# Fields that carry a tunable value we expect to see mirrored in the
# markdown tables. run_id and member are bookkeeping, not values to match.
VALUE_FIELDS = {
    "learning_rate", "gamma", "batch_size",
    "exploration_initial_eps", "exploration_final_eps", "exploration_fraction",
}

# Matches integers, decimals, and scientific notation like 1e-6 or 3e-3.
_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def numbers_in(text: str) -> list[float]:
    """Every float-parseable token in a line of markdown."""
    out = []
    for tok in _NUMBER.findall(text):
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def find_row(lines: list[str], run_id: str) -> str | None:
    """The single markdown table row whose first cell is this run_id."""
    matches = [ln for ln in lines if re.match(rf"^\|\s*{re.escape(run_id)}\s*\|", ln)]
    if len(matches) == 1:
        return matches[0]
    return None  # 0 = missing, >1 = ambiguous; both are failures upstream


def main() -> int:
    if not os.path.isfile(EXPERIMENTS_MD):
        print(f"ERROR: {EXPERIMENTS_MD} not found.")
        return 1

    with open(EXPERIMENTS_MD) as f:
        lines = f.read().splitlines()

    errors: list[str] = []

    for key, overrides in PRESETS.items():
        # 1. dict key must equal the preset's own run_id.
        run_id = overrides.get("run_id")
        if run_id != key:
            errors.append(f"{key}: dict key != run_id field ({run_id!r})")
            continue

        # 2. run_id must appear in exactly one table row.
        row = find_row(lines, run_id)
        if row is None:
            errors.append(f"{run_id}: not found (or ambiguous) in EXPERIMENTS.md")
            continue

        # 3. every overridden numeric value must appear in that row.
        row_numbers = numbers_in(row)
        for field, value in overrides.items():
            if field not in VALUE_FIELDS:
                continue
            if not any(abs(value - n) <= 1e-12 + 1e-9 * abs(value)
                       for n in row_numbers):
                errors.append(
                    f"{run_id}: {field}={value} not present in its "
                    f"EXPERIMENTS.md row (row has {row_numbers})"
                )

    if errors:
        print("config.py and EXPERIMENTS.md disagree:\n")
        for e in errors:
            print(f"  - {e}")
        print(f"\n{len(errors)} mismatch(es). Fix both files so they agree.")
        return 1

    print(f"OK: all {len(PRESETS)} presets match EXPERIMENTS.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
