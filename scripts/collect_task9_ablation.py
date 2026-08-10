#!/usr/bin/env python3
"""Collect task-9 evaluation metrics from completed RLinf logs.

Usage:
  python scripts/collect_task9_ablation.py --output results/task9_ablation.csv \
    A0_seen=/path/to/task9_a0_seen/run.log \
    A0_unseen=/path/to/task9_a0_unseen/run.log
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


METRICS = ("num_trajectories", "success_once", "success_at_end", "reward")


def metric(log: str, name: str) -> str:
    matches = re.findall(rf"{name}=([0-9.eE+-]+)", log)
    if not matches:
        raise ValueError(f"Missing {name} in evaluation log")
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "logs",
        nargs="+",
        metavar="GROUP_WINDOW=LOG",
        help="for example B20_seen=/path/to/run.log",
    )
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for spec in args.logs:
        label, raw_path = spec.split("=", 1)
        group, window = label.rsplit("_", 1)
        text = Path(raw_path).read_text(errors="replace")
        row = {"group": group, "window": window}
        row.update({name: metric(text, name) for name in METRICS})
        row["success_once_percent"] = f"{100 * float(row['success_once']):.1f}"
        row["success_at_end_percent"] = f"{100 * float(row['success_at_end']):.1f}"
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["group", "window", *METRICS, "success_once_percent", "success_at_end_percent"]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
