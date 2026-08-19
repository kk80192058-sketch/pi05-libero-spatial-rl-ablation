#!/usr/bin/env python3
"""Extract one fixed-window LIBERO result from each task console log.

Usage:
  python scripts/collect_spatial_diagnostic.py /path/to/diagnostic output.csv
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


def last_metric(text: str, metric: str) -> str | None:
    matches = re.findall(rf"{re.escape(metric)}=([0-9.]+)", text)
    return matches[-1] if matches else None


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: collect_spatial_diagnostic.py DIAGNOSTIC_DIR OUTPUT_CSV")

    root = Path(sys.argv[1])
    rows = []
    for log_path in sorted(root.glob("task_*_resets_*_*/console.log")):
        match = re.fullmatch(r"task_(\d+)_resets_(\d+)_(\d+)", log_path.parent.name)
        if not match:
            continue
        task_id, reset_start, reset_end = map(int, match.groups())
        text = log_path.read_text(errors="replace")
        success_once = last_metric(text, "success_once")
        success_at_end = last_metric(text, "success_at_end")
        reward = last_metric(text, "reward")
        if success_once is None:
            continue
        rows.append(
            {
                "task_id": task_id,
                "reset_window": f"{reset_start}-{reset_end}",
                "success_once": float(success_once),
                "success_at_end": float(success_at_end) if success_at_end else "",
                "reward": float(reward) if reward else "",
            }
        )

    rows.sort(key=lambda row: (row["success_once"], row["task_id"]))
    with Path(sys.argv[2]).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", "reset_window", "success_once", "success_at_end", "reward"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} task results to {sys.argv[2]}")


if __name__ == "__main__":
    main()
