#!/usr/bin/env python3
"""Render the final task-9 ablation CSV as a small reproducible figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


LABELS = {"A0": "A0\nbase", "A1": "A1\nPPO", "B20": "B20\n20-demo SFT", "C20": "C20\nSFT + PPO"}
COLORS = {"seen": "#4C78A8", "unseen": "#F58518"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    data = {(row["group"], row["window"]): float(row["success_once_percent"]) for row in rows}
    groups = [group for group in ("A0", "A1", "B20", "C20") if any(row["group"] == group for row in rows)]
    x = list(range(len(groups)))
    width = 0.34
    figure, axis = plt.subplots(figsize=(8, 4.6), dpi=180)
    for offset, window in ((-width / 2, "seen"), (width / 2, "unseen")):
        values = [data[(group, window)] for group in groups]
        bars = axis.bar([value + offset for value in x], values, width, label=window, color=COLORS[window])
        axis.bar_label(bars, labels=[f"{value:.0f}%" for value in values], padding=3, fontsize=9)
    axis.set_xticks(x, [LABELS[group] for group in groups])
    axis.set_ylim(0, 108)
    axis.set_ylabel("success_once (%)")
    axis.set_title("LIBERO-Spatial task 9: fixed 10-trial evaluation windows")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="reset window")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)


if __name__ == "__main__":
    main()
