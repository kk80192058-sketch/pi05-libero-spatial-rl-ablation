#!/usr/bin/env python3
"""Create a contiguous few-shot LeRobot dataset for any LIBERO-Spatial task."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def source_parquet(root: Path, episode_id: int) -> Path:
    return root / "data" / f"chunk-{episode_id // 1000:03d}" / f"episode_{episode_id:06d}.parquet"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    manifest = json.loads((source / "task_manifest.json").read_text())
    episode_ids = list(manifest["episode_ids"])
    if len(episode_ids) < args.train_count:
        raise ValueError(f"Need {args.train_count} demos but only found {len(episode_ids)}")

    shuffled = episode_ids.copy()
    random.Random(args.seed).shuffle(shuffled)
    train_ids = sorted(shuffled[: args.train_count])
    holdout_ids = sorted(shuffled[args.train_count :])

    source_info = json.loads((source / "meta" / "info.json").read_text())
    source_episodes = {row["episode_index"]: row for row in read_jsonl(source / "meta" / "episodes.jsonl")}
    output.mkdir(parents=True)
    shutil.copytree(source / "meta", output / "meta")
    out_data = output / "data" / "chunk-000"
    out_data.mkdir(parents=True)

    new_episodes: list[dict] = []
    frame_index = 0
    for new_id, old_id in enumerate(train_ids):
        table = pq.read_table(source_parquet(source, old_id))
        length = table.num_rows
        for name, values in {
            "episode_index": pa.array([new_id] * length, type=pa.int64()),
            "task_index": pa.array([0] * length, type=pa.int64()),
            "index": pa.array(range(frame_index, frame_index + length), type=pa.int64()),
        }.items():
            table = table.set_column(table.schema.get_field_index(name), name, values)
        pq.write_table(table, out_data / f"episode_{new_id:06d}.parquet")
        new_episodes.append({"episode_index": new_id, "tasks": source_episodes[old_id]["tasks"], "length": length})
        frame_index += length

    source_info.update({"total_episodes": len(train_ids), "total_frames": frame_index, "total_tasks": 1, "total_chunks": 1, "splits": {"train": f"0:{len(train_ids)}"}})
    (output / "meta" / "info.json").write_text(json.dumps(source_info, indent=2) + "\n")
    write_jsonl(output / "meta" / "episodes.jsonl", new_episodes)
    write_jsonl(output / "meta" / "tasks.jsonl", [{"task_index": 0, "task": manifest["task"]}])
    with (output / "split.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode_id", "split"])
        writer.writeheader()
        writer.writerows({"episode_id": episode_id, "split": "train"} for episode_id in train_ids)
        writer.writerows({"episode_id": episode_id, "split": "holdout"} for episode_id in holdout_ids)
    print(f"Created {len(train_ids)}-demo subset ({frame_index} frames); {len(holdout_ids)} holdout demos")


if __name__ == "__main__":
    main()
