#!/usr/bin/env python3
"""Build a self-contained 20-episode LeRobot subset for task-6 few-shot SFT.

The source LIBERO dataset has sparse, global episode indices. OpenPI's
LeRobot loader expects a contiguous local dataset, so selected trajectories
are re-indexed from 0 to N-1 without changing observations or actions.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


TRAIN_EPISODE_IDS = (
    20,
    46,
    51,
    57,
    70,
    100,
    106,
    115,
    143,
    179,
    187,
    204,
    214,
    270,
    283,
    288,
    306,
    314,
    316,
    376,
)


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
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    source_info = json.loads((source / "meta" / "info.json").read_text())
    source_episodes = {
        row["episode_index"]: row
        for row in read_jsonl(source / "meta" / "episodes.jsonl")
    }

    output.mkdir(parents=True)
    shutil.copytree(source / "meta", output / "meta")
    output_data = output / "data" / "chunk-000"
    output_data.mkdir(parents=True)

    new_episodes: list[dict] = []
    total_frames = 0
    global_frame_index = 0
    for new_id, old_id in enumerate(TRAIN_EPISODE_IDS):
        old_path = source_parquet(source, old_id)
        if not old_path.is_file():
            raise FileNotFoundError(old_path)
        table = pq.read_table(old_path)
        num_rows = table.num_rows
        for column_name, values in {
            "episode_index": pa.array([new_id] * num_rows, type=pa.int64()),
            "task_index": pa.array([0] * num_rows, type=pa.int64()),
            "index": pa.array(range(global_frame_index, global_frame_index + num_rows), type=pa.int64()),
        }.items():
            column_index = table.schema.get_field_index(column_name)
            table = table.set_column(column_index, column_name, values)
        pq.write_table(table, output_data / f"episode_{new_id:06d}.parquet")

        source_episode = source_episodes[old_id]
        new_episodes.append(
            {
                "episode_index": new_id,
                "tasks": source_episode["tasks"],
                "length": num_rows,
            }
        )
        total_frames += num_rows
        global_frame_index += num_rows

    source_info.update(
        {
            "total_episodes": len(TRAIN_EPISODE_IDS),
            "total_frames": total_frames,
            "total_tasks": 1,
            "total_chunks": 1,
            "splits": {"train": f"0:{len(TRAIN_EPISODE_IDS)}"},
        }
    )
    (output / "meta" / "info.json").write_text(json.dumps(source_info, indent=2) + "\n")
    write_jsonl(output / "meta" / "episodes.jsonl", new_episodes)
    write_jsonl(
        output / "meta" / "tasks.jsonl",
        [{"task_index": 0, "task": new_episodes[0]["tasks"][0]}],
    )
    print(f"Created {len(TRAIN_EPISODE_IDS)}-episode subset with {total_frames} frames: {output}")


if __name__ == "__main__":
    main()
