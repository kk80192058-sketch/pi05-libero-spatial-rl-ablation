#!/usr/bin/env python3
"""Download all public LIBERO demonstrations belonging to one Spatial task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True, help="LIBERO-Spatial task id, 0-9")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.task_id <= 9:
        raise SystemExit("--task-id must be in [0, 9] for LIBERO-Spatial")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="physical-intelligence/libero",
        repo_type="dataset",
        local_dir=output,
        allow_patterns=["meta/*"],
    )

    tasks = {row["task_index"]: row["task"] for row in read_jsonl(output / "meta" / "tasks.jsonl")}
    task_text = tasks[args.task_id]
    episode_ids = [
        row["episode_index"]
        for row in read_jsonl(output / "meta" / "episodes.jsonl")
        if row["tasks"] == [task_text]
    ]
    if not episode_ids:
        raise RuntimeError(f"No demonstrations found for task {args.task_id}: {task_text}")

    patterns = [f"data/chunk-{episode_id // 1000:03d}/episode_{episode_id:06d}.parquet" for episode_id in episode_ids]
    snapshot_download(
        repo_id="physical-intelligence/libero",
        repo_type="dataset",
        local_dir=output,
        allow_patterns=patterns,
    )
    manifest = {
        "task_id": args.task_id,
        "task": task_text,
        "episode_ids": episode_ids,
    }
    (output / "task_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Downloaded {len(episode_ids)} demos for task {args.task_id}: {task_text}")


if __name__ == "__main__":
    main()
