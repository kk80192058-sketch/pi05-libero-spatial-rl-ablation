#!/usr/bin/env python3
"""Download the fixed 10-demonstration subset for LIBERO-Spatial task 6."""

from pathlib import Path

from huggingface_hub import snapshot_download


EPISODE_IDS = (10, 20, 23, 46, 51, 54, 57, 67, 70, 73)
DEFAULT_OUTPUT = "/root/autodl-tmp/vla-rl/data/libero_spatial_task6_10demos"


def main() -> None:
    patterns = ["meta/*"] + [
        f"data/chunk-000/episode_{episode_id:06d}.parquet"
        for episode_id in EPISODE_IDS
    ]
    output_dir = Path(DEFAULT_OUTPUT)
    snapshot_download(
        repo_id="physical-intelligence/libero",
        repo_type="dataset",
        local_dir=output_dir,
        allow_patterns=patterns,
    )
    print(f"Downloaded task-6 subset to {output_dir}")


if __name__ == "__main__":
    main()
