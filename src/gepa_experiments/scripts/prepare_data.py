"""Write the synthetic BRG dataset to disk."""

from __future__ import annotations

import argparse

from gepa_experiments.cli import add_common_args, config_from_args
from gepa_experiments.data import write_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare synthetic BRG dataset")
    add_common_args(parser)
    parser.add_argument(
        "--output",
        default=None,
        help="Override output path (default: paths.dataset from config)",
    )
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    path = cfg.resolve_path(args.output) if args.output else cfg.dataset_path()
    written = write_dataset(path)
    print(f"Wrote synthetic dataset to {written}")


if __name__ == "__main__":
    main()
