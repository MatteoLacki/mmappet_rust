"""Python mmappet as the oracle for the Rust crate's tests (`tests/oracle.rs`)."""

import argparse
from pathlib import Path

import numpy as np
from mmappet import DatasetWriter


def smoke(path: Path) -> None:
    with DatasetWriter(path, overwrite_dir=True) as writer:
        writer.append(
            id=np.array([1, 2, 3], dtype=np.uint32),
            score=np.array([0.5, 1.5, 2.5], dtype=np.float64),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke_parser = commands.add_parser("smoke", help="write a fixed 3-row dataset")
    smoke_parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.path)


if __name__ == "__main__":
    main()
