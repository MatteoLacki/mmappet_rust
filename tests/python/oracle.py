"""Python mmappet as the oracle for the Rust crate's tests (`tests/oracle.rs`)."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from mmappet import DatasetWriter

# Every dtype the crate supports. Time columns hold only the units pandas keeps.
DTYPES = [
    *"uint8 int8 uint16 int16 uint32 int32 uint64 int64".split(),
    *"float16 float32 float64 complex64 complex128 bool".split(),
    *(f"{kind}[{unit}]" for kind in ("datetime64", "timedelta64") for unit in ("s", "ms", "us", "ns")),
]


def smoke(path: Path) -> None:
    with DatasetWriter(path, overwrite_dir=True) as writer:
        writer.append(
            id=np.array([1, 2, 3], dtype=np.uint32),
            score=np.array([0.5, 1.5, 2.5], dtype=np.float64),
        )


def dtypes() -> None:
    """Print numpy's name and itemsize of each supported dtype, tab-separated."""
    for name in DTYPES:
        dtype = np.dtype(name)
        print(f"{dtype}\t{dtype.itemsize}")


def new_types(path: Path) -> None:
    """A 3-row dataset with one column of each type 0.1.0 did not read."""
    frame = pd.DataFrame(
        {
            "half": np.array([0.5, -2.0, np.inf], dtype=np.float16),
            "c64": np.array([1 + 2j, -0.5j, 3], dtype=np.complex64),
            "c128": np.array([1 + 2j, -0.5j, 3], dtype=np.complex128),
            "flag": np.array([True, False, True]),
            "when": np.array([0, 1500, "NaT"], dtype="datetime64[ms]"),
            "took": np.array([1, -2, "NaT"], dtype="timedelta64[ns]"),
        }
    )
    with DatasetWriter(path, overwrite_dir=True) as writer:
        writer.append_df(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke_parser = commands.add_parser("smoke", help="write a fixed 3-row dataset")
    smoke_parser.add_argument("path", type=Path)
    commands.add_parser("dtypes", help="print name and itemsize of each supported dtype")
    new_types_parser = commands.add_parser("new-types", help="write one column per new type")
    new_types_parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.path)
    elif args.command == "dtypes":
        dtypes()
    elif args.command == "new-types":
        new_types(args.path)


if __name__ == "__main__":
    main()
