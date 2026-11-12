"""Python mmappet as the oracle for the Rust crate's tests (`tests/oracle.rs`)."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from mmappet import DatasetWriter, open_dataset_dct

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


def spaced_names(path: Path) -> None:
    """2 rows; column names with spaces in them."""
    frame = pd.DataFrame(
        {
            "retention time": np.array([1.5, 2.5]),
            "charge state": np.array([2, 3], dtype=np.int8),
        }
    )
    with DatasetWriter(path, overwrite_dir=True) as writer:
        writer.append_df(frame)


def no_rows(path: Path) -> None:
    """A schema and empty column files."""
    DatasetWriter.new(path, overwrite_dir=True, id=np.uint32, score=np.float64).close()


FILLED = {"id": "uint32", "score": "float64", "when": "datetime64[us]"}


def filled_values(n: int) -> dict:
    """What `preallocate`d columns hold once filled: row i is (3i, i/2, 1000i)."""
    i = np.arange(n)
    return {
        "id": (3 * i).astype(np.uint32),
        "score": i / 2,
        "when": (1000 * i).astype("datetime64[us]"),
    }


def preallocate(path: Path, nrows: int) -> None:
    schema = pd.DataFrame({name: np.empty(0, dtype=dtype) for name, dtype in FILLED.items()})
    DatasetWriter.preallocate_dataset(path, schema, nrows=nrows, overwrite_dir=True)


def check_filled(path: Path) -> None:
    """Raises unless the dataset at `path` holds `filled_values`."""
    columns = open_dataset_dct(path)
    expected = filled_values(len(columns["id"]))
    for name, dtype in FILLED.items():
        assert columns[name].dtype == np.dtype(dtype), (name, columns[name].dtype)
        assert np.array_equal(columns[name], expected[name]), (name, columns[name])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke_parser = commands.add_parser("smoke", help="write a fixed 3-row dataset")
    smoke_parser.add_argument("path", type=Path)
    commands.add_parser("dtypes", help="print name and itemsize of each supported dtype")
    new_types_parser = commands.add_parser("new-types", help="write one column per new type")
    new_types_parser.add_argument("path", type=Path)
    for name in ("spaced-names", "no-rows", "check-filled"):
        commands.add_parser(name).add_argument("path", type=Path)
    preallocate_parser = commands.add_parser("preallocate")
    preallocate_parser.add_argument("path", type=Path)
    preallocate_parser.add_argument("nrows", type=int)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke(args.path)
    elif args.command == "dtypes":
        dtypes()
    elif args.command == "new-types":
        new_types(args.path)
    elif args.command == "spaced-names":
        spaced_names(args.path)
    elif args.command == "no-rows":
        no_rows(args.path)
    elif args.command == "preallocate":
        preallocate(args.path, args.nrows)
    elif args.command == "check-filled":
        check_filled(args.path)


if __name__ == "__main__":
    main()
