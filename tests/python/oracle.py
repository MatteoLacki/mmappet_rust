"""Python mmappet as the oracle for the Rust crate's tests (`tests/oracle.rs`)."""

import argparse
import fcntl
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from mmappet import DatasetWriter, open_dataset_dct
from mmappet.dirty_hacks_experimental_never_use import crop_dataset

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


def reference_frame(start: int, stop: int) -> pd.DataFrame:
    """Rows [start, stop) of the reference dataset: column `col <dtype>` for each dtype,
    row i holding 7i (unsigned), 7i - 10 (signed), i/4 - 1 (float), i/4 - 1 + (i/2)j
    (complex), i odd (bool) or 1000i - 2000 ticks (time). `tests/oracle.rs` builds the same."""
    i = np.arange(start, stop)
    columns = {}
    for name in DTYPES:
        dtype = np.dtype(name)
        if dtype.kind == "u":
            values = (7 * i).astype(dtype)
        elif dtype.kind == "i":
            values = (7 * i - 10).astype(dtype)
        elif dtype.kind == "f":
            values = (i / 4 - 1).astype(dtype)
        elif dtype.kind == "c":
            values = (i / 4 - 1 + 1j * (i / 2)).astype(dtype)
        elif dtype.kind == "b":
            values = i % 2 == 1
        else:
            values = (1000 * i - 2000).astype(dtype)
        columns[f"col {name}"] = values
    return pd.DataFrame(columns)


def reference(path: Path, start: int, stop: int) -> None:
    """Appends reference rows [start, stop) to the dataset at `path`, creating it if needed."""
    with DatasetWriter(path, append_ok=path.exists()) as writer:
        writer.append_df(reference_frame(start, stop))


def check_extra(path: Path, nrows: int) -> None:
    """Raises unless `path` holds reference rows [0, nrows) plus `extra` = 3i as uint16."""
    columns = open_dataset_dct(path)
    expected = reference_frame(0, nrows)
    assert list(columns) == [*expected.columns, "extra"], list(columns)
    for name in expected.columns:
        assert columns[name].dtype == expected[name].dtype, name
        assert np.array_equal(columns[name], expected[name].to_numpy()), name
    assert columns["extra"].dtype == np.uint16
    assert np.array_equal(columns["extra"], 3 * np.arange(nrows)), columns["extra"]


def hold_lock(path: Path, kind: str) -> None:
    """flock the dataset directory (`directory`) or its `write.lock` (`file`), print
    `locked`, and hold the lock until stdin closes or the process is killed."""
    if kind == "directory":
        fd = os.open(path, os.O_RDONLY)
    else:
        fd = os.open(path / "write.lock", os.O_WRONLY | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print("locked", flush=True)
    sys.stdin.read()


# ---- Random round trips (`tests/oracle.rs` draws the same shapes) ----

NAMES = ["a", "mz", "rt", "retention time", "charge state", "x y z", "id", "score", "intensity", "col 7"]


def random_values(rng: np.random.Generator, name: str, nrows: int) -> np.ndarray:
    """Random bits for numbers (NaNs included), 0/1 for bool, and ticks within pandas' range
    with some NaT for time."""
    dtype = np.dtype(name)
    if dtype.kind == "b":
        return rng.integers(0, 2, nrows).astype(bool)
    if dtype.kind in "mM":
        ticks = rng.integers(-(2**40), 2**40, nrows)
        ticks[rng.random(nrows) < 0.1] = np.iinfo(np.int64).min
        return ticks.astype(dtype)
    return np.frombuffer(rng.bytes(nrows * dtype.itemsize), dtype=dtype).copy()


def random_schema(rng: np.random.Generator) -> tuple[dict, int]:
    ncols = int(rng.integers(1, 9))
    names = rng.choice(NAMES, ncols, replace=False)
    dtypes = rng.choice(DTYPES, ncols)
    nrows = int(rng.choice([0, 1, int(rng.integers(2, 1001))]))
    return dict(zip(map(str, names), map(str, dtypes))), nrows


def random_chunks(rng: np.random.Generator, nrows: int) -> list[tuple[int, int]]:
    cuts = rng.integers(1, nrows, int(rng.integers(0, 5))) if nrows > 1 else []
    bounds = [0, *sorted(set(int(c) for c in cuts)), nrows]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]


def write_in_chunks(rng: np.random.Generator, path: Path, columns: dict) -> None:
    """Writes `columns` with DatasetWriter in random chunks, sometimes closing and reopening
    with append_ok between them."""
    dtypes = {name: values.dtype for name, values in columns.items()}
    writer = DatasetWriter.new(path, overwrite_dir=True, **dtypes)
    nrows = len(next(iter(columns.values())))
    for start, stop in random_chunks(rng, nrows):
        if rng.random() < 0.3:
            writer.close()
            writer = DatasetWriter(path, append_ok=True)
        writer.append_df(pd.DataFrame({name: values[start:stop] for name, values in columns.items()}))
    writer.close()


def gen(root: Path, count: int) -> None:
    """Random datasets `root/<seed>` for seeds 0..count, and `root/<seed>.crop`, its first
    rows by crop_dataset, the count in `root/<seed>.crop.rows`."""
    for seed in range(count):
        rng = np.random.default_rng([0, seed])
        schema, nrows = random_schema(rng)
        columns = {name: random_values(rng, dtype, nrows) for name, dtype in schema.items()}
        write_in_chunks(rng, root / str(seed), columns)
        keep = int(rng.integers(0, nrows + 1))
        crop_dataset(root / str(seed), root / f"{seed}.crop", keep)
        (root / f"{seed}.crop.rows").write_text(str(keep))


def copy(source: Path, target: Path, count: int) -> None:
    """Reads `source/<seed>` and rewrites it to `target/<seed>` in random chunks."""
    for seed in range(count):
        rng = np.random.default_rng([1, seed])
        columns = open_dataset_dct(source / str(seed))
        write_in_chunks(rng, target / str(seed), columns)


def preallocate_random(root: Path, count: int) -> None:
    """Random schemas preallocated at `root/<seed>`, zero-filled."""
    for seed in range(count):
        rng = np.random.default_rng([2, seed])
        schema, nrows = random_schema(rng)
        frame = pd.DataFrame({name: np.empty(0, dtype=dtype) for name, dtype in schema.items()})
        DatasetWriter.preallocate_dataset(root / str(seed), frame, nrows=nrows, overwrite_dir=True)


def fill(root: Path, count: int) -> None:
    """Fills the preallocated `root/<seed>` in place with random values, and writes the same
    values with DatasetWriter to `root/<seed>.written`."""
    for seed in range(count):
        rng = np.random.default_rng([3, seed])
        columns = open_dataset_dct(root / str(seed), read_write=True)
        for name, values in columns.items():
            values[:] = random_values(rng, str(values.dtype), len(values))
        write_in_chunks(rng, root / f"{seed}.written", columns)
        del columns


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
    for name in ("reference",):
        reference_parser = commands.add_parser(name)
        reference_parser.add_argument("path", type=Path)
        reference_parser.add_argument("start", type=int)
        reference_parser.add_argument("stop", type=int)
    check_extra_parser = commands.add_parser("check-extra")
    check_extra_parser.add_argument("path", type=Path)
    check_extra_parser.add_argument("nrows", type=int)
    crop_parser = commands.add_parser("crop")
    crop_parser.add_argument("source", type=Path)
    crop_parser.add_argument("target", type=Path)
    crop_parser.add_argument("nrows", type=int)
    hold_lock_parser = commands.add_parser("hold-lock")
    hold_lock_parser.add_argument("path", type=Path)
    hold_lock_parser.add_argument("kind", choices=["directory", "file"])
    for name in ("gen", "preallocate-random", "fill"):
        many_parser = commands.add_parser(name)
        many_parser.add_argument("root", type=Path)
        many_parser.add_argument("count", type=int)
    copy_parser = commands.add_parser("copy")
    copy_parser.add_argument("source", type=Path)
    copy_parser.add_argument("target", type=Path)
    copy_parser.add_argument("count", type=int)
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
    elif args.command == "reference":
        reference(args.path, args.start, args.stop)
    elif args.command == "check-extra":
        check_extra(args.path, args.nrows)
    elif args.command == "crop":
        crop_dataset(args.source, args.target, args.nrows)
    elif args.command == "hold-lock":
        hold_lock(args.path, args.kind)
    elif args.command == "gen":
        gen(args.root, args.count)
    elif args.command == "copy":
        copy(args.source, args.target, args.count)
    elif args.command == "preallocate-random":
        preallocate_random(args.root, args.count)
    elif args.command == "fill":
        fill(args.root, args.count)


if __name__ == "__main__":
    main()
