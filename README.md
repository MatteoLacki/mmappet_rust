# mmappet (Rust)

Rust library for mmappet datasets - memory-mapped columnar data format.

This is the Rust equivalent of the Python [mmappet](../mmappet/) library.

## Status

**Reading: Complete** - every dtype above, read-only or read-write.

**Writing: Not implemented** - Future work.

## Installation

```toml
[dependencies]
mmappet = { path = "../mmappet_rust" }
```

## Usage

### Library API

```rust
use mmappet::{Dataset, DatasetMut, Typed};

let ds = Dataset::open("precursors.mmappet")?;
println!("{} rows, columns {:?}", ds.len(), ds.schema().column_names());

// Borrow: slices live as long as `ds`; the dtype is checked against the Rust type.
let mz: &[f64] = ds.get("mz")?;
let charge: &[i64] = ds.get("charges")?;
let ppm_lo: Option<&[f64]> = ds.get_opt("ppm_tol_lo")?; // missing -> None, wrong dtype -> Err

// Raw: dtype and bytes, e.g. to pick f32 or f64 at run time.
let column = ds.column("mz")?;
println!("{} values of {}", column.len(), column.dtype());
```

A column can also outlive its `Dataset`, to be kept in a struct or handed to another
owner. There are two ways:

```rust
// Share: `ds` keeps the column too. Clones share one map, freed with the last of them.
let ds = Dataset::open("speclib/arrays.mmappet")?;
let intensity: Typed<mmappet::f16> = ds.typed("predicted_intensity")?;
let annotation = ds.column("annotation_id")?.clone(); // raw, `AsRef<[u8]>`
drop(ds);
let first: mmappet::f16 = intensity[0]; // `Typed<T>` derefs to `&[T]`

// Move: the column leaves `ds`, and the caller is its only owner.
let mut ds = Dataset::open("speclib/index.mmappet")?;
let start: Typed<u64> = ds.take("start")?;
assert!(ds.get::<u64>("start").is_err()); // gone from `ds`
drop(ds);
let bounds: &[u64] = &start;
```

Read-write, to fill a dataset in place (e.g. one made with Python's
`DatasetWriter.preallocate_dataset`). Its columns can only be borrowed:

```rust
let mut ds = DatasetMut::open("scores.mmappet")?;
ds.get_mut::<f32>("score")?.fill(0.0);
ds.flush()?;
```

Time columns keep their unit in the dtype:

```rust
use mmappet::{DType, Datetime64, TimeUnit};
let when: &[Datetime64] = ds.get("when")?; // any of s, ms, us, ns
assert_eq!(ds.column("when")?.dtype(), DType::Datetime64(TimeUnit::Millis));
```

With the `ndarray` feature, `ds.get_array::<T>(name)` gives an `ArrayView1<T>` and
`column.as_typed_array()` a `TypedArrayView` matched on the dtype.

### CLI Tool

```bash
# Build
cargo build --release --features cli

# Show dataset info
cargo run --features cli --bin mmappet-cli -- info path/to/dataset.mmappet

# Show first N rows
cargo run --features cli --bin mmappet-cli -- head path/to/dataset.mmappet -n 10

# Show first N rows of specific columns
cargo run --features cli --bin mmappet-cli -- head path/to/dataset.mmappet -n 5 --columns tof,mz

# Show statistics for numeric columns
cargo run --features cli --bin mmappet-cli -- stats path/to/dataset.mmappet
```

## Supported Data Types

| Schema String | Rust Type | Aliases |
|---------------|-----------|---------|
| `uint8` | `u8` | `u8` |
| `int8` | `i8` | `i8` |
| `uint16` | `u16` | `u16` |
| `int16` | `i16` | `i16` |
| `uint32` | `u32` | `u32` |
| `int32` | `i32` | `i32` |
| `uint64` | `u64` | `u64`, `size_t` |
| `int64` | `i64` | `i64` |
| `float16` | `mmappet::f16` | `f16`, `half` |
| `float32` | `f32` | `f32` |
| `float64` | `f64` | `f64`, `double` |
| `complex64` | `mmappet::Complex<f32>` | `c64` |
| `complex128` | `mmappet::Complex<f64>` | `c128` |
| `bool` | `u8` | `boolean` |
| `datetime64[s\|ms\|us\|ns]` | `mmappet::Datetime64` (i64 ticks, NaT = `i64::MIN`) | |
| `timedelta64[s\|ms\|us\|ns]` | `mmappet::Timedelta64` | |

Other numpy time units (`D`, `M`, `10ms`, ...) are refused: Python mmappet reads them
under a wrong unit (see `AI.md`).

## File Format

mmappet datasets are directories containing:

```
dataset.mmappet/
├── schema.txt     # Text file: "{dtype} {colname}" per line
├── 0.bin          # Binary column data (column 0)
├── 1.bin          # Binary column data (column 1)
└── ...
```

**schema.txt example:**
```
uint32 tof
uint32 intensity
float32 score
float32 mz
```

Binary files contain raw packed data in native byte order.

## Example: pmsms.mmappet

The repository includes a test dataset at `../pmsms.mmappet`:

```
Rows: 76,733,051
Columns: tof (uint32), intensity (uint32), score (float32), mz (float32)

First 5 rows:
tof       intensity   score      mz
202989    0           0.540689   677.962402
202990    610         0.680852   677.966492
202991    538         0.680852   677.970642
229175    0           0.701620   789.985779
229176    1042        0.564873   789.990234

Statistics:
tof: min=49262, max=393752, mean=192758.68
intensity: min=0, max=416199, mean=692.70
score: min=0.500000, max=0.999849, mean=0.595158
mz: min=192.982727, max=1690.030396, mean=654.013775
```

## Project Structure

```
src/
├── lib.rs          # Public API re-exports
├── error.rs        # MmappetError enum
├── dtype.rs        # DType enum, MmappetType trait
├── schema.rs       # Schema parsing
├── column.rs       # Column, TypedArrayView
├── dataset.rs      # Dataset (main entry point)
└── bin/
    └── mmappet_cli.rs  # CLI tool
```

## Dependencies

- `memmap2` - Memory-mapped file I/O
- `ndarray` - N-dimensional arrays
- `bytemuck` - Zero-copy type casting
- `thiserror` - Error derive macros
- `clap` - CLI argument parsing
- `anyhow` - CLI error handling

## Future Work

- [ ] Write support (`DatasetWriter` equivalent)
- [ ] Append to existing datasets
- [ ] Pre-allocation for zero-copy writes
- [ ] Lazy column loading (only mmap on first access)
- [ ] Iterator support for row-wise access
- [ ] Polars/Arrow integration

## Running Tests

```bash
cargo test   # oracle tests build .venv via `make venv` (needs uv, git)
```

## Building

```bash
cargo build --release --features cli
```
