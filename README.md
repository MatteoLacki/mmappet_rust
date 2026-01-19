# mmappet (Rust)

Rust library for reading mmappet datasets - memory-mapped columnar data format.

This is the Rust equivalent of the Python [mmappet](../mmappet/) library.

## Status

**Reading: Complete** - Full support for reading mmappet datasets with all dtypes.

**Writing: Not implemented** - Future work.

## Installation

```toml
[dependencies]
mmappet = { path = "../mmappet_rust" }
```

## Usage

### Library API

```rust
use mmappet::Dataset;

// Open a dataset
let ds = Dataset::open("data.mmappet")?;

// Check schema
println!("Rows: {}", ds.len());
println!("Columns: {:?}", ds.schema().column_names());

// Typed access (compile-time checked)
let tof: &[u32] = ds.get("tof")?;
let mz: &[f32] = ds.get("mz")?;

// ArrayView1 for ndarray operations
use mmappet::ArrayView1;
let scores: ArrayView1<f32> = ds.get_array("score")?;
println!("Mean score: {}", scores.mean().unwrap());

// Dictionary-style access (runtime type)
let col = &ds["intensity"];
println!("dtype: {}, len: {}", col.dtype(), col.len());

// Dynamic typed access
use mmappet::TypedArrayView;
match ds["mz"].as_typed_array() {
    TypedArrayView::Float32(arr) => println!("First mz: {}", arr[0]),
    _ => {}
}
```

### CLI Tool

```bash
# Build
cargo build --release

# Show dataset info
cargo run --bin mmappet-cli -- info path/to/dataset.mmappet

# Show first N rows
cargo run --bin mmappet-cli -- head path/to/dataset.mmappet -n 10

# Show first N rows of specific columns
cargo run --bin mmappet-cli -- head path/to/dataset.mmappet -n 5 --columns tof,mz

# Show statistics for numeric columns
cargo run --bin mmappet-cli -- stats path/to/dataset.mmappet
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
| `float32` | `f32` | `f32` |
| `float64` | `f64` | `f64`, `double` |
| `bool` | `u8` | `boolean` |

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
cargo test
```

## Building

```bash
cargo build --release
```
