# mmappet_rust Development Log

## Session: 2026-01-19

### Phase 1: Core Types (COMPLETED)

**Created `src/error.rs`:**
- `MmappetError` enum with variants: Io, SchemaParse, UnknownDType, ColumnNotFound, TypeMismatch, LengthMismatch, MissingSchema, MissingColumnFile, InvalidFileSize, DuplicateColumnName
- `Result<T>` type alias

**Created `src/dtype.rs`:**
- `DType` enum: UInt8, Int8, UInt16, Int16, UInt32, Int32, UInt64, Int64, Float32, Float64, Bool
- `DType::from_str()` - parses schema strings like "uint32", "float64", "size_t"
- `DType::size_bytes()` - returns byte size
- `DType::as_str()` - canonical string representation
- `MmappetType` trait implemented for u8, i8, u16, i16, u32, i32, u64, i64, f32, f64
- Unit tests for parsing and size

### Phase 2: Schema Parsing (IN PROGRESS)

**Creating `src/schema.rs`:**
- `ColumnDef` struct: index, name, dtype
- `Schema` struct: columns vec, name-to-index map
- Parse format: `{dtype} {colname}` per line

### Phase 3: Column Implementation (COMPLETED)

**Created `src/column.rs`:**
- `Column` struct holding Mmap, dtype, and length
- `Column::open()` - opens binary file and validates size
- `as_slice<T>()` - returns typed slice if T matches dtype
- `as_array<T>()` - returns ArrayView1 if T matches dtype
- `as_typed_array()` - returns TypedArrayView enum
- `TypedArrayView` enum with variants for all dtypes

### Phase 4: Dataset Implementation (COMPLETED)

**Created `src/dataset.rs`:**
- `Dataset` struct as main entry point
- `Dataset::open()` - loads schema and all columns
- Validates all columns have equal length
- `get<T>()` and `get_array<T>()` for typed access
- `Index<&str>` trait for dictionary-style access: `ds["colname"]`

### Phase 5: Library Setup (COMPLETED)

**Created `src/lib.rs`:**
- Module structure and public re-exports
- Doc comments with usage example

**Updated `Cargo.toml`:**
- Renamed package to "mmappet"
- Added `[lib]` section
- Added `[[bin]]` section for CLI
- Added `thiserror = "1.0"` dependency

### Phase 6: CLI Binary (COMPLETED)

**Created `src/bin/mmappet_cli.rs`:**
- `info` subcommand: shows schema and row count
- `head` subcommand: prints first N rows with optional column selection
- `stats` subcommand: shows min/max/mean for numeric columns

### Verification (COMPLETED)

**Build:** `cargo build` - success
**Tests:** `cargo test` - 7 unit tests + 1 doc test pass
**CLI test against pmsms.mmappet:**
- `info` shows 76,733,051 rows, 4 columns (tof, intensity, score, mz)
- `head` output matches Python mmappet output exactly
- `stats` computes statistics across 77M rows in seconds

### Key Decisions

1. Using `thiserror` for error derive macros
2. Supporting case-insensitive dtype parsing
3. Supporting "size_t" as alias for uint64 (seen in pmsms.mmappet/dataindex.mmappet)
4. Bool stored as single byte (same as Python mmappet)
5. Using enum-based TypedArrayView rather than trait objects for dynamic access
6. Both slice and ndarray access patterns supported
