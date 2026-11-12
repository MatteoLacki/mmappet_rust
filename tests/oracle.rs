//! Python mmappet is the oracle: these tests run `tests/python/oracle.py` in the `.venv`
//! that `make venv` builds (needs `uv` and `git` on PATH).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

fn crate_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// The venv's Python, after `make venv` (once per test binary; a no-op once built).
fn python() -> &'static Path {
    static PYTHON: OnceLock<PathBuf> = OnceLock::new();
    PYTHON.get_or_init(|| {
        let status = Command::new("make")
            .arg("-C")
            .arg(crate_root())
            .arg("venv")
            .status()
            .expect("could not run `make`");
        assert!(
            status.success(),
            "`make venv` failed; it needs `uv` and `git` on PATH"
        );
        crate_root().join(".venv/bin/python")
    })
}

/// Runs `oracle.py args...` and returns its stdout.
fn oracle(args: &[&str]) -> String {
    let output = Command::new(python())
        .arg(crate_root().join("tests/python/oracle.py"))
        .args(args)
        .output()
        .expect("could not run oracle.py");
    assert!(
        output.status.success(),
        "oracle.py {args:?} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("oracle")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn reads_what_python_writes() {
    let dir = scratch("smoke");
    oracle(&["smoke", dir.to_str().unwrap()]);
    let ds = mmappet::Dataset::open(&dir).unwrap();
    assert_eq!(ds.get::<u32>("id").unwrap(), &[1, 2, 3]);
    assert_eq!(ds.get::<f64>("score").unwrap(), &[0.5, 1.5, 2.5]);
}

#[test]
fn dtype_names_and_sizes_match_numpy() {
    let listing = oracle(&["dtypes"]);
    assert!(listing.lines().count() >= 22);
    for line in listing.lines() {
        let (name, itemsize) = line.split_once('\t').unwrap();
        let dtype = mmappet::DType::from_str(name).unwrap();
        assert_eq!(dtype.to_string(), name);
        assert_eq!(
            dtype.size_bytes(),
            itemsize.parse::<usize>().unwrap(),
            "{name}"
        );
    }
}

#[test]
fn reads_the_new_types_python_writes() {
    use mmappet::{f16, Complex, DType, Datetime64, TimeUnit, Timedelta64};
    let dir = scratch("new_types");
    oracle(&["new-types", dir.to_str().unwrap()]);
    let ds = mmappet::Dataset::open(&dir).unwrap();

    let half: &[f16] = ds.get("half").unwrap();
    assert_eq!(
        half,
        &[f16::from_f32(0.5), f16::from_f32(-2.0), f16::INFINITY]
    );
    let expected = [
        Complex::new(1.0, 2.0),
        Complex::new(0.0, -0.5),
        Complex::new(3.0, 0.0),
    ];
    assert_eq!(ds.get::<Complex<f32>>("c64").unwrap(), &expected);
    assert_eq!(
        ds.get::<Complex<f64>>("c128").unwrap(),
        &expected.map(|c| Complex::new(c.re as f64, c.im as f64))
    );
    assert_eq!(ds.get::<u8>("flag").unwrap(), &[1, 0, 1]);

    assert_eq!(
        ds.column("when").unwrap().dtype(),
        DType::Datetime64(TimeUnit::Millis)
    );
    let when: &[Datetime64] = ds.get("when").unwrap();
    assert_eq!(when, &[Datetime64(0), Datetime64(1500), Datetime64::NAT]);
    assert_eq!(
        ds.column("took").unwrap().dtype(),
        DType::Timedelta64(TimeUnit::Nanos)
    );
    let took: &[Timedelta64] = ds.get("took").unwrap();
    assert_eq!(took, &[Timedelta64(1), Timedelta64(-2), Timedelta64::NAT]);

    let error = ds.get::<i64>("when").unwrap_err().to_string();
    assert!(
        error.contains("when") && error.contains("datetime64[ms]"),
        "{error}"
    );
}

#[test]
fn reads_names_with_spaces() {
    let dir = scratch("spaced_names");
    oracle(&["spaced-names", dir.to_str().unwrap()]);
    let ds = mmappet::Dataset::open(&dir).unwrap();
    assert_eq!(ds.get::<f64>("retention time").unwrap(), &[1.5, 2.5]);
    assert_eq!(ds.get::<i8>("charge state").unwrap(), &[2, 3]);
}

#[test]
fn reads_a_dataset_without_rows() {
    let dir = scratch("no_rows");
    oracle(&["no-rows", dir.to_str().unwrap()]);
    let ds = mmappet::Dataset::open(&dir).unwrap();
    assert_eq!(ds.len(), 0);
    assert_eq!(ds.get::<u32>("id").unwrap(), &[] as &[u32]);
    let score = ds.typed::<f64>("score").unwrap();
    assert!(score.is_empty());
    let mut ds = mmappet::DatasetMut::open(&dir).unwrap();
    assert!(ds.get_mut::<u32>("id").unwrap().is_empty());
}

#[test]
fn typed_columns_outlive_their_dataset() {
    let dir = scratch("outlive");
    oracle(&["smoke", dir.to_str().unwrap()]);
    let (ids, score_bytes) = {
        let ds = mmappet::Dataset::open(&dir).unwrap();
        (
            ds.typed::<u32>("id").unwrap(),
            ds.column("score").unwrap().clone(),
        )
    };
    assert_eq!(&*ids, &[1, 2, 3]);
    assert_eq!(
        score_bytes.as_ref(),
        bytemuck::cast_slice::<f64, u8>(&[0.5, 1.5, 2.5])
    );
}

#[test]
fn optional_columns() {
    let dir = scratch("optional");
    oracle(&["smoke", dir.to_str().unwrap()]);
    let ds = mmappet::Dataset::open(&dir).unwrap();
    assert_eq!(ds.get_opt::<u32>("id").unwrap(), Some(&[1, 2, 3][..]));
    assert_eq!(ds.get_opt::<u32>("absent").unwrap(), None);
    assert!(ds.get_opt::<f32>("id").is_err());
}

#[test]
fn python_reads_what_rust_fills_in_place() {
    use mmappet::Datetime64;
    let dir = scratch("filled");
    let nrows = 1000;
    oracle(&["preallocate", dir.to_str().unwrap(), &nrows.to_string()]);
    let mut ds = mmappet::DatasetMut::open(&dir).unwrap();
    assert_eq!(ds.len(), nrows);
    for (i, id) in ds.get_mut::<u32>("id").unwrap().iter_mut().enumerate() {
        *id = 3 * i as u32;
    }
    for (i, score) in ds.get_mut::<f64>("score").unwrap().iter_mut().enumerate() {
        *score = i as f64 / 2.0;
    }
    for (i, when) in ds
        .get_mut::<Datetime64>("when")
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        *when = Datetime64(1000 * i as i64);
    }
    ds.flush().unwrap();
    drop(ds);
    oracle(&["check-filled", dir.to_str().unwrap()]);
}

#[test]
fn taken_columns_leave_the_dataset_and_outlive_it() {
    let dir = scratch("take");
    oracle(&["smoke", dir.to_str().unwrap()]);
    let ids = {
        let mut ds = mmappet::Dataset::open(&dir).unwrap();
        assert!(ds.take::<f32>("id").is_err());
        assert!(
            ds.get::<u32>("id").is_ok(),
            "a wrong dtype leaves the column"
        );
        let ids = ds.take::<u32>("id").unwrap();
        assert!(ds.get::<u32>("id").is_err());
        assert!(ds.take_column("id").is_err());
        assert_eq!(ds.get::<f64>("score").unwrap(), &[0.5, 1.5, 2.5]);
        ids
    };
    assert_eq!(&*ids, &[1, 2, 3]);
}

// ---- Writer ----

use mmappet::{DType, DatasetWriter, Mode, Value, Values};

/// The dtypes the oracle supports, in its order (`oracle.py dtypes`).
fn oracle_dtypes() -> Vec<DType> {
    let listing = oracle(&["dtypes"]);
    let mut dtypes = Vec::new();
    for line in listing.lines() {
        let (name, _) = line.split_once('\t').unwrap();
        dtypes.push(DType::from_str(name).unwrap());
    }
    dtypes
}

/// Row `i` of reference column `dtype`, as `oracle.py`'s `reference_frame` builds it.
fn reference_value(dtype: DType, i: i64) -> Value {
    use mmappet::{f16, Complex, Datetime64, Timedelta64};
    let float = i as f64 / 4.0 - 1.0;
    let imag = i as f64 / 2.0;
    match dtype {
        DType::UInt8 => Value::UInt8((7 * i) as u8),
        DType::UInt16 => Value::UInt16((7 * i) as u16),
        DType::UInt32 => Value::UInt32((7 * i) as u32),
        DType::UInt64 => Value::UInt64((7 * i) as u64),
        DType::Int8 => Value::Int8((7 * i - 10) as i8),
        DType::Int16 => Value::Int16((7 * i - 10) as i16),
        DType::Int32 => Value::Int32((7 * i - 10) as i32),
        DType::Int64 => Value::Int64(7 * i - 10),
        DType::Float16 => Value::Float16(f16::from_f64(float)),
        DType::Float32 => Value::Float32(float as f32),
        DType::Float64 => Value::Float64(float),
        DType::Complex64 => Value::Complex64(Complex::new(float as f32, imag as f32)),
        DType::Complex128 => Value::Complex128(Complex::new(float, imag)),
        DType::Bool => Value::Bool(i % 2 == 1),
        DType::Datetime64(unit) => Value::Datetime64(unit, Datetime64(1000 * i - 2000)),
        DType::Timedelta64(unit) => Value::Timedelta64(unit, Timedelta64(1000 * i - 2000)),
    }
}

fn reference_name(dtype: DType) -> String {
    format!("col {dtype}")
}

/// Appends reference rows `rows` in one `append`, each column as raw bytes.
fn append_reference(writer: &mut DatasetWriter, dtypes: &[DType], rows: std::ops::Range<i64>) {
    let names: Vec<String> = dtypes.iter().map(|&d| reference_name(d)).collect();
    let mut bytes: Vec<Vec<u8>> = vec![Vec::new(); dtypes.len()];
    for i in rows {
        for (column, &dtype) in bytes.iter_mut().zip(dtypes) {
            column.extend_from_slice(reference_value(dtype, i).as_values().bytes());
        }
    }
    let mut columns = Vec::new();
    for ((name, column), &dtype) in names.iter().zip(&bytes).zip(dtypes) {
        columns.push((name.as_str(), Values::Raw(dtype, column)));
    }
    writer.append(&columns).unwrap();
}

/// Appends reference row `i` with `append_row`, each value typed.
fn append_reference_row(writer: &mut DatasetWriter, dtypes: &[DType], i: i64) {
    let names: Vec<String> = dtypes.iter().map(|&d| reference_name(d)).collect();
    let mut row = Vec::new();
    for (name, &dtype) in names.iter().zip(dtypes) {
        row.push((name.as_str(), reference_value(dtype, i)));
    }
    writer.append_row(&row).unwrap();
}

/// Every file under `dir`, by relative path, with its bytes.
fn tree(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        files.push((name, std::fs::read(&path).unwrap()));
    }
    files.sort();
    files
}

fn assert_same_tree(rust: &Path, python: &Path) {
    let (rust_tree, python_tree) = (tree(rust), tree(python));
    let names = |t: &[(String, Vec<u8>)]| t.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>();
    assert_eq!(names(&rust_tree), names(&python_tree));
    for ((name, rust_bytes), (_, python_bytes)) in rust_tree.iter().zip(&python_tree) {
        assert!(
            rust_bytes == python_bytes,
            "{name} differs:\n rust   {:?}\n python {:?}",
            String::from_utf8_lossy(rust_bytes),
            String::from_utf8_lossy(python_bytes)
        );
    }
}

fn python_reference(name: &str, chunks: &[(i64, i64)]) -> PathBuf {
    let dir = scratch(name);
    for (start, stop) in chunks {
        let (start, stop) = (start.to_string(), stop.to_string());
        oracle(&["reference", dir.to_str().unwrap(), &start, &stop]);
    }
    dir
}

#[test]
fn rust_writes_the_bytes_python_writes() {
    let dtypes = oracle_dtypes();
    let python = python_reference("reference_python", &[(0, 3), (3, 5)]);
    let rust = scratch("reference_rust");
    let mut writer = DatasetWriter::new(&rust, Mode::New, None).unwrap();
    append_reference(&mut writer, &dtypes, 0..3);
    append_reference_row(&mut writer, &dtypes, 3);
    append_reference(&mut writer, &dtypes, 4..5);
    assert_eq!(writer.len(), 5);
    writer.close().unwrap();
    assert_same_tree(&rust, &python);
}

#[test]
fn rust_and_python_append_to_each_other() {
    let dtypes = oracle_dtypes();
    let whole = python_reference("append_whole", &[(0, 5)]);

    let python_first = python_reference("append_python_first", &[(0, 3)]);
    let mut writer = DatasetWriter::new(&python_first, Mode::AppendOk, None).unwrap();
    assert_eq!(writer.len(), 3);
    append_reference(&mut writer, &dtypes, 3..5);
    writer.close().unwrap();
    assert_same_tree(&python_first, &whole);

    let rust_first = scratch("append_rust_first");
    let mut writer = DatasetWriter::new(&rust_first, Mode::New, None).unwrap();
    append_reference(&mut writer, &dtypes, 0..3);
    writer.close().unwrap();
    oracle(&["reference", rust_first.to_str().unwrap(), "3", "5"]);
    assert_same_tree(&rust_first, &whole);
}

#[test]
fn python_reads_a_column_rust_appends() {
    let dtypes = oracle_dtypes();
    let dir = scratch("append_column");
    let mut writer = DatasetWriter::new(&dir, Mode::New, None).unwrap();
    append_reference(&mut writer, &dtypes, 0..5);
    let extra: Vec<u16> = (0..5).map(|i| 3 * i).collect();
    writer.append_column("extra", (&extra).into()).unwrap();
    writer.close().unwrap();
    oracle(&["check-extra", dir.to_str().unwrap(), "5"]);
}

#[test]
fn rust_preallocates_what_python_preallocates() {
    use mmappet::{Datetime64, TimeUnit};
    let nrows = 1000;
    let python = scratch("preallocate_python");
    oracle(&["preallocate", python.to_str().unwrap(), &nrows.to_string()]);
    let rust = scratch("preallocate_rust");
    let schema = [
        ("id", DType::UInt32),
        ("score", DType::Float64),
        ("when", DType::Datetime64(TimeUnit::Micros)),
    ];
    DatasetWriter::preallocate(&rust, &schema, nrows, false).unwrap();
    assert_same_tree(&rust, &python);

    let mut ds = mmappet::DatasetMut::open(&rust).unwrap();
    for (i, id) in ds.get_mut::<u32>("id").unwrap().iter_mut().enumerate() {
        *id = 3 * i as u32;
    }
    for (i, score) in ds.get_mut::<f64>("score").unwrap().iter_mut().enumerate() {
        *score = i as f64 / 2.0;
    }
    for (i, when) in ds
        .get_mut::<Datetime64>("when")
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        *when = Datetime64(1000 * i as i64);
    }
    drop(ds);
    oracle(&["check-filled", rust.to_str().unwrap()]);
}

// ---- Truncate and locks ----

#[test]
fn truncate_matches_python_crop_dataset() {
    let dtypes = oracle_dtypes();
    let python = scratch("crop_python");
    let whole = python_reference("crop_source", &[(0, 5)]);
    oracle(&[
        "crop",
        whole.to_str().unwrap(),
        python.to_str().unwrap(),
        "3",
    ]);

    let rust = scratch("truncate_rust");
    let mut writer = DatasetWriter::new(&rust, Mode::New, None).unwrap();
    append_reference(&mut writer, &dtypes, 0..5);
    writer.close().unwrap();
    assert!(mmappet::truncate(&rust, 6).is_err());
    mmappet::truncate(&rust, 3).unwrap();
    assert_same_tree(&rust, &python);
    mmappet::truncate(&rust, 0).unwrap();
    assert_eq!(mmappet::Dataset::open(&rust).unwrap().len(), 0);
}

/// A Python process holding `kind`'s flock on `dir` until killed.
struct PythonLock(std::process::Child);

impl PythonLock {
    fn hold(dir: &Path, kind: &str) -> Self {
        use std::io::BufRead;
        let mut child = Command::new(python())
            .arg(crate_root().join("tests/python/oracle.py"))
            .args(["hold-lock", dir.to_str().unwrap(), kind])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut line = String::new();
        let stdout = child.stdout.take().unwrap();
        std::io::BufReader::new(stdout)
            .read_line(&mut line)
            .unwrap();
        assert_eq!(line.trim(), "locked");
        PythonLock(child)
    }

    /// Kills the holder, as a crash would, and waits for it.
    fn kill(mut self) {
        self.0.kill().unwrap();
        self.0.wait().unwrap();
    }
}

#[test]
fn a_python_flock_stops_rust_writers_until_its_holder_dies() {
    use mmappet::{Lock, MmappetError};
    for (kind, lock) in [("directory", Lock::Directory), ("file", Lock::File)] {
        let dir = scratch(&format!("python_lock_{kind}"));
        oracle(&["smoke", dir.to_str().unwrap()]);
        let holder = PythonLock::hold(&dir, kind);
        let writer = DatasetWriter::new_locked(&dir, Mode::AppendOk, None, lock);
        assert!(matches!(writer, Err(MmappetError::Locked(_))), "{kind}");
        let cut = mmappet::truncate_locked(&dir, 1, lock);
        assert!(matches!(cut, Err(MmappetError::Locked(_))), "{kind}");
        holder.kill();
        let writer = DatasetWriter::new_locked(&dir, Mode::AppendOk, None, lock).unwrap();
        assert_eq!(writer.len(), 3);
    }
}

// ---- Random round trips (`oracle.py` draws the same shapes) ----

const SEEDS: usize = 20;
const NAMES: [&str; 10] = [
    "a",
    "mz",
    "rt",
    "retention time",
    "charge state",
    "x y z",
    "id",
    "score",
    "intensity",
    "col 7",
];

/// splitmix64: a few lines, no `rand`.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn chance(&mut self, p: f64) -> bool {
        ((self.next() >> 11) as f64 / (1u64 << 53) as f64) < p
    }
}

/// Runs `$body` with `$t` the Rust type `$dtype` is read as.
macro_rules! with_type {
    ($dtype:expr, $t:ident => $body:expr) => {
        match $dtype {
            DType::UInt8 | DType::Bool => {
                type $t = u8;
                $body
            }
            DType::Int8 => {
                type $t = i8;
                $body
            }
            DType::UInt16 => {
                type $t = u16;
                $body
            }
            DType::Int16 => {
                type $t = i16;
                $body
            }
            DType::UInt32 => {
                type $t = u32;
                $body
            }
            DType::Int32 => {
                type $t = i32;
                $body
            }
            DType::UInt64 => {
                type $t = u64;
                $body
            }
            DType::Int64 => {
                type $t = i64;
                $body
            }
            DType::Float16 => {
                type $t = mmappet::f16;
                $body
            }
            DType::Float32 => {
                type $t = f32;
                $body
            }
            DType::Float64 => {
                type $t = f64;
                $body
            }
            DType::Complex64 => {
                type $t = mmappet::Complex<f32>;
                $body
            }
            DType::Complex128 => {
                type $t = mmappet::Complex<f64>;
                $body
            }
            DType::Datetime64(_) => {
                type $t = mmappet::Datetime64;
                $body
            }
            DType::Timedelta64(_) => {
                type $t = mmappet::Timedelta64;
                $body
            }
        }
    };
}

/// Random bits for numbers, 0/1 for bool, and ticks within pandas' range with some NaT for
/// time, as `oracle.py`'s `random_values`.
fn random_bytes(rng: &mut SplitMix, dtype: DType, nrows: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(nrows * dtype.size_bytes());
    for _ in 0..nrows {
        match dtype {
            DType::Bool => bytes.push(rng.below(2) as u8),
            DType::Datetime64(_) | DType::Timedelta64(_) => {
                let ticks = if rng.chance(0.1) {
                    i64::MIN
                } else {
                    rng.below(1 << 41) as i64 - (1 << 40)
                };
                bytes.extend_from_slice(&ticks.to_ne_bytes());
            }
            _ => {
                for _ in 0..dtype.size_bytes() {
                    bytes.push(rng.next() as u8);
                }
            }
        }
    }
    bytes
}

type Schema = Vec<(String, DType)>;

fn random_schema(rng: &mut SplitMix, dtypes: &[DType]) -> (Schema, usize) {
    let mut names: Vec<&str> = NAMES.to_vec();
    let ncols = 1 + rng.below(8);
    let mut schema = Vec::new();
    for _ in 0..ncols {
        let name = names.remove(rng.below(names.len()));
        schema.push((name.to_string(), dtypes[rng.below(dtypes.len())]));
    }
    let nrows = match rng.below(3) {
        0 => 0,
        1 => 1,
        _ => 2 + rng.below(999),
    };
    (schema, nrows)
}

fn random_chunks(rng: &mut SplitMix, nrows: usize) -> Vec<std::ops::Range<usize>> {
    let mut bounds = vec![0, nrows];
    if nrows > 1 {
        for _ in 0..rng.below(5) {
            bounds.push(1 + rng.below(nrows - 1));
        }
    }
    bounds.sort();
    bounds.dedup();
    bounds.windows(2).map(|w| w[0]..w[1]).collect()
}

/// Writes `columns` (stored bytes, in `schema` order) in random chunks, sometimes closing
/// and reopening with `AppendOk` between them.
fn write_in_chunks(rng: &mut SplitMix, path: &Path, schema: &Schema, columns: &[Vec<u8>]) {
    let schema_refs: Vec<(&str, DType)> = schema.iter().map(|(n, d)| (n.as_str(), *d)).collect();
    let nrows = columns[0].len() / schema[0].1.size_bytes();
    let mut writer = DatasetWriter::new(path, Mode::OverwriteDir, Some(&schema_refs)).unwrap();
    for rows in random_chunks(rng, nrows) {
        if rng.chance(0.3) {
            writer.close().unwrap();
            writer = DatasetWriter::new(path, Mode::AppendOk, Some(&schema_refs)).unwrap();
        }
        let mut chunk = Vec::new();
        for ((name, dtype), bytes) in schema.iter().zip(columns) {
            let size = dtype.size_bytes();
            chunk.push((
                name.as_str(),
                Values::Raw(*dtype, &bytes[rows.start * size..rows.end * size]),
            ));
        }
        writer.append(&chunk).unwrap();
    }
    writer.close().unwrap();
}

#[test]
fn rust_rewrites_random_python_datasets_byte_for_byte() {
    let python = scratch("random_python");
    oracle(&["gen", python.to_str().unwrap(), &SEEDS.to_string()]);
    let rust = scratch("random_python_rewritten");
    for seed in 0..SEEDS {
        eprintln!("seed {seed}");
        let source = python.join(seed.to_string());
        let ds = mmappet::Dataset::open(&source).unwrap();
        let schema: Schema = ds
            .schema()
            .columns()
            .map(|d| (d.name.clone(), d.dtype))
            .collect();
        let mut columns = Vec::new();
        for (name, dtype) in &schema {
            let raw = ds.column(name).unwrap().as_bytes().to_vec();
            let typed: Vec<u8> =
                with_type!(*dtype, T => bytemuck::cast_slice(ds.get::<T>(name).unwrap()).to_vec());
            assert_eq!(typed, raw, "{name}");
            columns.push(raw);
        }
        let target = rust.join(seed.to_string());
        write_in_chunks(&mut SplitMix(seed as u64), &target, &schema, &columns);
        assert_same_tree(&target, &source);

        let keep: usize = std::fs::read_to_string(python.join(format!("{seed}.crop.rows")))
            .unwrap()
            .parse()
            .unwrap();
        mmappet::truncate(&target, keep).unwrap();
        assert_same_tree(&target, &python.join(format!("{seed}.crop")));
    }
}

#[test]
fn python_rewrites_random_rust_datasets_byte_for_byte() {
    let dtypes = oracle_dtypes();
    let rust = scratch("random_rust");
    for seed in 0..SEEDS {
        let mut rng = SplitMix(100 + seed as u64);
        let (schema, nrows) = random_schema(&mut rng, &dtypes);
        let columns: Vec<Vec<u8>> = schema
            .iter()
            .map(|(_, d)| random_bytes(&mut rng, *d, nrows))
            .collect();
        write_in_chunks(&mut rng, &rust.join(seed.to_string()), &schema, &columns);
    }
    let python = scratch("random_rust_rewritten");
    oracle(&[
        "copy",
        rust.to_str().unwrap(),
        python.to_str().unwrap(),
        &SEEDS.to_string(),
    ]);
    for seed in 0..SEEDS {
        eprintln!("seed {seed}");
        assert_same_tree(&python.join(seed.to_string()), &rust.join(seed.to_string()));
    }
}

#[test]
fn rust_fills_random_python_preallocations_as_its_writer_writes() {
    let root = scratch("random_python_preallocated");
    oracle(&[
        "preallocate-random",
        root.to_str().unwrap(),
        &SEEDS.to_string(),
    ]);
    for seed in 0..SEEDS {
        eprintln!("seed {seed}");
        let filled = root.join(seed.to_string());
        let mut rng = SplitMix(200 + seed as u64);
        let mut ds = mmappet::DatasetMut::open(&filled).unwrap();
        let schema: Schema = ds
            .schema()
            .columns()
            .map(|d| (d.name.clone(), d.dtype))
            .collect();
        let mut columns = Vec::new();
        for (name, dtype) in &schema {
            let bytes = random_bytes(&mut rng, *dtype, ds.len());
            with_type!(*dtype, T => bytemuck::cast_slice_mut::<T, u8>(ds.get_mut::<T>(name).unwrap()).copy_from_slice(&bytes));
            columns.push(bytes);
        }
        drop(ds);
        let written = root.join(format!("{seed}.written"));
        write_in_chunks(&mut rng, &written, &schema, &columns);
        assert_same_tree(&filled, &written);
    }
}

#[test]
fn python_fills_random_rust_preallocations_as_its_writer_writes() {
    let dtypes = oracle_dtypes();
    let root = scratch("random_rust_preallocated");
    std::fs::create_dir_all(&root).unwrap();
    for seed in 0..SEEDS {
        let mut rng = SplitMix(300 + seed as u64);
        let (schema, nrows) = random_schema(&mut rng, &dtypes);
        let schema_refs: Vec<(&str, DType)> =
            schema.iter().map(|(n, d)| (n.as_str(), *d)).collect();
        DatasetWriter::preallocate(root.join(seed.to_string()), &schema_refs, nrows, false)
            .unwrap();
    }
    oracle(&["fill", root.to_str().unwrap(), &SEEDS.to_string()]);
    for seed in 0..SEEDS {
        eprintln!("seed {seed}");
        assert_same_tree(
            &root.join(seed.to_string()),
            &root.join(format!("{seed}.written")),
        );
    }
}
