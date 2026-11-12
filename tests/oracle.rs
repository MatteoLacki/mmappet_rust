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
