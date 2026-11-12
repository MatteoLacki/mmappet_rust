//! What the writer refuses, where Python's would carry on (see `AI.md`).

use mmappet::{DType, DatasetWriter, Mode, Values};
use std::path::{Path, PathBuf};

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("writer")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn two_rows(dir: &Path) {
    let mut writer = DatasetWriter::new(dir, Mode::New, None).unwrap();
    let ids: Vec<u32> = vec![1, 2];
    let scores: Vec<f32> = vec![0.5, 1.5];
    writer
        .append(&[("id", (&ids).into()), ("score", (&scores).into())])
        .unwrap();
    writer.close().unwrap();
}

#[test]
fn new_refuses_an_existing_directory() {
    let dir = scratch("exists");
    two_rows(&dir);
    assert!(DatasetWriter::new(&dir, Mode::New, None).is_err());
    let writer = DatasetWriter::new(&dir, Mode::OverwriteDir, None).unwrap();
    assert_eq!(writer.len(), 0);
    assert!(!dir.join("0.bin").exists());
}

#[test]
fn append_ok_refuses_another_schema() {
    let dir = scratch("schema");
    two_rows(&dir);
    let other = [("id", DType::UInt64), ("score", DType::Float32)];
    assert!(DatasetWriter::new(&dir, Mode::AppendOk, Some(&other)).is_err());
    let same = [("id", DType::UInt32), ("score", DType::Float32)];
    assert_eq!(
        DatasetWriter::new(&dir, Mode::AppendOk, Some(&same))
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn append_refuses_bad_columns() {
    let dir = scratch("bad_columns");
    let schema = [("id", DType::UInt32), ("score", DType::Float32)];
    let mut writer = DatasetWriter::new(&dir, Mode::New, Some(&schema)).unwrap();
    let ids: Vec<u32> = vec![1, 2];
    let short: Vec<f32> = vec![0.5];
    let wrong: Vec<f64> = vec![0.5, 1.5];
    let scores: Vec<f32> = vec![0.5, 1.5];
    assert!(writer
        .append(&[("id", (&ids).into()), ("score", (&short).into())])
        .is_err());
    assert!(writer
        .append(&[("id", (&ids).into()), ("score", (&wrong).into())])
        .is_err());
    assert!(writer.append(&[("id", (&ids).into())]).is_err());
    assert!(writer
        .append(&[("id", (&ids).into()), ("id", (&ids).into())])
        .is_err());
    let odd = [0u8; 3];
    assert!(writer
        .append(&[
            ("id", Values::Raw(DType::UInt32, &odd[..])),
            ("score", (&scores).into())
        ])
        .is_err());
    assert_eq!(writer.len(), 0, "a refused append writes nothing");
    writer
        .append(&[("score", (&scores).into()), ("id", (&ids).into())])
        .unwrap();
    assert!(writer.append_column("id", (&ids).into()).is_err());
    assert!(writer.append_column("short", (&short).into()).is_err());
    writer.close().unwrap();
    let ds = mmappet::Dataset::open(&dir).unwrap();
    assert_eq!(ds.get::<u32>("id").unwrap(), &[1, 2]);
}

#[test]
fn schemas_refuse_names_schema_txt_cannot_hold() {
    for name in ["", " lead", "two\nlines"] {
        let dir = scratch("names");
        assert!(
            DatasetWriter::new(&dir, Mode::New, Some(&[(name, DType::UInt8)])).is_err(),
            "{name:?}"
        );
    }
    assert!(DatasetWriter::new(scratch("empty"), Mode::New, Some(&[])).is_err());
}
