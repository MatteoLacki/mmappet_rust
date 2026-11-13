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

#[test]
fn a_second_writer_is_refused_until_the_first_is_dropped() {
    use mmappet::{Lock, MmappetError};
    for lock in [Lock::Directory, Lock::File] {
        let dir = scratch("second_writer");
        two_rows(&dir);
        let first = DatasetWriter::new_locked(&dir, Mode::AppendOk, None, lock).unwrap();
        let second = DatasetWriter::new_locked(&dir, Mode::AppendOk, None, lock);
        assert!(matches!(second, Err(MmappetError::Locked(_))), "{lock:?}");
        assert!(matches!(
            mmappet::truncate_locked(&dir, 1, lock),
            Err(MmappetError::Locked(_))
        ));
        assert!(DatasetWriter::new_locked(&dir, Mode::AppendOk, None, Lock::None).is_ok());
        drop(first);
        assert!(DatasetWriter::new_locked(&dir, Mode::AppendOk, None, lock).is_ok());
    }
}

#[test]
fn locks_leave_a_file_only_when_asked_and_overwrite_keeps_it() {
    use mmappet::{Lock, LOCK_FILE};
    let dir = scratch("lock_file");
    two_rows(&dir);
    assert!(
        !dir.join(LOCK_FILE).exists(),
        "Lock::Directory leaves no file"
    );
    let schema = [("id", DType::UInt32)];
    let writer =
        DatasetWriter::new_locked(&dir, Mode::OverwriteDir, Some(&schema), Lock::File).unwrap();
    assert!(dir.join(LOCK_FILE).exists());
    drop(writer);
    let writer =
        DatasetWriter::new_locked(&dir, Mode::OverwriteDir, Some(&schema), Lock::File).unwrap();
    assert!(
        dir.join(LOCK_FILE).exists(),
        "OverwriteDir keeps the lock file"
    );
    assert!(!dir.join("1.bin").exists());
    drop(writer);
}

#[test]
fn sync_leaves_every_appended_row_in_the_files() {
    let dir = scratch("sync");
    let mut writer = DatasetWriter::new(&dir, Mode::New, None).unwrap();
    let ids: Vec<u32> = vec![1, 2];
    writer.append(&[("id", (&ids).into())]).unwrap();
    writer.sync().unwrap();
    assert_eq!(std::fs::read(dir.join("0.bin")).unwrap().len(), 8);
}
