//! Reader behaviour Python has no counterpart for.

use mmappet::{Dataset, DatasetWriter, Mode};
use std::path::Path;

#[test]
fn column_path_names_the_file_the_column_maps() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("reader_column_path");
    let _ = std::fs::remove_dir_all(&dir);
    let mut writer = DatasetWriter::new(&dir, Mode::New, None).unwrap();
    let ids: Vec<u32> = vec![1, 2];
    let scores: Vec<f32> = vec![0.5, 1.5];
    writer
        .append(&[("id", (&ids).into()), ("a score", (&scores).into())])
        .unwrap();
    writer.close().unwrap();

    let dataset = Dataset::open(&dir).unwrap();
    let path = dataset.column_path("a score").unwrap();
    assert_eq!(path, dir.join("1.bin"));
    assert_eq!(
        std::fs::read(path).unwrap(),
        dataset.column("a score").unwrap().as_bytes()
    );
    assert!(dataset.column_path("missing").is_err());
}
