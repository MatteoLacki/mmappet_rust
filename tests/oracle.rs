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

fn oracle(args: &[&str]) {
    let status = Command::new(python())
        .arg(crate_root().join("tests/python/oracle.py"))
        .args(args)
        .status()
        .expect("could not run oracle.py");
    assert!(status.success(), "oracle.py {args:?} failed");
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
