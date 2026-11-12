//! Advisory locks that keep two writers off one dataset: `flock(2)` through std's
//! `File::try_lock`, the call Python's `fcntl.flock` makes, so the two see each other's
//! locks. The kernel drops a lock when its holder exits, however it exits.

use std::fs::{File, OpenOptions, TryLockError};
use std::path::Path;

use crate::error::{MmappetError, Result};

/// The lock file of [`Lock::File`], as mmappeteer's caches name theirs.
pub const LOCK_FILE: &str = "write.lock";

/// What a writer locks while it runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Lock {
    /// The dataset directory itself; leaves no file behind.
    #[default]
    Directory,
    /// `<dir>/write.lock`, for sharing the lock with code that flocks that file. Never
    /// delete it while a writer may hold it: the next one would lock a new file.
    File,
    /// No lock, like Python mmappet's writer.
    None,
}

/// A lock held until dropped.
pub(crate) struct Held {
    _file: Option<File>,
}

/// Locks dataset directory `dir` (which must exist), or fails at once if another holder
/// has it.
pub(crate) fn acquire(dir: &Path, lock: Lock) -> Result<Held> {
    let file = match lock {
        Lock::None => return Ok(Held { _file: None }),
        Lock::Directory => File::open(dir)?,
        Lock::File => OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(dir.join(LOCK_FILE))?,
    };
    match file.try_lock() {
        Ok(()) => Ok(Held { _file: Some(file) }),
        Err(TryLockError::WouldBlock) => Err(MmappetError::Locked(dir.to_path_buf())),
        Err(TryLockError::Error(error)) => Err(error.into()),
    }
}
