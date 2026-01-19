//! Error types for the mmappet library.

use std::path::PathBuf;
use thiserror::Error;

use crate::dtype::DType;

/// Errors that can occur when working with mmappet datasets.
#[derive(Error, Debug)]
pub enum MmappetError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Schema parse error at line {line}: {message}")]
    SchemaParse { line: usize, message: String },

    #[error("Unknown dtype: {0}")]
    UnknownDType(String),

    #[error("Column not found: {0}")]
    ColumnNotFound(String),

    #[error("Type mismatch: expected {expected:?}, got {actual:?}")]
    TypeMismatch { expected: DType, actual: DType },

    #[error("Column length mismatch: column '{name}' has {actual} elements, expected {expected}")]
    LengthMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },

    #[error("Missing schema.txt in {0}")]
    MissingSchema(PathBuf),

    #[error("Missing column file: {0}")]
    MissingColumnFile(PathBuf),

    #[error("Invalid column file size: {path} has {actual} bytes, expected multiple of {element_size}")]
    InvalidFileSize {
        path: PathBuf,
        actual: usize,
        element_size: usize,
    },

    #[error("Duplicate column name: {0}")]
    DuplicateColumnName(String),
}

/// Result type for mmappet operations.
pub type Result<T> = std::result::Result<T, MmappetError>;
