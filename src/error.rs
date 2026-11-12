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

    #[error("Column '{column}' is {actual}, not readable as {expected}")]
    TypeMismatch {
        column: String,
        expected: &'static str,
        actual: DType,
    },

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

    #[error(
        "Invalid column file size: {path} has {actual} bytes, expected multiple of {element_size}"
    )]
    InvalidFileSize {
        path: PathBuf,
        actual: usize,
        element_size: usize,
    },

    #[error("Duplicate column name: {0}")]
    DuplicateColumnName(String),

    #[error("{0} exists; open it with Mode::AppendOk or Mode::OverwriteDir")]
    AlreadyExists(PathBuf),

    #[error("{path} holds schema [{stored}], not the given [{given}]")]
    SchemaMismatch {
        path: PathBuf,
        stored: String,
        given: String,
    },

    #[error("Invalid column name {0:?}: empty, starting with whitespace or holding a newline")]
    InvalidColumnName(String),

    #[error("A schema needs at least one column")]
    EmptySchema,

    #[error("Cannot append: {0}")]
    Append(String),
}

/// Result type for mmappet operations.
pub type Result<T> = std::result::Result<T, MmappetError>;
