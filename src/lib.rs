//! # mmappet
//!
//! Memory-mapped columnar dataset library for efficient I/O.
//!
//! mmappet provides zero-copy access to column-oriented datasets stored on disk.
//! It's the Rust equivalent of the Python mmappet library.
//!
//! ## Example
//!
//! ```rust,no_run
//! use mmappet::Dataset;
//!
//! // Open a dataset
//! let ds = mmappet::Dataset::open("data.mmappet").unwrap();
//!
//! // Check schema
//! println!("Columns: {:?}", ds.schema().column_names());
//! println!("Rows: {}", ds.len());
//!
//! // Get typed data
//! let scores: &[f32] = ds.get("score").unwrap();
//! let ids: &[u32] = ds.get("id").unwrap();
//! ```

mod column;
mod dataset;
mod dtype;
mod error;
mod schema;

pub use column::{Column, TypedArrayView};
pub use dataset::Dataset;
pub use dtype::{DType, MmappetType};
pub use error::{MmappetError, Result};
pub use schema::{ColumnDef, Schema};

// Re-export commonly used ndarray types for convenience
pub use ndarray::ArrayView1;
