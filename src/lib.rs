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
mod writer;

#[cfg(feature = "ndarray")]
pub use column::TypedArrayView;
pub use column::{Column, Typed};
pub use dataset::{Dataset, DatasetMut};
pub use dtype::{DType, Datetime64, MmappetType, TimeUnit, Timedelta64};
pub use error::{MmappetError, Result};
pub use half::f16;
pub use num_complex::Complex;
pub use schema::{ColumnDef, Schema};
pub use writer::{DatasetWriter, Mode, Value, Values};

#[cfg(feature = "ndarray")]
pub use ndarray::ArrayView1;
