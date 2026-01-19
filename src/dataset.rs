//! Dataset type for mmappet - the main entry point.

use std::collections::HashMap;
use std::ops::Index;
use std::path::{Path, PathBuf};

use ndarray::ArrayView1;

use crate::column::Column;
use crate::dtype::MmappetType;
use crate::error::{MmappetError, Result};
use crate::schema::Schema;

/// Main entry point - a memory-mapped mmappet dataset.
pub struct Dataset {
    path: PathBuf,
    schema: Schema,
    columns: HashMap<String, Column>,
    row_count: usize,
}

impl Dataset {
    /// Open a dataset from a directory path.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Parse schema
        let schema = Schema::from_path(&path)?;

        // Load all columns
        let mut columns = HashMap::new();
        let mut row_count: Option<usize> = None;

        for col_def in schema.columns() {
            let col_path = path.join(format!("{}.bin", col_def.index));

            if !col_path.exists() {
                return Err(MmappetError::MissingColumnFile(col_path));
            }

            let column = Column::open(&col_path, col_def.dtype)?;

            // Validate all columns have same length
            match row_count {
                None => row_count = Some(column.len()),
                Some(expected) => {
                    if column.len() != expected {
                        return Err(MmappetError::LengthMismatch {
                            name: col_def.name.clone(),
                            expected,
                            actual: column.len(),
                        });
                    }
                }
            }

            columns.insert(col_def.name.clone(), column);
        }

        Ok(Dataset {
            path,
            schema,
            columns,
            row_count: row_count.unwrap_or(0),
        })
    }

    /// Get the schema.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Get a column by name.
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.get(name)
    }

    /// Get a typed slice directly by name.
    ///
    /// Returns an error if the column doesn't exist or the type doesn't match.
    pub fn get<T: MmappetType>(&self, name: &str) -> Result<&[T]> {
        let column = self
            .columns
            .get(name)
            .ok_or_else(|| MmappetError::ColumnNotFound(name.to_string()))?;

        column.as_slice::<T>().ok_or_else(|| MmappetError::TypeMismatch {
            expected: T::DTYPE,
            actual: column.dtype(),
        })
    }

    /// Get a typed ArrayView1 directly by name.
    ///
    /// Returns an error if the column doesn't exist or the type doesn't match.
    pub fn get_array<T: MmappetType>(&self, name: &str) -> Result<ArrayView1<T>> {
        let column = self
            .columns
            .get(name)
            .ok_or_else(|| MmappetError::ColumnNotFound(name.to_string()))?;

        column
            .as_array::<T>()
            .ok_or_else(|| MmappetError::TypeMismatch {
                expected: T::DTYPE,
                actual: column.dtype(),
            })
    }

    /// Number of rows (all columns have same length).
    pub fn len(&self) -> usize {
        self.row_count
    }

    /// Check if dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.row_count == 0
    }

    /// Number of columns.
    pub fn num_columns(&self) -> usize {
        self.schema.len()
    }

    /// Iterate over column names.
    pub fn column_names(&self) -> impl Iterator<Item = &str> {
        self.schema.column_names().into_iter()
    }

    /// Get dataset path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

// Dictionary-style indexing via Index trait
impl Index<&str> for Dataset {
    type Output = Column;

    fn index(&self, name: &str) -> &Self::Output {
        self.columns
            .get(name)
            .unwrap_or_else(|| panic!("Column not found: {}", name))
    }
}
