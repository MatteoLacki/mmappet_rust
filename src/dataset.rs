//! Datasets: a directory of columns, opened read-only ([`Dataset`]) or read-write
//! ([`DatasetMut`]).

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::ops::Index;
use std::path::{Path, PathBuf};

use memmap2::MmapMut;
#[cfg(feature = "ndarray")]
use ndarray::ArrayView1;

use crate::column::{cast, cast_mut, Column, Typed};
use crate::dtype::{DType, MmappetType};
use crate::error::{MmappetError, Result};
use crate::schema::Schema;

/// A dataset mapped read-only. Columns can be borrowed (`get`) or shared (`typed`, `column`
/// clones), which keeps their maps alive after the `Dataset` is dropped.
pub struct Dataset {
    path: PathBuf,
    schema: Schema,
    columns: HashMap<String, Column>,
    row_count: usize,
}

impl Dataset {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let schema = Schema::from_path(&path)?;
        let files = open_column_files(&path, &schema, false)?;
        let mut columns = HashMap::new();
        for (def, ColumnFile { file, bytes }) in schema.columns().zip(&files) {
            columns.insert(def.name.clone(), Column::map(file, *bytes, def.dtype)?);
        }
        Ok(Dataset {
            row_count: row_count(&schema, &files),
            path,
            schema,
            columns,
        })
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// The column `name`: its dtype and raw bytes. Clone it to keep it past the `Dataset`.
    pub fn column(&self, name: &str) -> Result<&Column> {
        self.columns
            .get(name)
            .ok_or_else(|| MmappetError::ColumnNotFound(name.to_string()))
    }

    /// The column `name` as `T`, borrowed from the `Dataset`.
    pub fn get<T: MmappetType>(&self, name: &str) -> Result<&[T]> {
        let column = self.column(name)?;
        column
            .as_slice::<T>()
            .ok_or_else(|| type_mismatch::<T>(name, column.dtype()))
    }

    /// Like [`get`](Self::get), but a missing column is `None`; a wrong dtype is still an
    /// error.
    pub fn get_opt<T: MmappetType>(&self, name: &str) -> Result<Option<&[T]>> {
        if !self.columns.contains_key(name) {
            return Ok(None);
        }
        self.get(name).map(Some)
    }

    /// The column `name` as `T`, sharing its map: it lives on after the `Dataset` is
    /// dropped.
    pub fn typed<T: MmappetType>(&self, name: &str) -> Result<Typed<T>> {
        let column = self.column(name)?;
        Typed::new(column.clone()).ok_or_else(|| type_mismatch::<T>(name, column.dtype()))
    }

    #[cfg(feature = "ndarray")]
    pub fn get_array<T: MmappetType>(&self, name: &str) -> Result<ArrayView1<'_, T>> {
        self.get(name).map(ArrayView1::from)
    }

    /// Number of rows (all columns have same length).
    pub fn len(&self) -> usize {
        self.row_count
    }

    pub fn is_empty(&self) -> bool {
        self.row_count == 0
    }

    pub fn num_columns(&self) -> usize {
        self.schema.len()
    }

    pub fn column_names(&self) -> impl Iterator<Item = &str> {
        self.schema.column_names().into_iter()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Index<&str> for Dataset {
    type Output = Column;

    fn index(&self, name: &str) -> &Self::Output {
        self.columns
            .get(name)
            .unwrap_or_else(|| panic!("Column not found: {}", name))
    }
}

/// A dataset mapped read-write: values written through `get_mut` land in its files, as
/// with Python's `open_dataset_dct(path, read_write=True)`. Its columns are only borrowed,
/// never shared, so no other reference can see a value change under it.
pub struct DatasetMut {
    path: PathBuf,
    schema: Schema,
    columns: HashMap<String, ColumnMut>,
    row_count: usize,
}

struct ColumnMut {
    /// `None` for an empty file, which cannot be mapped.
    map: Option<MmapMut>,
    dtype: DType,
}

impl ColumnMut {
    fn bytes(&self) -> &[u8] {
        match &self.map {
            Some(map) => &map[..],
            None => &[],
        }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        match &mut self.map {
            Some(map) => &mut map[..],
            None => &mut [],
        }
    }
}

impl DatasetMut {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let schema = Schema::from_path(&path)?;
        let files = open_column_files(&path, &schema, true)?;
        let mut columns = HashMap::new();
        for (def, ColumnFile { file, bytes }) in schema.columns().zip(&files) {
            let map = match bytes {
                0 => None,
                // SAFETY: as for `Column::map`: the file is not truncated while mapped.
                _ => Some(unsafe { MmapMut::map_mut(file)? }),
            };
            let dtype = def.dtype;
            columns.insert(def.name.clone(), ColumnMut { map, dtype });
        }
        Ok(DatasetMut {
            row_count: row_count(&schema, &files),
            path,
            schema,
            columns,
        })
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    fn column(&self, name: &str) -> Result<&ColumnMut> {
        self.columns
            .get(name)
            .ok_or_else(|| MmappetError::ColumnNotFound(name.to_string()))
    }

    pub fn dtype(&self, name: &str) -> Result<DType> {
        Ok(self.column(name)?.dtype)
    }

    pub fn get<T: MmappetType>(&self, name: &str) -> Result<&[T]> {
        let column = self.column(name)?;
        if !T::accepts(column.dtype) {
            return Err(type_mismatch::<T>(name, column.dtype));
        }
        Ok(cast(column.bytes()))
    }

    pub fn get_mut<T: MmappetType>(&mut self, name: &str) -> Result<&mut [T]> {
        let column = self
            .columns
            .get_mut(name)
            .ok_or_else(|| MmappetError::ColumnNotFound(name.to_string()))?;
        if !T::accepts(column.dtype) {
            return Err(type_mismatch::<T>(name, column.dtype));
        }
        Ok(cast_mut(column.bytes_mut()))
    }

    /// Writes every changed page to its file now, rather than when the OS gets to it.
    pub fn flush(&self) -> Result<()> {
        for column in self.columns.values() {
            if let Some(map) = &column.map {
                map.flush()?;
            }
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.row_count
    }

    pub fn is_empty(&self) -> bool {
        self.row_count == 0
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

struct ColumnFile {
    file: File,
    bytes: usize,
}

/// Each schema column's `<index>.bin`, checked to hold whole values of its dtype and as
/// many values as every other column.
fn open_column_files(path: &Path, schema: &Schema, writable: bool) -> Result<Vec<ColumnFile>> {
    let mut files = Vec::with_capacity(schema.len());
    let mut expected_len: Option<usize> = None;
    for def in schema.columns() {
        let file_path = path.join(format!("{}.bin", def.index));
        if !file_path.exists() {
            return Err(MmappetError::MissingColumnFile(file_path));
        }
        let file = OpenOptions::new()
            .read(true)
            .write(writable)
            .open(&file_path)?;
        let bytes = file.metadata()?.len() as usize;
        let element_size = def.dtype.size_bytes();
        if bytes % element_size != 0 {
            return Err(MmappetError::InvalidFileSize {
                path: file_path,
                actual: bytes,
                element_size,
            });
        }
        let len = bytes / element_size;
        match expected_len {
            None => expected_len = Some(len),
            Some(expected) if expected != len => {
                return Err(MmappetError::LengthMismatch {
                    name: def.name.clone(),
                    expected,
                    actual: len,
                })
            }
            Some(_) => {}
        }
        files.push(ColumnFile { file, bytes });
    }
    Ok(files)
}

fn row_count(schema: &Schema, files: &[ColumnFile]) -> usize {
    match (schema.columns().next(), files.first()) {
        (Some(def), Some(file)) => file.bytes / def.dtype.size_bytes(),
        _ => 0,
    }
}

fn type_mismatch<T>(column: &str, actual: DType) -> MmappetError {
    MmappetError::TypeMismatch {
        column: column.to_string(),
        expected: std::any::type_name::<T>(),
        actual,
    }
}
