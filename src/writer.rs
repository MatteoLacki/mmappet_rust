//! Writing datasets, after Python's `mmappet.DatasetWriter`: append-only, one buffered file
//! per column, `schema.txt` written when the schema is fixed.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use half::f16;
use num_complex::Complex;

use crate::dtype::{DType, Datetime64, TimeUnit, Timedelta64};
use crate::error::{MmappetError, Result};
use crate::schema::Schema;

/// Python's `open(..., buffering=10240)`.
const BUFFER_BYTES: usize = 10240;

/// What `DatasetWriter::new` does with an existing directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// The directory must not exist.
    New,
    /// Append to the dataset there, or create one.
    AppendOk,
    /// Delete whatever is there first.
    OverwriteDir,
}

/// One column's values for [`DatasetWriter::append`], borrowed. Time values carry their
/// unit.
#[derive(Debug, Clone, Copy)]
pub enum Values<'a> {
    UInt8(&'a [u8]),
    Int8(&'a [i8]),
    UInt16(&'a [u16]),
    Int16(&'a [i16]),
    UInt32(&'a [u32]),
    Int32(&'a [i32]),
    UInt64(&'a [u64]),
    Int64(&'a [i64]),
    Float16(&'a [f16]),
    Float32(&'a [f32]),
    Float64(&'a [f64]),
    Complex64(&'a [Complex<f32>]),
    Complex128(&'a [Complex<f64>]),
    Bool(&'a [bool]),
    Datetime64(TimeUnit, &'a [Datetime64]),
    Timedelta64(TimeUnit, &'a [Timedelta64]),
    /// Values already in their stored form, e.g. `[u8; 4]` structs written as `uint32`; the
    /// byte count must be a whole number of values.
    Raw(DType, &'a [u8]),
}

/// One value for [`DatasetWriter::append_row`].
#[derive(Debug, Clone, Copy)]
pub enum Value {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    UInt64(u64),
    Int64(i64),
    Float16(f16),
    Float32(f32),
    Float64(f64),
    Complex64(Complex<f32>),
    Complex128(Complex<f64>),
    Bool(bool),
    Datetime64(TimeUnit, Datetime64),
    Timedelta64(TimeUnit, Timedelta64),
}

/// Calls `$m!(Variant, RustType)` for every variant whose dtype has no unit.
macro_rules! plain_types {
    ($m:ident) => {
        $m!(UInt8, u8);
        $m!(Int8, i8);
        $m!(UInt16, u16);
        $m!(Int16, i16);
        $m!(UInt32, u32);
        $m!(Int32, i32);
        $m!(UInt64, u64);
        $m!(Int64, i64);
        $m!(Float16, f16);
        $m!(Float32, f32);
        $m!(Float64, f64);
        $m!(Complex64, Complex<f32>);
        $m!(Complex128, Complex<f64>);
        $m!(Bool, bool);
    };
}

macro_rules! from_plain {
    ($variant:ident, $rust:ty) => {
        impl<'a> From<&'a [$rust]> for Values<'a> {
            fn from(values: &'a [$rust]) -> Self {
                Values::$variant(values)
            }
        }
        impl<'a> From<&'a Vec<$rust>> for Values<'a> {
            fn from(values: &'a Vec<$rust>) -> Self {
                Values::$variant(values)
            }
        }
        impl From<$rust> for Value {
            fn from(value: $rust) -> Self {
                Value::$variant(value)
            }
        }
    };
}
plain_types!(from_plain);

impl<'a> Values<'a> {
    pub fn dtype(&self) -> DType {
        match self {
            Values::UInt8(_) => DType::UInt8,
            Values::Int8(_) => DType::Int8,
            Values::UInt16(_) => DType::UInt16,
            Values::Int16(_) => DType::Int16,
            Values::UInt32(_) => DType::UInt32,
            Values::Int32(_) => DType::Int32,
            Values::UInt64(_) => DType::UInt64,
            Values::Int64(_) => DType::Int64,
            Values::Float16(_) => DType::Float16,
            Values::Float32(_) => DType::Float32,
            Values::Float64(_) => DType::Float64,
            Values::Complex64(_) => DType::Complex64,
            Values::Complex128(_) => DType::Complex128,
            Values::Bool(_) => DType::Bool,
            Values::Datetime64(unit, _) => DType::Datetime64(*unit),
            Values::Timedelta64(unit, _) => DType::Timedelta64(*unit),
            Values::Raw(dtype, _) => *dtype,
        }
    }

    pub fn len(&self) -> usize {
        self.bytes().len() / self.dtype().size_bytes()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The values as stored: native byte order, no padding.
    pub fn bytes(&self) -> &'a [u8] {
        use bytemuck::cast_slice;
        match *self {
            Values::UInt8(v) => v,
            Values::Int8(v) => cast_slice(v),
            Values::UInt16(v) => cast_slice(v),
            Values::Int16(v) => cast_slice(v),
            Values::UInt32(v) => cast_slice(v),
            Values::Int32(v) => cast_slice(v),
            Values::UInt64(v) => cast_slice(v),
            Values::Int64(v) => cast_slice(v),
            Values::Float16(v) => cast_slice(v),
            Values::Float32(v) => cast_slice(v),
            Values::Float64(v) => cast_slice(v),
            Values::Complex64(v) => cast_slice(v),
            Values::Complex128(v) => cast_slice(v),
            Values::Bool(v) => cast_slice(v),
            Values::Datetime64(_, v) => cast_slice(v),
            Values::Timedelta64(_, v) => cast_slice(v),
            Values::Raw(_, bytes) => bytes,
        }
    }
}

impl Value {
    /// The one value as a one-element [`Values`].
    pub fn as_values(&self) -> Values<'_> {
        use std::slice::from_ref;
        match self {
            Value::UInt8(v) => Values::UInt8(from_ref(v)),
            Value::Int8(v) => Values::Int8(from_ref(v)),
            Value::UInt16(v) => Values::UInt16(from_ref(v)),
            Value::Int16(v) => Values::Int16(from_ref(v)),
            Value::UInt32(v) => Values::UInt32(from_ref(v)),
            Value::Int32(v) => Values::Int32(from_ref(v)),
            Value::UInt64(v) => Values::UInt64(from_ref(v)),
            Value::Int64(v) => Values::Int64(from_ref(v)),
            Value::Float16(v) => Values::Float16(from_ref(v)),
            Value::Float32(v) => Values::Float32(from_ref(v)),
            Value::Float64(v) => Values::Float64(from_ref(v)),
            Value::Complex64(v) => Values::Complex64(from_ref(v)),
            Value::Complex128(v) => Values::Complex128(from_ref(v)),
            Value::Bool(v) => Values::Bool(from_ref(v)),
            Value::Datetime64(unit, v) => Values::Datetime64(*unit, from_ref(v)),
            Value::Timedelta64(unit, v) => Values::Timedelta64(*unit, from_ref(v)),
        }
    }
}

/// Appends rows to a dataset directory. The schema is given to `new` or fixed by the first
/// `append`; each column then goes to its own buffered `<index>.bin`.
pub struct DatasetWriter {
    path: PathBuf,
    /// Empty until the schema is fixed.
    columns: Vec<(String, DType)>,
    files: Vec<BufWriter<File>>,
    len: usize,
}

impl DatasetWriter {
    /// Opens `path` as `mode` says. With `schema`, the columns are fixed and `schema.txt`
    /// written now; without it, the first `append` fixes them (or, under `AppendOk`, the
    /// dataset already there does).
    pub fn new<P: AsRef<Path>>(
        path: P,
        mode: Mode,
        schema: Option<&[(&str, DType)]>,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut writer = DatasetWriter {
            path,
            columns: Vec::new(),
            files: Vec::new(),
            len: 0,
        };
        let exists = writer.path.exists();
        match mode {
            Mode::New if exists => return Err(MmappetError::AlreadyExists(writer.path)),
            Mode::OverwriteDir if exists => std::fs::remove_dir_all(&writer.path)?,
            Mode::AppendOk if exists => {
                writer.reopen()?;
                if let Some(schema) = schema {
                    writer.check_same_schema(schema)?;
                }
                return Ok(writer);
            }
            _ => {}
        }
        std::fs::create_dir_all(&writer.path)?;
        if let Some(schema) = schema {
            writer.fix_schema(schema)?;
        }
        Ok(writer)
    }

    /// Creates a dataset of `nrows` zeroed rows, to fill in place with
    /// [`DatasetMut`](crate::DatasetMut), like Python's `DatasetWriter.preallocate_dataset`.
    pub fn preallocate<P: AsRef<Path>>(
        path: P,
        schema: &[(&str, DType)],
        nrows: usize,
        overwrite_dir: bool,
    ) -> Result<()> {
        let mode = if overwrite_dir {
            Mode::OverwriteDir
        } else {
            Mode::New
        };
        let mut writer = DatasetWriter::new(path, mode, Some(schema))?;
        for (file, (_, dtype)) in writer.files.iter_mut().zip(&writer.columns) {
            file.flush()?;
            file.get_ref()
                .set_len((nrows * dtype.size_bytes()) as u64)?;
        }
        Ok(())
    }

    /// Rows written so far, counting those already there under `AppendOk`.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Appends rows: every column once, by name in any order, all of one length and each of
    /// its column's exact dtype. The first call fixes the schema if nothing has, in the
    /// order given.
    pub fn append(&mut self, columns: &[(&str, Values)]) -> Result<()> {
        if self.columns.is_empty() {
            let schema: Vec<(&str, DType)> = columns
                .iter()
                .map(|(name, values)| (*name, values.dtype()))
                .collect();
            self.fix_schema(&schema)?;
        }
        if columns.len() != self.columns.len() {
            return Err(MmappetError::Append(format!(
                "{} columns given, the dataset has {}",
                columns.len(),
                self.columns.len()
            )));
        }
        let rows = columns[0].1.len();
        let mut ordered = Vec::with_capacity(columns.len());
        for (stored_name, stored_dtype) in &self.columns {
            let mut found = None;
            for (name, values) in columns {
                if name == stored_name {
                    if found.is_some() {
                        return Err(MmappetError::DuplicateColumnName(stored_name.clone()));
                    }
                    found = Some(values);
                }
            }
            let Some(values) = found else {
                return Err(MmappetError::Append(format!(
                    "column '{stored_name}' not given"
                )));
            };
            if values.dtype() != *stored_dtype {
                return Err(MmappetError::Append(format!(
                    "column '{stored_name}' is {stored_dtype}, given {}",
                    values.dtype()
                )));
            }
            check_whole_values(stored_name, values)?;
            if values.len() != rows {
                return Err(MmappetError::Append(format!(
                    "column '{stored_name}' given {} values, others {rows}",
                    values.len()
                )));
            }
            ordered.push(values);
        }
        for (file, values) in self.files.iter_mut().zip(ordered) {
            file.write_all(values.bytes())?;
        }
        self.len += rows;
        Ok(())
    }

    /// Appends one row, as [`append`](Self::append) does.
    pub fn append_row(&mut self, row: &[(&str, Value)]) -> Result<()> {
        let columns: Vec<(&str, Values)> = row
            .iter()
            .map(|(name, value)| (*name, value.as_values()))
            .collect();
        self.append(&columns)
    }

    /// Adds a column `name` holding one value per row written so far.
    pub fn append_column(&mut self, name: &str, values: Values) -> Result<()> {
        if self.columns.is_empty() {
            return Err(MmappetError::Append(
                "no schema yet to add a column to".into(),
            ));
        }
        check_name(name)?;
        if self.columns.iter().any(|(stored, _)| stored == name) {
            return Err(MmappetError::DuplicateColumnName(name.to_string()));
        }
        check_whole_values(name, &values)?;
        if values.len() != self.len {
            return Err(MmappetError::Append(format!(
                "column '{name}' given {} values, the dataset has {} rows",
                values.len(),
                self.len
            )));
        }
        let file_path = self.path.join(format!("{}.bin", self.columns.len()));
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&file_path)?;
        file.write_all(values.bytes())?;
        drop(file);
        self.columns.push((name.to_string(), values.dtype()));
        self.write_schema()?;
        self.files.push(open_append(&file_path)?);
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        for file in &mut self.files {
            file.flush()?;
        }
        Ok(())
    }

    /// Flushes every column, reporting errors that dropping the writer would swallow.
    pub fn close(mut self) -> Result<()> {
        self.flush()
    }

    fn fix_schema(&mut self, schema: &[(&str, DType)]) -> Result<()> {
        if schema.is_empty() {
            return Err(MmappetError::EmptySchema);
        }
        for (index, (name, _)) in schema.iter().enumerate() {
            check_name(name)?;
            if schema[..index].iter().any(|(earlier, _)| earlier == name) {
                return Err(MmappetError::DuplicateColumnName(name.to_string()));
            }
        }
        self.columns = schema
            .iter()
            .map(|(name, dtype)| (name.to_string(), *dtype))
            .collect();
        self.write_schema()?;
        for index in 0..self.columns.len() {
            self.files
                .push(open_append(&self.path.join(format!("{index}.bin")))?);
        }
        Ok(())
    }

    /// Takes the schema and length of the dataset already at `path`.
    fn reopen(&mut self) -> Result<()> {
        let schema = Schema::from_path(&self.path)?;
        let mut len = None;
        for def in schema.columns() {
            let file_path = self.path.join(format!("{}.bin", def.index));
            let bytes = std::fs::metadata(&file_path)
                .map(|m| m.len() as usize)
                .unwrap_or(0);
            let rows = bytes / def.dtype.size_bytes();
            if bytes % def.dtype.size_bytes() != 0 || len.is_some_and(|len| len != rows) {
                return Err(MmappetError::Append(format!(
                    "corrupted dataset: columns of unequal lengths in {}",
                    self.path.display()
                )));
            }
            len = Some(rows);
            self.columns.push((def.name.clone(), def.dtype));
            self.files.push(open_append(&file_path)?);
        }
        self.len = len.unwrap_or(0);
        Ok(())
    }

    fn check_same_schema(&self, schema: &[(&str, DType)]) -> Result<()> {
        let same = schema.len() == self.columns.len()
            && schema.iter().zip(&self.columns).all(
                |((name, dtype), (stored_name, stored_dtype))| {
                    name == stored_name && dtype == stored_dtype
                },
            );
        if same {
            return Ok(());
        }
        let show = |columns: Vec<String>| columns.join(", ");
        Err(MmappetError::SchemaMismatch {
            path: self.path.clone(),
            stored: show(
                self.columns
                    .iter()
                    .map(|(n, d)| format!("{d} {n}"))
                    .collect(),
            ),
            given: show(schema.iter().map(|(n, d)| format!("{d} {n}")).collect()),
        })
    }

    /// `schema.txt` as Python writes it: one `<dtype> <name>` line per column, no trailing
    /// newline.
    fn write_schema(&self) -> Result<()> {
        let lines: Vec<String> = self
            .columns
            .iter()
            .map(|(name, dtype)| format!("{dtype} {name}"))
            .collect();
        std::fs::write(self.path.join("schema.txt"), lines.join("\n"))?;
        Ok(())
    }
}

fn check_whole_values(name: &str, values: &Values) -> Result<()> {
    let size = values.dtype().size_bytes();
    if values.bytes().len() % size != 0 {
        return Err(MmappetError::Append(format!(
            "column '{name}' given {} bytes, not a whole number of {size}-byte values",
            values.bytes().len()
        )));
    }
    Ok(())
}

/// A name `schema.txt` can hold and read back: non-empty, not starting with whitespace (the
/// reader skips it), one line.
fn check_name(name: &str) -> Result<()> {
    if name.is_empty() || name.starts_with(char::is_whitespace) || name.contains(['\n', '\r']) {
        return Err(MmappetError::InvalidColumnName(name.to_string()));
    }
    Ok(())
}

fn open_append(path: &Path) -> Result<BufWriter<File>> {
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    Ok(BufWriter::with_capacity(BUFFER_BYTES, file))
}
