//! Column types for mmappet datasets.

use std::fs::File;
use std::path::Path;

#[cfg(feature = "ndarray")]
use crate::dtype::{Datetime64, TimeUnit, Timedelta64};
use bytemuck::cast_slice;
#[cfg(feature = "ndarray")]
use half::f16;
use memmap2::Mmap;
#[cfg(feature = "ndarray")]
use ndarray::ArrayView1;
#[cfg(feature = "ndarray")]
use num_complex::Complex;

use crate::dtype::{DType, MmappetType};
use crate::error::{MmappetError, Result};

/// Type-erased column data holding the mmap and metadata.
pub struct Column {
    mmap: Mmap,
    dtype: DType,
    len: usize,
}

impl Column {
    /// Open a column from a binary file.
    pub fn open<P: AsRef<Path>>(path: P, dtype: DType) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let element_size = dtype.size_bytes();
        let file_size = mmap.len();

        if file_size % element_size != 0 {
            return Err(MmappetError::InvalidFileSize {
                path: path.to_path_buf(),
                actual: file_size,
                element_size,
            });
        }

        let len = file_size / element_size;

        Ok(Column { mmap, dtype, len })
    }

    /// Get the data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if column is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get raw bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap[..]
    }

    /// Try to get as a typed slice.
    ///
    /// Returns `None` if the requested type doesn't match the column's dtype.
    pub fn as_slice<T: MmappetType>(&self) -> Option<&[T]> {
        if T::accepts(self.dtype) {
            Some(cast_slice(&self.mmap[..]))
        } else {
            None
        }
    }

    #[cfg(feature = "ndarray")]
    /// Try to get as ndarray ArrayView1.
    ///
    /// Returns `None` if the requested type doesn't match the column's dtype.
    pub fn as_array<T: MmappetType>(&self) -> Option<ArrayView1<'_, T>> {
        self.as_slice::<T>().map(ArrayView1::from)
    }

    #[cfg(feature = "ndarray")]
    /// Get as dynamically-typed array enum.
    pub fn as_typed_array(&self) -> TypedArrayView<'_> {
        let bytes = &self.mmap[..];
        match self.dtype {
            DType::UInt8 => TypedArrayView::UInt8(view(bytes)),
            DType::Int8 => TypedArrayView::Int8(view(bytes)),
            DType::UInt16 => TypedArrayView::UInt16(view(bytes)),
            DType::Int16 => TypedArrayView::Int16(view(bytes)),
            DType::UInt32 => TypedArrayView::UInt32(view(bytes)),
            DType::Int32 => TypedArrayView::Int32(view(bytes)),
            DType::UInt64 => TypedArrayView::UInt64(view(bytes)),
            DType::Int64 => TypedArrayView::Int64(view(bytes)),
            DType::Float16 => TypedArrayView::Float16(view(bytes)),
            DType::Float32 => TypedArrayView::Float32(view(bytes)),
            DType::Float64 => TypedArrayView::Float64(view(bytes)),
            DType::Complex64 => TypedArrayView::Complex64(view(bytes)),
            DType::Complex128 => TypedArrayView::Complex128(view(bytes)),
            DType::Bool => TypedArrayView::Bool(view(bytes)),
            DType::Datetime64(unit) => TypedArrayView::Datetime64(unit, view(bytes)),
            DType::Timedelta64(unit) => TypedArrayView::Timedelta64(unit, view(bytes)),
        }
    }
}

#[cfg(feature = "ndarray")]
fn view<T: bytemuck::Pod>(bytes: &[u8]) -> ArrayView1<'_, T> {
    ArrayView1::from(cast_slice(bytes))
}

/// Enum for dynamically-typed array access.
#[cfg(feature = "ndarray")]
pub enum TypedArrayView<'a> {
    UInt8(ArrayView1<'a, u8>),
    Int8(ArrayView1<'a, i8>),
    UInt16(ArrayView1<'a, u16>),
    Int16(ArrayView1<'a, i16>),
    UInt32(ArrayView1<'a, u32>),
    Int32(ArrayView1<'a, i32>),
    UInt64(ArrayView1<'a, u64>),
    Int64(ArrayView1<'a, i64>),
    Float16(ArrayView1<'a, f16>),
    Float32(ArrayView1<'a, f32>),
    Float64(ArrayView1<'a, f64>),
    Complex64(ArrayView1<'a, Complex<f32>>),
    Complex128(ArrayView1<'a, Complex<f64>>),
    Bool(ArrayView1<'a, u8>),
    Datetime64(TimeUnit, ArrayView1<'a, Datetime64>),
    Timedelta64(TimeUnit, ArrayView1<'a, Timedelta64>),
}

#[cfg(feature = "ndarray")]
impl<'a> TypedArrayView<'a> {
    /// Get the number of elements.
    pub fn len(&self) -> usize {
        match self {
            TypedArrayView::UInt8(arr) => arr.len(),
            TypedArrayView::Int8(arr) => arr.len(),
            TypedArrayView::UInt16(arr) => arr.len(),
            TypedArrayView::Int16(arr) => arr.len(),
            TypedArrayView::UInt32(arr) => arr.len(),
            TypedArrayView::Int32(arr) => arr.len(),
            TypedArrayView::UInt64(arr) => arr.len(),
            TypedArrayView::Int64(arr) => arr.len(),
            TypedArrayView::Float16(arr) => arr.len(),
            TypedArrayView::Float32(arr) => arr.len(),
            TypedArrayView::Float64(arr) => arr.len(),
            TypedArrayView::Complex64(arr) => arr.len(),
            TypedArrayView::Complex128(arr) => arr.len(),
            TypedArrayView::Bool(arr) => arr.len(),
            TypedArrayView::Datetime64(_, arr) => arr.len(),
            TypedArrayView::Timedelta64(_, arr) => arr.len(),
        }
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the dtype.
    pub fn dtype(&self) -> DType {
        match self {
            TypedArrayView::UInt8(_) => DType::UInt8,
            TypedArrayView::Int8(_) => DType::Int8,
            TypedArrayView::UInt16(_) => DType::UInt16,
            TypedArrayView::Int16(_) => DType::Int16,
            TypedArrayView::UInt32(_) => DType::UInt32,
            TypedArrayView::Int32(_) => DType::Int32,
            TypedArrayView::UInt64(_) => DType::UInt64,
            TypedArrayView::Int64(_) => DType::Int64,
            TypedArrayView::Float16(_) => DType::Float16,
            TypedArrayView::Float32(_) => DType::Float32,
            TypedArrayView::Float64(_) => DType::Float64,
            TypedArrayView::Complex64(_) => DType::Complex64,
            TypedArrayView::Complex128(_) => DType::Complex128,
            TypedArrayView::Bool(_) => DType::Bool,
            TypedArrayView::Datetime64(unit, _) => DType::Datetime64(*unit),
            TypedArrayView::Timedelta64(unit, _) => DType::Timedelta64(*unit),
        }
    }
}
