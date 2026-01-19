//! Column types for mmappet datasets.

use std::fs::File;
use std::path::Path;

use bytemuck::cast_slice;
use memmap2::Mmap;
use ndarray::ArrayView1;

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
        if T::DTYPE == self.dtype {
            Some(cast_slice(&self.mmap[..]))
        } else {
            None
        }
    }

    /// Try to get as ndarray ArrayView1.
    ///
    /// Returns `None` if the requested type doesn't match the column's dtype.
    pub fn as_array<T: MmappetType>(&self) -> Option<ArrayView1<T>> {
        self.as_slice::<T>().map(ArrayView1::from)
    }

    /// Get as dynamically-typed array enum.
    pub fn as_typed_array(&self) -> TypedArrayView<'_> {
        match self.dtype {
            DType::UInt8 => TypedArrayView::UInt8(ArrayView1::from(cast_slice(&self.mmap[..]))),
            DType::Int8 => TypedArrayView::Int8(ArrayView1::from(cast_slice(&self.mmap[..]))),
            DType::UInt16 => TypedArrayView::UInt16(ArrayView1::from(cast_slice(&self.mmap[..]))),
            DType::Int16 => TypedArrayView::Int16(ArrayView1::from(cast_slice(&self.mmap[..]))),
            DType::UInt32 => TypedArrayView::UInt32(ArrayView1::from(cast_slice(&self.mmap[..]))),
            DType::Int32 => TypedArrayView::Int32(ArrayView1::from(cast_slice(&self.mmap[..]))),
            DType::UInt64 => TypedArrayView::UInt64(ArrayView1::from(cast_slice(&self.mmap[..]))),
            DType::Int64 => TypedArrayView::Int64(ArrayView1::from(cast_slice(&self.mmap[..]))),
            DType::Float32 => TypedArrayView::Float32(ArrayView1::from(cast_slice(&self.mmap[..]))),
            DType::Float64 => TypedArrayView::Float64(ArrayView1::from(cast_slice(&self.mmap[..]))),
            DType::Bool => TypedArrayView::Bool(ArrayView1::from(cast_slice(&self.mmap[..]))),
        }
    }
}

/// Enum for dynamically-typed array access.
pub enum TypedArrayView<'a> {
    UInt8(ArrayView1<'a, u8>),
    Int8(ArrayView1<'a, i8>),
    UInt16(ArrayView1<'a, u16>),
    Int16(ArrayView1<'a, i16>),
    UInt32(ArrayView1<'a, u32>),
    Int32(ArrayView1<'a, i32>),
    UInt64(ArrayView1<'a, u64>),
    Int64(ArrayView1<'a, i64>),
    Float32(ArrayView1<'a, f32>),
    Float64(ArrayView1<'a, f64>),
    Bool(ArrayView1<'a, u8>), // Bool stored as u8
}

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
            TypedArrayView::Float32(arr) => arr.len(),
            TypedArrayView::Float64(arr) => arr.len(),
            TypedArrayView::Bool(arr) => arr.len(),
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
            TypedArrayView::Float32(_) => DType::Float32,
            TypedArrayView::Float64(_) => DType::Float64,
            TypedArrayView::Bool(_) => DType::Bool,
        }
    }
}
