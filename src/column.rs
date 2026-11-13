//! Column types for mmappet datasets.

use std::fs::File;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;

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
use crate::error::Result;

/// One column: its file mapped read-only and shared, so a page the kernel drops is reread
/// from the file (an empty file has no map). Clones share the map, which lives as long as
/// any clone, so a column can outlive the `Dataset` it came from.
#[derive(Clone)]
pub struct Column {
    /// `None` for an empty file, which cannot be mapped.
    map: Option<Arc<Mmap>>,
    dtype: DType,
    len: usize,
}

impl std::fmt::Debug for Column {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Column")
            .field("dtype", &self.dtype)
            .field("len", &self.len)
            .finish()
    }
}

impl Column {
    /// Map `file`, which holds `bytes` bytes of `dtype` values.
    pub(crate) fn map(file: &File, bytes: usize, dtype: DType) -> Result<Self> {
        let map = match bytes {
            0 => None,
            // SAFETY: a mapped file must not be truncated while mapped; datasets are not
            // written while read.
            _ => Some(Arc::new(unsafe { Mmap::map(file)? })),
        };
        Ok(Column {
            map,
            dtype,
            len: bytes / dtype.size_bytes(),
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_bytes(&self) -> &[u8] {
        match &self.map {
            Some(map) => &map[..],
            None => &[],
        }
    }

    /// The values as `T`, or `None` if `T` cannot hold this column's dtype.
    pub fn as_slice<T: MmappetType>(&self) -> Option<&[T]> {
        if T::accepts(self.dtype) {
            Some(cast(self.as_bytes()))
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
        let bytes = self.as_bytes();
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

impl AsRef<[u8]> for Column {
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

/// A column checked once to hold `T`; derefs to `&[T]`. Clones share the map, like
/// [`Column`]'s.
pub struct Typed<T> {
    column: Column,
    element: PhantomData<T>,
}

impl<T: MmappetType> Typed<T> {
    /// `column` as `T`, or `None` if `T` cannot hold its dtype.
    pub fn new(column: Column) -> Option<Self> {
        if !T::accepts(column.dtype) {
            return None;
        }
        Some(Typed {
            column,
            element: PhantomData,
        })
    }

    pub fn column(&self) -> &Column {
        &self.column
    }
}

impl<T> std::fmt::Debug for Typed<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Typed").field(&self.column).finish()
    }
}

impl<T> Clone for Typed<T> {
    fn clone(&self) -> Self {
        Typed {
            column: self.column.clone(),
            element: PhantomData,
        }
    }
}

impl<T: MmappetType> Deref for Typed<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        cast(self.column.as_bytes())
    }
}

/// `bytes` as `T` values. A mapped file starts page-aligned and was checked to hold whole
/// values; an empty column has no map, so its empty slice may be unaligned.
pub(crate) fn cast<T: bytemuck::Pod>(bytes: &[u8]) -> &[T] {
    if bytes.is_empty() {
        return &[];
    }
    cast_slice(bytes)
}

pub(crate) fn cast_mut<T: bytemuck::Pod>(bytes: &mut [u8]) -> &mut [T] {
    if bytes.is_empty() {
        return &mut [];
    }
    bytemuck::cast_slice_mut(bytes)
}

#[cfg(feature = "ndarray")]
fn view<T: bytemuck::Pod>(bytes: &[u8]) -> ArrayView1<'_, T> {
    ArrayView1::from(cast(bytes))
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
