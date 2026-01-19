//! Data type definitions for mmappet columns.

use crate::error::{MmappetError, Result};

/// Represents all supported mmappet data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    UInt8,
    Int8,
    UInt16,
    Int16,
    UInt32,
    Int32,
    UInt64,
    Int64,
    Float32,
    Float64,
    Bool,
}

impl DType {
    /// Size of this dtype in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::UInt8 | DType::Int8 | DType::Bool => 1,
            DType::UInt16 | DType::Int16 => 2,
            DType::UInt32 | DType::Int32 | DType::Float32 => 4,
            DType::UInt64 | DType::Int64 | DType::Float64 => 8,
        }
    }

    /// Parse from schema string (e.g., "uint32").
    pub fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_lowercase().as_str() {
            "uint8" | "u8" => Ok(DType::UInt8),
            "int8" | "i8" => Ok(DType::Int8),
            "uint16" | "u16" => Ok(DType::UInt16),
            "int16" | "i16" => Ok(DType::Int16),
            "uint32" | "u32" => Ok(DType::UInt32),
            "int32" | "i32" => Ok(DType::Int32),
            "uint64" | "u64" | "size_t" => Ok(DType::UInt64),
            "int64" | "i64" => Ok(DType::Int64),
            "float32" | "f32" => Ok(DType::Float32),
            "float64" | "f64" | "double" => Ok(DType::Float64),
            "bool" | "boolean" => Ok(DType::Bool),
            _ => Err(MmappetError::UnknownDType(s.to_string())),
        }
    }

    /// Convert to canonical string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            DType::UInt8 => "uint8",
            DType::Int8 => "int8",
            DType::UInt16 => "uint16",
            DType::Int16 => "int16",
            DType::UInt32 => "uint32",
            DType::Int32 => "int32",
            DType::UInt64 => "uint64",
            DType::Int64 => "int64",
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Bool => "bool",
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Trait for Rust types that correspond to mmappet dtypes.
///
/// This trait is sealed and only implemented for supported primitive types.
pub trait MmappetType: bytemuck::Pod + 'static {
    /// The corresponding DType for this Rust type.
    const DTYPE: DType;
}

impl MmappetType for u8 {
    const DTYPE: DType = DType::UInt8;
}

impl MmappetType for i8 {
    const DTYPE: DType = DType::Int8;
}

impl MmappetType for u16 {
    const DTYPE: DType = DType::UInt16;
}

impl MmappetType for i16 {
    const DTYPE: DType = DType::Int16;
}

impl MmappetType for u32 {
    const DTYPE: DType = DType::UInt32;
}

impl MmappetType for i32 {
    const DTYPE: DType = DType::Int32;
}

impl MmappetType for u64 {
    const DTYPE: DType = DType::UInt64;
}

impl MmappetType for i64 {
    const DTYPE: DType = DType::Int64;
}

impl MmappetType for f32 {
    const DTYPE: DType = DType::Float32;
}

impl MmappetType for f64 {
    const DTYPE: DType = DType::Float64;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_from_str() {
        assert_eq!(DType::from_str("uint32").unwrap(), DType::UInt32);
        assert_eq!(DType::from_str("float64").unwrap(), DType::Float64);
        assert_eq!(DType::from_str("UINT32").unwrap(), DType::UInt32);
        assert_eq!(DType::from_str("size_t").unwrap(), DType::UInt64);
        assert!(DType::from_str("invalid").is_err());
    }

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::UInt8.size_bytes(), 1);
        assert_eq!(DType::UInt32.size_bytes(), 4);
        assert_eq!(DType::Float64.size_bytes(), 8);
    }

    #[test]
    fn test_mmappet_type_trait() {
        assert_eq!(u32::DTYPE, DType::UInt32);
        assert_eq!(f32::DTYPE, DType::Float32);
        assert_eq!(i64::DTYPE, DType::Int64);
    }
}
