//! Data type definitions for mmappet columns, named as numpy names them.

use half::f16;
use num_complex::Complex;

use crate::error::{MmappetError, Result};

/// Every supported mmappet column type.
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
    Float16,
    Float32,
    Float64,
    Complex64,
    Complex128,
    Bool,
    Datetime64(TimeUnit),
    Timedelta64(TimeUnit),
}

/// The time units Python mmappet keeps: pandas, which builds its schema, relabels every other
/// numpy unit (`datetime64[D]` reads back as `datetime64[s]`) and refuses multiples (`10ms`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeUnit {
    Seconds,
    Millis,
    Micros,
    Nanos,
}

impl TimeUnit {
    pub const ALL: [TimeUnit; 4] = [
        TimeUnit::Seconds,
        TimeUnit::Millis,
        TimeUnit::Micros,
        TimeUnit::Nanos,
    ];

    /// numpy's code for the unit, as inside `datetime64[...]`.
    pub fn code(self) -> &'static str {
        match self {
            TimeUnit::Seconds => "s",
            TimeUnit::Millis => "ms",
            TimeUnit::Micros => "us",
            TimeUnit::Nanos => "ns",
        }
    }

    pub fn parse(code: &str) -> Option<Self> {
        for unit in TimeUnit::ALL {
            if unit.code() == code {
                return Some(unit);
            }
        }
        None
    }
}

impl std::fmt::Display for TimeUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.code())
    }
}

impl DType {
    /// Size of one value in bytes.
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::UInt8 | DType::Int8 | DType::Bool => 1,
            DType::UInt16 | DType::Int16 | DType::Float16 => 2,
            DType::UInt32 | DType::Int32 | DType::Float32 => 4,
            DType::UInt64 | DType::Int64 | DType::Float64 | DType::Complex64 => 8,
            DType::Datetime64(_) | DType::Timedelta64(_) => 8,
            DType::Complex128 => 16,
        }
    }

    /// Parse a `schema.txt` type: numpy's `str(dtype)` (`uint32`, `datetime64[ms]`) or one
    /// of the short aliases (`u32`, `f64`, `size_t`, ...).
    pub fn from_str(s: &str) -> Result<Self> {
        let s = s.trim();
        if let Some(dtype) = Self::parse_time(s) {
            return Ok(dtype);
        }
        match s.to_lowercase().as_str() {
            "uint8" | "u8" => Ok(DType::UInt8),
            "int8" | "i8" => Ok(DType::Int8),
            "uint16" | "u16" => Ok(DType::UInt16),
            "int16" | "i16" => Ok(DType::Int16),
            "uint32" | "u32" => Ok(DType::UInt32),
            "int32" | "i32" => Ok(DType::Int32),
            "uint64" | "u64" | "size_t" => Ok(DType::UInt64),
            "int64" | "i64" => Ok(DType::Int64),
            "float16" | "f16" | "half" => Ok(DType::Float16),
            "float32" | "f32" => Ok(DType::Float32),
            "float64" | "f64" | "double" => Ok(DType::Float64),
            "complex64" | "c64" => Ok(DType::Complex64),
            "complex128" | "c128" => Ok(DType::Complex128),
            "bool" | "boolean" => Ok(DType::Bool),
            _ => Err(MmappetError::UnknownDType(s.to_string())),
        }
    }

    fn parse_time(s: &str) -> Option<Self> {
        let unit = |prefix: &str| {
            let inner = s.strip_prefix(prefix)?.strip_suffix(']')?;
            TimeUnit::parse(inner)
        };
        if let Some(unit) = unit("datetime64[") {
            return Some(DType::Datetime64(unit));
        }
        if let Some(unit) = unit("timedelta64[") {
            return Some(DType::Timedelta64(unit));
        }
        None
    }
}

/// numpy's `str(dtype)`, as the writer puts it in `schema.txt`.
impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            DType::UInt8 => "uint8",
            DType::Int8 => "int8",
            DType::UInt16 => "uint16",
            DType::Int16 => "int16",
            DType::UInt32 => "uint32",
            DType::Int32 => "int32",
            DType::UInt64 => "uint64",
            DType::Int64 => "int64",
            DType::Float16 => "float16",
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Complex64 => "complex64",
            DType::Complex128 => "complex128",
            DType::Bool => "bool",
            DType::Datetime64(unit) => return write!(f, "datetime64[{unit}]"),
            DType::Timedelta64(unit) => return write!(f, "timedelta64[{unit}]"),
        };
        f.write_str(name)
    }
}

/// A `datetime64` value: ticks of its column's unit since 1970-01-01T00:00.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Datetime64(pub i64);

/// A `timedelta64` value: ticks of its column's unit.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Timedelta64(pub i64);

impl Datetime64 {
    /// numpy's not-a-time.
    pub const NAT: Datetime64 = Datetime64(i64::MIN);
}

impl Timedelta64 {
    /// numpy's not-a-time.
    pub const NAT: Timedelta64 = Timedelta64(i64::MIN);
}

// SAFETY: one i64 each (`repr(transparent)`); every bit pattern is a value.
unsafe impl bytemuck::Zeroable for Datetime64 {}
unsafe impl bytemuck::Pod for Datetime64 {}
unsafe impl bytemuck::Zeroable for Timedelta64 {}
unsafe impl bytemuck::Pod for Timedelta64 {}

/// Rust types a column can be read as.
pub trait MmappetType: bytemuck::Pod + 'static {
    /// Whether a column of `dtype` holds values of this type.
    fn accepts(dtype: DType) -> bool;
}

macro_rules! exact_type {
    ($($rust:ty => $dtype:ident),* $(,)?) => {
        $(impl MmappetType for $rust {
            fn accepts(dtype: DType) -> bool {
                dtype == DType::$dtype
            }
        })*
    };
}

exact_type!(
    i8 => Int8,
    u16 => UInt16,
    i16 => Int16,
    u32 => UInt32,
    i32 => Int32,
    u64 => UInt64,
    i64 => Int64,
    f16 => Float16,
    f32 => Float32,
    f64 => Float64,
    Complex<f32> => Complex64,
    Complex<f64> => Complex128,
);

/// `bool` columns are read as `u8`: Rust's `bool` may only hold 0 or 1.
impl MmappetType for u8 {
    fn accepts(dtype: DType) -> bool {
        matches!(dtype, DType::UInt8 | DType::Bool)
    }
}

/// Any unit; the column's `DType` says which.
impl MmappetType for Datetime64 {
    fn accepts(dtype: DType) -> bool {
        matches!(dtype, DType::Datetime64(_))
    }
}

/// Any unit; the column's `DType` says which.
impl MmappetType for Timedelta64 {
    fn accepts(dtype: DType) -> bool {
        matches!(dtype, DType::Timedelta64(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_names_and_aliases() {
        assert_eq!(DType::from_str("uint32").unwrap(), DType::UInt32);
        assert_eq!(DType::from_str("UINT32").unwrap(), DType::UInt32);
        assert_eq!(DType::from_str("size_t").unwrap(), DType::UInt64);
        assert_eq!(DType::from_str("float16").unwrap(), DType::Float16);
        assert_eq!(DType::from_str("complex128").unwrap(), DType::Complex128);
        assert!(DType::from_str("invalid").is_err());
    }

    #[test]
    fn time_units_are_those_python_keeps() {
        let ms = DType::from_str("timedelta64[ms]").unwrap();
        assert_eq!(ms, DType::Timedelta64(TimeUnit::Millis));
        assert_eq!(ms.to_string(), "timedelta64[ms]");
        for refused in [
            "datetime64[D]",
            "datetime64[M]",
            "datetime64[10ms]",
            "datetime64[ps]",
            "datetime64",
        ] {
            assert!(DType::from_str(refused).is_err(), "{refused}");
        }
    }

    #[test]
    fn sizes() {
        assert_eq!(DType::Float16.size_bytes(), 2);
        assert_eq!(DType::Complex64.size_bytes(), 8);
        assert_eq!(DType::Complex128.size_bytes(), 16);
        assert_eq!(DType::Timedelta64(TimeUnit::Seconds).size_bytes(), 8);
    }

    #[test]
    fn rust_types_accept_their_dtypes() {
        assert!(u8::accepts(DType::Bool));
        assert!(!i8::accepts(DType::Bool));
        assert!(f16::accepts(DType::Float16));
        assert!(Datetime64::accepts(DType::Datetime64(TimeUnit::Nanos)));
        assert!(!Datetime64::accepts(DType::Timedelta64(TimeUnit::Nanos)));
        assert!(!i64::accepts(DType::Datetime64(TimeUnit::Nanos)));
    }
}
