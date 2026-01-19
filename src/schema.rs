//! Schema parsing for mmappet datasets.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::dtype::DType;
use crate::error::{MmappetError, Result};

/// A single column definition from the schema.
#[derive(Debug, Clone)]
pub struct ColumnDef {
    /// Position in schema (0, 1, 2...).
    pub index: usize,
    /// Column name.
    pub name: String,
    /// Data type.
    pub dtype: DType,
}

/// Parsed schema from schema.txt.
#[derive(Debug, Clone)]
pub struct Schema {
    columns: Vec<ColumnDef>,
    name_to_index: HashMap<String, usize>,
}

impl Schema {
    /// Parse schema from schema.txt content.
    ///
    /// Format: `{dtype} {colname}` per line (e.g., "uint32 tof")
    pub fn parse(content: &str) -> Result<Self> {
        let mut columns = Vec::new();
        let mut name_to_index = HashMap::new();

        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();

            // Skip empty lines
            if line.is_empty() {
                continue;
            }

            // Split into dtype and name
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 2 {
                return Err(MmappetError::SchemaParse {
                    line: line_num + 1,
                    message: format!("Expected 'dtype name', got: {}", line),
                });
            }

            let dtype = DType::from_str(parts[0])?;
            let name = parts[1].to_string();

            // Check for duplicates
            if name_to_index.contains_key(&name) {
                return Err(MmappetError::DuplicateColumnName(name));
            }

            let index = columns.len();
            name_to_index.insert(name.clone(), index);
            columns.push(ColumnDef { index, name, dtype });
        }

        Ok(Schema {
            columns,
            name_to_index,
        })
    }

    /// Load schema from a directory path.
    pub fn from_path<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let schema_path = dir.as_ref().join("schema.txt");
        if !schema_path.exists() {
            return Err(MmappetError::MissingSchema(dir.as_ref().to_path_buf()));
        }
        let content = fs::read_to_string(&schema_path)?;
        Self::parse(&content)
    }

    /// Get column definition by name.
    pub fn get(&self, name: &str) -> Option<&ColumnDef> {
        self.name_to_index.get(name).map(|&idx| &self.columns[idx])
    }

    /// Get column definition by index.
    pub fn get_by_index(&self, index: usize) -> Option<&ColumnDef> {
        self.columns.get(index)
    }

    /// Number of columns.
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// Check if schema is empty.
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// Iterate over column definitions.
    pub fn columns(&self) -> impl Iterator<Item = &ColumnDef> {
        self.columns.iter()
    }

    /// Get all column names.
    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|c| c.name.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_schema() {
        let content = "uint32 tof\nuint32 intensity\nfloat32 score\nfloat32 mz";
        let schema = Schema::parse(content).unwrap();

        assert_eq!(schema.len(), 4);
        assert_eq!(schema.column_names(), vec!["tof", "intensity", "score", "mz"]);

        let tof = schema.get("tof").unwrap();
        assert_eq!(tof.index, 0);
        assert_eq!(tof.dtype, DType::UInt32);

        let mz = schema.get("mz").unwrap();
        assert_eq!(mz.index, 3);
        assert_eq!(mz.dtype, DType::Float32);
    }

    #[test]
    fn test_parse_schema_with_empty_lines() {
        let content = "\nuint32 a\n\nfloat64 b\n";
        let schema = Schema::parse(content).unwrap();
        assert_eq!(schema.len(), 2);
    }

    #[test]
    fn test_parse_schema_duplicate_error() {
        let content = "uint32 col\nfloat32 col";
        let result = Schema::parse(content);
        assert!(matches!(result, Err(MmappetError::DuplicateColumnName(_))));
    }

    #[test]
    fn test_parse_schema_invalid_format() {
        let content = "invalid line format here";
        let result = Schema::parse(content);
        assert!(matches!(result, Err(MmappetError::SchemaParse { .. })));
    }
}
