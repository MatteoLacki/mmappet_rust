//! mmappet CLI - command line tool for inspecting mmappet datasets.

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use mmappet::{Dataset, TypedArrayView};

#[derive(Parser)]
#[command(name = "mmappet-cli")]
#[command(about = "Inspect mmappet datasets", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show dataset info (schema, row count)
    Info {
        /// Path to the mmappet dataset directory
        path: PathBuf,
    },

    /// Print first N rows of specified columns
    Head {
        /// Path to the mmappet dataset directory
        path: PathBuf,

        /// Number of rows to show
        #[arg(short, long, default_value = "10")]
        n: usize,

        /// Columns to show (comma-separated, or all if not specified)
        #[arg(short, long)]
        columns: Option<String>,
    },

    /// Show statistics for numeric columns
    Stats {
        /// Path to the mmappet dataset directory
        path: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Info { path } => cmd_info(&path),
        Commands::Head { path, n, columns } => cmd_head(&path, n, columns),
        Commands::Stats { path } => cmd_stats(&path),
    }
}

fn cmd_info(path: &PathBuf) -> Result<()> {
    let ds = Dataset::open(path)?;

    println!("Dataset: {}", path.display());
    println!("Rows: {}", ds.len());
    println!("Columns: {}", ds.num_columns());
    println!();
    println!("Schema:");
    for col_def in ds.schema().columns() {
        println!("  {:>2}. {} ({})", col_def.index, col_def.name, col_def.dtype);
    }

    Ok(())
}

fn cmd_head(path: &PathBuf, n: usize, columns: Option<String>) -> Result<()> {
    let ds = Dataset::open(path)?;

    let col_names: Vec<&str> = match &columns {
        Some(cols) => cols.split(',').map(|s| s.trim()).collect(),
        None => ds.schema().column_names(),
    };

    let n = n.min(ds.len());

    // Print header
    for (i, name) in col_names.iter().enumerate() {
        if i > 0 {
            print!("\t");
        }
        print!("{}", name);
    }
    println!();

    // Print rows
    for row_idx in 0..n {
        for (col_idx, name) in col_names.iter().enumerate() {
            if col_idx > 0 {
                print!("\t");
            }
            let col = &ds[*name];
            match col.as_typed_array() {
                TypedArrayView::UInt8(arr) => print!("{}", arr[row_idx]),
                TypedArrayView::Int8(arr) => print!("{}", arr[row_idx]),
                TypedArrayView::UInt16(arr) => print!("{}", arr[row_idx]),
                TypedArrayView::Int16(arr) => print!("{}", arr[row_idx]),
                TypedArrayView::UInt32(arr) => print!("{}", arr[row_idx]),
                TypedArrayView::Int32(arr) => print!("{}", arr[row_idx]),
                TypedArrayView::UInt64(arr) => print!("{}", arr[row_idx]),
                TypedArrayView::Int64(arr) => print!("{}", arr[row_idx]),
                TypedArrayView::Float32(arr) => print!("{:.6}", arr[row_idx]),
                TypedArrayView::Float64(arr) => print!("{:.6}", arr[row_idx]),
                TypedArrayView::Bool(arr) => print!("{}", arr[row_idx] != 0),
            }
        }
        println!();
    }

    Ok(())
}

fn cmd_stats(path: &PathBuf) -> Result<()> {
    let ds = Dataset::open(path)?;

    println!("Dataset: {}", path.display());
    println!("Rows: {}", ds.len());
    println!();

    for col_def in ds.schema().columns() {
        let col = &ds[&col_def.name];
        print!("{} ({}):", col_def.name, col_def.dtype);

        match col.as_typed_array() {
            TypedArrayView::UInt32(arr) => {
                let min = arr.iter().min().copied().unwrap_or(0);
                let max = arr.iter().max().copied().unwrap_or(0);
                let sum: u64 = arr.iter().map(|&x| x as u64).sum();
                let mean = sum as f64 / arr.len() as f64;
                println!(" min={}, max={}, mean={:.2}", min, max, mean);
            }
            TypedArrayView::UInt64(arr) => {
                let min = arr.iter().min().copied().unwrap_or(0);
                let max = arr.iter().max().copied().unwrap_or(0);
                let sum: u128 = arr.iter().map(|&x| x as u128).sum();
                let mean = sum as f64 / arr.len() as f64;
                println!(" min={}, max={}, mean={:.2}", min, max, mean);
            }
            TypedArrayView::Float32(arr) => {
                let min = arr.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = arr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f64 = arr.iter().map(|&x| x as f64).sum();
                let mean = sum / arr.len() as f64;
                println!(" min={:.6}, max={:.6}, mean={:.6}", min, max, mean);
            }
            TypedArrayView::Float64(arr) => {
                let min = arr.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = arr.iter().sum();
                let mean = sum / arr.len() as f64;
                println!(" min={:.6}, max={:.6}, mean={:.6}", min, max, mean);
            }
            _ => println!(" (stats not available for this type)"),
        }
    }

    Ok(())
}
