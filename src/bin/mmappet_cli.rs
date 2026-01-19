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

    /// Plot numeric column values as ASCII bars
    Plot {
        /// Path to the mmappet dataset directory
        path: PathBuf,

        /// Number of rows to show
        #[arg(short, long, default_value = "30")]
        n: usize,

        /// Column to plot (uses first numeric column if not specified)
        #[arg(short, long)]
        column: Option<String>,

        /// Width of the plot in characters
        #[arg(short, long, default_value = "60")]
        width: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Info { path } => cmd_info(&path),
        Commands::Head { path, n, columns } => cmd_head(&path, n, columns),
        Commands::Stats { path } => cmd_stats(&path),
        Commands::Plot { path, n, column, width } => cmd_plot(&path, n, column, width),
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

fn cmd_plot(path: &PathBuf, n: usize, column: Option<String>, width: usize) -> Result<()> {
    let ds = Dataset::open(path)?;

    // Find column to plot
    let col_name = match column {
        Some(name) => name,
        None => ds.schema().column_names().first()
            .ok_or_else(|| anyhow::anyhow!("Dataset has no columns"))?
            .to_string(),
    };

    let col = ds.column(&col_name)
        .ok_or_else(|| anyhow::anyhow!("Column not found: {}", col_name))?;

    let n = n.min(ds.len());

    // Extract values as f64 for plotting
    let values: Vec<f64> = match col.as_typed_array() {
        TypedArrayView::UInt8(arr) => arr.iter().take(n).map(|&x| x as f64).collect(),
        TypedArrayView::Int8(arr) => arr.iter().take(n).map(|&x| x as f64).collect(),
        TypedArrayView::UInt16(arr) => arr.iter().take(n).map(|&x| x as f64).collect(),
        TypedArrayView::Int16(arr) => arr.iter().take(n).map(|&x| x as f64).collect(),
        TypedArrayView::UInt32(arr) => arr.iter().take(n).map(|&x| x as f64).collect(),
        TypedArrayView::Int32(arr) => arr.iter().take(n).map(|&x| x as f64).collect(),
        TypedArrayView::UInt64(arr) => arr.iter().take(n).map(|&x| x as f64).collect(),
        TypedArrayView::Int64(arr) => arr.iter().take(n).map(|&x| x as f64).collect(),
        TypedArrayView::Float32(arr) => arr.iter().take(n).map(|&x| x as f64).collect(),
        TypedArrayView::Float64(arr) => arr.iter().take(n).copied().collect(),
        TypedArrayView::Bool(arr) => arr.iter().take(n).map(|&x| x as f64).collect(),
    };

    if values.is_empty() {
        println!("No data to plot");
        return Ok(());
    }

    // Find min/max for scaling
    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;

    // Print header
    println!("Column: {} ({})  Rows: 0..{}", col_name, col.dtype(), n);
    println!("Range: [{:.4}, {:.4}]", min_val, max_val);
    println!();

    // Calculate label width for alignment
    let max_idx_width = format!("{}", n - 1).len();
    let val_width = 12;

    // Plot each value as a horizontal bar
    for (i, &val) in values.iter().enumerate() {
        let bar_len = if range > 0.0 {
            ((val - min_val) / range * width as f64).round() as usize
        } else {
            width / 2
        };

        let bar: String = "█".repeat(bar_len);

        println!("{:>idx_w$} │ {:>val_w$.4} │{}",
            i, val, bar,
            idx_w = max_idx_width,
            val_w = val_width);
    }

    Ok(())
}
