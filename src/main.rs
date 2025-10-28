use std::fs::File;
use std::path::Path;
use memmap2::Mmap;
use ndarray::{ArrayView1, s};
use std::env;
use anyhow::Result;
use bytemuck::cast_slice;

fn main() -> Result<()> {
    // Read the first command-line argument as the path
    let folder = env::args().nth(1)
        .ok_or_else(|| anyhow::anyhow!("Usage: cargo run -- <folder-path>"))?;


    let path_f32 = Path::new(&folder).join("0.bin");
    let path_u32 = Path::new(&folder).join("1.bin");

    // --- Open and memory-map bin.0 as f32 ---
    let file_f32 = File::open(&path_f32)?;
    let mmap_f32 = unsafe { Mmap::map(&file_f32)? };
    let data_f32: &[f32] = cast_slice(&mmap_f32[..]);
    let array_f32 = ArrayView1::from(data_f32);

    println!("Loaded {} f32 elements from {:?}", array_f32.len(), path_f32);
    println!("First few f32 elements: {:?}", array_f32.slice(s![..5]));
    let n_f32 = 3.min(array_f32.len());
    println!("Last {} f32 elements: {:?}", n_f32, array_f32.slice(s![array_f32.len()-n_f32..]));

    // --- Open and memory-map bin.1 as u32 ---
    let file_u32 = File::open(&path_u32)?;
    let mmap_u32 = unsafe { Mmap::map(&file_u32)? };
    let data_u32: &[u32] = cast_slice(&mmap_u32[..]);
    let array_u32 = ArrayView1::from(data_u32);

    println!("Loaded {} u32 elements from {:?}", array_u32.len(), path_u32);
    println!("First few u32 elements: {:?}", array_u32.slice(s![..5]));
    let n_u32 = 3.min(array_u32.len());
    println!("Last {} u32 elements: {:?}", n_u32, array_u32.slice(s![array_u32.len()-n_u32..]));

    Ok(())
}

