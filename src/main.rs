use std::fs::File;
use std::mem;
use memmap2::Mmap;
use ndarray::{ArrayView1, s};

fn main() -> anyhow::Result<()> {
    // Path to one binary column file
    let path = "dataset/0.bin";

    // Open and memory-map the file
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };

    // Interpret the bytes as a slice of f32 values
    let bytes = &mmap[..];
    let len = bytes.len() / mem::size_of::<f32>();
    let data: &[f32] = unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len)
    };

    // Wrap as ndarray view (zero-copy)
    let array = ArrayView1::from(data);
    println!("Loaded {} elements from {}", array.len(), path);
    println!("First few elements: {:?}", &array.slice(s![..5]));

    Ok(())
}

