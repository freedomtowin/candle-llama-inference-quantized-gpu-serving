#![allow(unused)]
use anyhow::{Context, Result};
use std::io::Write;
use std::path::PathBuf;
use std::env; // Ensure std::env is in scope

const CL_EXE_PATH: &str = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.38.33130/bin/Hostx64/x64";

struct KernelDirectories {
    kernel_glob: &'static str,
    rust_target: &'static str,
    include_dirs: &'static [&'static str],
}

const KERNEL_DIRS: [KernelDirectories; 1] = [KernelDirectories {
    kernel_glob: "examples/custom-ops/kernels/*.cu",
    rust_target: "examples/custom-ops/cuda_kernels.rs",
    include_dirs: &[],
}];

fn main() -> Result<()> {
    // Get the current PATH environment variable
    let mut path = env::var("PATH").unwrap_or_else(|_| String::new());

    // Append the new path to the existing PATH
    path.push_str(";");
    path.push_str(CL_EXE_PATH);

    // Set the updated PATH environment variable
    env::set_var("PATH", path);

    // Ensure the build script is rerun if `build.rs` changes
    println!("cargo:rerun-if-changed=build.rs");

    // CUDA feature gating
    #[cfg(feature = "cuda")]
    {
        for kdir in KERNEL_DIRS.iter() {
            let builder = bindgen_cuda::Builder::default().kernel_paths_glob(kdir.kernel_glob);
            println!("cargo:info={:?}", builder);
            let bindings = builder.build_ptx().unwrap();
            bindings.write(kdir.rust_target).unwrap();
        }
    }
    Ok(())
}

