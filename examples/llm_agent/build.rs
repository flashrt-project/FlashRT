// Link the FlashRT execution-contract C ABI (libflashrt_exec).
// Point FLASHRT_EXEC_LIB_DIR at the directory holding libflashrt_exec.so
// (e.g. <repo>/exec/build).
fn main() {
    if let Ok(dir) = std::env::var("FLASHRT_EXEC_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
    }
    println!("cargo:rustc-link-lib=dylib=flashrt_exec");
}
