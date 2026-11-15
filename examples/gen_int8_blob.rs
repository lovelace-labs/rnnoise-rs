//! Regenerate `models/rnnoise_default_int8.bin` from the committed float blob.
//!
//! Everything here goes through the crate's own code: the quantization is
//! `RnnModel::quantized()` and the serialization is `RnnModel::to_bytes()`, so the artifact is
//! exactly what those two produce. `tests/model_artifacts.rs` asserts the committed file against
//! that, which is what keeps the tool and the library from drifting apart.
//!
//! ```text
//! cargo run --release --example gen_int8_blob -- \
//!     models/rnnoise_default.bin models/rnnoise_default_int8.bin
//! ```
//!
//! The output is the `RNNS` container, so a loaded model is version- and checksum-verified.

use std::env;
use std::process::ExitCode;

use rnnoise::RnnModel;

/// `RNNS` container header: magic + format version + record count + checksum.
const HEADER_SIZE: usize = 16;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <float blob> <int8 blob out>", args[0]);
        return ExitCode::FAILURE;
    }
    match run(&args[1], &args[2]) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(input: &str, output: &str) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = std::fs::read(input)?;
    let model = RnnModel::from_bytes(&bytes)?.quantized();
    let blob = model.to_bytes();
    std::fs::write(output, &blob)?;
    let record_count = u32::from_le_bytes([blob[8], blob[9], blob[10], blob[11]]);
    println!(
        "wrote {output}: {} bytes, {record_count} records, record-stream checksum {:#010x}",
        blob.len(),
        rnnoise::crc32(&blob[HEADER_SIZE..]),
    );
    Ok(())
}
