//! Command-line denoiser, equivalent to upstream `examples/rnnoise_demo`.
//!
//! Operates on RAW 16-bit (machine-endian) mono PCM at 48 kHz:
//!
//! ```text
//! rnnoise-demo <noisy.raw> <denoised.raw>
//! ```
//!
//! As with the C demo, the first frame of output is dropped (it is all-zero
//! due to the algorithm's one-frame look-ahead).

use std::env;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::process::ExitCode;

use rnnoise::{DenoiseState, FRAME_SIZE};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <noisy speech> <output denoised>", args[0]);
        eprintln!("  (RAW 16-bit machine-endian mono PCM @ 48 kHz, not WAV)");
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

fn run(in_path: &str, out_path: &str) -> io::Result<()> {
    let mut reader = BufReader::new(File::open(in_path)?);
    let mut writer = BufWriter::new(File::create(out_path)?);
    let mut st = DenoiseState::new();

    let mut in_buf = [0.0f32; FRAME_SIZE];
    let mut out_buf = [0.0f32; FRAME_SIZE];
    let mut bytes = [0u8; FRAME_SIZE * 2];
    let mut first = true;

    loop {
        match read_full(&mut reader, &mut bytes)? {
            n if n < bytes.len() => break, // upstream stops at the first short/empty read
            _ => {}
        }
        for (i, c) in bytes.chunks_exact(2).enumerate() {
            in_buf[i] = i16::from_ne_bytes([c[0], c[1]]) as f32;
        }
        st.process_frame(&mut out_buf, &in_buf);
        if !first {
            let mut ob = [0u8; FRAME_SIZE * 2];
            for (i, s) in out_buf.iter().enumerate() {
                // Match the C demo: plain float->short conversion (truncates toward zero).
                let v = *s as i16;
                ob[2 * i..2 * i + 2].copy_from_slice(&v.to_ne_bytes());
            }
            writer.write_all(&ob)?;
        }
        first = false;
    }
    writer.flush()
}

fn read_full(r: &mut impl Read, buf: &mut [u8]) -> io::Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match r.read(&mut buf[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    Ok(filled)
}
