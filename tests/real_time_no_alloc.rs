//! Real-time gate: once the state is built, `process_frame` must perform
//! **no heap allocation**.
//!
//! Exactly one `#[test]` in this file on purpose: `#[global_allocator]` is process-wide,
//! so a test running concurrently with another one would see that test's allocations. The
//! counter is scoped to the thread that owns the window (`OWNER`), which makes it robust
//! even under the default parallel harness.
//!
//! This is the enforceable half of the contract documented in `doc/realtime.md`: the frame
//! path uses fixed-size buffers, so a refactor that reaches for a `Vec`, a `Box`, a
//! `format!` or a lazy static inside `process_frame` fails here rather than in the field.
#![cfg(any(feature = "embedded-model", feature = "embedded-model-int8"))]
// These tests need a model; with `default-features = false` there is no default one, so
// they compile out rather than failing. The real-time gate itself still runs in the
// all-features job.

// Test island: `GlobalAlloc` is unsafe by trait definition, and the counting allocator is
// the whole point of this file.
#![allow(unsafe_code)]

// SAFETY: every method reads two already-initialised statics, increments a static atomic
// counter, and then delegates to the system allocator. Nothing is read through a pointer
// the allocator did not get from `System`, no layout is recomputed, and the counting check
// itself allocates nothing — which is what would make recursion possible.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use rnnoise::{warmup, DenoiseState, RnnModel, FRAME_SIZE};

static COUNT: AtomicUsize = AtomicUsize::new(0);
static ARMED: AtomicBool = AtomicBool::new(false);
static OWNER: OnceLock<std::thread::ThreadId> = OnceLock::new();

#[must_use]
fn counting_this_thread() -> bool {
    ARMED.load(Ordering::Relaxed) && Some(std::thread::current().id()) == OWNER.get().copied()
}

struct CountingAllocator;

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if counting_this_thread() {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if counting_this_thread() {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        System.realloc(ptr, layout, new_size)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if counting_this_thread() {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        System.alloc_zeroed(layout)
    }
}

const WARMUP_FRAMES: usize = 50;
const MEASURED_FRAMES: usize = 250;

/// An active (never-silent) frame: the worst case, because it runs the network.
#[must_use]
fn active_frame() -> [f32; FRAME_SIZE] {
    let mut seed = 1u32;
    let mut frame = [0.0f32; FRAME_SIZE];
    for (i, s) in frame.iter_mut().enumerate() {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let noise = (seed >> 8) as f32 / 16_777_216.0 - 0.5;
        *s = 3000.0 * (i as f32 * 0.05).sin() + 2000.0 * noise;
    }
    frame
}

#[must_use]
fn allocations_while_processing(
    st: &mut DenoiseState,
    frame: &[f32; FRAME_SIZE],
    out: &mut [f32; FRAME_SIZE],
    frames: usize,
) -> usize {
    let before = COUNT.load(Ordering::Relaxed);
    for _ in 0..frames {
        st.process_frame(out, frame);
    }
    COUNT.load(Ordering::Relaxed) - before
}

#[test]
fn process_frame_allocates_nothing_after_construction() {
    OWNER.set(std::thread::current().id()).ok();
    ARMED.store(true, Ordering::Relaxed);

    warmup();
    let frame = active_frame();
    let mut out = [0.0f32; FRAME_SIZE];

    // Construction is where the state is built, so the frame path must be clean from the
    // first frame on. The very first call on a fresh harness thread can include one-time
    // runtime bookkeeping on that thread — lazy TLS/CRT init, which is not the frame path
    // and which Windows CI showed as exactly 1 block on the first window. That is measured
    // with a documented tolerance; the strict claim is the second, identical call: if the
    // same code path allocates once and not the next time, the allocation was not the frame
    // path's.
    let mut st = DenoiseState::new();
    let cold = allocations_while_processing(&mut st, &frame, &mut out, 1);
    assert!(
        cold <= 1,
        "the first frame allocated {cold} times — more than one-time runtime bookkeeping"
    );
    assert_eq!(
        allocations_while_processing(&mut st, &frame, &mut out, 1),
        0,
        "the frame path after construction must not allocate"
    );

    for (label, model) in [
        ("float", Arc::new(RnnModel::default())),
        ("int8", Arc::new(RnnModel::default().quantized())),
    ] {
        let mut st = DenoiseState::new_with_model(model);
        // Warm up before the window so first-touch and page-fault costs are not counted.
        for _ in 0..WARMUP_FRAMES {
            st.process_frame(&mut out, &frame);
        }
        let during = allocations_while_processing(&mut st, &frame, &mut out, MEASURED_FRAMES);
        assert_eq!(
            during, 0,
            "{label} path allocated {during} times over {MEASURED_FRAMES} frames"
        );
    }

    #[cfg(feature = "legacy-model")]
    {
        use rnnoise::DenoiseStateV1;

        let mut st = DenoiseStateV1::new();
        for _ in 0..WARMUP_FRAMES {
            st.process_frame(&mut out, &frame);
        }
        let before = COUNT.load(Ordering::Relaxed);
        for _ in 0..MEASURED_FRAMES {
            st.process_frame(&mut out, &frame);
        }
        let during = COUNT.load(Ordering::Relaxed) - before;
        assert_eq!(
            during, 0,
            "legacy path allocated {during} times over {MEASURED_FRAMES} frames"
        );
    }

    ARMED.store(false, Ordering::Relaxed);
}
