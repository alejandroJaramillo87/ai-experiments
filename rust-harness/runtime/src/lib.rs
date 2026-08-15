/*
 * runtime — inference engine library crate.
 *
 * Linked into the harness binary; owns model weights, the KV cache, sampling,
 * and grammar-constrained decoding. The harness talks to it through `Session`
 * handles over the `Model` trait, so everything above the forward pass builds
 * and tests without a GPU.
 */

// This crate produces the training and interp numerics, so it holds the
// strongest guarantee the language offers: no unsafe, and no way to write an
// `allow` that admits one.
//
// It was `deny` between P9a and P10.7, for a single memory map in
// `checkpoint::CheckpointFiles::read` -- `Mmap::map` is unsafe and has no safe
// wrapper, and `forbid` cannot be lifted at a call site. That map was there to
// avoid a private copy of a 15 GB checkpoint on top of the loader's own
// buffers. It stopped earning that once the storage-preserving loader began
// copying every tensor out regardless, and positioned reads now do the same
// job with less resident memory. Removing it also removed the one obligation
// the SAFETY comment could not discharge: a file rewritten underneath a live
// mapping is undefined behaviour, while a file rewritten underneath a `pread`
// is an error value.
//
// The GPU backend will reach hardware through Candle, which owns that FFI.
#![forbid(unsafe_code)]

pub mod align;
pub mod checkpoint;
pub mod conformance;
pub mod digest;
pub mod envelope;
pub mod eval;
pub mod grammar;
pub mod kv;
pub mod ladder;
pub mod metrics;
pub mod model;
pub mod npz;
pub mod pyenv;
pub mod reference;
pub mod sampler;
pub mod session;
pub mod status;
pub mod synth;
pub mod tensor;
pub mod testlog;
pub mod tokenizer;

/// Peak resident set size in bytes, from `/proc/self/status`'s `VmHWM`.
///
/// Reported at exit so a memory claim is checkable from the run's own output
/// rather than from remembering to wrap it in `/usr/bin/time -v`. The repo has
/// twice reasoned about resident weights from a prediction and been wrong; a
/// number printed by every run is what turns that into a measurement nobody
/// has to opt into.
///
/// `VmHWM` is the high-water mark, not the current size, and it counts mapped
/// file pages while they are resident. That is the wanted behaviour: the
/// checkpoint mapping is exactly what this is watching for, and a metric that
/// excluded it would have shown no change across the two loader fixes.
///
/// `None` off Linux, and on a Linux without `/proc`. A caller prints nothing
/// rather than guessing -- an absent measurement must not read as a small one.
pub fn peak_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let field = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    // "VmHWM:\t   29524612 kB" -- the kernel states the unit and has always
    // stated kB, but parse it rather than assume it.
    let mut parts = field.split_whitespace().skip(1);
    let value: u64 = parts.next()?.parse().ok()?;
    match parts.next() {
        Some("kB") => Some(value * 1024),
        Some(_) | None => None,
    }
}

/// Resident set size *now*, in bytes, from `/proc/self/status` (`VmRSS`).
///
/// The companion to `peak_rss_bytes`, and not a substitute for it. `VmHWM` is
/// a high-water mark, so a series built from it can only ever climb; a run
/// that loads 14 GiB and then drops the checkpoint mapping renders as a
/// staircase that never comes back down. This is the figure a memory chart
/// wants, because the fall is the part worth seeing.
///
/// `None` off Linux, and on a Linux without `/proc`, for the same reason as
/// its companion: an absent measurement must not read as a small one.
pub fn resident_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let field = status.lines().find(|l| l.starts_with("VmRSS:"))?;
    let mut parts = field.split_whitespace().skip(1);
    let value: u64 = parts.next()?.parse().ok()?;
    match parts.next() {
        Some("kB") => Some(value * 1024),
        Some(_) | None => None,
    }
}

#[cfg(test)]
mod peak_rss_tests {
    /// A live process has touched pages, so the floor is "more than nothing".
    /// The upper bound is deliberately loose -- this pins the parse and the
    /// unit conversion, not the allocator's behaviour under a test harness.
    #[test]
    #[cfg(target_os = "linux")]
    fn peak_rss_reports_a_plausible_live_figure() {
        let bytes = super::peak_rss_bytes().expect("Linux with /proc reports VmHWM");
        assert!(
            (1 << 20..1 << 40).contains(&bytes),
            "VmHWM of {bytes} bytes is outside anything a test process could hold, \
             which means the unit conversion is wrong rather than the memory"
        );
    }

    /// The resident figure parses, and sits at or below the high-water mark.
    ///
    /// The ordering is the assertion worth making: it is the property that
    /// distinguishes the two fields, so reading VmHWM into both -- the
    /// copy-paste this function invites -- would still pass a bounds check
    /// and fail here only if the bound were `<`. It is not: the two are equal
    /// in a process that has never released a page, which is the common case
    /// for a short test.
    #[test]
    #[cfg(target_os = "linux")]
    fn resident_rss_is_never_above_the_peak() {
        let now = super::resident_rss_bytes().expect("Linux with /proc reports VmRSS");
        let peak = super::peak_rss_bytes().expect("Linux with /proc reports VmHWM");
        assert!(
            (1 << 20..1 << 40).contains(&now),
            "VmRSS of {now} bytes is outside anything a test process could hold"
        );
        assert!(now <= peak, "resident {now} exceeds peak {peak}");
    }
}
