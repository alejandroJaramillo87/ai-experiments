/*
 * tensor — tensor payload access and dequantization to f32.
 *
 * One consumer now: the safetensors checkpoint loader, which widens through
 * the two decoders below and builds the `Weight` values beside them. The
 * GGUF-side helpers -- `element_count`, `byte_size`, `dequantize`,
 * `tensor_bytes`, `load_tensor` and the GGML storage-type ids -- went with
 * R10, which retired the last GGUF artifact this workspace read.
 *
 * The CPU backend computes in plain f32, so a narrower storage type widens at
 * load time. Supported types are F32, F16 and BF16, the ones Gemma 4
 * checkpoints actually use; anything else is a hard error naming the tensor,
 * raised by the loader, so a new dtype shows up at load time rather than as
 * silent garbage.
 *
 * BF16 is the published storage type for Gemma 4 and the format the training
 * tier produces, so widening it to f32 is lossless: the checkpoint's own
 * values reach the forward pass unrounded.
 */

/// bfloat16 to f32. Exact and total, for every input.
///
/// bfloat16 is the top 16 bits of an f32: one sign bit, the same 8-bit
/// exponent, and 7 of the 23 mantissa bits. Widening is therefore a shift,
/// with none of the exponent rebiasing and subnormal renormalization that
/// `f16_to_f32` needs — normals, subnormals, infinities and NaN payloads all
/// map by construction. Do not route bfloat16 through `f16_to_f32`: IEEE
/// half splits 1/5/10 where bfloat16 splits 1/8/7, so the two formats share
/// a width and nothing else.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// IEEE 754 half-precision to f32, including subnormals, infinities, NaN.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;
    let word = match (exponent, mantissa) {
        (0, 0) => sign,
        // Subnormal: renormalize into the f32 exponent range.
        (0, m) => {
            let shift = m.leading_zeros() - 21; // bits above the 10-bit field
            let m = (m << shift) & 0x3ff;
            // Top set bit at position 10 - shift gives exponent 2^(-15 + (10 - shift) - 10).
            sign | ((113 - shift) << 23) | (m << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(word)
}

/// Independent partial sums in `Weight::row_dot_lanes`.
///
/// Sixteen and not eight because sixteen f32 is one zmm, two ymm, or four xmm:
/// the same sixteen accumulators exist whatever vector width the backend
/// chooses, so the arithmetic does not change with `-C target-cpu`. Eight
/// would share that property; sixteen additionally gives four independent
/// chains on the baseline build the sandbox gate compiles, where the loop does
/// not vectorize at all and instruction-level parallelism is all there is.
///
/// Not a tunable, and deliberately not reachable from a command line. A
/// selectable lane count is a selectable answer; changing it is a commit.
pub const DOT_LANES: usize = 16;

/// The lane-partitioned fold behind `Weight::row_dot_lanes`.
fn lane_fold<T: Copy>(w: &[T], x: &[f32], widen: impl Fn(T) -> f32) -> f32 {
    let mut acc = [0.0f32; DOT_LANES];
    let mut wc = w.chunks_exact(DOT_LANES);
    let mut xc = x.chunks_exact(DOT_LANES);
    for (wb, xb) in wc.by_ref().zip(xc.by_ref()) {
        for l in 0..DOT_LANES {
            acc[l] += widen(wb[l]) * xb[l];
        }
    }
    // The tail keeps its own accumulator and joins last, so the result does
    // not depend on `w.len() % DOT_LANES`.
    let mut tail = 0.0f32;
    for (&a, b) in wc.remainder().iter().zip(xc.remainder()) {
        tail += widen(a) * b;
    }
    // Lane 0 through lane 15, left to right, as an accumulating loop rather
    // than `acc.iter().sum()` -- both compile the same today, but one of them
    // says the order is the contract rather than the current behaviour of a
    // library function.
    let mut total = 0.0f32;
    for a in acc {
        total += a;
    }
    total + tail
}

/// The block fold behind `Weight::row_dot_block`.
///
/// `R` is a const parameter rather than a runtime length because the whole
/// effect depends on the inner loop being fully unrolled into `R` independent
/// accumulators that the backend can then place in vector lanes. A runtime
/// count compiles to a loop over an array and reverts to scalar.
///
/// The rows are sliced **before** the `k` loop, and that is load-bearing
/// rather than tidy: indexing `w[(o + j) * n + k]` inside the loop keeps a
/// bounds check per access that the vectorizer will not hoist, and the whole
/// body stays scalar. Measured, not guessed -- the two spellings differ by
/// every packed instruction in the loop.
#[inline]
fn block_fold<T: Copy, const R: usize>(
    w: &[T],
    o: usize,
    x: &[f32],
    widen: impl Fn(T) -> f32,
    out: &mut [f32; R],
) {
    let n = x.len();
    let rows: [&[T]; R] = std::array::from_fn(|j| &w[(o + j) * n..(o + j + 1) * n]);
    let mut acc = [0.0f32; R];
    for (k, &xk) in x.iter().enumerate() {
        for j in 0..R {
            acc[j] += widen(rows[j][k]) * xk;
        }
    }
    *out = acc;
}

/// `block_fold` over `R` rows that a caller has already widened to f32.
///
/// **Bit-identical to `Weight::row_dot_block` by shared implementation rather
/// than by argument.** This is the same `block_fold` the `F32` arm of that
/// method runs, over the same values in the same k-outer order, into the same
/// `R` accumulators. There are not two kernels here to reason about, only two
/// callers of one.
///
/// The values are the same because widening is exact: `bf16_to_f32` is a shift
/// and `f16_to_f32` lands 10 mantissa bits in a 23-bit field, so converting a
/// stored element before the fold and converting it inside the fold produce
/// the same f32. That is the argument already made above for widening at use
/// versus at load, applied one level down.
///
/// The reason a caller would want this is that the conversion is otherwise
/// repeated: a batched matmul folding `R` weight rows against `n` positions
/// widens each stored element `n` times to use it `n` times. See
/// `model::cpu::fill_panel_widened`.
pub fn widened_row_dot_block<const R: usize>(rows: &[f32], x: &[f32], out: &mut [f32; R]) {
    debug_assert_eq!(rows.len(), R * x.len(), "widened block is R whole rows");
    block_fold(rows, 0, x, |a| a, out);
}

/// A weight matrix held in the width its checkpoint published.
///
/// **Widening at use is bit-identical to widening at load**, which is what
/// makes this a memory and bandwidth decision rather than a numerics one.
/// `bf16_to_f32` is a 16-bit shift and `f16_to_f32` lands 10 mantissa bits in a
/// 23-bit field, so both are exact for every input; the f32 a fold multiplies
/// is the same f32 either way, in the same order. The same argument
/// `cpu::par_rows` makes about splitting rows, applied to storage.
///
/// The point is bytes moved. Decode reads every weight once per token and is
/// bound by memory bandwidth, not arithmetic, so halving the width of the
/// weights nearly halves the time. Holding a bf16 checkpoint at f32 also costs
/// a second copy of it on the heap, which is what took E4B to 42 GB resident
/// for a 15 GB file.
///
/// **Storage-preserving, never narrowing.** An f32 checkpoint stays f32. This
/// type records what a file declared; it does not convert between widths, and
/// a loader that narrowed an f32 tensor to bf16 to save memory would be
/// changing the model.
#[derive(Debug, Clone, PartialEq)]
pub enum Weight {
    F32(Vec<f32>),
    F16(Vec<u16>),
    Bf16(Vec<u16>),
}

impl Weight {
    /// Element count, whatever the width.
    pub fn len(&self) -> usize {
        match self {
            Weight::F32(v) => v.len(),
            Weight::F16(v) | Weight::Bf16(v) => v.len(),
        }
    }

    /// Whether the tensor holds no elements.
    ///
    /// Present because a shared-KV layer leaves `wk`, `wv` empty rather than
    /// allocating weights the modeling code never evaluates.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// One element, widened.
    ///
    /// For the readers that walk a row alongside another buffer and have no
    /// f32 slice to widen into: the packed per-layer embedding table is summed
    /// straight into the projected rows.
    #[inline]
    pub fn get(&self, i: usize) -> f32 {
        match self {
            Weight::F32(v) => v[i],
            Weight::F16(v) => f16_to_f32(v[i]),
            Weight::Bf16(v) => bf16_to_f32(v[i]),
        }
    }

    /// Dot `x` with row `o` of a row-major `[out, in]` matrix.
    ///
    /// One sequential fold over `x.len()` in index order — identical to the
    /// f32 loop this replaced, operand for operand. The dtype is matched once
    /// per row, never per element, so the inner loop stays a plain zip.
    #[inline]
    pub fn row_dot(&self, o: usize, x: &[f32]) -> f32 {
        let (lo, hi) = (o * x.len(), (o + 1) * x.len());
        match self {
            Weight::F32(v) => v[lo..hi].iter().zip(x).map(|(a, b)| a * b).sum(),
            Weight::F16(v) => v[lo..hi]
                .iter()
                .zip(x)
                .map(|(&a, b)| f16_to_f32(a) * b)
                .sum(),
            Weight::Bf16(v) => v[lo..hi]
                .iter()
                .zip(x)
                .map(|(&a, b)| bf16_to_f32(a) * b)
                .sum(),
        }
    }

    /// Dot `x` with row `o`, accumulating into `DOT_LANES` partial sums.
    ///
    /// **Not bit-identical to `row_dot`, and that is the entire point of it
    /// existing separately.** Float addition is not associative, so sixteen
    /// interleaved accumulators give a different last bit from one sequential
    /// fold. Both are correct; they are different correct answers. `row_dot`
    /// remains the reference the golden tests pin, and a caller reaches this
    /// only through `model::cpu::DotKind::Lanes16`.
    ///
    /// This shape is what a vector unit can execute. A sequential `.sum()` is
    /// a dependency chain LLVM will not vectorize without fast-math, so the
    /// bf16 widening is paid per element; with sixteen independent chains the
    /// same loop compiles to a widen-multiply-accumulate over whole vectors.
    /// Verified rather than assumed -- under the pinned toolchain with
    /// `-C target-cpu=native` the bf16 arm emits `vpmovzxwd` / `vpslld` /
    /// `vmulps` / `vaddps`, and without it stays scalar. Loop shape alone
    /// changes nothing; the target matters too.
    ///
    /// Accuracy moves in the same direction as speed, which is worth stating
    /// because the opposite is the usual assumption. Naive sequential
    /// summation accumulates error as O(n * eps); k independent accumulators
    /// bound it by O((n/k + k) * eps). For a 4096-wide row at sixteen lanes
    /// that is a tighter bound, not a looser one.
    ///
    /// Two ordering rules are fixed here in source rather than left to the
    /// vectorizer, so the result does not vary with `-C target-cpu`, with
    /// thread width, or between machines:
    ///
    /// - **The tail** -- elements past the last full group of `DOT_LANES` --
    ///   folds into its own accumulator in index order and joins at the very
    ///   end. Folding it back into `acc[0..r]` instead would make the answer
    ///   depend on `in_dim % DOT_LANES`, so the same weights zero-padded to a
    ///   round width would produce different logits.
    /// - **The lanes combine left to right**, lane 0 through lane 15, written
    ///   as an explicit loop. LLVM may not reassociate it, so the horizontal
    ///   reduction is pinned rather than whatever tree the backend prefers.
    #[inline]
    pub fn row_dot_lanes(&self, o: usize, x: &[f32]) -> f32 {
        let (lo, hi) = (o * x.len(), (o + 1) * x.len());
        match self {
            Weight::F32(v) => lane_fold(&v[lo..hi], x, |a| a),
            Weight::F16(v) => lane_fold(&v[lo..hi], x, f16_to_f32),
            Weight::Bf16(v) => lane_fold(&v[lo..hi], x, bf16_to_f32),
        }
    }

    /// Dot `x` with rows `o..o + R` at once, one accumulator per row.
    ///
    /// **Bit-identical to calling `row_dot` on each of the `R` rows**, and it
    /// is the same argument `par_rows` makes one level up: nothing is
    /// reordered. Accumulator `j` sums row `j`'s products over `k` in index
    /// order, exactly as the single-row fold does. What changes is only that
    /// `R` of those independent folds are in flight together, so the machine
    /// has something to issue while each one waits on its own add.
    ///
    /// That is the difference from `row_dot_lanes`, and it is the whole reason
    /// this exists. Both turn one dependency chain into several; only this one
    /// does it without touching summation order, because the independence was
    /// already there in the problem -- separate output rows never shared an
    /// accumulator to begin with.
    ///
    /// The backend places the `R` accumulators in the lanes of a vector
    /// register and broadcasts `x[k]` across them, so a fold that is
    /// bit-identical by construction still compiles to packed multiply-add.
    /// Verified under the pinned toolchain: at `R` of 4, 8 and 16 the bf16 arm
    /// emits no scalar float instruction at all, and unlike `row_dot_lanes` it
    /// vectorizes on the baseline build too, with no `-C target-cpu` needed.
    ///
    /// Rows past the last full block are the caller's problem, and they need
    /// no rule: `row_dot` on the remainder gives the same bits, so a partial
    /// block is a fallback rather than a special case.
    #[inline]
    pub fn row_dot_block<const R: usize>(&self, o: usize, x: &[f32], out: &mut [f32; R]) {
        match self {
            Weight::F32(v) => block_fold(v, o, x, |a| a, out),
            Weight::F16(v) => block_fold(v, o, x, f16_to_f32, out),
            Weight::Bf16(v) => block_fold(v, o, x, bf16_to_f32, out),
        }
    }

    /// Widen row `o` of a row-major `[_, dst.len()]` matrix into `dst`.
    ///
    /// For the callers that need the values themselves rather than a product:
    /// the token embedding lookup and the per-layer embedding table.
    pub fn widen_row_into(&self, o: usize, dst: &mut [f32]) {
        let (lo, hi) = (o * dst.len(), (o + 1) * dst.len());
        match self {
            Weight::F32(v) => dst.copy_from_slice(&v[lo..hi]),
            Weight::F16(v) => {
                for (d, &s) in dst.iter_mut().zip(&v[lo..hi]) {
                    *d = f16_to_f32(s);
                }
            }
            Weight::Bf16(v) => {
                for (d, &s) in dst.iter_mut().zip(&v[lo..hi]) {
                    *d = bf16_to_f32(s);
                }
            }
        }
    }

    /// Bytes of heap this tensor occupies, for the loader's resident report.
    pub fn stored_bytes(&self) -> usize {
        match self {
            Weight::F32(v) => v.len() * 4,
            Weight::F16(v) | Weight::Bf16(v) => v.len() * 2,
        }
    }

    /// The whole tensor widened to f32, for tests and diagnostics.
    ///
    /// Never called on a hot path: it allocates the expansion this type exists
    /// to avoid.
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            Weight::F32(v) => v.clone(),
            Weight::F16(v) => v.iter().map(|&b| f16_to_f32(b)).collect(),
            Weight::Bf16(v) => v.iter().map(|&b| bf16_to_f32(b)).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_decodes_normals_specials_and_subnormals() {
        // Literal bit patterns rather than an encoder round trip: this crate
        // only ever decodes f16, so a round trip would have proven the two
        // directions agreed with each other rather than with IEEE 754.
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xb800), -0.5);
        assert_eq!(f16_to_f32(0x7bff), 65504.0); // f16 max
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        // Smallest positive subnormal: 2^-24.
        assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24));
        // Largest subnormal: (1023/1024) * 2^-14.
        assert_eq!(f16_to_f32(0x03ff), (1023.0 / 1024.0) * 2.0f32.powi(-14));
    }

    #[test]
    fn bf16_widening_is_a_shift_for_every_input() {
        // The whole contract. bfloat16 is the high half of an f32, so the
        // widened bits must equal the input shifted left by 16 — no rounding,
        // no special case. Strided so the sweep stays a unit test.
        for bits in (0..=u16::MAX).step_by(7) {
            assert_eq!(
                bf16_to_f32(bits).to_bits(),
                (bits as u32) << 16,
                "bf16 {bits:#06x} did not widen by shift"
            );
        }
    }

    #[test]
    fn bf16_decodes_values_infinities_and_subnormals() {
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xbfc0), -1.5);
        assert_eq!(bf16_to_f32(0x7f80), f32::INFINITY);
        assert_eq!(bf16_to_f32(0xff80), f32::NEG_INFINITY);
        assert!(bf16_to_f32(0xffc1).is_nan());
        // An f32 subnormal: no renormalization needed, unlike IEEE half.
        assert_eq!(bf16_to_f32(0x0001).to_bits(), 1u32 << 16);
        // The two formats share a width and nothing else: this bit pattern
        // is 1.0 as bfloat16 and 0.00097... as IEEE half.
        assert_ne!(bf16_to_f32(0x3f80), f16_to_f32(0x3f80));
    }

    /// A row of plausible activation and weight magnitudes, deterministic and
    /// in-tree. Values straddle zero so cancellation is exercised: a fold that
    /// only ever adds same-signed terms would hide an ordering difference.
    fn probe_row(n: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1 << 24) as f32) - 0.5
        };
        (
            (0..n).map(|_| next()).collect(),
            (0..n).map(|_| next()).collect(),
        )
    }

    /// `row_dot_lanes` accumulates exactly the way its doc comment says:
    /// sixteen interleaved lanes, a separate tail, lanes combined left to
    /// right. Written as an independent transcription rather than by calling
    /// `lane_fold`, so a change to the ordering rules fails here instead of
    /// silently redefining the contract.
    #[test]
    fn row_dot_lanes_accumulates_in_the_order_it_documents() {
        // 37 is not a multiple of 16, so the tail rule is exercised; 64 is,
        // so the tail path is empty and must contribute nothing.
        for n in [37usize, 64] {
            let (w, x) = probe_row(n, 7);
            let mut acc = [0.0f32; DOT_LANES];
            let full = n / DOT_LANES * DOT_LANES;
            for base in (0..full).step_by(DOT_LANES) {
                for l in 0..DOT_LANES {
                    acc[l] += w[base + l] * x[base + l];
                }
            }
            let mut tail = 0.0f32;
            for i in full..n {
                tail += w[i] * x[i];
            }
            let mut want = 0.0f32;
            for a in acc {
                want += a;
            }
            want += tail;

            let f32_weight = Weight::F32(w.clone());
            assert_eq!(
                f32_weight.row_dot_lanes(0, &x),
                want,
                "n={n}: the lane fold departed from its documented order"
            );

            // Same claim for the widening arms, which must differ from the
            // f32 arm only in the value each element widens to.
            let bf16: Vec<u16> = w.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            let widened: Vec<f32> = bf16.iter().map(|&b| bf16_to_f32(b)).collect();
            assert_eq!(
                Weight::Bf16(bf16).row_dot_lanes(0, &x),
                Weight::F32(widened).row_dot_lanes(0, &x),
                "n={n}: the bf16 lane fold is not the widened f32 lane fold"
            );
        }
    }

    /// The claim `row_dot_block` is built on, asserted the only way it can
    /// honestly be asserted: exact equality, per row, per dtype. A tolerance
    /// here would be admitting the property does not hold.
    #[test]
    fn a_block_of_rows_is_bit_identical_to_the_rows_folded_one_at_a_time() {
        const IN: usize = 67; // not a multiple of any block width tested
        const OUT: usize = 16;
        let (w, x) = probe_row(IN * OUT, 11);
        let x = &x[..IN];

        let bf16: Vec<u16> = w.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let f16: Vec<u16> = w.iter().map(|v| (v.to_bits() >> 17) as u16).collect();
        let cases = [
            ("f32", Weight::F32(w.clone())),
            ("bf16", Weight::Bf16(bf16)),
            ("f16", Weight::F16(f16)),
        ];

        for (name, weight) in cases {
            // Every block origin, not just zero: an off-by-one in the row
            // slicing would still pass at o = 0.
            for o in [0usize, 1, 4, 7] {
                let mut got = [0.0f32; 4];
                weight.row_dot_block::<4>(o, x, &mut got);
                for (j, g) in got.iter().enumerate() {
                    assert_eq!(
                        *g,
                        weight.row_dot(o + j, x),
                        "{name}: block of 4 at row {o} disagreed with row_dot on row {}",
                        o + j
                    );
                }
            }
            let mut wide = [0.0f32; 8];
            weight.row_dot_block::<8>(0, x, &mut wide);
            for (j, g) in wide.iter().enumerate() {
                assert_eq!(
                    *g,
                    weight.row_dot(j, x),
                    "{name}: block of 8 differed at row {j}"
                );
            }
        }
    }

    /// Widening before the fold and widening inside it are the same fold.
    ///
    /// Asserted directly, one level below the batched matmul that motivates
    /// it, so a failure names the conversion rather than the panel machinery.
    /// Exact equality for the same reason as the block test above: a tolerance
    /// would be conceding the property does not hold.
    #[test]
    fn a_widened_block_is_bit_identical_to_folding_the_stored_rows() {
        const IN: usize = 67;
        const OUT: usize = 16;
        let (w, x) = probe_row(IN * OUT, 23);
        let x = &x[..IN];

        let bf16: Vec<u16> = w.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let f16: Vec<u16> = w.iter().map(|v| (v.to_bits() >> 17) as u16).collect();
        let cases = [
            ("f32", Weight::F32(w.clone())),
            ("bf16", Weight::Bf16(bf16)),
            ("f16", Weight::F16(f16)),
        ];

        for (name, weight) in cases {
            // Off zero, so a block origin dropped somewhere in the widening
            // cannot pass by folding row 0 four times.
            for o in [0usize, 1, 7] {
                let mut wide = vec![0.0f32; 4 * IN];
                for j in 0..4 {
                    weight.widen_row_into(o + j, &mut wide[j * IN..(j + 1) * IN]);
                }
                let (mut stored, mut hoisted) = ([0.0f32; 4], [0.0f32; 4]);
                weight.row_dot_block::<4>(o, x, &mut stored);
                widened_row_dot_block::<4>(&wide, x, &mut hoisted);
                assert_eq!(
                    stored, hoisted,
                    "{name}: hoisting the widen changed the fold at row {o}"
                );
            }
        }
    }

    /// The numerics claim, and it is a tolerance because a tolerance is the
    /// honest assertion: the two folds are different correct answers, not the
    /// same answer computed twice.
    #[test]
    fn row_dot_lanes_agrees_with_row_dot_to_within_rounding() {
        // 4096 is E4B's embedding width doubled -- long enough that sequential
        // summation's O(n * eps) error growth is visible if it is going to be.
        let (w, x) = probe_row(4096, 19);
        let weight = Weight::F32(w);
        let (seq, lanes) = (weight.row_dot(0, &x), weight.row_dot_lanes(0, &x));
        // Relative to the magnitude of the terms summed, not to the total: the
        // total is a near-cancelling sum of ~4096 signed products, so its own
        // magnitude is small and a relative bound against it would be
        // meaningless.
        let scale: f32 = match &weight {
            Weight::F32(v) => v.iter().zip(&x).map(|(a, b)| (a * b).abs()).sum(),
            _ => unreachable!(),
        };
        let rel = (seq - lanes).abs() / scale;
        assert!(
            rel < 1e-6,
            "sequential {seq} and lane {lanes} differ by {rel} of the summed magnitude, \
             which is more than rounding can explain"
        );
    }

    /// Sixteen lanes must not become sixteen lanes plus whatever the tail
    /// happened to be: a row of zeros in the tail is the degenerate case that
    /// would pass a weaker test.
    #[test]
    fn a_lane_fold_of_one_short_row_is_all_tail() {
        let (w, x) = probe_row(5, 3);
        let mut want = 0.0f32;
        for i in 0..5 {
            want += w[i] * x[i];
        }
        // Shorter than one lane group, so every element takes the tail path
        // and the result is the plain sequential fold.
        assert_eq!(Weight::F32(w.clone()).row_dot_lanes(0, &x), want);
        assert_eq!(Weight::F32(w).row_dot(0, &x), want);
    }

    /// The little-endian byte pairs a bf16 payload is made of, decoded the
    /// way the checkpoint loader decodes them. Retained from the deleted
    /// dequantize tests: the widening is the part with a wrong answer
    /// available, and it outlived the container it used to arrive in.
    #[test]
    fn bf16_payload_pairs_decode_little_endian() {
        let bytes = [0x80, 0x3f, 0xc0, 0xbf];
        let got: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        assert_eq!(got, vec![1.0, -1.5], "{got:?}");
    }
}
