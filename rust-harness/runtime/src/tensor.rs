/*
 * tensor — GGUF tensor payload access and dequantization to f32.
 *
 * The CPU backend computes in plain f32, so every supported storage type
 * dequantizes to a dense Vec<f32> at load time. Supported types are the ones
 * the target checkpoints actually use: F32, F16, BF16, and Q8_0; anything
 * else is a hard error naming the tensor, so a new quantization shows up at
 * load time rather than as silent garbage.
 *
 * BF16 is the published storage type for Gemma 4 and the format the training
 * tier produces, so widening it to f32 is lossless: the checkpoint's own
 * values reach the forward pass unrounded.
 */

use anyhow::{bail, ensure, Result};

use super::gguf::{GgufFile, TensorInfo};

// GGML storage type ids, from the GGUF specification.
pub const GGML_F32: u32 = 0;
pub const GGML_F16: u32 = 1;
pub const GGML_Q8_0: u32 = 8;
pub const GGML_BF16: u32 = 30;

/// Q8_0 block layout: an f16 scale followed by 32 signed byte weights.
const Q8_0_BLOCK_ELEMENTS: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 2 + Q8_0_BLOCK_ELEMENTS;

/// Number of elements a tensor holds; empty dims is a scalar.
///
/// Dimensions come straight from the file, and GGUF bounds only their count
/// (at most four), not their magnitude. The product is therefore computed
/// with checked arithmetic: a hostile or corrupt header must be a positioned
/// error, never an overflow panic (debug) or a wrapped small size (release).
pub fn element_count(dims: &[u64]) -> Result<usize> {
    let mut n: u64 = 1;
    for &d in dims {
        n = n
            .checked_mul(d)
            .ok_or_else(|| anyhow::anyhow!("tensor dimensions overflow: {dims:?}"))?;
    }
    usize::try_from(n).map_err(|_| anyhow::anyhow!("tensor element count {n} exceeds usize"))
}

/// Stored byte size for `n` elements of the given GGML type.
pub fn byte_size(ggml_type: u32, n: usize) -> Result<usize> {
    let scaled = |factor: usize| {
        n.checked_mul(factor)
            .ok_or_else(|| anyhow::anyhow!("tensor byte size overflows for {n} elements"))
    };
    Ok(match ggml_type {
        GGML_F32 => scaled(4)?,
        GGML_F16 => scaled(2)?,
        GGML_BF16 => scaled(2)?,
        GGML_Q8_0 => {
            ensure!(
                n.is_multiple_of(Q8_0_BLOCK_ELEMENTS),
                "Q8_0 element count {n} is not a multiple of the block size {Q8_0_BLOCK_ELEMENTS}"
            );
            (n / Q8_0_BLOCK_ELEMENTS) * Q8_0_BLOCK_BYTES
        }
        t => bail!("unsupported GGML tensor type {t}"),
    })
}

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

/// Dequantize `bytes` holding `n` elements of `ggml_type` into f32.
pub fn dequantize(ggml_type: u32, bytes: &[u8], n: usize) -> Result<Vec<f32>> {
    let expected = byte_size(ggml_type, n)?;
    ensure!(
        bytes.len() == expected,
        "tensor payload is {} bytes, type {ggml_type} with {n} elements needs {expected}",
        bytes.len()
    );
    Ok(match ggml_type {
        GGML_F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        GGML_F16 => bytes
            .chunks_exact(2)
            .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
            .collect(),
        GGML_BF16 => bytes
            .chunks_exact(2)
            .map(|c| bf16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
            .collect(),
        GGML_Q8_0 => {
            let mut out = Vec::with_capacity(n);
            for block in bytes.chunks_exact(Q8_0_BLOCK_BYTES) {
                let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                out.extend(block[2..].iter().map(|&b| (b as i8) as f32 * scale));
            }
            out
        }
        t => bail!("unsupported GGML tensor type {t}"),
    })
}

/// The stored payload of one tensor, bounds-checked against the file.
///
/// Split out from `load_tensor` so a storage format that keeps its own bits
/// rather than widening to f32 reuses these offset and overflow checks
/// instead of restating them. Returns the element count alongside the bytes,
/// since every caller needs both and recomputing invites disagreement.
pub fn tensor_bytes<'a>(
    file: &GgufFile,
    data: &'a [u8],
    info: &TensorInfo,
) -> Result<(&'a [u8], usize)> {
    let n = element_count(&info.dims).map_err(|e| anyhow::anyhow!("tensor {}: {e}", info.name))?;
    let size =
        byte_size(info.ggml_type, n).map_err(|e| anyhow::anyhow!("tensor {}: {e}", info.name))?;
    let start = file
        .data_offset
        .checked_add(usize::try_from(info.offset).ok().unwrap_or(usize::MAX));
    let end = start.and_then(|s| s.checked_add(size));
    let (start, end) = match (start, end) {
        (Some(s), Some(e)) if e <= data.len() => (s, e),
        _ => bail!(
            "tensor {}: payload at offset {} ({} bytes) exceeds file of {} bytes",
            info.name,
            info.offset,
            size,
            data.len()
        ),
    };
    Ok((&data[start..end], n))
}

/// Load one tensor from a full GGUF image as dense f32, bounds-checked
/// against the file.
pub fn load_tensor(file: &GgufFile, data: &[u8], info: &TensorInfo) -> Result<Vec<f32>> {
    let (bytes, n) = tensor_bytes(file, data, info)?;
    dequantize(info.ggml_type, bytes, n).map_err(|e| anyhow::anyhow!("tensor {}: {e}", info.name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_payload_round_trips() {
        let values = [0.0f32, 1.5, -2.25, 1e-8, 3.4e38];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(dequantize(GGML_F32, &bytes, 5).unwrap(), values);
    }

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
    fn q8_0_block_dequantizes_scale_times_weight() {
        // One block: scale 0.5 (f16 0x3800), weights 0..31 alternating sign.
        let mut bytes = 0x3800u16.to_le_bytes().to_vec();
        let weights: Vec<i8> = (0..32).map(|i| if i % 2 == 0 { i } else { -i }).collect();
        bytes.extend(weights.iter().map(|&w| w as u8));
        let out = dequantize(GGML_Q8_0, &bytes, 32).unwrap();
        for (i, (&w, &v)) in weights.iter().zip(&out).enumerate() {
            assert_eq!(v, w as f32 * 0.5, "element {i}");
        }
    }

    #[test]
    fn q8_0_requires_whole_blocks() {
        assert!(byte_size(GGML_Q8_0, 33).is_err());
        // Hostile header: dims bound only by count, so the product and the
        // byte size must be errors rather than overflow panics.
        assert!(element_count(&[u64::MAX, 2]).is_err());
        // Fits u64 but no payload can be that large: the byte size overflows.
        let huge = element_count(&[u64::from(u32::MAX), u64::from(u32::MAX)]).unwrap();
        assert!(byte_size(GGML_F32, huge).is_err());
        assert_eq!(element_count(&[]).unwrap(), 1);
        assert_eq!(element_count(&[2, 3, 4]).unwrap(), 24);
        assert!(byte_size(GGML_F32, usize::MAX / 2).is_err());
        assert_eq!(byte_size(GGML_Q8_0, 64).unwrap(), 68);
    }

    #[test]
    fn wrong_payload_size_is_an_error() {
        let err = dequantize(GGML_F32, &[0u8; 7], 2).unwrap_err();
        assert!(err.to_string().contains("needs 8"), "{err}");
    }

    #[test]
    fn load_tensor_reads_relative_to_the_data_section() {
        use crate::gguf::{GgufFile, TensorInfo};
        // Data section at byte 64; one f32 tensor of 2 elements at offset 8.
        let mut data = vec![0u8; 64 + 8];
        data.extend(1.5f32.to_le_bytes());
        data.extend((-2.0f32).to_le_bytes());
        let file = GgufFile {
            metadata: Default::default(),
            tensors: vec![],
            data_offset: 64,
        };
        let info = TensorInfo {
            name: "blk.0.attn_q.weight".into(),
            dims: vec![2],
            ggml_type: GGML_F32,
            offset: 8,
        };
        assert_eq!(load_tensor(&file, &data, &info).unwrap(), vec![1.5, -2.0]);
        // Truncated file: positioned error naming the tensor.
        let err = load_tensor(&file, &data[..70], &info).unwrap_err();
        assert!(err.to_string().contains("blk.0.attn_q.weight"), "{err}");
    }

    #[test]
    fn unsupported_type_is_an_error() {
        assert!(byte_size(12, 32).is_err()); // Q4_K, deliberately unsupported
        assert!(dequantize(12, &[], 0).is_err());
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

    #[test]
    fn bf16_tensors_size_and_dequantize() {
        assert_eq!(byte_size(GGML_BF16, 5).unwrap(), 10);
        let bytes = [0x80, 0x3f, 0xc0, 0xbf];
        assert_eq!(dequantize(GGML_BF16, &bytes, 2).unwrap(), vec![1.0, -1.5]);
        // Payload length is checked against the type, as for every type.
        assert!(dequantize(GGML_BF16, &bytes, 3).is_err());
    }
}
