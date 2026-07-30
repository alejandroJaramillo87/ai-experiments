/*
 * tensor — GGUF tensor payload access and dequantization to f32.
 *
 * The CPU backend computes in plain f32, so every supported storage type
 * dequantizes to a dense Vec<f32> at load time. Supported types are the ones
 * the target checkpoints actually use: F32, F16, and Q8_0; anything else is
 * a hard error naming the tensor, so a new quantization shows up at load
 * time rather than as silent garbage.
 */

use anyhow::{bail, ensure, Result};

use super::gguf::{GgufFile, TensorInfo};

// GGML storage type ids, from the GGUF specification.
pub const GGML_F32: u32 = 0;
pub const GGML_F16: u32 = 1;
pub const GGML_Q8_0: u32 = 8;

/// Q8_0 block layout: an f16 scale followed by 32 signed byte weights.
const Q8_0_BLOCK_ELEMENTS: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 2 + Q8_0_BLOCK_ELEMENTS;

/// Number of elements a tensor holds; empty dims is a scalar.
pub fn element_count(dims: &[u64]) -> usize {
    dims.iter().product::<u64>() as usize
}

/// Stored byte size for `n` elements of the given GGML type.
pub fn byte_size(ggml_type: u32, n: usize) -> Result<usize> {
    Ok(match ggml_type {
        GGML_F32 => n * 4,
        GGML_F16 => n * 2,
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

/// f32 to IEEE 754 half precision with round-to-nearest-even, covering
/// subnormals, overflow to infinity, and NaN. Matches numpy's
/// `astype(np.float16)`, which the interp capture format requires.
pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        // Infinity or NaN; keep NaN quiet with a non-zero mantissa.
        let payload = if mantissa == 0 { 0 } else { 0x200 };
        return sign | 0x7c00 | payload;
    }
    let e = exponent - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow to infinity
    }
    // Round a mantissa down to 10 bits, carrying into the exponent field
    // naturally on overflow (all-ones mantissa rounds up).
    let round = |value: u32, shift: u32| -> u16 {
        let kept = value >> shift;
        let rem = value & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        let up = u32::from(rem > half || (rem == half && kept & 1 == 1));
        (kept + up) as u16
    };
    if e <= 0 {
        // Result is subnormal (or rounds to zero below the smallest one).
        if e < -10 {
            return sign;
        }
        let full = mantissa | 0x80_0000; // make the implicit bit explicit
        return sign | round(full, (13 + 1 - e) as u32);
    }
    sign | round(((e as u32) << 23) | mantissa, 13)
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

/// Load one tensor from a full GGUF image as dense f32, bounds-checked
/// against the file.
pub fn load_tensor(file: &GgufFile, data: &[u8], info: &TensorInfo) -> Result<Vec<f32>> {
    let n = element_count(&info.dims);
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
    dequantize(info.ggml_type, &data[start..end], n)
        .map_err(|e| anyhow::anyhow!("tensor {}: {e}", info.name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_to_f16_covers_rounding_subnormals_and_specials() {
        // Exact round trips.
        for v in [0.0f32, -0.0, 1.0, -0.5, 65504.0, 2.0f32.powi(-24)] {
            assert_eq!(f16_to_f32(f32_to_f16(v)), v, "{v}");
        }
        // Round to nearest, ties to even: 1 + 2^-11 sits exactly between
        // 1.0 and the next f16 (1 + 2^-10); the even mantissa wins.
        assert_eq!(f32_to_f16(1.0 + 2.0f32.powi(-11)), f32_to_f16(1.0));
        // 1 + 1.5 * 2^-10 ties between mantissa 1 and 2; even (2) wins.
        assert_eq!(
            f32_to_f16(1.0 + 3.0 * 2.0f32.powi(-11)),
            f32_to_f16(1.0 + 2.0f32.powi(-9))
        );
        // Overflow and specials.
        assert_eq!(f32_to_f16(1e6), 0x7c00);
        assert_eq!(f32_to_f16(-1e6), 0xfc00);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
        // Below half the smallest subnormal rounds to zero.
        assert_eq!(f32_to_f16(2.0f32.powi(-26)), 0x0000);
        // Largest subnormal survives the round trip.
        assert_eq!(f32_to_f16((1023.0 / 1024.0) * 2.0f32.powi(-14)), 0x03ff);
    }

    #[test]
    fn f32_payload_round_trips() {
        let values = [0.0f32, 1.5, -2.25, 1e-8, 3.4e38];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(dequantize(GGML_F32, &bytes, 5).unwrap(), values);
    }

    #[test]
    fn f16_decodes_normals_specials_and_subnormals() {
        assert_eq!(f16_to_f32(f32_to_f16(1.0)), 1.0);
        assert_eq!(f16_to_f32(f32_to_f16(-0.5)), -0.5);
        assert_eq!(f16_to_f32(f32_to_f16(65504.0)), 65504.0); // f16 max
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
        // One block: scale 0.5, weights 0..31 alternating sign.
        let mut bytes = f32_to_f16(0.5).to_le_bytes().to_vec();
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
}
