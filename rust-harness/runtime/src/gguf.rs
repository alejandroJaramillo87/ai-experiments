/*
 * gguf — GGUF container parsing: header, metadata, tensor directory.
 *
 * Parses the GGUF v3 layout used by quantized Gemma checkpoints. Only the
 * metadata value types the loader actually consumes are supported; anything
 * else is a hard error rather than a silent skip, so a format surprise shows
 * up at load time. The parser keeps every tensor entry; the loader filters
 * vision-tower tensors with `is_vision_tensor` — the design skips
 * multimodality at load time.
 */

use std::collections::BTreeMap;

use anyhow::{bail, ensure, Context, Result};

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
pub const GGUF_VERSION: u32 = 3;

#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U32(u32),
    I32(i32),
    U64(u64),
    F32(f32),
    Bool(bool),
    Str(String),
    // Array element types the loader consumes: the tokenizer vocab needs
    // tokens (str), scores (f32), and token_type (i32); Gemma 4 converters
    // publish the per-layer attention schedule as a bool array.
    ArrStr(Vec<String>),
    ArrF32(Vec<f32>),
    ArrI32(Vec<i32>),
    // Parsing a bool array is not consuming it. Gemma 4 writes
    // `attention.sliding_window_pattern` as a per-layer bool array where
    // Config::layer_kind expects a uint32 stride, so the key now parses but
    // the CPU backend still ignores it — which would build a full-attention
    // model for an interleaved checkpoint. check_supported must refuse such
    // a file until the schedule is read; see docs/rust-harness/gemma-4-ingestion.md.
    ArrBool(Vec<bool>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

#[derive(Debug)]
pub struct GgufFile {
    pub metadata: BTreeMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
    /// Absolute byte offset of the tensor data section: the directory end
    /// rounded up to `general.alignment` (default 32). `TensorInfo::offset`
    /// values are relative to this.
    pub data_offset: usize,
}

/// True for tensors belonging to the vision tower, which the loader skips.
pub fn is_vision_tensor(name: &str) -> bool {
    name.starts_with("v.") || name.starts_with("mm.") || name.contains("vision")
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        // checked_add: an adversarial u64 length must not wrap the bounds
        // check into a panic; it reports as truncation like any short read.
        let end = self.pos.checked_add(n);
        ensure!(
            end.is_some_and(|e| e <= self.data.len()),
            "truncated GGUF at byte {}",
            self.pos
        );
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn string(&mut self) -> Result<String> {
        let len = usize::try_from(self.u64()?).context("GGUF string length overflows usize")?;
        String::from_utf8(self.take(len)?.to_vec()).context("non-UTF8 GGUF string")
    }
}

/// Fixed-width little-endian reads share one shape.
macro_rules! le_reader {
    ($name:ident, $ty:ty, $n:expr) => {
        impl<'a> Reader<'a> {
            fn $name(&mut self) -> Result<$ty> {
                Ok(<$ty>::from_le_bytes(self.take($n)?.try_into().unwrap()))
            }
        }
    };
}

le_reader!(u32, u32, 4);
le_reader!(i32, i32, 4);
le_reader!(u64, u64, 8);
le_reader!(f32, f32, 4);

impl Reader<'_> {
    /// Array metadata: element type, count, then packed elements. Only the
    /// element types the loader consumes are supported.
    fn array(&mut self, key: &str) -> Result<MetaValue> {
        let elem_type = self.u32()?;
        let count = usize::try_from(self.u64()?).context("GGUF array count overflows usize")?;
        // Reject counts that cannot fit in the remaining bytes before
        // allocating. The bound is per element type: a bool is one byte, so
        // a flat four-byte minimum would reject bool arrays that do fit.
        let min_elem_bytes = match elem_type {
            7 => 1,
            8 => 8, // a GGUF string is at least its u64 length prefix
            _ => 4,
        };
        ensure!(
            count
                .checked_mul(min_elem_bytes)
                .is_some_and(|n| n <= self.data.len() - self.pos),
            "metadata key {key}: array count {count} exceeds file size"
        );
        Ok(match elem_type {
            5 => MetaValue::ArrI32((0..count).map(|_| self.i32()).collect::<Result<_>>()?),
            6 => MetaValue::ArrF32((0..count).map(|_| self.f32()).collect::<Result<_>>()?),
            7 => MetaValue::ArrBool(
                (0..count)
                    .map(|_| Ok(self.take(1)?[0] != 0))
                    .collect::<Result<_>>()?,
            ),
            8 => MetaValue::ArrStr((0..count).map(|_| self.string()).collect::<Result<_>>()?),
            t => bail!("metadata key {key}: unsupported GGUF array element type {t}"),
        })
    }
}

/// Parse header, metadata, and tensor directory from an in-memory GGUF
/// image. Tensor data itself is not touched; `offset` indexes into the data
/// section that follows the directory.
pub fn parse(data: &[u8]) -> Result<GgufFile> {
    let mut r = Reader { data, pos: 0 };
    ensure!(r.u32()? == GGUF_MAGIC, "not a GGUF file (bad magic)");
    let version = r.u32()?;
    ensure!(
        version == GGUF_VERSION,
        "unsupported GGUF version {version}, need {GGUF_VERSION}"
    );
    let tensor_count = r.u64()?;
    let kv_count = r.u64()?;

    let mut metadata = BTreeMap::new();
    for _ in 0..kv_count {
        let key = r.string()?;
        let value_type = r.u32()?;
        let value = match value_type {
            4 => MetaValue::U32(r.u32()?),
            5 => MetaValue::I32(r.i32()?),
            10 => MetaValue::U64(r.u64()?),
            6 => MetaValue::F32(r.f32()?),
            7 => MetaValue::Bool(r.take(1)?[0] != 0),
            8 => MetaValue::Str(r.string()?),
            9 => r.array(&key)?,
            t => bail!("metadata key {key}: unsupported GGUF value type {t}"),
        };
        metadata.insert(key, value);
    }

    let mut tensors = Vec::new();
    for _ in 0..tensor_count {
        let name = r.string()?;
        let n_dims = r.u32()? as usize;
        ensure!(
            n_dims <= 4,
            "tensor {name}: {n_dims} dims exceeds GGUF maximum of 4"
        );
        let dims = (0..n_dims).map(|_| r.u64()).collect::<Result<Vec<_>>>()?;
        let ggml_type = r.u32()?;
        let offset = r.u64()?;
        tensors.push(TensorInfo {
            name,
            dims,
            ggml_type,
            offset,
        });
    }
    let alignment = match metadata.get("general.alignment") {
        Some(MetaValue::U32(a)) if *a > 0 => *a as usize,
        Some(other) => bail!("general.alignment: expected positive u32, got {other:?}"),
        None => 32,
    };
    let data_offset = r.pos.div_ceil(alignment) * alignment;
    Ok(GgufFile {
        metadata,
        tensors,
        data_offset,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-built GGUF image builder for fixtures.
    struct Builder(Vec<u8>);

    impl Builder {
        fn new(tensor_count: u64, kv_count: u64) -> Self {
            let mut b = Vec::new();
            b.extend(GGUF_MAGIC.to_le_bytes());
            b.extend(GGUF_VERSION.to_le_bytes());
            b.extend(tensor_count.to_le_bytes());
            b.extend(kv_count.to_le_bytes());
            Self(b)
        }

        fn string(&mut self, s: &str) -> &mut Self {
            self.0.extend((s.len() as u64).to_le_bytes());
            self.0.extend(s.as_bytes());
            self
        }

        fn kv_u32(&mut self, key: &str, v: u32) -> &mut Self {
            self.string(key);
            self.0.extend(4u32.to_le_bytes());
            self.0.extend(v.to_le_bytes());
            self
        }

        fn kv_str(&mut self, key: &str, v: &str) -> &mut Self {
            self.string(key);
            self.0.extend(8u32.to_le_bytes());
            self.string(v);
            self
        }

        fn kv_u64(&mut self, key: &str, v: u64) -> &mut Self {
            self.string(key);
            self.0.extend(10u32.to_le_bytes());
            self.0.extend(v.to_le_bytes());
            self
        }

        fn kv_f32(&mut self, key: &str, v: f32) -> &mut Self {
            self.string(key);
            self.0.extend(6u32.to_le_bytes());
            self.0.extend(v.to_le_bytes());
            self
        }

        fn kv_bool(&mut self, key: &str, v: bool) -> &mut Self {
            self.string(key);
            self.0.extend(7u32.to_le_bytes());
            self.0.push(v as u8);
            self
        }

        fn kv_arr_str(&mut self, key: &str, vs: &[&str]) -> &mut Self {
            self.string(key);
            self.0.extend(9u32.to_le_bytes());
            self.0.extend(8u32.to_le_bytes());
            self.0.extend((vs.len() as u64).to_le_bytes());
            for v in vs {
                self.string(v);
            }
            self
        }

        fn kv_arr_f32(&mut self, key: &str, vs: &[f32]) -> &mut Self {
            self.string(key);
            self.0.extend(9u32.to_le_bytes());
            self.0.extend(6u32.to_le_bytes());
            self.0.extend((vs.len() as u64).to_le_bytes());
            for v in vs {
                self.0.extend(v.to_le_bytes());
            }
            self
        }

        fn kv_arr_i32(&mut self, key: &str, vs: &[i32]) -> &mut Self {
            self.string(key);
            self.0.extend(9u32.to_le_bytes());
            self.0.extend(5u32.to_le_bytes());
            self.0.extend((vs.len() as u64).to_le_bytes());
            for v in vs {
                self.0.extend(v.to_le_bytes());
            }
            self
        }

        fn kv_arr_bool(&mut self, key: &str, vs: &[bool]) -> &mut Self {
            self.string(key);
            self.0.extend(9u32.to_le_bytes());
            self.0.extend(7u32.to_le_bytes());
            self.0.extend((vs.len() as u64).to_le_bytes());
            for v in vs {
                self.0.push(*v as u8);
            }
            self
        }

        fn tensor(&mut self, name: &str, dims: &[u64], ggml_type: u32, offset: u64) -> &mut Self {
            self.string(name);
            self.0.extend((dims.len() as u32).to_le_bytes());
            for d in dims {
                self.0.extend(d.to_le_bytes());
            }
            self.0.extend(ggml_type.to_le_bytes());
            self.0.extend(offset.to_le_bytes());
            self
        }
    }

    #[test]
    fn parses_header_metadata_and_tensor_directory() {
        let mut b = Builder::new(2, 2);
        b.kv_str("general.architecture", "gemma4");
        b.kv_u32("gemma4.context_length", 131072);
        b.tensor("token_embd.weight", &[256000, 2048], 8, 0);
        b.tensor("blk.0.attn_q.weight", &[2048, 2048], 8, 4096);
        let f = parse(&b.0).unwrap();
        assert_eq!(
            f.metadata["general.architecture"],
            MetaValue::Str("gemma4".into())
        );
        assert_eq!(f.metadata["gemma4.context_length"], MetaValue::U32(131072));
        assert_eq!(f.tensors.len(), 2);
        assert_eq!(f.tensors[1].name, "blk.0.attn_q.weight");
        assert_eq!(f.tensors[1].dims, vec![2048, 2048]);
        assert_eq!(f.tensors[1].offset, 4096);
    }

    #[test]
    fn rejects_bad_magic_and_wrong_version() {
        assert!(parse(b"noturightfileatall").is_err());
        let mut wrong_version = Builder::new(0, 0).0;
        wrong_version[4..8].copy_from_slice(&2u32.to_le_bytes());
        let err = parse(&wrong_version).unwrap_err();
        assert!(err.to_string().contains("version 2"));
    }

    #[test]
    fn truncated_file_is_a_positioned_error() {
        let mut b = Builder::new(1, 0);
        b.string("blk.0.attn_q.weight"); // tensor entry cut off after the name
        let err = parse(&b.0).unwrap_err();
        assert!(err.to_string().contains("truncated"));
    }

    #[test]
    fn unsupported_metadata_type_names_the_key() {
        let mut b = Builder::new(0, 1);
        b.string("tokenizer.ggml.merges");
        b.0.extend(12u32.to_le_bytes()); // f64, unsupported
        let err = parse(&b.0).unwrap_err();
        assert!(err.to_string().contains("tokenizer.ggml.merges"));
    }

    #[test]
    fn string_f32_and_i32_arrays_decode() {
        let mut b = Builder::new(0, 3);
        b.kv_arr_str("tokenizer.ggml.tokens", &["<pad>", "\u{2581}hello"]);
        b.kv_arr_f32("tokenizer.ggml.scores", &[0.0, -1.5]);
        b.kv_arr_i32("tokenizer.ggml.token_type", &[3, 1]);
        let f = parse(&b.0).unwrap();
        assert_eq!(
            f.metadata["tokenizer.ggml.tokens"],
            MetaValue::ArrStr(vec!["<pad>".into(), "\u{2581}hello".into()])
        );
        assert_eq!(
            f.metadata["tokenizer.ggml.scores"],
            MetaValue::ArrF32(vec![0.0, -1.5])
        );
        assert_eq!(
            f.metadata["tokenizer.ggml.token_type"],
            MetaValue::ArrI32(vec![3, 1])
        );
    }

    #[test]
    fn a_bool_array_decodes_as_the_gemma4_layer_schedule() {
        // Gemma 4 converters write attention.sliding_window_pattern as one
        // bool per layer, true marking a sliding-window layer. Refusing the
        // element type made four of the six published files unparseable
        // before any tensor was read.
        let mut b = Builder::new(0, 1);
        b.kv_arr_bool(
            "gemma4.attention.sliding_window_pattern",
            &[true, true, true, true, false],
        );
        let f = parse(&b.0).unwrap();
        assert_eq!(
            f.metadata["gemma4.attention.sliding_window_pattern"],
            MetaValue::ArrBool(vec![true, true, true, true, false])
        );
    }

    #[test]
    fn a_bool_array_denser_than_four_bytes_per_element_still_fits() {
        // The pre-allocation guard bounds the count by the remaining bytes.
        // A flat four-byte minimum would reject this array, which occupies
        // exactly one byte per element and does fit.
        let values = vec![true; 64];
        let mut b = Builder::new(0, 1);
        b.kv_arr_bool("gemma4.attention.sliding_window_pattern", &values);
        let remaining_after_header = b.0.len() - 24;
        assert!(
            values.len() * 4 > remaining_after_header,
            "fixture must exceed the old four-byte-per-element bound: {} vs {}",
            values.len() * 4,
            remaining_after_header
        );
        let f = parse(&b.0).unwrap();
        assert_eq!(
            f.metadata["gemma4.attention.sliding_window_pattern"],
            MetaValue::ArrBool(values)
        );
    }

    #[test]
    fn adversarial_array_count_is_an_error_not_an_allocation() {
        let mut b = Builder::new(0, 1);
        b.string("tokenizer.ggml.scores");
        b.0.extend(9u32.to_le_bytes()); // value type: array
        b.0.extend(6u32.to_le_bytes()); // element type: f32
        b.0.extend(u64::MAX.to_le_bytes()); // absurd count
        let err = parse(&b.0).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("exceeds") || msg.contains("overflows"),
            "{msg}"
        );
    }

    #[test]
    fn unsupported_array_element_type_names_the_key() {
        let mut b = Builder::new(0, 1);
        b.string("tokenizer.ggml.token_type");
        b.0.extend(9u32.to_le_bytes()); // value type: array
        b.0.extend(10u32.to_le_bytes()); // element type: uint64, unsupported
        b.0.extend(0u64.to_le_bytes());
        let err = parse(&b.0).unwrap_err();
        assert!(err.to_string().contains("tokenizer.ggml.token_type"));
    }

    #[test]
    fn u64_f32_and_bool_metadata_decode() {
        let mut b = Builder::new(0, 3);
        b.kv_u64("gemma4.file_size", 17_000_000_000);
        b.kv_f32("gemma4.rope_theta", 10000.0);
        b.kv_bool("gemma4.attention.causal", true);
        let f = parse(&b.0).unwrap();
        assert_eq!(
            f.metadata["gemma4.file_size"],
            MetaValue::U64(17_000_000_000)
        );
        assert_eq!(f.metadata["gemma4.rope_theta"], MetaValue::F32(10000.0));
        assert_eq!(f.metadata["gemma4.attention.causal"], MetaValue::Bool(true));
    }

    #[test]
    fn adversarial_string_length_is_an_error_not_a_panic() {
        // A u64::MAX length must not wrap the bounds check.
        let mut b = Builder::new(0, 1).0;
        b.extend(u64::MAX.to_le_bytes()); // key length
        let err = parse(&b).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("truncated") || msg.contains("overflows"),
            "{msg}"
        );
    }

    #[test]
    fn too_many_dims_is_an_error_naming_the_tensor() {
        let mut b = Builder::new(1, 0);
        b.tensor("blk.0.bad", &[1, 2, 3, 4, 5], 8, 0);
        let err = parse(&b.0).unwrap_err();
        assert!(err.to_string().contains("blk.0.bad"));
    }

    #[test]
    fn vision_tensors_are_identified_for_skipping() {
        assert!(is_vision_tensor("v.blk.0.attn_q.weight"));
        assert!(is_vision_tensor("mm.input_projection.weight"));
        assert!(!is_vision_tensor("blk.0.attn_q.weight"));
        assert!(!is_vision_tensor("token_embd.weight"));
    }
}
