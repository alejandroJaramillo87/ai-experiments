/*
 * gguf — GGUF container parsing: header, metadata, tensor directory.
 *
 * Parses the GGUF v3 layout used by quantized Gemma checkpoints. Only the
 * metadata value types the loader actually consumes are supported; anything
 * else is a hard error rather than a silent skip, so a format surprise shows
 * up at load time. Vision-tower tensors are filtered by name — the design
 * skips multimodality at load time.
 */

use std::collections::BTreeMap;

use anyhow::{bail, ensure, Context, Result};

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
pub const GGUF_VERSION: u32 = 3;

#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U32(u32),
    U64(u64),
    F32(f32),
    Bool(bool),
    Str(String),
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
        ensure!(
            self.pos + n <= self.data.len(),
            "truncated GGUF at byte {}",
            self.pos
        );
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        String::from_utf8(self.take(len)?.to_vec()).context("non-UTF8 GGUF string")
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
            10 => MetaValue::U64(r.u64()?),
            6 => MetaValue::F32(r.f32()?),
            7 => MetaValue::Bool(r.take(1)?[0] != 0),
            8 => MetaValue::Str(r.string()?),
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
    Ok(GgufFile { metadata, tensors })
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
        b.string("tokenizer.ggml.tokens");
        b.0.extend(9u32.to_le_bytes()); // array type, unsupported
        let err = parse(&b.0).unwrap_err();
        assert!(err.to_string().contains("tokenizer.ggml.tokens"));
    }

    #[test]
    fn vision_tensors_are_identified_for_skipping() {
        assert!(is_vision_tensor("v.blk.0.attn_q.weight"));
        assert!(is_vision_tensor("mm.input_projection.weight"));
        assert!(!is_vision_tensor("blk.0.attn_q.weight"));
        assert!(!is_vision_tensor("token_embd.weight"));
    }
}
