/*
 * hf_format_smoke — pins the safetensors and tokenizers API surface the
 * checkpoint port depends on, so a dependency bump that changes behaviour
 * fails here rather than inside the loader.
 *
 * These two crates are HuggingFace's own, and they are what let this workspace
 * read a Gemma 4 checkpoint directly instead of a llama.cpp translation of one.
 * The translation is what produced the tokenizer defect this port exists to
 * fix: the GGUF carries a "gemma4" type tag over both a score table and a merge
 * list, the backend read it as SentencePiece unigram, and the real vocabulary is
 * BPE. A tag that has to be guessed is the failure mode; an explicit
 * `"type": "BPE"` is not.
 *
 * Every fixture here is built in-test. Nothing is read from fixtures/ and
 * nothing is written to it — the committed fixtures arrive with the loader.
 */

use std::collections::HashMap;

use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;
use tokenizers::models::bpe::{Vocab, BPE};
use tokenizers::normalizers::Replace;
use tokenizers::pre_tokenizers::split::Split;
use tokenizers::{SplitDelimiterBehavior, Tokenizer};

/// The three dtypes a BF16 Gemma checkpoint puts in front of the loader.
/// Anything else is a checkpoint this harness does not claim to read.
#[test]
fn safetensors_round_trips_the_three_dtypes_the_loader_reads() {
    // Distinct byte patterns per tensor so a mixed-up offset shows as a value
    // mismatch rather than passing on coincidentally equal bytes.
    let f32_bytes: Vec<u8> = [1.0f32, -2.0, 0.5, 1024.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let f16_bytes: Vec<u8> = [0x3c00u16, 0xc000, 0x0001, 0x7c00]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let bf16_bytes: Vec<u8> = [0x3f80u16, 0xc000, 0x4049, 0xff80]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let tensors = vec![
        (
            "model.layers.0.self_attn.q_proj.weight",
            TensorView::new(Dtype::F32, vec![2, 2], &f32_bytes).unwrap(),
        ),
        (
            "model.layers.0.input_layernorm.weight",
            TensorView::new(Dtype::F16, vec![4], &f16_bytes).unwrap(),
        ),
        (
            "model.embed_tokens.weight",
            TensorView::new(Dtype::BF16, vec![2, 2], &bf16_bytes).unwrap(),
        ),
    ];

    let serialized = safetensors::serialize(tensors, None).unwrap();
    let parsed = SafeTensors::deserialize(&serialized).unwrap();

    let mut names = parsed.names();
    names.sort_unstable();
    assert_eq!(
        names,
        vec![
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ],
        "tensor names did not survive the round trip"
    );

    // Shape and dtype are how the loader derives geometry and picks a decoder,
    // so both are asserted per tensor rather than only on the one being decoded.
    for (name, dtype, shape, bytes) in [
        (
            "model.layers.0.self_attn.q_proj.weight",
            Dtype::F32,
            &[2usize, 2][..],
            &f32_bytes,
        ),
        (
            "model.layers.0.input_layernorm.weight",
            Dtype::F16,
            &[4usize][..],
            &f16_bytes,
        ),
        (
            "model.embed_tokens.weight",
            Dtype::BF16,
            &[2usize, 2][..],
            &bf16_bytes,
        ),
    ] {
        let view = parsed.tensor(name).unwrap();
        assert_eq!(view.dtype(), dtype, "{name}: dtype changed");
        assert_eq!(view.shape(), shape, "{name}: shape changed");
        assert_eq!(view.data(), bytes.as_slice(), "{name}: payload changed");
    }
}

/// safetensors stores BF16 as little-endian u16 in the layout `tensor.rs`'s
/// `bf16_to_f32` already expects, so the loader reuses that decoder instead of
/// growing a second one. This asserts the two agree; if a dependency bump ever
/// changes the byte order, the port has one decoder to fix and this test names
/// it.
#[test]
fn safetensors_bf16_payload_decodes_through_the_existing_widener() {
    // 1.0, -2.0, pi truncated to bf16, and -inf: a normal, a sign, a rounded
    // mantissa and a non-finite, which together catch a byte swap and a
    // mistaken route through the f16 path (which splits 1/5/10, not 1/8/7).
    let bits: [u16; 4] = [0x3f80, 0xc000, 0x4049, 0xff80];
    let want = [1.0f32, -2.0, 3.140_625, f32::NEG_INFINITY];

    let bytes: Vec<u8> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tensors = vec![(
        "w",
        TensorView::new(Dtype::BF16, vec![bits.len()], &bytes).unwrap(),
    )];
    let serialized = safetensors::serialize(tensors, None).unwrap();
    let parsed = SafeTensors::deserialize(&serialized).unwrap();
    let view = parsed.tensor("w").unwrap();

    let got: Vec<f32> = view
        .data()
        .chunks_exact(2)
        .map(|c| runtime::tensor::bf16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
        .collect();

    assert_eq!(
        got.len(),
        want.len(),
        "decoded {} values from {} bytes, expected {}",
        got.len(),
        view.data().len(),
        want.len()
    );
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g, w, "position {i}: decoded {g}, expected {w}");
    }
}

/// A tokenizer.json in the shape the real Gemma 4 file uses: a BPE model behind
/// a Replace normalizer and a Split pre-tokenizer. Built through the API rather
/// than hand-written so the test cannot encode a guess about the JSON schema,
/// and returned as a `Tokenizer` plus its serialized bytes so both the
/// in-memory and the on-disk paths are covered.
fn tiny_bpe() -> (Tokenizer, Vec<u8>) {
    // A vocabulary where the merge order is the whole answer: "ab" and "abc"
    // both exist, so a tokenizer that ignores merges and picks greedily by
    // length would return a different segmentation than BPE does.
    let vocab: Vocab = [
        ("<unk>", 0u32),
        ("a", 1),
        ("b", 2),
        ("c", 3),
        ("ab", 4),
        ("abc", 5),
        ("\u{2581}", 6),
    ]
    .into_iter()
    .map(|(s, i)| (s.to_string(), i))
    .collect();

    let merges = vec![
        ("a".to_string(), "b".to_string()),
        ("ab".to_string(), "c".to_string()),
    ];

    let model = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .unk_token("<unk>".to_string())
        .build()
        .unwrap();

    let mut tokenizer = Tokenizer::new(model);
    // The real file replaces the space with U+2581 before splitting, the same
    // convention the outgoing SentencePiece implementation used.
    let _ = tokenizer.with_normalizer(Some(Replace::new(" ", "\u{2581}").unwrap()));
    let _ = tokenizer.with_pre_tokenizer(Some(
        Split::new("\u{2581}", SplitDelimiterBehavior::Isolated, false).unwrap(),
    ));

    let json = tokenizer.to_string(false).unwrap().into_bytes();
    (tokenizer, json)
}

#[test]
fn bpe_encode_follows_the_merge_list() {
    let (tokenizer, _) = tiny_bpe();

    // "abc" merges a+b then ab+c, reaching the single token 5. A unigram or
    // longest-match reading of the same vocabulary would also land on 5 here,
    // so the discriminating case is below.
    let encoding = tokenizer.encode("abc", false).unwrap();
    assert_eq!(
        encoding.get_ids(),
        &[5],
        "expected the fully merged token, got {:?} for {:?}",
        encoding.get_ids(),
        encoding.get_tokens()
    );

    // "cab" cannot reach "abc": the merge list applies left to right within a
    // word, so c stays alone and a+b merges. A longest-match tokenizer that
    // ignored merges would produce the same ids by luck, but one that scored
    // pieces independently would not.
    let encoding = tokenizer.encode("cab", false).unwrap();
    assert_eq!(
        encoding.get_ids(),
        &[3, 4],
        "expected c then the ab merge, got {:?} for {:?}",
        encoding.get_ids(),
        encoding.get_tokens()
    );
}

#[test]
fn tokenizer_json_round_trips_through_bytes_and_a_file() {
    let (tokenizer, json) = tiny_bpe();
    let want = tokenizer.encode("ab c", false).unwrap();

    // The schema is explicit, not a tag to be interpreted: assert the model
    // type is named in the serialized form, since that is the property whose
    // absence in GGUF caused the defect this port fixes.
    let doc: serde_json::Value = serde_json::from_slice(&json).unwrap();
    assert_eq!(
        doc["model"]["type"], "BPE",
        "serialized tokenizer did not declare its model type: {}",
        doc["model"]
    );
    assert_eq!(
        doc["normalizer"]["type"], "Replace",
        "normalizer missing from the serialized form: {}",
        doc["normalizer"]
    );
    assert_eq!(
        doc["pre_tokenizer"]["type"], "Split",
        "pre_tokenizer missing from the serialized form: {}",
        doc["pre_tokenizer"]
    );

    let from_bytes = Tokenizer::from_bytes(&json).unwrap();
    let got = from_bytes.encode("ab c", false).unwrap();
    assert_eq!(
        got.get_ids(),
        want.get_ids(),
        "from_bytes changed the encoding: {:?} became {:?}",
        want.get_ids(),
        got.get_ids()
    );

    // from_file is the call the loader will actually make, and it is a distinct
    // code path from from_bytes.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.json");
    std::fs::write(&path, &json).unwrap();
    let from_file = Tokenizer::from_file(&path).unwrap();
    let got = from_file.encode("ab c", false).unwrap();
    assert_eq!(
        got.get_ids(),
        want.get_ids(),
        "from_file changed the encoding: {:?} became {:?}",
        want.get_ids(),
        got.get_ids()
    );
}

/// The loader derives geometry from `config.json` and cross-checks it against
/// tensor shapes, so shape introspection has to survive a checkpoint whose
/// tensors are stored in the HF `[out, in]` convention — the transpose of the
/// GGUF layout `matvec_into` assumes today. Pinning it here is what makes the
/// port's `dims[]` validation checkable.
#[test]
fn tensor_shapes_report_the_hf_out_in_convention() {
    let in_dim = 3usize;
    let out_dim = 2usize;
    let bytes = vec![0u8; in_dim * out_dim * 4];
    let tensors = vec![(
        "model.layers.0.mlp.down_proj.weight",
        TensorView::new(Dtype::F32, vec![out_dim, in_dim], &bytes).unwrap(),
    )];

    let serialized = safetensors::serialize(tensors, None).unwrap();
    let parsed = SafeTensors::deserialize(&serialized).unwrap();
    let view = parsed
        .tensor("model.layers.0.mlp.down_proj.weight")
        .unwrap();

    assert_eq!(
        view.shape(),
        &[out_dim, in_dim],
        "shape is stored row-major as [out, in]; got {:?}",
        view.shape()
    );
    assert_eq!(
        view.data().len(),
        out_dim * in_dim * 4,
        "payload length disagrees with the shape: {} bytes for {:?}",
        view.data().len(),
        view.shape()
    );
}

/// A missing tensor must be an error the loader can report, not a panic or a
/// silent default. The port names the tensor it wanted; this pins that the
/// crate gives it something to name.
#[test]
fn a_missing_tensor_is_an_error() {
    let bytes = vec![0u8; 4];
    let tensors = vec![(
        "present",
        TensorView::new(Dtype::F32, vec![1], &bytes).unwrap(),
    )];
    let serialized = safetensors::serialize(tensors, None).unwrap();
    let parsed = SafeTensors::deserialize(&serialized).unwrap();

    let err = parsed.tensor("absent").unwrap_err();
    let rendered = format!("{err:?}");
    assert!(
        rendered.contains("absent") || rendered.contains("TensorNotFound"),
        "error does not identify the missing tensor: {rendered}"
    );
}

/// Truncated or corrupt input must fail rather than read past the header.
#[test]
fn a_truncated_file_is_an_error() {
    let bytes = vec![0u8; 8];
    let tensors = vec![("w", TensorView::new(Dtype::F32, vec![2], &bytes).unwrap())];
    let serialized = safetensors::serialize(tensors, None).unwrap();

    let truncated = &serialized[..serialized.len() - 4];
    assert!(
        SafeTensors::deserialize(truncated).is_err(),
        "a {}-byte file truncated to {} bytes deserialized without error",
        serialized.len(),
        truncated.len()
    );
}

/// A HashMap iterates in an unspecified order, and the loader will build its
/// tensor map from one. Serialization must not depend on that order, or a
/// checkpoint written twice from the same weights would differ byte for byte
/// and break the reproducible-build contract the container exists to provide.
#[test]
fn serialization_is_independent_of_insertion_order() {
    let a = vec![0u8; 4];
    let b = vec![1u8; 4];

    let forward: HashMap<&str, TensorView> = [
        ("alpha", TensorView::new(Dtype::F32, vec![1], &a).unwrap()),
        ("beta", TensorView::new(Dtype::F32, vec![1], &b).unwrap()),
    ]
    .into_iter()
    .collect();
    let reverse: HashMap<&str, TensorView> = [
        ("beta", TensorView::new(Dtype::F32, vec![1], &b).unwrap()),
        ("alpha", TensorView::new(Dtype::F32, vec![1], &a).unwrap()),
    ]
    .into_iter()
    .collect();

    let first = safetensors::serialize(forward, None).unwrap();
    let second = safetensors::serialize(reverse, None).unwrap();
    assert_eq!(
        first,
        second,
        "the same tensors serialized to {} and {} bytes depending on map order",
        first.len(),
        second.len()
    );
}
