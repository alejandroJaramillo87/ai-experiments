"""Independent reference forward pass for the tiny synthetic models.

Reads the committed checkpoint fixtures (config.json and model.safetensors,
written by the tiny-model bin), runs the
same Gemma-4-style forward pass in numpy float32, and writes the per-position
logits for a fixed prompt to <stem>-logits.json, together with the
last-position per-layer residuals (the Tap seam observable) and the
last-position logits under a fixed synthetic control vector (the steering
observable). The Rust golden tests (runtime/tests/cpu_reference.rs) assert
the CPU backend matches within f32 tolerance.

Two fixtures. tiny-gemma4 is uniform and declares none of Gemma 4's optional
features, so it holds the absent-safe path. tiny-gemma4-e4b declares all
seven: per-layer head width, per-layer feed-forward width, dual RoPE with
proportional divisors, per-layer input embeddings, cross-layer KV sharing,
the per-layer output scale, and final logit softcapping. Everything below is
driven by the file's own metadata, so one implementation serves both.

This file is written from the architecture documents -- docs/models/gemma-4.md
and the upstream transcript beside it -- and not from the Rust. A reference
transcribed from the implementation it checks is not independent, and would
agree with it about a misreading.

Regenerate with:

    cargo run -p runtime --bin tiny-model -- fixtures/tiny-gemma4
    cargo run -p runtime --bin tiny-model -- fixtures/tiny-gemma4-e4b e4b
    .venv/bin/python3 fixtures/reference_logits.py

The interpreter matters: numpy lives in the project venv, and a bare python3
on either the workstation or the sandbox may not have it.
"""

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

FIXTURE_DIR = Path(__file__).parent
PROMPT = [1, 5, 9, 13, 3, 8, 21, 2]

# Synthetic steering fixture, mirrored by the Rust golden test: direction
# dir[j] = 0.05 * (j - 8) added to the residual after 0-based layer 1 at
# scale 0.5.
STEER_LAYER = 1
STEER_SCALE = 0.5


def steer_direction(d):
    return np.array([0.05 * (j - 8.0) for j in range(d)], dtype=np.float32)


# Synthetic probe fixture, mirrored by the Rust golden test: a linear probe
# over 0-based layer 1 scored at every position with the score.py math,
# sigmoid(((x - mu) / sigma) . w + bias).
PROBE_LAYER = 1
PROBE_BIAS = np.float32(0.1)


def probe_params(d):
    weights = np.array([0.1 * (j - 4.0) for j in range(d)], dtype=np.float32)
    mu = np.array([0.01 * j for j in range(d)], dtype=np.float32)
    sigma = np.array([1.0 + 0.1 * j for j in range(d)], dtype=np.float32)
    return weights, mu, sigma


def probe_score(x, d):
    weights, mu, sigma = probe_params(d)
    logit = ((x - mu) / sigma) @ weights + PROBE_BIAS
    return float(1.0 / (1.0 + np.exp(-logit, dtype=np.float32)))


# Hugging Face tensor name for each name this reference uses internally.
#
# Transcribed from the published checkpoint's key list, not from the Rust
# table it mirrors. That is the point: the Rust loader maps these names in the
# other direction, and if the two transcriptions disagree the goldens below
# stop matching. A shared table would make the disagreement unobservable.
#
# Verified against google/gemma-4-E4B (2130 tensors, read 2026-08-10). Gemma 4
# publishes as Gemma4ForConditionalGeneration, so the text stack sits under
# model.language_model. beside the audio and vision towers this reference does
# not read.
TEXT_PREFIX = "model.language_model."
HF_NAMES = {
    "token_embd.weight": TEXT_PREFIX + "embed_tokens.weight",
    "output_norm.weight": TEXT_PREFIX + "norm.weight",
    "per_layer_token_embd.weight": TEXT_PREFIX + "embed_tokens_per_layer.weight",
    "per_layer_model_proj.weight": TEXT_PREFIX + "per_layer_model_projection.weight",
    "per_layer_proj_norm.weight": TEXT_PREFIX + "per_layer_projection_norm.weight",
}
HF_LAYER_NAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.weight": "pre_feedforward_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "post_ffw_norm.weight": "post_feedforward_layernorm.weight",
    # The one published name without a .weight suffix: upstream declares it
    # nn.Buffer(torch.ones(1)) and multiplies it into the layer output. Trained
    # away from ones -- the released checkpoints carry distinct per-layer
    # values between 0.049 and 0.879 -- so it is read, never assumed.
    "layer_output_scale.weight": "layer_scalar",
    "inp_gate.weight": "per_layer_input_gate.weight",
    "proj.weight": "per_layer_projection.weight",
    "post_norm.weight": "post_per_layer_input_norm.weight",
}

# RMSNorm weights, which the checkpoint stores unshifted. See `rms_norm`.
NORM_SUFFIX = "norm.weight"


def parse_safetensors(data):
    """Tensors from a model.safetensors, keyed by this reference's own names.

    The container is simple enough that reading it directly costs less than a
    dependency: an 8-byte little-endian header length, a JSON header mapping
    each name to its dtype, shape and byte range, then the payload. The venv
    carries no safetensors package, and adding one to read 40 lines of format
    would put a third party between the reference and the bytes it exists to
    check independently.

    F32 only, by assertion, as the GGUF reader was. The committed fixtures are
    F32 and numpy has no native bfloat16; the bf16 path is proven in Rust
    instead, against a model whose exact logits are known.
    """
    (header_len,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + header_len])
    payload = 8 + header_len

    inverse = {v: k for k, v in HF_NAMES.items()}
    tensors = {}
    for hf_name, entry in header.items():
        if hf_name == "__metadata__":
            continue
        assert entry["dtype"] == "F32", (
            f"{hf_name} is {entry['dtype']}; this reference reads F32 only"
        )
        start, end = entry["data_offsets"]
        shape = entry["shape"]
        count = math.prod(shape) if shape else 1
        flat = np.frombuffer(data, dtype="<f4", count=count, offset=payload + start)
        assert end - start == count * 4, f"{hf_name}: byte range disagrees with shape"

        # Rename to the internal spelling. Safetensors shapes are already
        # row-major [out, in], the same layout and the same bytes the GGUF
        # held under reversed dims, so nothing is transposed here.
        name = inverse.get(hf_name)
        if name is None:
            prefix = TEXT_PREFIX + "layers."
            assert hf_name.startswith(prefix), f"unrecognized tensor {hf_name}"
            index, tail = hf_name[len(prefix) :].split(".", 1)
            suffix = next((k for k, v in HF_LAYER_NAMES.items() if v == tail), None)
            assert suffix is not None, f"unrecognized tensor {hf_name}"
            name = f"blk.{index}.{suffix}"

        # Undo the shift the checkpoint stores norms without, so every tensor
        # in this dict is the effective multiplier `rms_norm` applies.
        values = flat.reshape(shape).astype(np.float32)
        if name.endswith(NORM_SUFFIX):
            values = values + np.float32(1.0)
        tensors[name] = values
    return tensors


def rms_norm(x, w, eps):
    # `w` is the effective multiplier, with Gemma's runtime +1 already in it.
    # A published checkpoint stores the raw parameter and transformers adds
    # one at every use; parse_safetensors folds that in as it reads, so the
    # shift happens once per weight here rather than once per call. The Rust
    # loader makes the same choice, which is why the two agree bit for bit.
    inv = np.float32(1.0) / np.sqrt(np.mean(x * x, dtype=np.float32) + np.float32(eps))
    return (x * inv * w).astype(np.float32)


def gelu(x):
    c = np.float32(0.7978846)
    inner = c * (x + np.float32(0.044715) * x * x * x)
    return (np.float32(0.5) * x * (np.float32(1.0) + np.tanh(inner))).astype(np.float32)


def rope(head, position, freq_base, divisors=None):
    # `divisors` are the proportional-RoPE entries of rope_freqs.weight: ones
    # over the rotating pairs and 1e30 over the rest, which drive those
    # inverse frequencies to zero and leave those pairs unrotated. Gemma 4
    # expresses partial rotation this way rather than by slicing the head.
    half = head.shape[0] // 2
    out = head.copy()
    for j in range(half):
        freq = np.float32(freq_base) ** np.float32(-2.0 * j / head.shape[0])
        if divisors is not None:
            freq = np.float32(freq / np.float32(divisors[j]))
        angle = np.float32(position) * freq
        sin, cos = np.sin(angle, dtype=np.float32), np.cos(angle, dtype=np.float32)
        a, b = head[j], head[j + half]
        out[j] = a * cos - b * sin
        out[j + half] = a * sin + b * cos
    return out


def forward(config, tensors, prompt, steered=False):
    def m(key):
        return config[key]

    def opt(key, default=None):
        return config.get(key, default)

    d = m("hidden_size")
    n_layers = m("num_hidden_layers")
    n_heads = m("num_attention_heads")
    n_kv = m("num_key_value_heads")
    eps = m("rms_norm_eps")
    window = m("sliding_window")
    heads_per_kv = n_heads // n_kv

    # An explicit per-layer list, not a stride to infer. The GGUF spelling
    # could only express a periodic schedule; this one states each layer.
    layer_types = m("layer_types")
    assert len(layer_types) == n_layers, (
        f"layer_types has {len(layer_types)} entries for {n_layers} layers"
    )

    def is_global(i):
        kind = layer_types[i]
        assert kind in ("full_attention", "sliding_attention"), (
            f"layer_types[{i}] is {kind!r}"
        )
        return kind == "full_attention"

    # Per-layer head width. `head_dim` names the sliding regime and
    # `global_head_dim` the full-attention one; a checkpoint carrying only
    # the former is uniform.
    hd_swa = m("head_dim")
    hd_global = opt("global_head_dim", hd_swa)

    def head_dim(i):
        return hd_global if is_global(i) else hd_swa

    # Per-layer feed-forward width needs no lookup here: every matrix is
    # reshaped from its own tensor dimensions, so a stack whose widths vary
    # per layer is handled by reading the file. The metadata key is still
    # validated, because a checkpoint whose declared widths contradict its
    # tensors is a file the loader must reject rather than one this
    # reference should quietly follow.
    ffn_meta = m("intermediate_size")
    for i in range(n_layers):
        want = ffn_meta[i] if isinstance(ffn_meta, list) else ffn_meta
        got = tensors[f"blk.{i}.ffn_gate.weight"].shape[0]
        assert got == want, f"layer {i} declares ffn width {want}, tensor has {got}"

    # Dual RoPE: the two regimes carry separate bases, and only the global
    # one takes the proportional divisors.
    #
    # transformers 5.5 nests the three RoPE fields under rope_parameters,
    # keyed by attention regime; before that they were flat in text_config.
    # Both spellings are read, and the nested one wins, matching the Rust
    # accessors -- independently transcribed, as the whole file is.
    rope_params = opt("rope_parameters")
    if rope_params is not None:
        full = rope_params["full_attention"]
        sliding = rope_params.get("sliding_attention")
        for regime, block in (("full_attention", full), ("sliding_attention", sliding)):
            if block is None:
                continue
            kind = block["rope_type"]
            assert kind in ("default", "proportional"), (
                f"rope_parameters.{regime}.rope_type is {kind!r}"
            )
        freq_base = full["rope_theta"]
        freq_base_swa = sliding["rope_theta"] if sliding else freq_base
        factor = full.get("partial_rotary_factor")
    else:
        freq_base = m("rope_theta")
        freq_base_swa = opt("rope_local_base_freq", freq_base)
        factor = opt("partial_rotary_factor")

    # The divisors are derived from the rotation factor rather than read from
    # a tensor. llama.cpp precomputes them at conversion time and ships
    # rope_freqs.weight; a published checkpoint states partial_rotary_factor
    # and leaves the construction to the loader. Ones over the rotating pairs,
    # 1e30 over the rest -- large enough to drive those angles to zero.
    half = hd_global // 2
    rope_divisors = None
    if factor is not None:
        # Floor, matching transformers' int(rope_proportion * head_dim // 2)
        # and the Rust rope_rotating_pairs.
        rotating = int(factor * hd_global // 2)
        assert 0 < rotating <= half, f"partial_rotary_factor {factor} gives {rotating}"
        if rotating < half:
            rope_divisors = np.array(
                [1.0] * rotating + [1e30] * (half - rotating), dtype=np.float32
            )

    def rope_for(i):
        if is_global(i):
            return freq_base, rope_divisors
        return freq_base_swa, None

    # Cross-layer KV sharing. Layers at or past the boundary compute no key
    # or value and attend over a donor's cache; the donor is the last layer
    # below the boundary of the same kind, so there are two of them, one per
    # attention regime.
    boundary = n_layers - opt("num_kv_shared_layers", 0)

    def kv_source(i):
        if i < boundary:
            return i
        same = [j for j in range(boundary) if is_global(j) == is_global(i)]
        assert same, f"layer {i} is shared but no layer of its kind precedes {boundary}"
        return max(same)

    softcap = opt("final_logit_softcapping")
    ple_dim = opt("hidden_size_per_layer_input")

    embed = tensors["token_embd.weight"]  # [vocab, d]
    caches = [{"k": [], "v": []} for _ in range(n_layers)]
    all_logits = []
    residuals = []
    probe_scores = []
    for position, token in enumerate(prompt):
        residuals = []
        x = (embed[token] * np.float32(math.sqrt(d))).astype(np.float32)
        # Per-layer input embeddings, stage one: computed once per token from
        # the scaled embedding, before the stack. The token-identity table
        # carries its own embed scale of sqrt(hidden_size_per_layer_input);
        # the context projection is scaled by hidden_size ** -0.5 and
        # normalized over the per-layer width; their sum is scaled by
        # 2 ** -0.5. Both are packed [n_layers * ple_dim] and split into rows.
        per_layer = None
        if ple_dim:
            identity = (
                tensors["per_layer_token_embd.weight"][token]
                * np.float32(math.sqrt(ple_dim))
            ).astype(np.float32)
            projection = (
                (tensors["per_layer_model_proj.weight"] @ x) * np.float32(d**-0.5)
            ).astype(np.float32)
            proj_norm = tensors["per_layer_proj_norm.weight"]
            projection = np.stack(
                [
                    rms_norm(row, proj_norm, eps)
                    for row in projection.reshape(n_layers, ple_dim)
                ]
            )
            per_layer = (
                (projection + identity.reshape(n_layers, ple_dim))
                * np.float32(2.0**-0.5)
            ).astype(np.float32)
        for i in range(n_layers):

            def t(suffix, i=i):
                return tensors[f"blk.{i}.{suffix}"]

            hd = head_dim(i)
            base, divisors = rope_for(i)
            src = kv_source(i)
            h = rms_norm(x, t("attn_norm.weight"), eps)
            q = (t("attn_q.weight") @ h).astype(np.float32)
            q = np.stack(
                [
                    rope(
                        rms_norm(qh, t("attn_q_norm.weight"), eps),
                        position,
                        base,
                        divisors,
                    )
                    for qh in q.reshape(n_heads, hd)
                ]
            )
            # A shared layer computes no key or value: upstream builds neither
            # projection for it, and it reads the donor's cache instead.
            if src == i:
                k = (t("attn_k.weight") @ h).astype(np.float32)
                v = (t("attn_v.weight") @ h).astype(np.float32)
                k = np.stack(
                    [
                        rope(
                            rms_norm(kh, t("attn_k_norm.weight"), eps),
                            position,
                            base,
                            divisors,
                        )
                        for kh in k.reshape(n_kv, hd)
                    ]
                )
                caches[i]["k"].append(k)
                caches[i]["v"].append(v.reshape(n_kv, hd))
            # The window bound comes from the reading layer's own kind:
            # sharing supplies states, it does not change the mask.
            first = 0 if is_global(i) else max(0, position + 1 - window)
            attn = np.zeros((n_heads, hd), dtype=np.float32)
            for qh in range(n_heads):
                kvh = qh // heads_per_kv
                keys = np.stack(
                    [caches[src]["k"][p][kvh] for p in range(first, position + 1)]
                )
                vals = np.stack(
                    [caches[src]["v"][p][kvh] for p in range(first, position + 1)]
                )
                scores = (keys @ q[qh]) / np.float32(math.sqrt(hd))
                scores = np.exp(scores - scores.max(), dtype=np.float32)
                scores = (scores / scores.sum(dtype=np.float32)).astype(np.float32)
                attn[qh] = scores @ vals
            projected = (t("attn_output.weight") @ attn.reshape(-1)).astype(np.float32)
            x = x + rms_norm(projected, t("post_attention_norm.weight"), eps)
            h = rms_norm(x, t("ffn_norm.weight"), eps)
            gate = (t("ffn_gate.weight") @ h).astype(np.float32)
            up = (t("ffn_up.weight") @ h).astype(np.float32)
            down = (t("ffn_down.weight") @ (gelu(gate) * up)).astype(np.float32)
            x = x + rms_norm(down, t("post_ffw_norm.weight"), eps)
            # Per-layer input embeddings, stage two: a third residual update
            # after the feed-forward block. The gate is multiplicative and
            # acts in the per-layer space, not the model space.
            if per_layer is not None:
                g = gelu((t("inp_gate.weight") @ x).astype(np.float32))
                g = (g * per_layer[i]).astype(np.float32)
                x = x + rms_norm(
                    (t("proj.weight") @ g).astype(np.float32),
                    t("post_norm.weight"),
                    eps,
                )
            # The layer scalar is the module's last operation, after all three
            # residual updates and before the tap observes the row.
            if f"blk.{i}.layer_output_scale.weight" in tensors:
                x = (x * t("layer_output_scale.weight")[0]).astype(np.float32)
            if steered and i == STEER_LAYER:
                x = (x + np.float32(STEER_SCALE) * steer_direction(d)).astype(
                    np.float32
                )
            residuals.append(x.astype(np.float32).tolist())
            if i == PROBE_LAYER:
                probe_scores.append(probe_score(x.astype(np.float32), d))
        x = rms_norm(x, tensors["output_norm.weight"], eps)
        logits = (embed @ x).astype(np.float32)
        # Final logit softcapping: tanh(logits / cap) * cap, bounding the
        # head's output to (-cap, cap). The LM head is tied to token_embd.
        if softcap is not None:
            cap = np.float32(softcap)
            logits = (np.tanh(logits / cap, dtype=np.float32) * cap).astype(np.float32)
        all_logits.append(logits.tolist())
    # residuals holds the last position's per-layer rows (rewritten each
    # position), matching the capture format's position == "last";
    # probe_scores holds one score per position of the probe's layer.
    return all_logits, residuals, probe_scores


def generate(stem, regen_hint):
    checkpoint = FIXTURE_DIR / stem
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "model.safetensors"
    for path in (config_path, weights_path):
        if not path.exists():
            print(f"[ERROR] Fixture not found: {path}", file=sys.stderr)
            print(f"  Solution: {regen_hint}", file=sys.stderr)
            return 1
    config = json.loads(config_path.read_text())["text_config"]
    tensors = parse_safetensors(weights_path.read_bytes())
    logits, residuals, probe_scores = forward(config, tensors, PROMPT)
    steered_logits, _, _ = forward(config, tensors, PROMPT, steered=True)
    out = {
        "prompt": PROMPT,
        "logits": logits,
        "last_residuals": residuals,
        "steer_layer": STEER_LAYER,
        "steer_scale": STEER_SCALE,
        "steered_last_logits": steered_logits[-1],
        "probe_layer": PROBE_LAYER,
        "probe_scores": probe_scores,
    }
    (FIXTURE_DIR / f"{stem}-logits.json").write_text(json.dumps(out))
    print(
        f"[OK] {stem}: logits, residuals, and steered logits "
        f"for {len(PROMPT)} positions",
        file=sys.stderr,
    )
    return 0


def main():
    # Both committed fixtures run through the same reference. The uniform one
    # exercises the absent-safe path -- no sharing, no per-layer embeddings,
    # no softcapping -- and the E4B-shaped one exercises all seven features.
    failures = 0
    for stem, variant in (("tiny-gemma4", ""), ("tiny-gemma4-e4b", " e4b")):
        failures += generate(
            stem,
            f"cargo run -p runtime --bin tiny-model -- fixtures/{stem}{variant}",
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
