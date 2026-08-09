"""Independent reference forward pass for the tiny synthetic models.

Reads the committed GGUF fixtures (written by the tiny-model bin), runs the
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

    cargo run -p runtime --bin tiny-model -- fixtures/tiny-gemma4.gguf
    cargo run -p runtime --bin tiny-model -- fixtures/tiny-gemma4-e4b.gguf e4b
    python3 fixtures/reference_logits.py
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


def parse_gguf(data):
    pos = [0]

    def take(n):
        chunk = data[pos[0] : pos[0] + n]
        assert len(chunk) == n, "truncated GGUF"
        pos[0] += n
        return chunk

    def u32():
        return struct.unpack("<I", take(4))[0]

    def u64():
        return struct.unpack("<Q", take(8))[0]

    def string():
        return take(u64()).decode()

    assert take(4) == b"GGUF" and u32() == 3
    tensor_count, kv_count = u64(), u64()
    metadata = {}
    for _ in range(kv_count):
        key = string()
        value_type = u32()
        if value_type == 4:
            metadata[key] = u32()
        elif value_type == 6:
            metadata[key] = struct.unpack("<f", take(4))[0]
        elif value_type == 8:
            metadata[key] = string()
        elif value_type == 9:
            elem, count = u32(), u64()
            if elem == 5:
                metadata[key] = list(struct.unpack(f"<{count}i", take(4 * count)))
            elif elem == 6:
                metadata[key] = list(struct.unpack(f"<{count}f", take(4 * count)))
            elif elem == 8:
                metadata[key] = [string() for _ in range(count)]
            else:
                raise ValueError(f"array element type {elem}")
        else:
            raise ValueError(f"value type {value_type}")
    directory = []
    for _ in range(tensor_count):
        name = string()
        dims = [u64() for _ in range(u32())]
        ggml_type, offset = u32(), u64()
        assert ggml_type == 0, "reference reader handles F32 only"
        directory.append((name, dims, offset))
    alignment = metadata.get("general.alignment", 32)
    data_start = (pos[0] + alignment - 1) // alignment * alignment
    tensors = {}
    for name, dims, offset in directory:
        count = math.prod(dims)
        start = data_start + offset
        flat = np.frombuffer(data, dtype="<f4", count=count, offset=start)
        # GGUF layout [ne0=in, ne1=out]: row-major with ne0 contiguous.
        tensors[name] = flat.reshape(list(reversed(dims))).astype(np.float32)
    return metadata, tensors


def rms_norm(x, w, eps):
    # The stored tensor is the effective multiplier: the GGUF converter
    # resolves Gemma's runtime +1 before writing (folded into the tensor for
    # gemma1-3, already shifted for gemma3n/gemma4), and the llama.cpp graph
    # applies a plain multiply.
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


def forward(metadata, tensors, prompt, steered=False):
    arch = metadata["general.architecture"]

    def m(key):
        return metadata[f"{arch}.{key}"]

    def opt(key, default=None):
        return metadata.get(f"{arch}.{key}", default)

    d = m("embedding_length")
    n_layers = m("block_count")
    n_heads = m("attention.head_count")
    n_kv = m("attention.head_count_kv")
    eps = m("attention.layer_norm_rms_epsilon")
    window = m("attention.sliding_window")
    pattern = m("attention.sliding_window_pattern")
    heads_per_kv = n_heads // n_kv

    def is_global(i):
        return (i + 1) % pattern == 0

    # Per-layer head width. The unsuffixed key names the global regime and
    # the _swa suffix the sliding one; a file carrying only the former is
    # uniform.
    hd_global = m("attention.key_length")
    hd_swa = opt("attention.key_length_swa", hd_global)

    def head_dim(i):
        return hd_global if is_global(i) else hd_swa

    # Per-layer feed-forward width needs no lookup here: every matrix is
    # reshaped from its own tensor dimensions, so a stack whose widths vary
    # per layer is handled by reading the file. The metadata key is still
    # validated, because a checkpoint whose declared widths contradict its
    # tensors is a file the loader must reject rather than one this
    # reference should quietly follow.
    ffn_meta = m("feed_forward_length")
    for i in range(n_layers):
        want = ffn_meta[i] if isinstance(ffn_meta, list) else ffn_meta
        got = tensors[f"blk.{i}.ffn_gate.weight"].shape[0]
        assert got == want, f"layer {i} declares ffn width {want}, tensor has {got}"

    # Dual RoPE: the two regimes carry separate bases, and only the global
    # one takes the proportional divisors.
    freq_base = m("rope.freq_base")
    freq_base_swa = opt("rope.freq_base_swa", freq_base)
    rope_divisors = tensors.get("rope_freqs.weight")

    def rope_for(i):
        if is_global(i):
            return freq_base, rope_divisors
        return freq_base_swa, None

    # Cross-layer KV sharing. Layers at or past the boundary compute no key
    # or value and attend over a donor's cache; the donor is the last layer
    # below the boundary of the same kind, so there are two of them, one per
    # attention regime.
    boundary = n_layers - opt("attention.shared_kv_layers", 0)

    def kv_source(i):
        if i < boundary:
            return i
        same = [j for j in range(boundary) if is_global(j) == is_global(i)]
        assert same, f"layer {i} is shared but no layer of its kind precedes {boundary}"
        return max(same)

    softcap = opt("final_logit_softcapping")
    ple_dim = opt("embedding_length_per_layer_input")

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
    gguf_path = FIXTURE_DIR / f"{stem}.gguf"
    if not gguf_path.exists():
        print(f"[ERROR] Fixture not found: {gguf_path}", file=sys.stderr)
        print(f"  Solution: {regen_hint}", file=sys.stderr)
        return 1
    metadata, tensors = parse_gguf(gguf_path.read_bytes())
    logits, residuals, probe_scores = forward(metadata, tensors, PROMPT)
    steered_logits, _, _ = forward(metadata, tensors, PROMPT, steered=True)
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
            f"cargo run -p runtime --bin tiny-model -- fixtures/{stem}.gguf{variant}",
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
