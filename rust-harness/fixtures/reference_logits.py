"""Independent reference forward pass for the tiny synthetic model.

Reads fixtures/tiny-gemma4.gguf (written by the tiny-model bin), runs the
same Gemma-4-style forward pass in numpy float32, and writes the per-position
logits for a fixed prompt to tiny-gemma4-logits.json, together with the
last-position per-layer residuals (the Tap seam observable) and the
last-position logits under a fixed synthetic control vector (the steering
observable). The Rust golden tests (runtime/tests/cpu_reference.rs) assert
the CPU backend matches within f32 tolerance. Regenerate with:

    cargo run -p runtime --bin tiny-model -- fixtures/tiny-gemma4.gguf
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
    inv = np.float32(1.0) / np.sqrt(np.mean(x * x, dtype=np.float32) + np.float32(eps))
    return (x * inv * (np.float32(1.0) + w)).astype(np.float32)


def gelu(x):
    c = np.float32(0.7978846)
    inner = c * (x + np.float32(0.044715) * x * x * x)
    return (np.float32(0.5) * x * (np.float32(1.0) + np.tanh(inner))).astype(np.float32)


def rope(head, position, freq_base):
    half = head.shape[0] // 2
    out = head.copy()
    for j in range(half):
        freq = np.float32(freq_base) ** np.float32(-2.0 * j / head.shape[0])
        angle = np.float32(position) * freq
        sin, cos = np.sin(angle, dtype=np.float32), np.cos(angle, dtype=np.float32)
        a, b = head[j], head[j + half]
        out[j] = a * cos - b * sin
        out[j + half] = a * sin + b * cos
    return out


def forward(metadata, tensors, prompt, steered=False):
    arch = metadata["general.architecture"]
    m = lambda key: metadata[f"{arch}.{key}"]
    d = m("embedding_length")
    n_layers = m("block_count")
    n_heads = m("attention.head_count")
    n_kv = m("attention.head_count_kv")
    hd = m("attention.key_length")
    eps = m("attention.layer_norm_rms_epsilon")
    window = m("attention.sliding_window")
    pattern = m("attention.sliding_window_pattern")
    freq_base = m("rope.freq_base")
    heads_per_kv = n_heads // n_kv

    embed = tensors["token_embd.weight"]  # [vocab, d]
    caches = [{"k": [], "v": []} for _ in range(n_layers)]
    all_logits = []
    residuals = []
    probe_scores = []
    for position, token in enumerate(prompt):
        residuals = []
        x = (embed[token] * np.float32(math.sqrt(d))).astype(np.float32)
        for i in range(n_layers):
            t = lambda suffix: tensors[f"blk.{i}.{suffix}"]
            h = rms_norm(x, t("attn_norm.weight"), eps)
            q = (t("attn_q.weight") @ h).astype(np.float32)
            k = (t("attn_k.weight") @ h).astype(np.float32)
            v = (t("attn_v.weight") @ h).astype(np.float32)
            q = q.reshape(n_heads, hd)
            k = k.reshape(n_kv, hd)
            q = np.stack(
                [rope(rms_norm(qh, t("attn_q_norm.weight"), eps), position, freq_base) for qh in q]
            )
            k = np.stack(
                [rope(rms_norm(kh, t("attn_k_norm.weight"), eps), position, freq_base) for kh in k]
            )
            caches[i]["k"].append(k)
            caches[i]["v"].append(v.reshape(n_kv, hd))
            is_global = (i + 1) % pattern == 0
            first = 0 if is_global else max(0, position + 1 - window)
            attn = np.zeros((n_heads, hd), dtype=np.float32)
            for qh in range(n_heads):
                kvh = qh // heads_per_kv
                keys = np.stack([caches[i]["k"][p][kvh] for p in range(first, position + 1)])
                vals = np.stack([caches[i]["v"][p][kvh] for p in range(first, position + 1)])
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
            if steered and i == STEER_LAYER:
                x = (x + np.float32(STEER_SCALE) * steer_direction(d)).astype(np.float32)
            residuals.append(x.astype(np.float32).tolist())
            if i == PROBE_LAYER:
                probe_scores.append(probe_score(x.astype(np.float32), d))
        x = rms_norm(x, tensors["output_norm.weight"], eps)
        all_logits.append((embed @ x).astype(np.float32).tolist())
    # residuals holds the last position's per-layer rows (rewritten each
    # position), matching the capture format's position == "last";
    # probe_scores holds one score per position of the probe's layer.
    return all_logits, residuals, probe_scores


def main():
    gguf_path = FIXTURE_DIR / "tiny-gemma4.gguf"
    if not gguf_path.exists():
        print(f"[ERROR] Fixture not found: {gguf_path}", file=sys.stderr)
        print(
            "  Solution: cargo run -p runtime --bin tiny-model -- fixtures/tiny-gemma4.gguf",
            file=sys.stderr,
        )
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
    (FIXTURE_DIR / "tiny-gemma4-logits.json").write_text(json.dumps(out))
    print(
        f"[OK] Wrote logits, residuals, and steered logits for {len(PROMPT)} positions",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
