"""Build the tiny Gemma 4 that transformers itself authors, and capture it.

The oracle problem, stated once. Every numeric test in this workspace has
compared runtime/src/model/cpu.rs against fixtures/reference_logits.py, on
random weights. Both were transcribed from the same architecture documents by
the same reader, so a misreading lands in both and they agree about it
perfectly -- and the fixture generator encodes the same beliefs at the other
end, so nothing inside this repository can contradict one. That gate stayed
green through nine work packages while the forward pass produced noise.

This script removes our transcription from the oracle position. transformers
constructs the config from its own classes, instantiates its own Gemma 4,
and runs its own forward pass; we supply only the shapes and the weights. What
comes out is a checkpoint small enough to commit whole and a capture of what
the reference implementation computed from it, so the sandbox gate can measure
against a real Gemma 4 without a GPU, a 15 GB download, or torch.

Workstation only -- it needs torch and transformers. The artifacts it writes
are committed, which is why rust-harness/fixtures/ carries a .gitignore
negation for safetensors: unlike the synth.rs fixtures beside them, these
cannot be regenerated in a sandbox.

Usage:

    .venv/bin/python3 rust-harness/fixtures/build_tiny_reference.py \\
        --out rust-harness/fixtures/tiny-hf-gemma4

Writes into that directory: config.json and model.safetensors (transformers'
own save_pretrained), capture.json (the reference forward pass), and
manifest.json (sha256 of each of those, of this script, of the transformers
and torch versions, and of the seed).

FIRST DRAFT. The field names below are this reader's best understanding of
Gemma4TextConfig and have not been run against an installed transformers --
that is the one step of this package a sandbox cannot do, and it is the step
with real unknown-unknowns. The first run confirmed it, rejecting
`intermediate_size` as a list. Fix SHAPE here rather than working around the
rejection at the call site: a config built by hand around a rejected field is
the transcription this script exists to remove.

Two aids for that, because the author of this script cannot reach a machine
with transformers on it and one rejected keyword per round trip is the wrong
shape for the work. `--describe` prints the field vocabulary the installed
class declares. Failing that, the build preflights every SHAPE entry
separately and reports all the rejections at once instead of only the first.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Every shape the fixture carries, and why it is not the default.
#
# A tiny fixture is worth exactly the features it exercises. Two layer types
# in a pattern, a non-zero shared-KV count, per-layer input embeddings, head
# widths that differ between the two attention regimes, and a non-default
# rms_norm_eps are each a place the CPU backend can be wrong; a fixture
# without them is green about code it never ran. These mirror the feature set
# of fixtures/tiny-gemma4-e4b/config.json, which synth.rs writes -- the
# difference is only that transformers serializes these.
SEED = 20260810
SHAPE = {
    "hidden_size": 32,
    "intermediate_size": [48, 48, 64, 64],
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "global_head_dim": 12,
    "num_kv_shared_layers": 2,
    "hidden_size_per_layer_input": 4,
    "sliding_window": 3,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 1024,
    "final_logit_softcapping": 30.0,
    "tie_word_embeddings": True,
    "layer_types": [
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    ],
    "rope_parameters": {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {
            "rope_type": "proportional",
            "rope_theta": 1000000.0,
            "partial_rotary_factor": 1.0 / 3.0,
        },
    },
}

# The prompt the capture runs. Short and fixed: the point is a forward pass
# through every layer type, not coverage of the vocabulary.
PROMPT_IDS = [2, 17, 43, 8, 99, 12]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="destination directory")
    p.add_argument(
        "--tokenizer",
        default=str(Path(__file__).parent / "tiny-gemma4-e4b.tokenizer.json"),
        help="tokenizer whose vocab size the fixture matches (default: the "
        "committed tiny BPE, so L0 has a tokenizer both sides already agree "
        "exists)",
    )
    p.add_argument(
        "--describe",
        action="store_true",
        help="print the fields Gemma4TextConfig declares and exit, without "
        "building anything (workstation only, and the fastest way to correct "
        "SHAPE against the installed transformers)",
    )
    return p.parse_args()


def vocab_size_of(tokenizer_path):
    """Vocabulary size of a committed tokenizer.json.

    Read rather than chosen, so the fixture and the tokenizer cannot disagree
    about how wide a logits row is -- a mismatch there fails as a confusing
    shape error deep in the comparison rather than here, where it is one line.
    """
    with open(tokenizer_path) as f:
        model = json.load(f)["model"]
    return len(model["vocab"])


def config_class():
    from transformers.models.gemma4 import Gemma4TextConfig

    return Gemma4TextConfig


def describe_config():
    """Print every field Gemma4TextConfig declares, with its type and default.

    The whole point of this fixture is that transformers, not this reader,
    decides what a Gemma 4 config is. That only works if the field vocabulary
    is read off the installed class rather than recalled, so this mode exists
    to be run once on the workstation and pasted back. It is also the answer
    to the open question SHAPE cannot settle from a sandbox: whether per-layer
    feed-forward widths are a list on `intermediate_size` or are derived by
    transformers from a scalar plus `use_double_wide_mlp`.
    """
    import dataclasses

    cls = config_class()
    fields = getattr(cls, "__dataclass_fields__", None)
    if not fields:
        # Not a dataclass in this release; fall back to a default instance,
        # which is the only other place the field names are written down.
        instance = cls()
        for name, value in sorted(vars(instance).items()):
            print(f"{name}: {type(value).__name__} = {value!r}")
        return 0
    for name, field in sorted(fields.items()):
        default = field.default
        if default is dataclasses.MISSING:
            default = "<required>"
        print(f"{name}: {field.type} = {default!r}")
    return 0


def rejected_fields(vocab_size):
    """Every SHAPE entry the installed Gemma4TextConfig will not accept.

    One keyword at a time, so the report names all of them. The constructor
    raises on the first bad field and says nothing about the rest, which turns
    a config this reader got wrong in four places into four round trips to a
    machine the author of this script cannot reach. Each candidate is tried
    against an otherwise-default config, so the failures are independent.
    """
    cls = config_class()
    bad = []
    for key, value in SHAPE.items():
        try:
            cls(vocab_size=vocab_size, **{key: value})
        except Exception as e:  # noqa: BLE001 - any rejection is the signal
            bad.append((key, value, f"{type(e).__name__}: {e}"))
    return bad


def build_config(vocab_size):
    """A Gemma 4 config built from transformers' own class.

    Constructed rather than written as JSON on purpose. A hand-written
    config.json is another transcription: it would record what this reader
    believes the field names are, and transformers would fill the rest with
    defaults nobody chose. Passing them as keywords means an unknown field is
    an error here instead of a silent default.

    The preflight does not repair anything. A rejected field means SHAPE
    describes a model transformers would not build, and the repair is to
    correct SHAPE against the released config -- dropping the field, or
    coercing it to whatever the validator wanted, would produce a fixture that
    is quietly not the architecture the harness has to match.
    """
    bad = rejected_fields(vocab_size)
    if bad:
        print(
            "[ERROR] Config: transformers rejects these SHAPE entries", file=sys.stderr
        )
        for key, value, message in bad:
            print(f"  {key} = {value!r}", file=sys.stderr)
            print(f"    {message}", file=sys.stderr)
        print(
            "  Solution: correct SHAPE in this script against the released "
            "config.json and --describe output; do not drop the field",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return config_class()(vocab_size=vocab_size, **SHAPE)


def seeded_weights(model, seed):
    """Overwrite every parameter from a seeded generator.

    transformers' default initialization is not stable across versions, so a
    fixture that relied on it would silently become a different model on an
    upgrade -- and the capture committed beside it would then be a capture of
    something else. Every parameter is written here, from one generator, in
    the order named_parameters yields.

    Uniform on [-0.5, 0.5) rather than normal, and not scaled per tensor: the
    values only have to be distinct and finite. Norm weights are the one
    exception, centered on 1.0, because a norm weight near zero makes the
    normalized output arbitrarily small and the comparison numerically
    meaningless for a reason that has nothing to do with correctness.
    """
    import torch

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            values = torch.rand(param.shape, generator=generator) - 0.5
            if "norm" in name:
                values = values * 0.2 + 1.0
            param.copy_(values)
    return model


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    args = parse_args()
    out = Path(args.out)

    try:
        import torch
        import transformers
        from transformers import Gemma4ForCausalLM
    except ImportError as e:
        print(f"[ERROR] Import: {e}", file=sys.stderr)
        print(
            "  Solution: run on the workstation venv with torch and transformers",
            file=sys.stderr,
        )
        return 1

    if args.describe:
        return describe_config()

    sys.path.insert(0, str(Path(__file__).parent))
    from reference_capture import capture

    vocab_size = vocab_size_of(args.tokenizer)
    print(f"[OK] Tokenizer: {args.tokenizer}, vocab {vocab_size}", file=sys.stderr)

    print("[RUN] Build model from transformers' own config", file=sys.stderr)
    config = build_config(vocab_size)
    model = Gemma4ForCausalLM(config)
    seeded_weights(model, SEED)
    model.eval()

    # bfloat16 on disk, float32 in the capture. bfloat16 is what released
    # checkpoints carry, it halves the committed size, and it puts the Rust
    # loader's bf16_to_f32 on the tested path. Widening bfloat16 to float32 is
    # exact, so both sides then compute in f32 from identical values -- which
    # is why the tolerances for this fixture can be tight.
    out.mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    print(f"[OK] Wrote checkpoint: {out}", file=sys.stderr)

    reloaded = Gemma4ForCausalLM.from_pretrained(
        out,
        dtype=torch.float32,
        # Eager, so the capture is of the reference formulation of attention
        # rather than of a fused kernel's rounding.
        attn_implementation="eager",
    )
    reloaded.eval()
    with torch.no_grad():
        payload = capture(reloaded, PROMPT_IDS)
    payload["dtype"] = "float32"
    payload["storage_dtype"] = "bfloat16"
    (out / "capture.json").write_text(json.dumps(payload))
    print(
        f"[OK] Wrote capture: {payload['n_layers']} layers, "
        f"{len(payload['logits'])} logits",
        file=sys.stderr,
    )

    # The manifest is what makes an unregenerated artifact fail in the
    # sandbox, where there is no torch to regenerate anything. It hashes the
    # generator too, so editing this file without re-running it is caught --
    # the failure mode that matters, because the edit is the easy half.
    manifest = {
        "seed": SEED,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "generator_sha256": sha256_file(__file__),
        "artifacts": {
            name: sha256_file(out / name)
            for name in ("config.json", "model.safetensors", "capture.json")
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(
        f"[OK] Wrote manifest: transformers {manifest['transformers']}, "
        f"torch {manifest['torch']}, seed {SEED}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
