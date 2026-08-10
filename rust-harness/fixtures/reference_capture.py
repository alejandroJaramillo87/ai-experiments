"""Capture transformers' own residual stream and logits for one prompt.

The oracle this workspace did not write. fixtures/reference_logits.py is an
independent implementation, but independent only in the weak sense: the same
reader transcribed it from the same documents as the Rust, so a misreading
lands in both and the goldens agree about it. Two such defects survived nine
work packages that way (see that file's header). This script removes the
reader from the loop by running transformers itself.

Output feeds runtime/src/bin/layer-diff.rs, which prefills the same token IDs
through the CPU backend and reports the first layer whose residual stream
disagrees. First, not worst: once layer k is wrong every later layer is wrong
downstream of it, so a table sorted by magnitude points at the end of the
stack instead of at the cause.

Hooks fire on the decoder layer module, so the captured tensor is the layer
output after `hidden_states *= self.layer_scalar` -- the same seam the Rust
Tap observes, which runtime/src/model/cpu.rs documents and the interp probe
path already depends on.

Workstation only: this loads the full checkpoint. float32 by default, which
costs about 30 GB for E4B and removes dtype as an explanation for any
difference the comparison finds. Use --dtype bfloat16 on a smaller machine and
loosen layer-diff's tolerance to match.

Usage:

    .venv/bin/python3 rust-harness/fixtures/reference_capture.py \\
        --model /mnt/ai-data/models/hf/gemma-4-E4B \\
        --prompt "The capital of France is" \\
        --out /tmp/ref-capture.json

Writes JSON: the token IDs, one last-position residual row per layer, the
last-position logits, and the two rows that precede the stack -- the scaled
embedding and the composed per-layer inputs. Those two are what separate a
layer-0 divergence caused by a wrong input from one caused by a wrong layer 0.
Not committed -- it is 15 GB of weights reduced to a few MB of intermediate,
regenerated whenever the question changes.
"""

import argparse
import json
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="checkpoint directory")
    p.add_argument("--prompt", required=True, help="text to run")
    p.add_argument("--out", required=True, help="destination JSON")
    p.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16"],
        help="reference precision (default float32)",
    )
    p.add_argument(
        "--add-bos",
        action="store_true",
        help="prepend BOS, matching the harness's own prompt construction",
    )
    return p.parse_args()


def find_text_model(model):
    """The text model module, wherever this release nests it.

    Gemma 4 publishes as Gemma4ForConditionalGeneration, so the text stack is
    one tower among three and sits at model.model.language_model. A text-only
    release would put it at model.model. Both are probed rather than assumed,
    because a wrong guess here would hook nothing and the script would report
    a clean capture of zero layers.

    Returns the module itself and not just its `layers`, because the two
    points that precede layer 0 -- the scaled embedding and the composed
    per-layer inputs -- hang off this module rather than off any layer.
    """
    for path in ("model.language_model", "model", "language_model"):
        node = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if getattr(node, "layers", None) is not None:
            return path, node
    raise SystemExit("[ERROR] Decoder layers: no layers list found on the model")


def main():
    args = parse_args()

    # Imported here rather than at module scope so --help works on a machine
    # without torch, which is every sandbox.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"[ERROR] Import: {e}", file=sys.stderr)
        print(
            "  Solution: run on the workstation venv with torch installed",
            file=sys.stderr,
        )
        return 1

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"[RUN] Load tokenizer: {args.model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = tokenizer(args.prompt, add_special_tokens=args.add_bos)["input_ids"]
    print(f"[OK] Tokenized: {len(ids)} tokens {ids}", file=sys.stderr)

    print(
        f"[RUN] Load model: {args.dtype} on cpu, this reads the whole file",
        file=sys.stderr,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="cpu",
        # Eager, so the comparison is against the reference formulation of
        # attention rather than against a fused kernel's rounding.
        attn_implementation="eager",
    )
    model.eval()

    path, text_model = find_text_model(model)
    layers = text_model.layers
    print(f"[OK] Decoder layers: {len(layers)} at {path}.layers", file=sys.stderr)

    # A layer's output is a tensor for some releases and a tuple for others;
    # both are unwrapped rather than assumed, because indexing a tensor with
    # [0] silently yields its first row and would be captured as a plausible
    # short activation.
    captured = {}

    def hook(index):
        def fn(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[index] = tensor[0, -1, :].detach().to(torch.float32).tolist()

        return fn

    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]

    # The stages before layer 0. "Layer 0 diverges" has two unrelated causes
    # -- the stack was handed the wrong input, or layer 0 computed the wrong
    # thing with the right one -- and the per-layer table cannot distinguish
    # them, because layer 0 is the first row it has. The per-layer embeddings
    # then decompose the same way: a divergence in the composed rows is a
    # divergence in one of the two halves, or in the sum of them.
    #
    # Names and order must match runtime/src/model/cpu.rs PRELUDE_STAGES.
    prelude = {}

    def row(tensor):
        return tensor[0, -1].reshape(-1).detach().to(torch.float32).tolist()

    def stage_hook(name):
        def fn(_module, _inputs, output):
            prelude[name] = row(output)

        return fn

    # embed_tokens is a Gemma4TextScaledWordEmbedding, so its output is
    # already multiplied by sqrt(hidden_size) (upstream:1458-1459). That is
    # the tensor layer 0 consumes, and the one the Rust tap reports.
    handles.append(text_model.embed_tokens.register_forward_hook(stage_hook("embed")))

    # The token-identity half, likewise already scaled -- by
    # sqrt(hidden_size_per_layer_input) this time (upstream:1608). Captured
    # packed, before the reshape, which is the layout the Rust indexes.
    identity = getattr(text_model, "embed_tokens_per_layer", None)
    if identity is not None:
        handles.append(identity.register_forward_hook(stage_hook("ple_identity")))

    # The context half, after its norm and before the two are summed
    # (upstream:1796).
    context = getattr(text_model, "per_layer_projection_norm", None)
    if context is not None:
        handles.append(context.register_forward_hook(stage_hook("ple_context")))

    # project_per_layer_inputs is a method rather than a module, so there is
    # nothing to hang a forward hook on; it is wrapped instead. Its return
    # value is the composed [batch, seq, n_layers, ple_dim] tensor -- both
    # halves, both scales, and the 1/sqrt(2) -- which is exactly what the Rust
    # computes once before the layer loop.
    project = getattr(text_model, "project_per_layer_inputs", None)
    if project is not None:

        def wrapped(*a, **kw):
            out = project(*a, **kw)
            prelude["ple"] = row(out)
            return out

        text_model.project_per_layer_inputs = wrapped

    print("[RUN] Forward pass", file=sys.stderr)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]))
    for h in handles:
        h.remove()
    if project is not None:
        text_model.project_per_layer_inputs = project

    missing = [i for i in range(len(layers)) if i not in captured]
    if missing:
        print(f"[ERROR] Capture: layers {missing} produced no output", file=sys.stderr)
        return 1

    logits = out.logits[0, -1, :].detach().to(torch.float32).tolist()
    payload = {
        "prompt": args.prompt,
        "token_ids": ids,
        "dtype": args.dtype,
        "n_layers": len(layers),
        "layers": {str(i): captured[i] for i in range(len(layers))},
        "logits": logits,
    }
    payload.update(prelude)
    with open(args.out, "w") as f:
        json.dump(payload, f)

    top = max(range(len(logits)), key=logits.__getitem__)
    print(
        f"[OK] Wrote {args.out}: {len(layers)} layers x {len(captured[0])}, "
        f"{len(logits)} logits, prelude {sorted(prelude)}, "
        f"argmax {top} ({tokenizer.decode([top])!r})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
