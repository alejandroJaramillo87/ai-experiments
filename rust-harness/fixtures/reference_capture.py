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
import hashlib
import json
import sys
from pathlib import Path


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
    # BOS is on by default, from a measurement rather than from principle.
    # Captured against E4B on "The capital of France is": without BOS the top
    # prediction was token 563, ' is' -- the prompt's own last token, which is
    # what a model does with a sequence it does not believe has begun. With
    # BOS it was 496, ' a', an ordinary continuation. Gemma is trained with a
    # leading BOS and the harness's chat path emits one
    # (harness/src/template.rs), so a capture without one measures the model
    # off-distribution and predicts nothing about how the harness is used.
    p.add_argument(
        "--no-bos",
        action="store_false",
        dest="add_bos",
        default=True,
        help="omit the leading BOS; the model then runs off-distribution",
    )
    p.add_argument(
        "--generate",
        type=int,
        default=0,
        metavar="N",
        help="also greedily continue N tokens, for L4 and as a coherence check",
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


def capture(model, ids):
    """Run `ids` through `model` and return everything the ladder compares.

    Importable rather than inlined in main() because two callers need it: this
    script against the real checkpoint, and build_tiny_reference.py against
    the tiny transformers-authored fixture. One capture implementation, two
    callers, or the tiny gate stops predicting what the real one will find --
    which is the entire reason the tiny gate exists.

    Returns a dict with token_ids, n_layers, the per-layer last-position
    residual rows, the last-position logits, and whichever pre-stack stages
    this model reaches.
    """
    import torch

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
        raise SystemExit(f"[ERROR] Capture: layers {missing} produced no output")

    payload = {
        "token_ids": list(ids),
        "n_layers": len(layers),
        "layers": {str(i): captured[i] for i in range(len(layers))},
        "logits": out.logits[0, -1, :].detach().to(torch.float32).tolist(),
    }
    payload.update(prelude)
    return payload


def main():
    args = parse_args()

    # Prepared here rather than beside the write, because the write is the last
    # thing this script does and the forward pass before it is minutes of CPU on
    # a 42-layer checkpoint. The E4B reference directories are gitignored, so
    # they are absent in a fresh clone and this is the ordinary case, not an
    # edge one -- paying for a full model load to learn the path is wrong is a
    # cost with nothing to show for it.
    out = Path(args.out)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[ERROR] Destination: {out.parent} is not usable: {e}", file=sys.stderr)
        print("  Solution: name a writable directory for --out", file=sys.stderr)
        return 1

    # Imported here rather than at module scope so --help works on a machine
    # without torch, which is every sandbox.
    try:
        import torch
        import transformers
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
    bos = "with bos" if args.add_bos else "no bos"
    print(f"[OK] Tokenized: {len(ids)} tokens, {bos} {ids}", file=sys.stderr)

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

    payload = capture(model, ids)
    payload["prompt"] = args.prompt
    payload["dtype"] = args.dtype
    # Recorded because the two conventions produce different distributions from
    # the same prompt text, and an artifact that does not say which one it used
    # cannot be compared with another. Gemma is trained with a leading <bos>;
    # the harness's own chat path emits one (harness/src/template.rs), so a
    # capture taken without one measures the model off-distribution.
    payload["add_bos"] = args.add_bos

    # Greedy continuation: L4's reference, and the only cheap way to tell a
    # weak prompt from a broken model. A single argmax cannot -- ' a' after
    # "The capital of France is" is either a base model continuing fluently or
    # a checkpoint that is not what we think it is, and a dozen tokens
    # separate the two at a glance.
    #
    # do_sample=False rather than temperature=0, and it matches the harness's
    # own greedy path: sampler.rs resolves ties to the first index precisely so
    # that torch.argmax and our decoder cannot disagree on a tie.
    if args.generate > 0:
        with torch.no_grad():
            grown = model.generate(
                torch.tensor([ids]),
                max_new_tokens=args.generate,
                do_sample=False,
            )
        continuation = grown[0, len(ids) :].tolist()
        payload["continuation_ids"] = continuation
        payload["continuation"] = tokenizer.decode(continuation)
        print(
            f"[OK] Generated: {len(continuation)} tokens {payload['continuation']!r}",
            file=sys.stderr,
        )

    # Provenance, in the field names build_tiny_reference.py's manifest already
    # uses, so the two tiers spell the same facts the same way. A number nobody
    # can reproduce is not evidence, and these are what a reader needs to
    # reproduce one.
    #
    # The checkpoint is identified by its config.json hash rather than by
    # hashing the weights: the weights are tens of gigabytes, and a config
    # change is what silently alters the architecture under a stored capture.
    # No version is asserted anywhere against these. Tier T pins transformers
    # because its artifacts are committed and must not drift; a Tier E capture
    # is regenerated on every run, so a pin here would only produce friction.
    payload["model"] = str(Path(args.model))
    payload["transformers"] = transformers.__version__
    payload["torch"] = torch.__version__
    config = Path(args.model) / "config.json"
    if config.is_file():
        payload["checkpoint_config_sha256"] = hashlib.sha256(
            config.read_bytes()
        ).hexdigest()
    with open(out, "w") as f:
        json.dump(payload, f)

    logits = payload["logits"]
    prelude = sorted(
        k for k in payload if k in ("embed", "ple", "ple_identity", "ple_context")
    )
    top = max(range(len(logits)), key=logits.__getitem__)
    print(
        f"[OK] Wrote {out}: {payload['n_layers']} layers x "
        f"{len(payload['layers']['0'])}, {len(logits)} logits, prelude {prelude}, "
        f"argmax {top} ({tokenizer.decode([top])!r})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
