"""Facts every ladder reference artifact records, written once.

There is one generator per rung and they all answer the same two questions
before they answer their own: where is this going, and what produced it. Both
were first written inside reference_capture.py and both are the kind of thing
that drifts -- a second script spelling the transformers version
`transformers_version` would make two artifacts that cannot be compared by a
reader or by a script, for no reason beyond which day each was written.

An artifact that does not say what produced it cannot be reproduced, and a
number nobody can reproduce is not evidence.

Workstation only, like everything that imports it: `provenance` reads the torch
and transformers already loaded by its caller.
"""

import hashlib
import sys
from pathlib import Path


def ensure_destination(path):
    """Create the parent directory of `path`, or report why it cannot be.

    Called before the model load rather than beside the write. A generator's
    write is its last statement and the forward pass before it is minutes of
    CPU on a 42-layer checkpoint, so a mistyped --out that surfaces at the end
    costs the whole run. The reference directories are gitignored and therefore
    absent in a fresh clone, which makes creating them the ordinary case rather
    than an edge one.

    Returns True when the destination is usable, False after printing why.
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(
            f"[ERROR] Destination: {Path(path).parent} is not usable: {e}",
            file=sys.stderr,
        )
        print("  Solution: name a writable directory for --out", file=sys.stderr)
        return False
    return True


def provenance(model_path):
    """What produced an artifact, in the field names every tier already uses.

    `build_tiny_reference.py`'s manifest spells the versions the same way, so
    the two tiers state the same facts identically.

    The checkpoint is identified by the sha256 of its config.json rather than
    of its weights: the weights are tens of gigabytes, and it is a config
    change that silently alters the architecture under a stored artifact.

    No version is asserted against these anywhere. Tier T pins transformers
    because its artifacts are committed and must not drift; a Tier E artifact
    is regenerated on every run, so a pin here would produce friction and
    catch nothing.
    """
    import transformers

    fields = {
        "model": str(Path(model_path)),
        "transformers": transformers.__version__,
    }
    # Recorded when torch is importable, which is not the same as recorded when
    # torch was used -- and the difference is measurable. The tokenizer
    # generator loads no model, yet its 2026-08-14 artifact carries
    # torch 2.13.0+cu130, because the workstation environment has torch
    # installed and this import succeeds regardless of what ran.
    #
    # Read the field as "torch available to the process that wrote this", not
    # as "torch participated". Narrowing it to actual use would mean the
    # generators declaring their own dependencies rather than probing the
    # environment.
    try:
        import torch

        fields["torch"] = torch.__version__
    except ImportError:
        pass
    config = Path(model_path) / "config.json"
    if config.is_file():
        fields["checkpoint_config_sha256"] = hashlib.sha256(
            config.read_bytes()
        ).hexdigest()
    return fields
