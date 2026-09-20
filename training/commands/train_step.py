"""Step a recipe on the GPU and report what it held (the successor of the
memory-fit calculator's calibrate loop, 2026-09-20).

Builds the recipe's real model through the shared builders in
training/models/model.py (scratch recipes via LlamaConfig, model.hf recipes via
from_pretrained with the recipe's nf4 BitsAndBytesConfig, PEFT-wrapped through
training/models/peft_utils.py when peft.enabled; KB-03), runs a few optimizer
steps over synthetic batches through the trainer's chunked head loss
(training/models/head_loss.py, KB-87), and records the resident bytes after the
build, torch.cuda.max_memory_allocated/reserved and the NVML peak. The built
model's torch-reported parameter count is checked against the analytic count
(training/models/architecture.py) so drift in either is caught immediately
(KB-82); under PEFT the trainable count is checked the same way.

There is no estimate to hold the measurement to: the figures are the answer
to "does this recipe fit", read before a long run is launched. A step that
runs out of memory is reported with the resident and peak figures and the
tokens per micro-batch (KB-86).

The loop here is deliberately private: a few steps, no checkpointing,
tracking, or resume. It is not the D6 reference trainer
(training/runtime/loop.py), which owns those facilities. Out of scope,
refused with StepUnsupported: memory.offload=zero_offload recipes (the loop
is plain torch, not the DeepSpeed ZeRO engine).

GPU-only; torch is imported lazily so the CPU sandbox can import this module.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

from omegaconf import DictConfig

from common.output import Printer
from training.commands.recipe import load_validated
from training.models.architecture import spec_from_recipe, trainable_param_count
from training.models.head_loss import causal_lm_loss
from training.models.model import build_model, build_optimizer

DEFAULT_STEPS = 5
GIB = 1 << 30


class StepUnsupported(RuntimeError):
    """The recipe's runtime cannot be measured by the step loop."""


class StepOutOfMemory(RuntimeError):
    """A step exhausted the GPU: the recipe's geometry does not fit beside the
    built model (KB-86). Carries what was measured up to the failure, so the
    operator reads a resident figure and a peak rather than the allocator's
    message alone."""

    def __init__(
        self,
        step: int,
        tokens: int,
        static_measured: int,
        peak_allocated: int,
        total: int,
    ) -> None:
        super().__init__(
            f"out of GPU memory at step {step}: peak allocated {peak_allocated} "
            f"of {total} bytes at {tokens} tokens per micro-batch"
        )
        self.step = step
        self.tokens = tokens
        self.static_measured = static_measured
        self.peak_allocated = peak_allocated
        self.total = total


@dataclass(frozen=True)
class StepRecord:
    """What one recipe held on this machine over a few steps."""

    recipe_run_name: str
    steps: int
    model_param_count: int  # torch-reported, must match the analytic count
    analytic_param_count: int
    # Trainable-parameter drift check (KB-03): equals the totals above on a
    # full-parameter run; under PEFT the model count is the adapter.
    model_trainable_params: int
    analytic_trainable_params: int
    static_measured: int  # torch.cuda.memory_allocated right after the build
    measured_allocated: int  # torch.cuda.max_memory_allocated
    measured_reserved: int  # torch.cuda.max_memory_reserved
    measured_nvml_peak: int  # max NVML used-memory sample across steps
    tokens_per_micro_batch: int

    def as_dict(self) -> dict:
        return asdict(self)


def _nvml_used_bytes() -> int:
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
    finally:
        pynvml.nvmlShutdown()


def _check_supported(recipe: DictConfig) -> None:
    """Refuse recipes whose runtime the step loop cannot measure."""
    if recipe.memory.offload == "zero_offload":
        raise StepUnsupported(
            "memory.offload=zero_offload cannot be stepped here: the loop is "
            "plain torch, not the DeepSpeed ZeRO engine, so its measurement "
            "would not reflect the recipe's runtime"
        )
    # No cpu_adam guard is needed: validate.py requires zero_offload for
    # cpu_adam, so the branch above already refuses it.


def _build_step_model(recipe: DictConfig):
    """The recipe's actual model (KB-03): shared builders, PEFT-wrapped, in
    training mode as the Trainer runs it."""
    model = build_model(recipe)
    if recipe.peft.enabled:
        from training.models.peft_utils import (
            build_peft_model,
            resolve_target_modules,
        )

        resolve_target_modules(recipe, model.config.model_type)
        model = build_peft_model(
            model, recipe.peft, recipe.memory.gradient_checkpointing
        )
    # from_pretrained returns the model in eval mode and the PEFT wrap keeps
    # it there; transformers' GradientCheckpointingLayer checkpoints only
    # while self.training, so without this the loop kept every layer's
    # activations for the backward and measured a model the Trainer (which
    # calls train() itself) never runs (KB-89).
    model.train()
    return model


def step(recipe: DictConfig, steps: int = DEFAULT_STEPS) -> StepRecord:
    """Run the step loop and return what the recipe held."""
    import torch

    _check_supported(recipe)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Stepping a recipe requires the workstation GPU; run via "
            "'make train-step CONFIG=...' on the host, not in a sandbox."
        )

    spec = spec_from_recipe(recipe)
    device = torch.device("cuda:0")
    torch.manual_seed(recipe.run.seed)
    torch.cuda.init()  # memory-stats API does not lazy-init the CUDA context
    torch.cuda.reset_peak_memory_stats(device)

    model = _build_step_model(recipe)
    # A 4-bit base loads on the GPU at from_pretrained; .to() on it raises.
    if not getattr(model, "is_loaded_in_4bit", False):
        model = model.to(device)
    # What the built model holds resident: the weights as loaded (NF4
    # packing, towers and all; the embedding tables at the load dtype, KB-85).
    torch.cuda.synchronize(device)
    static_measured = torch.cuda.memory_allocated(device)
    if recipe.memory.compile:
        model = torch.compile(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(model, recipe, params=trainable)

    from training.models.peft_utils import trainable_parameter_report

    # Params4bit-aware logical counts (packed 4-bit elements count double)
    report = trainable_parameter_report(model)
    # Under PEFT the analytic total is base plus adapter.
    analytic_trainable = trainable_param_count(spec, recipe)
    analytic_total = spec.param_count()
    if recipe.peft.enabled:
        analytic_total += analytic_trainable

    tokens = recipe.data.micro_batch * recipe.data.seq_len
    nvml_peak = 0
    for step_index in range(1, steps + 1):
        batch = torch.randint(
            0, spec.vocab, (recipe.data.micro_batch, recipe.data.seq_len), device=device
        )
        try:
            # The trainer's loss path, not the HF forward with labels: the
            # head and cross-entropy run per chunk of positions (KB-87).
            loss = causal_lm_loss(
                model, batch, batch, chunk_tokens=recipe.memory.head_chunk_tokens
            )
            loss.backward()
            optimizer.step()
        except torch.cuda.OutOfMemoryError as exc:
            # The geometry is the finding, not the allocator's message; the
            # peak is readable without a synchronize (KB-86).
            raise StepOutOfMemory(
                step=step_index,
                tokens=tokens,
                static_measured=static_measured,
                peak_allocated=torch.cuda.max_memory_allocated(device),
                total=torch.cuda.get_device_properties(device).total_memory,
            ) from exc
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        nvml_peak = max(nvml_peak, _nvml_used_bytes())

    return StepRecord(
        recipe_run_name=recipe.run.name,
        steps=steps,
        model_param_count=report.total,
        analytic_param_count=analytic_total,
        model_trainable_params=report.trainable,
        analytic_trainable_params=analytic_trainable,
        static_measured=static_measured,
        measured_allocated=torch.cuda.max_memory_allocated(device),
        measured_reserved=torch.cuda.max_memory_reserved(device),
        measured_nvml_peak=nvml_peak,
        tokens_per_micro_batch=tokens,
    )


def write_record(record: StepRecord, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(record.as_dict(), f, indent=2)
        f.write("\n")


# -- CLI ---------------------------------------------------------------------


def gb(n: int) -> str:
    return f"{n / GIB:.2f} GB"


def configure(sub) -> None:
    parser = sub.add_parser(
        "step",
        help="run a few optimizer steps of a recipe on the GPU and report VRAM",
        description=(
            "Build the recipe's model, run a few optimizer steps through the "
            "trainer's loss, and report the resident and peak VRAM."
        ),
    )
    parser.add_argument("--config", required=True, help="path to a recipe YAML")
    parser.add_argument(
        "overrides", nargs="*", help="dotlist overrides, e.g. data.micro_batch=8"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"optimizer steps to run (default {DEFAULT_STEPS})",
    )
    parser.add_argument("--out", help="path for the step record JSON")
    parser.add_argument(
        "--json", action="store_true", help="emit only the record JSON on stdout"
    )
    parser.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    out = Printer(quiet=args.json)
    recipe = load_validated(args.config, args.overrides, out)
    if recipe is None:
        return 1
    out.run(f"Stepping {recipe.run.name}: {args.steps} steps on cuda:0")
    try:
        record = step(recipe, steps=args.steps)
    except StepUnsupported as exc:
        out.error(f"Step unsupported: {exc}")
        return 1
    except StepOutOfMemory as exc:
        # Before the generic arm: an OOM is a RuntimeError too, and its
        # remedy is the recipe's geometry, not the workstation (KB-86).
        out.error(
            f"Step ran out of GPU memory at step {exc.step}: resident after "
            f"build {gb(exc.static_measured)}, peak allocated "
            f"{gb(exc.peak_allocated)} of {gb(exc.total)}, at {exc.tokens:,} "
            "tokens per micro-batch"
        )
        out.line("  Solution: lower data.micro_batch or data.seq_len")
        return 1
    except (RuntimeError, ImportError) as exc:
        out.error(f"Step failed: {exc}")
        out.line("  Solution: run on the workstation via 'make train-step CONFIG=...'")
        return 1
    if args.out:
        write_record(record, args.out)
        out.ok(f"Step record written: {args.out}")

    if record.model_param_count != record.analytic_param_count:
        out.warn(
            f"Parameter count drift: model {record.model_param_count:,} vs "
            f"analytic {record.analytic_param_count:,}"
        )
        out.line(
            "  Solution: align the param formulas in training/models/architecture.py "
            "with the built model"
        )
    else:
        out.ok(
            f"Parameter count: {record.model_param_count:,} (analytic matches model)"
        )
    if record.model_trainable_params != record.analytic_trainable_params:
        out.warn(
            f"Trainable count drift: model {record.model_trainable_params:,} vs "
            f"analytic {record.analytic_trainable_params:,}"
        )
        out.line(
            "  Solution: align adapter_param_count in "
            "training/models/architecture.py with the built adapter"
        )
    out.line()
    out.line(f"Resident after build:   {gb(record.static_measured)}")
    out.line(f"Peak allocated:         {gb(record.measured_allocated)}")
    out.line(f"Peak reserved:          {gb(record.measured_reserved)}")
    out.line(f"NVML peak:              {gb(record.measured_nvml_peak)}")
    out.line(f"Tokens per micro-batch: {record.tokens_per_micro_batch:,}")
    out.ok(f"Stepped {record.steps} steps")
    if args.json:
        print(json.dumps(record.as_dict(), indent=2))
    return 0
