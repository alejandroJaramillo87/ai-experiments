"""Supervised fine-tuning entry point (Phase 1 F3).

One command takes a recipe and runs an SFT on TRL's SFTTrainer, wired into
the Phase 0 machinery: F1's prepared datasets (training/data/sft.py), F2's
PEFT wrapping (training/peft_utils.py), D6 tracking, D9's run lock, and the
D10 watchdog — all attached through training/sft_callbacks.py. The D6
reference loop is not used for SFT (resolved F3 design decision 1).

Preflight, in order (roadmap F3 interfaces): recipe validate() (the CLI runs
it), D4 memory-fit check (warn or refuse per force), D9 lock acquisition,
inference-services-down check. Checkpoints are transformers-native
checkpoint-<step>/ directories stamped with a D6-style meta.json (see
training/sft_callbacks.py).

Packing (owned by F3, not F1): prepared examples are already tokenized and
loss-masked, so TRL's text-level packing does not apply. pack_examples()
concatenates examples into seq_len sequences, preserving label masks verbatim
and resetting position_ids per example so rotary positions restart at each
boundary; flash-attn varlen treats the position reset as a sequence boundary.

torch/trl/transformers/datasets are imported lazily so the CPU sandbox can
import and unit-test this module without the training environment.
"""

from __future__ import annotations

import datetime
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from omegaconf import DictConfig

from training.data.sft import IGNORE_INDEX, load_sft_dataset
from training.lock import run_lock
from training.loop import snapshot_hash
from training.recipes import create_run_dir
from training.serving import SERVICES
from training.sft_callbacks import (
    build_callbacks,
    checkpoint_dir_name,
    latest_trl_checkpoint,
    write_checkpoint_meta,
)

PROBE_TIMEOUT_S = 3.0

# Recipe optimizer names mapped onto transformers optim identifiers. cpu_adam
# is DeepSpeed-bound and has no TRL equivalent; validate() permits it for the
# D6 loop, so the mapping (not the schema) rejects it for SFT.
_OPTIM_NAMES = {"adamw": "adamw_torch", "adamw8bit": "adamw_bnb_8bit"}

# Recipe schedule types mapped onto transformers lr_scheduler_type values.
_SCHEDULER_TYPES = {"cosine": "cosine", "wsd": "warmup_stable_decay"}


class SftPreflightError(RuntimeError):
    """A preflight check failed; nothing was started."""


@dataclass
class SftResult:
    run_dir: str
    checkpoint_root: str
    final_step: int
    backend: str
    train_loss: float | None = None
    holdout_loss: float | None = None
    trainable_report: str = ""
    packed_examples: int = 0  # sequences after packing (0 when packing off)
    losses: list = field(default_factory=list)  # (step, loss) from log history


# -- packing (F3 decision: ships in F3; F1 artifacts are unpacked) ----------

def pack_examples(samples: list[dict], seq_len: int) -> list[dict]:
    """Pack pre-masked examples into sequences of at most seq_len tokens.

    Greedy in-order first-fit: an example that no longer fits closes the
    current sequence. Labels are carried verbatim (masking correctness is
    independent of packing) and position_ids restart at 0 for each packed
    example, which flash-attn varlen interprets as a sequence boundary. An
    example longer than seq_len is truncated and dropped if no supervised
    token survives, mirroring F1's truncation rule.
    """
    packed: list[dict] = []
    cur_ids: list[int] = []
    cur_labels: list[int] = []
    cur_pos: list[int] = []

    def flush() -> None:
        if cur_ids:
            packed.append({"input_ids": list(cur_ids),
                           "labels": list(cur_labels),
                           "position_ids": list(cur_pos)})
            cur_ids.clear()
            cur_labels.clear()
            cur_pos.clear()

    for sample in samples:
        ids, labels = sample["input_ids"], sample["labels"]
        if len(ids) > seq_len:
            ids, labels = ids[:seq_len], labels[:seq_len]
            if all(lab == IGNORE_INDEX for lab in labels):
                continue  # truncation left nothing supervised
        if len(cur_ids) + len(ids) > seq_len:
            flush()
        cur_ids.extend(ids)
        cur_labels.extend(labels)
        cur_pos.extend(range(len(ids)))
    flush()
    return packed


def collate_padded(batch: list[dict], pad_id: int = 0) -> dict:
    """Pad a batch to its longest sample (pure lists; tensors at the edge).

    Padding positions carry attention_mask 0 and label IGNORE_INDEX, so the
    pad_id value never reaches attention or the loss.
    """
    width = max(len(s["input_ids"]) for s in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": [], "position_ids": []}
    for s in batch:
        n = len(s["input_ids"])
        pad = width - n
        out["input_ids"].append(list(s["input_ids"]) + [pad_id] * pad)
        out["labels"].append(list(s["labels"]) + [IGNORE_INDEX] * pad)
        out["attention_mask"].append([1] * n + [0] * pad)
        pos = s.get("position_ids") or list(range(n))
        out["position_ids"].append(list(pos) + [0] * pad)
    return out


class Collator:
    """Callable data collator turning collate_padded output into tensors."""

    def __init__(self, pad_id: int = 0) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        padded = collate_padded(batch, self.pad_id)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in padded.items()}


# -- preflight ---------------------------------------------------------------

def probe_inference_services(probe=None) -> list[str]:
    """Names of inference services currently answering their health endpoint.

    probe(url) -> bool is injectable for tests; the default GETs the URL with
    a short timeout and treats any HTTP response as "up" (an unhealthy-but-
    listening service still holds GPU memory).
    """
    if probe is None:
        def probe(url: str) -> bool:
            try:
                with urllib.request.urlopen(url, timeout=PROBE_TIMEOUT_S):
                    return True
            except urllib.error.HTTPError:
                return True
            except (urllib.error.URLError, OSError):
                return False
    return [name for name, spec in SERVICES.items() if probe(spec.health_url)]


def check_memory_fit(recipe: DictConfig, force: bool) -> tuple[bool, str]:
    """D4 memory-fit gate: (proceed, message). force downgrades refuse to warn."""
    from training.calculator.solve import GIB, VRAM_TOTAL_BYTES, estimate

    est = estimate(recipe)
    reserved_gb = est.reserved_estimate / GIB
    total_gb = VRAM_TOTAL_BYTES / GIB
    if est.fits:
        return True, (f"Memory fit: {reserved_gb:.2f} GB reserved estimate "
                      f"of {total_gb:.0f} GB VRAM")
    message = (f"Memory fit: predicted {reserved_gb:.2f} GB exceeds the "
               f"{total_gb:.0f} GB budget")
    if force:
        return True, message + " (--force: proceeding anyway)"
    return False, message + " (reduce micro_batch/seq_len, or pass --force)"


# -- build steps -------------------------------------------------------------

def build_dataset(recipe: DictConfig) -> tuple[list[dict], list[dict]]:
    """Load the prepared F1 dataset (hash-verified) and apply packing."""
    train, holdout = load_sft_dataset(recipe.sft.dataset,
                                      tokenizer=recipe.tokenizer.path)
    if recipe.sft.packing:
        train = pack_examples(train, recipe.data.seq_len)
        # The holdout stays unpacked: held-out loss is comparable across
        # runs only when its examples are identical, packed or not.
    return train, holdout


def _torch_dtype(name: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16,
            "fp32": torch.float32}[name]


def build_model_and_tokenizer(recipe: DictConfig):
    """Build (model, tokenizer, trainable_report) for the recipe's backend.

    The trl path loads through transformers (with F2's bitsandbytes config
    for QLoRA) and wraps with PEFT; the unsloth path swaps model loading to
    FastLanguageModel and keeps everything else identical (F3 decision 2).
    """
    from training.peft_utils import (
        bnb_quantization_config,
        build_peft_model,
        resolve_target_modules,
        trainable_parameter_report,
    )

    model_id = recipe.model.hf.id
    if recipe.sft.backend == "unsloth":
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=recipe.data.seq_len,
            load_in_4bit=recipe.peft.quantize_base == "nf4",
            dtype=_torch_dtype(recipe.peft.compute_dtype))
        if recipe.peft.enabled:
            modules = resolve_target_modules(recipe, model.config.model_type)
            model = FastLanguageModel.get_peft_model(
                model, r=recipe.peft.rank, lora_alpha=recipe.peft.alpha,
                lora_dropout=recipe.peft.dropout,
                target_modules=modules,
                use_dora=recipe.peft.method == "dora")
    else:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        config = AutoConfig.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=_torch_dtype(recipe.precision.dtype),
            quantization_config=bnb_quantization_config(recipe.peft))
        if recipe.peft.enabled:
            resolve_target_modules(recipe, config.model_type)
            model = build_peft_model(model, recipe.peft)
    return model, tokenizer, trainable_parameter_report(model)


def sft_config_kwargs(recipe: DictConfig, ckpt_root: str,
                      has_holdout: bool) -> dict:
    """Map the recipe onto TRL SFTConfig keyword arguments (pure, testable).

    Dataset preparation is skipped: F1 artifacts arrive tokenized and masked,
    and TRL's own packing stays off because pack_examples() already ran
    (F3 packing decision).
    """
    optim = _OPTIM_NAMES.get(recipe.optimizer.name)
    if optim is None:
        raise SftPreflightError(
            f"optimizer.name '{recipe.optimizer.name}' has no TRL mapping "
            f"{tuple(_OPTIM_NAMES)}; cpu_adam is a D6/DeepSpeed path")
    kwargs = {
        "output_dir": ckpt_root,
        "max_steps": recipe.schedule.total_steps,
        "per_device_train_batch_size": recipe.data.micro_batch,
        "gradient_accumulation_steps": recipe.data.grad_accum,
        "learning_rate": recipe.optimizer.lr,
        "adam_beta1": recipe.optimizer.betas[0],
        "adam_beta2": recipe.optimizer.betas[1],
        "adam_epsilon": recipe.optimizer.eps,
        "weight_decay": recipe.optimizer.weight_decay,
        "optim": optim,
        "lr_scheduler_type": _SCHEDULER_TYPES[recipe.schedule.type],
        "warmup_steps": recipe.schedule.warmup_steps,
        "bf16": recipe.precision.dtype == "bf16",
        "fp16": recipe.precision.dtype == "fp16",
        "gradient_checkpointing": recipe.memory.gradient_checkpointing,
        "max_length": recipe.data.seq_len,
        "packing": False,  # pack_examples() owns packing (pre-masked data)
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "remove_unused_columns": False,
        "save_strategy": "steps",
        "save_steps": recipe.checkpoint.interval_steps,
        "logging_steps": 10,
        "seed": recipe.run.seed,
        "dataloader_num_workers": recipe.data.workers,
        "dataloader_pin_memory": recipe.data.pin_memory,
        "report_to": [],  # the D6 Tracker owns metric fan-out
    }
    if has_holdout and recipe.sft.eval_steps > 0:
        kwargs["eval_strategy"] = "steps"
        kwargs["eval_steps"] = recipe.sft.eval_steps
        kwargs["per_device_eval_batch_size"] = recipe.data.micro_batch
    else:
        kwargs["eval_strategy"] = "no"
    return kwargs


def resolve_sft_resume(ckpt_root: str, resume: str) -> str:
    """checkpoint.resume -> a concrete TRL checkpoint path ("latest" or path)."""
    if resume == "latest":
        path = latest_trl_checkpoint(ckpt_root)
        if path is None:
            raise SftPreflightError(
                f"checkpoint.resume=latest but no complete checkpoint under "
                f"{ckpt_root}")
        return path
    if not os.path.isdir(resume):
        raise SftPreflightError(f"checkpoint.resume path not found: {resume}")
    return resume


# -- orchestration -----------------------------------------------------------

def train_sft(recipe: DictConfig, force: bool = False,
              service_probe=None) -> SftResult:
    """Run the recipe's supervised fine-tune end to end.

    Preflight (memory fit, then lock, then services-down) precedes any heavy
    import or model download. WatchdogAbort propagates to the caller after an
    emergency checkpoint is written; the D9 lock releases on every exit path
    via the context manager.
    """
    if not recipe.sft.enabled:
        raise SftPreflightError(
            "sft.enabled is false; train-sft needs an sft-enabled recipe "
            "(the reference is training/recipes/reference/sft-qlora-e2b.yaml)")

    fits, fit_message = check_memory_fit(recipe, force)
    if not fits:
        raise SftPreflightError(fit_message)

    with run_lock(recipe.run.name):
        up = probe_inference_services(service_probe)
        if up:
            raise SftPreflightError(
                f"Inference services answering health checks: {', '.join(up)}\n"
                "  Solution: stop them first (make down); training and "
                "serving must not share the GPU")
        return _train_locked(recipe)


def _train_locked(recipe: DictConfig) -> SftResult:
    import torch  # noqa: F401  (fail here, before run-dir side effects)

    from training.tracking import Tracker
    from training.watchdog import Watchdog, WatchdogAbort

    train_samples, holdout_samples = build_dataset(recipe)
    model, tokenizer, report = build_model_and_tokenizer(recipe)

    stamp = datetime.datetime.now()
    for _ in range(60):
        try:
            run_dir = create_run_dir(recipe, now=stamp)
            break
        except FileExistsError:
            stamp += datetime.timedelta(seconds=1)
    else:
        raise FileExistsError(
            f"Could not allocate a run directory under "
            f"{recipe.run.output_root}/{recipe.run.name}")
    ckpt_root = os.path.join(recipe.run.output_root, recipe.run.name,
                             "checkpoints")
    os.makedirs(ckpt_root, exist_ok=True)

    recipe_hash = snapshot_hash(recipe)
    tracker = Tracker(recipe.tracking, recipe.run.name, run_dir)
    watchdog = None
    if recipe.watchdog.enabled:
        watchdog = Watchdog(recipe.watchdog, run_dir, tracker=tracker,
                            disk_path=ckpt_root)
        watchdog.start()

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    kwargs = sft_config_kwargs(recipe, ckpt_root, bool(holdout_samples))
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(**kwargs),
        train_dataset=Dataset.from_list(train_samples),
        eval_dataset=(Dataset.from_list(holdout_samples)
                      if kwargs["eval_strategy"] != "no" else None),
        processing_class=tokenizer,
        data_collator=Collator(),
        callbacks=build_callbacks(recipe, ckpt_root, tracker, watchdog,
                                  recipe_hash, trainable_report=report))

    resume_path = None
    if recipe.checkpoint.resume is not None:
        resume_path = resolve_sft_resume(ckpt_root, recipe.checkpoint.resume)

    try:
        trainer.train(resume_from_checkpoint=resume_path)
    except WatchdogAbort:
        step = trainer.state.global_step
        emergency_dir = os.path.join(
            ckpt_root, f"{checkpoint_dir_name(step)}-emergency")
        trainer.save_model(emergency_dir)
        trainer.save_state()
        write_checkpoint_meta(
            emergency_dir, step, adapter_only=bool(recipe.peft.enabled),
            base_model=recipe.model.hf.id, backend=recipe.sft.backend,
            recipe_hash=recipe_hash, emergency=True)
        raise
    finally:
        if watchdog is not None:
            watchdog.stop()
        tracker.close()

    history = trainer.state.log_history
    losses = [(h["step"], h["loss"]) for h in history if "loss" in h]
    holdout = [h["eval_loss"] for h in history if "eval_loss" in h]
    return SftResult(
        run_dir=run_dir,
        checkpoint_root=ckpt_root,
        final_step=trainer.state.global_step,
        backend=recipe.sft.backend,
        train_loss=losses[-1][1] if losses else None,
        holdout_loss=holdout[-1] if holdout else None,
        trainable_report=str(report),
        packed_examples=len(train_samples) if recipe.sft.packing else 0,
        losses=losses)
