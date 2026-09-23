"""Supervised fine-tuning entry point (Phase 1 F3).

One command takes a recipe and runs an SFT on TRL's SFTTrainer, wired into
the Phase 0 machinery: F1's prepared datasets (training/data/sft.py), F2's
PEFT wrapping (training/models/peft_utils.py), D6 tracking, D9's run lock, and the
D10 watchdog — all attached through training/tune/sft_callbacks.py. The D6
reference loop is not used for SFT (resolved F3 design decision 1).

Preflight, in order (roadmap F3 interfaces): recipe validate() (the CLI runs
it), D9 lock acquisition. The memory-fit advisory left with the calculator
on 2026-09-20; `make train-step` measures a recipe before a long run.
Running inference containers are not probed; the operator manages containers
manually. Checkpoints are transformers-native
checkpoint-<step>/ directories stamped with a D6-style meta.json (see
training/tune/sft_callbacks.py).

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
import sys
from dataclasses import dataclass, field

from omegaconf import DictConfig

from common.obs import set_run_dir
from training.data.sft import IGNORE_INDEX, load_sft_dataset
from training.models.head_loss import causal_lm_loss
from training.recipes import create_run_dir, tokenizer_source
from training.runtime.lock import run_lock
from training.runtime.loop import snapshot_hash
from training.tune.sft_callbacks import (
    _attention_implementation,
    build_callbacks,
    checkpoint_dir_name,
    latest_trl_checkpoint,
    write_checkpoint_meta,
)

# Recipe optimizer names mapped onto transformers optim identifiers. cpu_adam
# is DeepSpeed-bound and has no TRL equivalent; validate() permits it for the
# D6 loop, so the mapping (not the schema) rejects it for SFT.
_OPTIM_NAMES = {
    "adamw": "adamw_torch",
    "adamw8bit": "adamw_bnb_8bit",
    "paged_adamw8bit": "paged_adamw_8bit",
}

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
            packed.append(
                {
                    "input_ids": list(cur_ids),
                    "labels": list(cur_labels),
                    "position_ids": list(cur_pos),
                }
            )
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
    out: dict[str, list[list[int]]] = {
        "input_ids": [],
        "labels": [],
        "attention_mask": [],
        "position_ids": [],
    }
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


# -- build steps -------------------------------------------------------------


def build_dataset(recipe: DictConfig) -> tuple[list[dict], list[dict]]:
    """Load the prepared F1 dataset (hash-verified) and apply packing.

    The hash check reads the tokenizer the base directory ships unless the
    recipe pins another (tokenizer_source), the same file prepare-sft was
    given, so an unstaged tokenizer.path cannot fail a run whose base is
    present (KB-80).
    """
    train, holdout = load_sft_dataset(
        recipe.sft.dataset, tokenizer=tokenizer_source(recipe)
    )
    if recipe.sft.packing:
        train = pack_examples(train, recipe.data.seq_len)
        # The holdout stays unpacked: held-out loss is comparable across
        # runs only when its examples are identical, packed or not.
    return train, holdout


def build_model_and_tokenizer(recipe: DictConfig):
    """Build (model, tokenizer, trainable_report) for the recipe's backend.

    Loads through transformers (with F2's bitsandbytes config for QLoRA)
    and wraps with PEFT.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from training.models.model import attention_implementation, torch_dtype
    from training.models.peft_utils import (
        bnb_quantization_config,
        build_peft_model,
        resolve_target_modules,
        trainable_parameter_report,
    )
    from training.models.staged import resolve_model_source

    # The staged directory when one exists, else the id (KB-79).
    model_id = resolve_model_source(recipe.model.hf.id)
    revision = recipe.model.hf.revision

    # local_files_only: training runs offline (see training/models/model.py).
    config = AutoConfig.from_pretrained(
        model_id, revision=revision, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=True,
        dtype=torch_dtype(recipe.precision.dtype),
        quantization_config=bnb_quantization_config(recipe.peft),
        # Explicit request with the flash->sdpa fallback (WI-7): the
        # transformers default never selects flash-attn on its own.
        # Resolved against the config's head width (KB-84).
        attn_implementation=attention_implementation(config),
    )
    if recipe.peft.enabled:
        resolve_target_modules(recipe, config.model_type)
        model = build_peft_model(
            model, recipe.peft, recipe.memory.gradient_checkpointing
        )
    return model, tokenizer, trainable_parameter_report(model)


def resolve_base_identity_or_none(recipe: DictConfig):
    """Best-effort base identity (revision, config hash) for meta.json.

    Runs after model loading so the config is already in the HF cache. A
    resolution failure must not kill a training run — the checkpoint's meta
    then records nulls and the F6 export preflight reports the fields as
    unrecorded instead of enforcing them.
    """
    from training.export.preflight import resolve_base_identity

    try:
        return resolve_base_identity(
            recipe.model.hf.id,
            revision=recipe.model.hf.revision,
            # Cache-only. This runs after the model has loaded, so the config is
            # already local; without this it would be the one train-time path
            # still able to reach the hub.
            local_files_only=True,
        )
    except Exception:
        return None


def sft_config_kwargs(recipe: DictConfig, ckpt_root: str, has_holdout: bool) -> dict:
    """Map the recipe onto TRL SFTConfig keyword arguments (pure, testable).

    Dataset preparation is skipped: F1 artifacts arrive tokenized and masked,
    and TRL's own packing stays off because pack_examples() already ran
    (F3 packing decision).
    """
    optim = _OPTIM_NAMES.get(recipe.optimizer.name)
    if optim is None:
        raise SftPreflightError(
            f"optimizer.name '{recipe.optimizer.name}' has no TRL mapping "
            f"{tuple(_OPTIM_NAMES)}; cpu_adam is a D6/DeepSpeed path"
        )
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
        "bf16": True,  # the run precision is bf16 (PRECISION_DTYPES)
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
        # Evaluation keeps the loss only: the chunked loss returns no logits,
        # and the Trainer default would otherwise accumulate every eval
        # batch's 262144-wide logits across the holdout (KB-88).
        "prediction_loss_only": True,
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
                f"checkpoint.resume=latest but no complete checkpoint under {ckpt_root}"
            )
        return path
    if not os.path.isdir(resume):
        raise SftPreflightError(f"checkpoint.resume path not found: {resume}")
    return resume


# -- loss --------------------------------------------------------------------


def compute_chunked_loss(
    chunk_tokens: int,
    model,
    inputs: dict,
    return_outputs: bool = False,
    num_items_in_batch=None,
):
    """The SFT loss through training/models/head_loss.py (KB-87), in the
    shape Trainer.compute_loss returns.

    Replaces TRL's compute_loss, which runs the HF forward with labels (the
    full micro_batch x seq_len x vocab logits through softcap, fp32 upcast
    and cross-entropy) and then reads those logits twice more for its
    entropy and token-accuracy metrics. Neither metric is kept: the tracker
    and SftResult read the loss. outputs carries only the loss, which is why
    sft_config_kwargs sets prediction_loss_only (KB-88).
    """
    loss = causal_lm_loss(
        model,
        inputs["input_ids"],
        inputs["labels"],
        chunk_tokens=chunk_tokens,
        attention_mask=inputs.get("attention_mask"),
        position_ids=inputs.get("position_ids"),
        num_items_in_batch=num_items_in_batch,
    )
    return (loss, {"loss": loss}) if return_outputs else loss


def chunked_loss_trainer(base, chunk_tokens: int):
    """SFTTrainer subclass whose compute_loss is compute_chunked_loss with
    the recipe's memory.head_chunk_tokens. base is passed in because trl is
    imported lazily, at the one call site that has it."""

    class ChunkedLossSFTTrainer(base):
        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            return compute_chunked_loss(
                chunk_tokens, model, inputs, return_outputs, num_items_in_batch
            )

    return ChunkedLossSFTTrainer


# -- orchestration -----------------------------------------------------------


def train_sft(recipe: DictConfig) -> SftResult:
    """Run the recipe's supervised fine-tune end to end.

    Preflight (the sft group, then the lock) precedes any heavy import or
    model download. Running inference containers are
    not probed: the operator manages containers manually and validates GPU
    exclusivity themselves. WatchdogAbort propagates to the caller after an
    emergency checkpoint is written; the D9 lock releases on every exit path
    via the context manager.
    """
    if not recipe.sft.enabled:
        raise SftPreflightError(
            "sft.enabled is false; train-sft needs an sft-enabled recipe "
            "(the reference is training/recipes/reference/sft-qlora-e2b.yaml)"
        )

    with run_lock(recipe.run.name):
        return _train_locked(recipe)


def _train_locked(recipe: DictConfig) -> SftResult:
    import torch  # noqa: F401  (fail here, before run-dir side effects)

    from training.runtime.tracking import Tracker
    from training.runtime.watchdog import Watchdog, WatchdogAbort

    train_samples, holdout_samples = build_dataset(recipe)
    model, tokenizer, report = build_model_and_tokenizer(recipe)
    base_identity = resolve_base_identity_or_none(recipe)

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
            f"{recipe.run.output_root}/{recipe.run.name}"
        )
    set_run_dir(run_dir)
    ckpt_root = os.path.join(recipe.run.output_root, recipe.run.name, "checkpoints")
    os.makedirs(ckpt_root, exist_ok=True)

    recipe_hash = snapshot_hash(recipe)
    tracker = Tracker(
        recipe.tracking,
        recipe.run.name,
        run_dir,
        header={
            "tier": "sft",
            "model_id": recipe.model.hf.id,
            "recipe_hash": recipe_hash,
            "schedule": {
                "type": recipe.schedule.type,
                "warmup_steps": recipe.schedule.warmup_steps,
                "total_steps": recipe.schedule.total_steps,
            },
        },
    )
    watchdog = None
    if recipe.watchdog.enabled:
        watchdog = Watchdog(
            recipe.watchdog, run_dir, tracker=tracker, disk_path=ckpt_root
        )
        watchdog.start()

    # The watchdog thread and tracker are live from here on: trainer
    # construction failures (bad optimizer mapping, resume path, TRL errors)
    # must still stop the thread and close the tracker, or the process hangs.
    try:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer

        kwargs = sft_config_kwargs(recipe, ckpt_root, bool(holdout_samples))
        trainer_class = chunked_loss_trainer(
            SFTTrainer, recipe.memory.head_chunk_tokens
        )
        trainer = trainer_class(
            model=model,
            args=SFTConfig(**kwargs),
            train_dataset=Dataset.from_list(train_samples),
            eval_dataset=(
                Dataset.from_list(holdout_samples)
                if kwargs["eval_strategy"] != "no"
                else None
            ),
            processing_class=tokenizer,
            data_collator=Collator(),
            callbacks=build_callbacks(
                recipe,
                ckpt_root,
                tracker,
                watchdog,
                recipe_hash,
                trainable_report=report,
                base_identity=base_identity,
            ),
        )

        resume_path = None
        if recipe.checkpoint.resume is not None:
            resume_path = resolve_sft_resume(ckpt_root, recipe.checkpoint.resume)

        try:
            trainer.train(resume_from_checkpoint=resume_path)
        except WatchdogAbort:
            # Best-effort final state for post-mortem and resume; the abort
            # must surface even if the save itself fails (mirrors the D6 loop).
            try:
                step = trainer.state.global_step
                emergency_dir = os.path.join(
                    ckpt_root, f"{checkpoint_dir_name(step)}-emergency"
                )
                trainer.save_model(emergency_dir)
                # save_state writes trainer_state.json to args.output_dir;
                # point it at the emergency directory so the checkpoint
                # resumes as saved instead of leaving the state file at the
                # checkpoint root.
                original_output_dir = trainer.args.output_dir
                trainer.args.output_dir = emergency_dir
                try:
                    trainer.save_state()
                finally:
                    trainer.args.output_dir = original_output_dir
                write_checkpoint_meta(
                    emergency_dir,
                    step,
                    adapter_only=bool(recipe.peft.enabled),
                    base_model=recipe.model.hf.id,
                    backend=recipe.sft.backend,
                    recipe_hash=recipe_hash,
                    base_revision=getattr(base_identity, "revision", None),
                    base_config_sha256=getattr(base_identity, "config_sha256", None),
                    emergency=True,
                    attention_implementation=_attention_implementation(trainer.model),
                )
            except Exception as exc:
                print(
                    f"[sft] WARN: emergency checkpoint failed: {exc}",
                    file=sys.stderr,
                )
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
        losses=losses,
    )
