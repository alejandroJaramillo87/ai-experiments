"""Distillation harness (Phase 1 F5).

Two tiers (resolved F5 decisions):

- tier: response — SFT on an F4 teacher-response dataset through the
  unmodified F3 trainer. train_distill() delegates to training/tune/sft.py; the
  recipe's distill block only adds provenance (the F4 teacher manifest),
  which flows into the run snapshot and checkpoint meta.json.
- tier: logit — token-level KL against teacher logits, teacher in-process.
  DistillTrainer subclasses TRL's SFTTrainer overriding compute_loss with
  L = alpha * CE + (1 - alpha) * T^2 * KL(student/T || teacher/T), KL computed
  only where F1's labels are unmasked (assistant tokens). GKDTrainer was
  evaluated and rejected: on-policy generation per step and a JSD loss do not
  match this offline setup, and it has no cached-teacher path (resolved F5
  decision 2).

Teacher execution modes: online runs a frozen no-grad teacher forward per
batch (optionally nf4); cached reads the top-K logprob cache built by
build_logit_cache() (training/data/distill.py) and loads no teacher, at the
cost of top-K truncation. Preflights extend F3's sequence (validate, memory
fit, lock, services down) with tokenizer-identity and logit-softcapping parity
checks: a hash mismatch or capped/uncapped mismatch silently skews the KL, so
both refuse at startup.

Held-out evaluation stays plain CE (no KL term), so the holdout curve is
comparable with F3 runs and usable as the F7 --baseline signal.

torch/trl/transformers are imported lazily so the CPU sandbox can import and
unit-test the pure loss math and preflight helpers without the training
environment.
"""

from __future__ import annotations

import datetime
import json
import os
import sys

from omegaconf import DictConfig

from common.obs import set_run_dir
from training.data.prepare import load_tokenizer
from training.data.sft import IGNORE_INDEX, TRAIN_FILE
from training.recipes import create_run_dir
from training.runtime.lock import run_lock
from training.runtime.loop import snapshot_hash
from training.tune.sft import (
    Collator,
    SftPreflightError,
    SftResult,
    build_dataset,
    build_model_and_tokenizer,
    check_memory_fit,
    resolve_base_identity_or_none,
    resolve_sft_resume,
    sft_config_kwargs,
)
from training.tune.sft_callbacks import (
    build_callbacks,
    checkpoint_dir_name,
    write_checkpoint_meta,
)

PROVENANCE_NAME = "distill_provenance.json"
CACHE_BATCH_EXAMPLES = 8  # pre-pass batch size; teacher-only, VRAM is ample
LSE_CHUNK_POSITIONS = 512  # positions per fp32 logsumexp chunk (cache pre-pass)


class DistillPreflightError(SftPreflightError):
    """A distillation preflight failed; nothing was started."""


# -- loss (pure given tensors; the golden-fixture tests exercise this) -------


def masked_distill_loss(
    student_logits,
    labels,
    *,
    alpha: float,
    temperature: float,
    kl_direction: str,
    teacher_logits=None,
    teacher_topk_ids=None,
    teacher_topk_logprobs=None,
):
    """L = alpha * CE + (1 - alpha) * T^2 * KL, on assistant tokens only.

    Applies the causal shift internally: position t's distributions predict
    token t+1, so logits[:, :-1] pair with labels[:, 1:], and the KL mask is
    exactly the shifted F1 label mask (prompt tokens contribute zero KL).

    Exactly one teacher form must be given: full logits (online mode, same
    shape as student_logits) or a top-K cache slice per input position
    (cached mode: ids [B, S, K] and logprobs [B, S, K]). Cached KL is computed
    over the teacher's top-K support with both distributions renormalized onto
    it — log-softmax over the K gathered values — so the truncation error is
    bounded by the mass outside the support (the online-versus-cached
    tolerance test measures it). Returns (loss, ce, kl) scalars.
    """
    import torch
    import torch.nn.functional as F

    if (teacher_logits is None) == (teacher_topk_ids is None):
        raise ValueError(
            "exactly one of teacher_logits or teacher_topk_ids/"
            "teacher_topk_logprobs is required"
        )

    s_logits = student_logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != IGNORE_INDEX

    ce = F.cross_entropy(
        s_logits.reshape(-1, s_logits.size(-1)),
        shifted_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )

    if teacher_logits is not None:
        s_lp = F.log_softmax(s_logits / temperature, dim=-1)
        t_lp = F.log_softmax(teacher_logits[:, :-1].float() / temperature, dim=-1)
    else:
        # Cache stores log p at T=1; dividing by T differs from logits/T only
        # by a per-position constant, which the renormalization absorbs.
        ids = teacher_topk_ids[:, :-1]
        t_top = teacher_topk_logprobs[:, :-1].float() / temperature
        t_lp = F.log_softmax(t_top, dim=-1)
        # Gather from raw logits: the full-vocab normalizer is a per-position
        # constant that the K-support renormalization cancels, so the
        # [B, S, V] student log-softmax never needs to be materialized.
        s_top = torch.gather(s_logits / temperature, dim=-1, index=ids)
        s_lp = F.log_softmax(s_top, dim=-1)

    if kl_direction == "forward":  # KL(teacher || student)
        kl_per_pos = (t_lp.exp() * (t_lp - s_lp)).sum(dim=-1)
    else:  # reverse: KL(student || teacher)
        kl_per_pos = (s_lp.exp() * (s_lp - t_lp)).sum(dim=-1)
    kl = (kl_per_pos * mask).sum() / mask.sum().clamp(min=1) * temperature * temperature

    loss = alpha * ce + (1.0 - alpha) * kl
    return loss, ce.detach(), kl.detach()


# -- preflights ---------------------------------------------------------------


def final_logit_softcapping(config):
    """The model's final_logit_softcapping, descending Gemma 4's text_config."""
    text = getattr(config, "text_config", None)
    if text is not None:
        config = text
    return getattr(config, "final_logit_softcapping", None)


def check_softcap_parity(student_config, teacher_config) -> None:
    """Refuse a capped/uncapped teacher-student pair: distilling capped
    logits against uncapped ones (or with different caps) silently skews
    the KL rather than erroring."""
    s_cap = final_logit_softcapping(student_config)
    t_cap = final_logit_softcapping(teacher_config)
    if s_cap != t_cap:
        raise DistillPreflightError(
            f"final_logit_softcapping mismatch: student {s_cap!r} vs teacher "
            f"{t_cap!r}; teacher and student logits must both be capped "
            "identically (or both uncapped) for the KL to be meaningful"
        )


def check_tokenizer_parity(student_source: str, teacher_source: str) -> None:
    """Refuse cross-tokenizer logit distillation (the D3 warning): teacher
    and student tokenizers must hash-match, or per-token KL is meaningless.
    Cross-family pairs (for example a Gemma teacher and a Qwen student) fail
    here by design; cross-tokenizer distillation is out of scope for Phase 1."""
    student_hash = load_tokenizer(student_source).content_hash
    teacher_hash = load_tokenizer(teacher_source).content_hash
    if student_hash != teacher_hash:
        raise DistillPreflightError(
            f"tokenizer mismatch: student {student_source} ({student_hash}) vs "
            f"teacher {teacher_source} ({teacher_hash}); logit distillation "
            "requires identical tokenizers. Cross-tokenizer distillation is "
            "out of scope for Phase 1 (use tier=response instead)"
        )


def _dataset_manifest(dataset_dir: str) -> dict:
    with open(os.path.join(dataset_dir, "manifest.json")) as f:
        return json.load(f)


def _train_file_hash(manifest: dict) -> str:
    for rec in manifest["files"]:
        if rec["file"] == TRAIN_FILE:
            return rec["xxhash64"]
    raise DistillPreflightError(f"dataset manifest lists no {TRAIN_FILE}")


# -- teacher loading ----------------------------------------------------------


def load_teacher(recipe: DictConfig):
    """Load the frozen in-process teacher for online mode.

    Frozen and eval-mode: no grads, no dropout, no gradient checkpointing.
    teacher_quantize: nf4 loads through bitsandbytes to trade exactness of
    dtype for VRAM (the KL is computed on its dequantized logits either way).
    """
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant = None
    if recipe.distill.teacher_quantize == "nf4":
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    from training.models.model import attention_implementation

    teacher = AutoModelForCausalLM.from_pretrained(
        recipe.distill.teacher,
        dtype=torch.bfloat16,
        quantization_config=quant,
        attn_implementation=attention_implementation(),
    )
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


def teacher_revision(teacher_model) -> str:
    return getattr(teacher_model.config, "_commit_hash", None) or "unknown"


# -- collation ----------------------------------------------------------------


class DistillCollator(Collator):
    """F3's padded collator plus per-example teacher cache slices.

    In cached mode each train sample carries a cache_index; the collator
    gathers that example's [tokens, K] slices, pads them to the batch width,
    and attaches teacher_topk_ids/teacher_topk_logprobs. Padded positions are
    zeros — the label mask already excludes them from the KL. Samples without
    a cache_index (the holdout split) collate exactly as F3.
    """

    def __init__(self, cache=None, pad_id: int = 0) -> None:
        super().__init__(pad_id)
        self.cache = cache

    def __call__(self, batch: list[dict]) -> dict:
        import numpy as np
        import torch

        out = super().__call__(batch)
        if self.cache is None or not all("cache_index" in s for s in batch):
            return out
        width = out["input_ids"].shape[1]
        top_k = self.cache.top_k
        ids = torch.zeros((len(batch), width, top_k), dtype=torch.long)
        logprobs = torch.zeros((len(batch), width, top_k), dtype=torch.float)
        for row, sample in enumerate(batch):
            idx_arr, lp_arr = self.cache.example(int(sample["cache_index"]))
            n = len(sample["input_ids"])
            if len(idx_arr) != n:
                from training.data.distill import LogitCacheError

                raise LogitCacheError(
                    f"cache example {sample['cache_index']} has {len(idx_arr)} "
                    f"token positions, sample has {n}; the cache was built "
                    "over a different preparation of this dataset"
                )
            ids[row, :n] = torch.from_numpy(np.asarray(idx_arr, np.int64))
            logprobs[row, :n] = torch.from_numpy(np.asarray(lp_arr, np.float32))
        out["teacher_topk_ids"] = ids
        out["teacher_topk_logprobs"] = logprobs
        return out


# -- trainer ------------------------------------------------------------------


def build_distill_trainer_cls():
    """DistillTrainer, built lazily so importing this module needs no trl."""
    from trl import SFTTrainer

    class DistillTrainer(SFTTrainer):
        """SFTTrainer with the F5 masked-KL compute_loss (resolved decision 2).

        Training batches get alpha-weighted CE plus temperature-scaled KL
        against the teacher (online forward or cached top-K). Evaluation
        stays plain CE so the holdout curve is comparable across F3/F5 runs.
        """

        def __init__(self, *args, teacher=None, distill_cfg=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.teacher = teacher
            self.distill_cfg = distill_cfg
            self.last_ce = None
            self.last_kl = None

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            import torch
            import torch.nn.functional as F

            labels = inputs.pop("labels")
            teacher_ids = inputs.pop("teacher_topk_ids", None)
            teacher_lps = inputs.pop("teacher_topk_logprobs", None)
            outputs = model(**inputs)

            if not model.training:
                logits = outputs.logits[:, :-1].float()
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=IGNORE_INDEX,
                )
                return (loss, outputs) if return_outputs else loss

            cfg = self.distill_cfg
            teacher_logits = None
            if self.teacher is not None:
                with torch.no_grad():
                    teacher_logits = self.teacher(**inputs).logits
            loss, ce, kl = masked_distill_loss(
                outputs.logits,
                labels,
                alpha=cfg.alpha,
                temperature=cfg.temperature,
                kl_direction=cfg.kl_direction,
                teacher_logits=teacher_logits,
                teacher_topk_ids=teacher_ids,
                teacher_topk_logprobs=teacher_lps,
            )
            self.last_ce, self.last_kl = float(ce), float(kl)
            return (loss, outputs) if return_outputs else loss

        def log(self, logs, *args, **kwargs):
            # Surface the last computed CE/KL decomposition beside the
            # windowed loss (OBS-10); the tracking callback maps them to
            # train/ce_loss and train/kl_loss. Guarded on "loss" so eval
            # logs pass untouched.
            if "loss" in logs and self.last_ce is not None:
                logs["ce_loss"], logs["kl_loss"] = self.last_ce, self.last_kl
            return super().log(logs, *args, **kwargs)

    return DistillTrainer


# -- cache pre-pass -----------------------------------------------------------


def build_logit_cache(
    recipe: DictConfig, out_dir: str | None = None, progress=None
) -> dict:
    """Materialize the teacher's top-K logprobs over the train split.

    A training-tier GPU job: takes the D9 lock like a run. The operator
    manages inference containers manually. The teacher runs alone (full
    budget, bf16), in dataset file order, unpacked — the alignment contract
    the reader enforces. Returns the written cache manifest.
    """
    d = recipe.distill
    if not d.enabled or d.tier != "logit":
        raise DistillPreflightError(
            "distill-cache needs distill.enabled=true and tier=logit"
        )
    if not d.teacher:
        raise DistillPreflightError("distill.teacher is required to build a cache")
    out_dir = out_dir or d.cache_path
    if not out_dir:
        raise DistillPreflightError(
            "no cache path: set distill.cache_path or pass --out"
        )

    import numpy as np
    import torch

    from training.data.distill import LogitCacheWriter
    from training.data.sft import load_sft_dataset

    check_tokenizer_parity(recipe.tokenizer.path, d.teacher)
    manifest = _dataset_manifest(recipe.sft.dataset)

    with run_lock(f"{recipe.run.name}-distill-cache"):
        train, _ = load_sft_dataset(recipe.sft.dataset, tokenizer=recipe.tokenizer.path)
        teacher = load_teacher_for_cache(recipe)
        device = next(teacher.parameters()).device
        writer = LogitCacheWriter(out_dir, d.top_k)
        collator = Collator()
        for start in range(0, len(train), CACHE_BATCH_EXAMPLES):
            batch = train[start : start + CACHE_BATCH_EXAMPLES]
            tensors = collator(batch)
            with torch.no_grad():
                logits = teacher(
                    input_ids=tensors["input_ids"].to(device),
                    attention_mask=tensors["attention_mask"].to(device),
                    position_ids=tensors["position_ids"].to(device),
                ).logits
            # topk over logits == topk over logprobs; subtract logsumexp once
            # for the K winners instead of materializing the full log-softmax.
            # topk ranks in the model dtype; only the normalizer needs fp32,
            # computed in position chunks so the [B, S, V] fp32 copy of the
            # logits (which roughly doubles peak memory at Gemma vocab sizes)
            # is never materialized.
            top_vals, top_ids = logits.topk(d.top_k, dim=-1)
            lse = torch.cat(
                [
                    torch.logsumexp(chunk.float(), dim=-1, keepdim=True)
                    for chunk in logits.split(LSE_CHUNK_POSITIONS, dim=1)
                ],
                dim=1,
            )
            logprobs = top_vals.float() - lse
            for row, sample in enumerate(batch):
                n = len(sample["input_ids"])
                writer.add_example(
                    np.asarray(top_ids[row, :n].cpu(), np.int32),
                    np.asarray(logprobs[row, :n].cpu(), np.float16),
                )
            if progress is not None:
                progress(min(start + CACHE_BATCH_EXAMPLES, len(train)), len(train))
        return writer.finalize(
            teacher=d.teacher,
            teacher_revision=teacher_revision(teacher),
            tokenizer_hash=manifest["tokenizer_hash"],
            dataset_dir=recipe.sft.dataset,
            dataset_train_hash=_train_file_hash(manifest),
            command=" ".join(sys.argv),
        )


def load_teacher_for_cache(recipe: DictConfig):
    """Pre-pass teacher: bf16, unquantized (it runs alone; validate() enforces
    teacher_quantize=none for cached recipes), on the available device."""
    import torch
    from transformers import AutoModelForCausalLM

    from training.models.model import attention_implementation

    teacher = AutoModelForCausalLM.from_pretrained(
        recipe.distill.teacher,
        dtype=torch.bfloat16,
        device_map="cuda:0" if torch.cuda.is_available() else None,
        attn_implementation=attention_implementation(),
    )
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


# -- orchestration ------------------------------------------------------------


def distill_provenance(
    recipe: DictConfig, revision: str | None = None, cache_manifest: dict | None = None
) -> dict:
    """The provenance record written into the run dir and checkpoint meta."""
    d = recipe.distill
    fields: dict = {"tier": d.tier}
    if d.tier == "response":
        fields["teacher_manifest"] = d.teacher_manifest
    else:
        fields.update(
            {
                "teacher": d.teacher,
                "teacher_revision": revision or "unknown",
                "mode": d.mode,
                "alpha": d.alpha,
                "temperature": d.temperature,
                "kl_direction": d.kl_direction,
            }
        )
        if d.mode == "cached":
            fields["cache_path"] = d.cache_path
            fields["top_k"] = d.top_k
            if cache_manifest is not None:
                fields["teacher_revision"] = cache_manifest["teacher_revision"]
    return fields


def train_distill(recipe: DictConfig, force: bool = False) -> SftResult:
    """Run the recipe's distillation end to end.

    tier=response delegates to F3's train_sft unchanged (the dataset is the
    F4 artifact; the distill block is provenance). tier=logit mirrors F3's
    preflight-then-train flow with the F5 additions: tokenizer parity before
    anything heavy, softcap parity after model load, teacher or cache wiring,
    and a provenance record in the run dir.
    """
    d = recipe.distill
    if not d.enabled:
        raise DistillPreflightError(
            "distill.enabled is false; train-distill needs a distill-enabled "
            "recipe (the reference is training/recipes/reference/distill-e2b.yaml)"
        )
    if d.tier == "response":
        from training.tune.sft import train_sft

        return train_sft(recipe, force=force)

    check_tokenizer_parity(recipe.tokenizer.path, d.teacher)
    fits, fit_message = check_memory_fit(recipe, force)
    if not fits:
        raise DistillPreflightError(fit_message)

    with run_lock(recipe.run.name):
        return _train_locked_distill(recipe)


def _train_locked_distill(recipe: DictConfig) -> SftResult:
    # Mirrors training/tune/sft.py _train_locked; kept separate rather than hooked
    # into it so F3's flow stays free of distillation branches. Divergences:
    # teacher/cache setup, DistillCollator, DistillTrainer, provenance record.
    import torch  # noqa: F401  (fail here, before run-dir side effects)

    from training.data.distill import LogitCache
    from training.runtime.tracking import Tracker
    from training.runtime.watchdog import Watchdog, WatchdogAbort

    d = recipe.distill
    train_samples, holdout_samples = build_dataset(recipe)

    cache = None
    cache_manifest = None
    if d.mode == "cached":
        cache = LogitCache(d.cache_path)
        manifest = _dataset_manifest(recipe.sft.dataset)
        cache.verify_binding(
            dataset_train_hash=_train_file_hash(manifest),
            tokenizer_hash=manifest["tokenizer_hash"],
            examples=len(train_samples),
        )
        cache_manifest = cache.manifest
        # Alignment index into the cache: samples stay in train-file order
        # (packing is off in cached mode; validate() enforced it).
        train_samples = [dict(s, cache_index=i) for i, s in enumerate(train_samples)]

    model, tokenizer, report = build_model_and_tokenizer(recipe)
    base_identity = resolve_base_identity_or_none(recipe)
    teacher = None
    revision = None
    if d.mode == "online":
        teacher = load_teacher(recipe)
        revision = teacher_revision(teacher)
        check_softcap_parity(model.config, teacher.config)

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

    provenance = distill_provenance(recipe, revision, cache_manifest)
    with open(os.path.join(run_dir, PROVENANCE_NAME), "w") as f:
        json.dump(provenance, f, indent=2)

    recipe_hash = snapshot_hash(recipe)
    tracker = Tracker(
        recipe.tracking,
        recipe.run.name,
        run_dir,
        header={
            "tier": "distill",
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

    from datasets import Dataset
    from trl import SFTConfig

    kwargs = sft_config_kwargs(recipe, ckpt_root, bool(holdout_samples))
    trainer_cls = build_distill_trainer_cls()
    trainer = trainer_cls(
        model=model,
        args=SFTConfig(**kwargs),
        train_dataset=Dataset.from_list(train_samples),
        eval_dataset=(
            Dataset.from_list(holdout_samples)
            if kwargs["eval_strategy"] != "no"
            else None
        ),
        processing_class=tokenizer,
        data_collator=DistillCollator(cache),
        callbacks=build_callbacks(
            recipe,
            ckpt_root,
            tracker,
            watchdog,
            recipe_hash,
            trainable_report=report,
            distill_meta=provenance,
            base_identity=base_identity,
        ),
        teacher=teacher,
        distill_cfg=d,
    )

    if (
        teacher is not None
        and not next(teacher.parameters()).is_cuda
        and torch.cuda.is_available()
    ):
        trainer.teacher = teacher.to(trainer.model.device)

    resume_path = None
    if recipe.checkpoint.resume is not None:
        resume_path = resolve_sft_resume(ckpt_root, recipe.checkpoint.resume)

    try:
        trainer.train(resume_from_checkpoint=resume_path)
    except WatchdogAbort:
        step = trainer.state.global_step
        emergency_dir = os.path.join(
            ckpt_root, f"{checkpoint_dir_name(step)}-emergency"
        )
        trainer.save_model(emergency_dir)
        trainer.save_state()
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
            distill=provenance,
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
