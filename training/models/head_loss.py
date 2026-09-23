"""Chunked causal-LM loss: the head, softcap and cross-entropy one slice of
tokens at a time (KB-87).

A causal-LM forward with labels materializes micro_batch x seq_len x vocab
logits and holds several copies of them through the loss: on Gemma 4 the
262144-wide head plus final_logit_softcapping (a divide, a tanh whose output
is saved, a multiply), transformers' fp32 upcast in ForCausalLMLoss, the saved
log-softmax, and the fp32 and bf16 gradients on the way back. About 6 bytes
per vocabulary entry per token, 24 GB at 4096 tokens per micro-batch, outside
every checkpointed decoder block; TRL's SFTTrainer reads the full logits twice
more for its entropy and accuracy metrics. FT.calibrate-qlora's fourth attempt
(2026-09-20) ran into it at 4096 tokens with 6.37 GB resident (a peak read at
an OOM is the card, not the term: the step loop was also running without
gradient checkpointing then, KB-89).

causal_lm_loss runs the decoder once and then computes the head and the loss
over chunk_tokens-sized slices of the flattened hidden states under
torch.utils.checkpoint, so one chunk's logits are alive at a time in the
forward and again in the backward. The value equals the HF loss (a sum over
supervised positions divided by the same denominator). Both the calibration
loop and the SFT trainer go through this function, so the calibrated peak is
the trainer's peak.

torch is imported lazily; the pure helpers are testable without it.
"""

from __future__ import annotations

IGNORE_INDEX = -100


def chunk_bounds(tokens: int, chunk_tokens: int) -> list[tuple[int, int]]:
    """[start, end) slices covering tokens in chunk_tokens-sized pieces; the
    last piece is shorter when chunk_tokens does not divide tokens."""
    if chunk_tokens <= 0:
        raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")
    return [
        (start, min(start + chunk_tokens, tokens))
        for start in range(0, tokens, chunk_tokens)
    ]


def shift_labels(labels: list[list[int]], ignore_index: int = IGNORE_INDEX):
    """Labels aligned with a next-token prediction, as ForCausalLMLoss shifts
    them: position t predicts label t+1, and the last position predicts the
    ignore index. Pure-list form for the sandbox tests; the tensor form is
    inline in causal_lm_loss."""
    return [row[1:] + [ignore_index] for row in labels]


def unwrap_causal_lm(model):
    """The HF causal-LM under a PEFT wrapper, or the model itself."""
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def final_logit_softcapping(config) -> float | None:
    """Gemma 4's head softcap, from the text config of a composite checkpoint
    or the config itself; None for a family without one (scratch Llama)."""
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    return getattr(text, "final_logit_softcapping", None)


def causal_lm_loss(
    model,
    input_ids,
    labels,
    *,
    chunk_tokens: int,
    attention_mask=None,
    position_ids=None,
    num_items_in_batch=None,
    ignore_index: int = IGNORE_INDEX,
):
    """The recipe model's next-token loss over the batch, chunked (KB-87).

    Runs the decoder (model.model: base_model_prefix on both
    Gemma4ForCausalLM and Gemma4ForConditionalGeneration) and the head
    (model.lm_head) of the HF model under any PEFT wrapper. LoRA layers are
    injected in place, so the adapter and the decoder's own gradient
    checkpointing are in effect. Reduction follows transformers: the sum over
    supervised positions divided by num_items_in_batch when the Trainer
    supplies it (the gradient-accumulation-correct denominator), else by the
    count of supervised positions in this batch.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    hf = unwrap_causal_lm(model)
    decoder, head = hf.model, hf.lm_head
    softcap = final_logit_softcapping(hf.config)

    kwargs = {"use_cache": False}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    hidden = decoder(input_ids=input_ids, **kwargs).last_hidden_state
    batch, seq_len, width = hidden.shape
    hidden = hidden.reshape(batch * seq_len, width)

    shifted = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
    shifted = shifted[..., 1:].reshape(-1).to(hidden.device)

    def chunk_loss(h, y):
        logits = head(h)
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        return torch.nn.functional.cross_entropy(
            logits.float(), y, ignore_index=ignore_index, reduction="sum"
        )

    parts = [
        checkpoint(
            chunk_loss, hidden[start:end], shifted[start:end], use_reentrant=False
        )
        for start, end in chunk_bounds(batch * seq_len, chunk_tokens)
    ]
    total = parts[0]
    for part in parts[1:]:
        total = total + part

    if num_items_in_batch is None:
        denominator = (shifted != ignore_index).sum().clamp(min=1)
    else:
        denominator = num_items_in_batch
    if torch.is_tensor(denominator):
        denominator = denominator.to(total.device)
    return total / denominator
