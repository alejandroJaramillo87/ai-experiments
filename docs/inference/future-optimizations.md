# Future inference optimizations

Parked findings from a survey of where custom CUDA or C++ work would help the training and
interpretability tiers. Nothing here is scheduled. Every item is gated behind the conditions in
[Gating conditions](#gating-conditions), and the gates are not met at the time of writing.

The purpose of this document is to record what the survey found so it is not derived a second
time, and to state plainly which findings are actionable now and which are not.

## Table of contents

- [Summary](#summary)
- [Gating conditions](#gating-conditions)
- [Actionable without custom kernels](#actionable-without-custom-kernels)
- [The activation read path](#the-activation-read-path)
- [Steering expressiveness](#steering-expressiveness)
- [The escalation ladder](#the-escalation-ladder)
- [Build details for any compiled target](#build-details-for-any-compiled-target)
- [What the survey ruled out](#what-the-survey-ruled-out)

## Summary

The training tier has no use for custom CUDA. Every hot path already runs on maintained,
hand-tuned kernels that are pinned and building.

The interpretability tier is where owning more of the inference stack would buy capabilities
rather than speed. The capabilities in question are an activation read path on the serving
model and steering operations richer than a static additive vector.

One finding is actionable now and requires no kernel work at all: a pinned dependency that
addresses the largest self-written tensor path in the repository is currently unused.

## Gating conditions

No work described here begins before all three hold:

1. The PENDING acceptance checks in `docs/verification/interp.md` have been executed and
   recorded. At the time of writing every check in that document is PENDING, along with 12 in
   `docs/verification/training-phase1.md` and seven in `docs/verification/training-phase2.md`.
2. A measured baseline exists. `docs/interpretability/platform.md` currently records no
   timings and no VRAM figures. Harvest tokens per second is already a manifest field
   (`interp/harvest.py:418-420`) and costs nothing to collect.
3. A named capability the current stack cannot deliver has been identified, with the
   measurement that demonstrates the shortfall.

Optimizing before measuring is what the pin-then-fix-then-update discipline in `AGENTS.md`
exists to prevent.

## Actionable without custom kernels

`liger-kernel==0.6.2` is pinned in `pyproject.toml` and built into the training image, but
nothing imports it except `training/commands/smoke_test.py`.

It ships a fused linear cross-entropy and fused distillation losses aimed at exactly the
problem in `training/tune/distill.py`: two full `[B, S, 262144]` float32 `log_softmax` tensors
at Gemma 4's vocabulary size. That path is the peak memory contributor in the distillation
recipe, and it is the reason top-K caching is a requirement rather than an optimization and the
reason cached mode forbids packing.

Try it during training verification. A hand-written kernel is defensible only if Liger's
variants cannot express the recipe's alpha, temperature-squared, masked, and top-K-renormalized
form, and even then the first attempt belongs in Triton rather than CUDA C++.

## The activation read path

`docs/interpretability/writeback.md` states that llama-server has no activation read path. That
is true of llama-server, but the underlying capability exists and the blocker is narrower than
the sentence implies.

- `ggml_backend_sched_set_eval_callback` invokes a callback per graph node during compute.
- It is reachable from the library API: `llama_context_params` carries `cb_eval` and
  `cb_eval_user_data`.
- Upstream ships `examples/eval-callback` demonstrating the mechanism.

What is actually missing:

- llama-server never sets `cb_eval`.
- `llama.h` exposes no per-layer hidden state. The only reads are `llama_get_embeddings`,
  `llama_get_embeddings_ith`, and `llama_get_embeddings_seq`, all of which return output
  embeddings.

So reading activations from a llama.cpp process needs server-side plumbing, not a new ggml
operation. Three constraints apply to any such work:

- The callback fires on every node and tensor data lives on the GPU, so a copy per node would
  destroy throughput. Filter by tensor name before copying.
- Graph node names are not a stable interface. Matching on them is the real re-verification
  burden across a pin bump, larger than whether a patch still applies.
- Fused kernels mean some intermediates do not exist as separate nodes.

## Steering expressiveness

Steering today is configuration only. `interp/steering/export.py` writes a control-vector GGUF
and `interp/steering/wiring.py` sets three environment variables; llama-server performs the
addition. The upstream interface is:

```c
int32_t llama_set_adapter_cvec(struct llama_context * ctx, const float * data, size_t len,
                               int32_t n_embd, int32_t il_start, int32_t il_end);
```

One buffer of `n_embd` by `n_layers`, added over one inclusive layer range. That is the entire
expressive surface. Not expressible through it:

- Conditional or per-token gated steering, which requires a per-token decision. ggml has no
  branch, so this becomes compute-always plus a mask multiply.
- Projection or ablation, subtracting a component rather than adding a direction.
- SAE feature clamping, which needs encoder and decoder matrices resident and a matrix multiply
  inserted mid-graph.

The last two require editing per-architecture graph construction, which is among the most
frequently changed code upstream. A carried patch there becomes a merge on every pin bump, and
it does not satisfy the upstreamable-by-default rule in `docker/interp/patches/README.md`.

## The escalation ladder

In cost order. Each rung is justified only by a measurement the previous rung could not
provide.

1. **Stock tooling.** Upstream binaries and the existing images, unmodified.
2. **An environment-gated activation tap**, built into a separate server image so the
   production serving path is untouched. Gated so that an unset variable preserves upstream
   behavior, which makes it plausibly upstreamable.
3. **A custom runtime.** Months of work. Justified only by a capability the first two rungs
   demonstrably cannot reach.

The expected outcome is stopping at rung one or two. Rung three exists to be skipped unless
evidence forces it.

## Build details for any compiled target

`docker/llama-gpu/Dockerfile.llama-gpu` forces the architecture explicitly:

```
-DCMAKE_CUDA_ARCHITECTURES=OFF
-DCMAKE_CUDA_FLAGS="-gencode arch=compute_120a,code=sm_120a"
```

Setting `CMAKE_CUDA_ARCHITECTURES` to `120a` does not work: CMake normalizes it and emits
`compute_120`, silently dropping the architecture-specific suffix. Any future compiled target
for this GPU needs the same explicit gencode, and its first acceptance check should be
`cuobjdump --dump-sass` reporting `sm_120a`.

Note also that `docker/interp/Dockerfile.interp` uses a CUDA runtime base by explicit decision,
because the interpretability stack is pure wheels and nothing compiles. Any compiled extension
in that image reverses that decision and needs a devel base.

## What the survey ruled out

- **CPU and GPU compute splitting in training.** The D9 heterogeneous-compute deliverable is
  DataLoader prefetch, pinned memory, CUDA streams, and core affinity, not compute
  partitioning. The genuine split is ZeRO-Offload, where the optimizer step runs on the CPU
  through DeepSpeed's CPUAdam, already compiled with Zen 5 flags.
- **Teacher generation and serve-evaluation.** Inference-bound, but the inference belongs to
  vLLM and llama.cpp. A hand-written runtime would lose to both.
- **SAE training.** Roughly 75 million parameters at 16k width on `d_model` 2304, which is
  trivial VRAM. The bottleneck is NVMe read throughput over the activation cache, making it an
  I/O problem rather than a compute one. Raise the runner's buffer sizes before considering
  anything else.
- **Attribution graphs at 32x transcoder width.** VRAM-bound, needing 40GB or more against a
  32GB budget. This is the one place a memory-frugal kernel would unlock a capability rather
  than accelerate one, but the upstream checkout is pinned and treated as unforkable.
