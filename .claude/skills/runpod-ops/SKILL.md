---
name: runpod-ops
description: RunPod endpoint operations runbook — checking worker state, forcing scale-down, reading the spend query, and cold-start expectations. Use before and after cloud inference sessions. Forward-looking — RunPod tooling lives outside this repo.
---
# RunPod operations runbook

Reference for managing RunPod serverless/pod endpoints used for cloud inference and
training. RunPod itself is external to this repository; this skill captures the operating
procedure so sessions do not leave workers running or misread spend.

> Status: forward-looking. Wire the concrete commands to whatever RunPod CLI / API wrapper
> the integration roadmap lands (`make agent-usage` becomes the spend entry point). Until
> then, treat the steps below as the checklist and use the RunPod console/API directly.

## Before a cloud session

- Confirm the target endpoint exists and its image tag is pinned (no `:latest`).
- Check current worker state: how many workers are warm vs cold, and max-worker cap.
- Note expected **cold start**: first request after scale-to-zero pays image pull + model
  load; do not attribute that latency to inference.

## During

- Keep an eye on active worker count; a stuck request can hold a worker (and spend) open.
- Route measurement work through the bench protocol (see the `benchmarking` skill) so cloud
  numbers are comparable to local ones and cold-start runs are discarded.

## After a cloud session (required)

1. Force scale-down to zero workers so idle GPUs stop billing.
2. Verify no workers remain warm.
3. Read the spend query (via `make agent-usage` once wired) and record it against the task.

The skeptic and the primary agent both run this checklist: primary before starting cloud
work, skeptic when verifying a "completed" cloud task actually released its resources.
