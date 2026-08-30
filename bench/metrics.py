"""Phase-separated metric reduction: per-request samples to reported statistics.

Pure functions over captured timings. Nothing here performs I/O or touches a service,
which is what makes the whole reduction testable in the sandbox against synthetic
arrival-time fixtures -- the half of a benchmark suite that can be proven without the
workstation.

The distinction this module exists to preserve is prefill versus decode. A streaming
client observes one prompt-shaped quantity (time to first token) and one
generation-shaped quantity (the gaps between subsequent tokens), and collapsing them
into a single tokens/second figure is what makes a number unpublishable: it moves when
either phase moves and identifies neither. The previous load generator captured the
per-request arrival array and then pooled every request's arrivals into one flat list,
so the per-token deltas were never differenced.

Every reported scalar is a distribution, not a point. `median` is the headline;
`mean` is carried only so figures recorded by the earlier tooling remain comparable.
"""

from __future__ import annotations

import dataclasses
import math
import statistics

# Below this sample count a 99th percentile is an order statistic of a handful of
# points -- it names the maximum, not a tail. Reported, but flagged, so a table can
# omit it rather than quoting a number that does not mean what it says.
P99_MIN_SAMPLES = 100


@dataclasses.dataclass(frozen=True)
class Sample:
    """One completed request, as the streaming client observed it.

    `arrivals` holds the wall-clock stamp of each content-bearing chunk, taken when the
    SSE line was decoded and before it was parsed. `prompt_tokens` and
    `completion_tokens` come from the server's own `usage` object, never from counting
    SSE frames: a server that packs more than one token into a chunk makes a frame count
    an undercount, silently. Reconciling the two counts is
    `arrivals_resolve_generation`, and what each metric below does when they disagree
    follows the single policy stated there.
    """

    t_submit: float
    arrivals: tuple[float, ...]
    prompt_tokens: int
    completion_tokens: int
    server_prompt_ms: float | None = None
    server_predicted_ms: float | None = None

    @property
    def chunks(self) -> int:
        return len(self.arrivals)

    @property
    def arrivals_resolve_generation(self) -> bool:
        """Whether the arrival series accounts for every token the server generated.

        Fewer arrivals than tokens has three causes that arrival stamps cannot tell
        apart: a server packing two tokens into one chunk, which vLLM does routinely
        under continuous batching; llama.cpp sending no frame at all for a token that
        leaves an incomplete UTF-8 sequence; and a frame that failed to parse being
        skipped here. Reconciliation against the server's own count is the only
        detector available, and it detects the condition, never the cause.

        The policy that follows, applied per metric here and again over a whole point
        in `reduce_samples`: report what packing -- the ordinary cause -- cannot bias,
        and refuse what it can. Packing cannot move the end-to-end span at all (the
        submit stamp at one end, and at the other the last token's text has to ride in
        the last content-bearing chunk), and can only move the first arrival later,
        which is the pessimistic direction. It does distort the first-to-last span,
        which then covers fewer observed intervals than the token count describes with
        nothing to say which went missing, so `tpot_ms` and `decode_tok_s` are refused.
        The rest are reported, with the condition declared in the point's block.
        """
        return self.chunks >= self.completion_tokens

    def ttft_ms(self) -> float | None:
        """Time to first token, client-side.

        At concurrency 1 this is prefill plus one decode step plus the round trip. Above
        concurrency 1 it also contains queue wait, which is why the caller labels the
        block `includes_queue` rather than this function pretending otherwise.

        Reported even when the arrivals do not resolve the generation: a first chunk
        carrying two tokens is stamped when the second was generated, so the value
        reads late and never early. Absent only when no chunk carried text at all.
        """
        if not self.arrivals:
            return None
        return (self.arrivals[0] - self.t_submit) * 1000.0

    def itl_ms(self) -> list[float]:
        """Inter-token latencies: the successive differences of the arrival stamps."""
        return [
            (b - a) * 1000.0
            for a, b in zip(self.arrivals, self.arrivals[1:], strict=False)
        ]

    def tpot_ms(self) -> float | None:
        """Mean time per output token, prefill excluded.

        Undefined for a single-token response: with one arrival there is no interval to
        measure, and returning the whole request duration would silently reintroduce
        prefill. The prefill arm generates exactly one token, so this is the common case
        rather than an edge one.
        """
        if self.completion_tokens < 2 or self.chunks < 2:
            return None
        if not self.arrivals_resolve_generation:
            # The span covers chunks-1 observed intervals and the denominator counts
            # completion_tokens-1 generated ones. Unequal, and which intervals went
            # missing is unknowable, so the quotient is refused rather than reported
            # low.
            return None
        span = self.arrivals[-1] - self.arrivals[0]
        return span / (self.completion_tokens - 1) * 1000.0

    def decode_tok_s(self) -> float | None:
        """Decode throughput, prefill excluded.

        Absent on an unresolved generation, where it reads high by an interval's worth.
        """
        if self.completion_tokens < 2 or self.chunks < 2:
            return None
        if not self.arrivals_resolve_generation:
            return None
        span = self.arrivals[-1] - self.arrivals[0]
        if span <= 0:
            return None
        return (self.completion_tokens - 1) / span

    def e2e_tok_s(self) -> float | None:
        """End-to-end throughput: completion tokens over the whole request.

        This is the figure the earlier tooling reported as "tokens/second", prefill
        folded in. Kept, and named for what it is, so new measurements can be reconciled
        against the numbers already quoted in docs/optimizations/.

        Reported on an unresolved generation, because both ends of this span are
        invariant under the packing that is the ordinary cause of one. A token withheld
        at the very end does shorten the span while the numerator stays at the server's
        count, but by less than the same lost arrival moves the decode rate (1.61% at
        the quick profile's 64 generated tokens, 0.39% at the publish profile's 256),
        since this span also carries prefill. Both are under the 5% threshold at which
        this suite reports a change.
        """
        if not self.arrivals:
            return None
        span = self.arrivals[-1] - self.t_submit
        if span <= 0:
            return None
        return self.completion_tokens / span


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile on a sorted copy; q in [0, 1]."""
    if not values:
        raise ValueError("percentile of an empty sample")
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def stats(values: list[float]) -> dict | None:
    """The reported shape for one scalar across repeats.

    Returns None for an empty sample rather than raising: a metric that does not apply
    to an arm (TPOT in the prefill arm, where every response is one token) is absent
    from the report, not zero.
    """
    if not values:
        return None
    n = len(values)
    block = {
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
        # statistics.stdev needs two points. One repeat has no spread to report, which
        # is a fact about the run rather than a spread of zero.
        "stdev": statistics.stdev(values) if n > 1 else None,
        "mean": statistics.fmean(values),
        "n": n,
    }
    if n < P99_MIN_SAMPLES:
        block["p99_meaningful"] = False
    return block


def _collect(samples: list[Sample], fn) -> list[float]:
    return [v for v in (fn(s) for s in samples) if v is not None]


def reduce_samples(samples: list[Sample], concurrency: int) -> dict:
    """Reduce one workload point's samples into its reported metric mapping.

    Returns statistics only. A point's identity -- which prompt length, which generation
    length, which concurrency -- is attached by the runner alongside this mapping rather
    than inside it, so that a reader keys on the identity and never on a statistic that
    happens to share its name. The observed token counts below are statistics: they
    record what the server actually tokenized, which is not always what was asked for.
    """
    if not samples:
        raise ValueError("no samples to reduce")

    itl: list[float] = []
    for sample in samples:
        itl.extend(sample.itl_ms())

    # Fewer arrivals than the server's token count, counted once. Both the packing flag
    # and the reconciliation count below are this one predicate reported two ways: a
    # sample whose mean tokens per chunk exceeds one is exactly a sample that did not
    # resolve its own generation.
    arrivals_incomplete = sum(1 for s in samples if not s.arrivals_resolve_generation)

    # Refusal is decided for the point, not sample by sample. Per sample it would take
    # the median over the samples that resolved, and those are not a random subset --
    # a sample loses arrivals when the server packed its chunks, which happens under
    # the heaviest batching, so the survivors are the faster requests and the median
    # would read better than the run was.
    span_resolved = arrivals_incomplete == 0

    # A chunk carrying more than one token makes the ITL series coarser than per-token.
    # Reporting it as a per-token latency anyway would be a quiet lie, so the block says
    # so. What it can say is each sample's mean tokens per chunk, and the largest of
    # those means across the samples -- not the worst single chunk, which arrival stamps
    # cannot observe: eight tokens in four chunks is a mean of 2.0 whether they packed
    # 2/2/2/2 or 5/1/1/1.
    tokens_per_chunk_mean_max = max(
        (s.completion_tokens / s.chunks for s in samples if s.chunks), default=1.0
    )

    block = {
        "n": len(samples),
        "concurrency": concurrency,
        # TTFT is queue-inclusive above concurrency 1. llama.cpp serves one request at a
        # time (common/serving.py), so this is only ever true for vLLM's batching sweep.
        "includes_queue": concurrency > 1,
        "ttft_ms": stats(_collect(samples, Sample.ttft_ms)),
        "itl_ms": stats(itl),
        "tpot_ms": (
            stats(_collect(samples, Sample.tpot_ms)) if span_resolved else None
        ),
        "decode_tok_s": (
            stats(_collect(samples, Sample.decode_tok_s)) if span_resolved else None
        ),
        "e2e_tok_s": stats(_collect(samples, Sample.e2e_tok_s)),
        "itl_approximate": arrivals_incomplete > 0,
        "tokens_per_chunk_mean_max": tokens_per_chunk_mean_max,
        # How many samples produced fewer arrivals than the server reported tokens.
        # Nonzero refuses `tpot_ms` and `decode_tok_s` for the whole point, and this
        # count against `n` above is what makes that absence reconcilable: a metric
        # nobody could measure versus a metric refused for a stated number of samples.
        # The other metrics still cover all `n`.
        "arrivals_incomplete": arrivals_incomplete,
        "prompt_tokens_observed": stats([float(s.prompt_tokens) for s in samples]),
        "completion_tokens_observed": stats(
            [float(s.completion_tokens) for s in samples]
        ),
    }

    server_prefill = _collect(
        samples,
        lambda s: (
            s.prompt_tokens / (s.server_prompt_ms / 1000.0)
            if s.server_prompt_ms
            else None
        ),
    )
    block["prefill_tok_s_server"] = stats(server_prefill)
    return block


def prefill_slope(points: list[tuple[float, float]]) -> dict | None:
    """Fit prefill throughput from a time-to-first-token sweep.

    `points` is (prompt_tokens, median ttft_ms). The token count is a float because the
    caller prefers the server's own observed median over the integer estimate that sliced
    the prompt; see bench/runner.py::slope_axis for which one a given fit used. Least squares through them gives a line
    whose slope is milliseconds per prompt token and whose intercept is the fixed
    per-request cost -- the HTTP round trip, tokenization, and the single decode step
    that a max_tokens=1 response still pays.

    The slope is what makes this honest. Dividing prompt tokens by TTFT at one length
    charges that fixed cost to prefill and understates it by however large the intercept
    is; taking the slope across lengths cancels it. This is the only defensible
    client-side prefill figure, and the only one available at all for an engine that
    reports no server-side timings.

    r2 is reported so a caller can tell a clean fit from a noisy one, and is absent
    below three points because a line through two of them leaves no residual to
    measure. It says nothing about a sweep too short to fit at all: that case returns
    no fit. The intercept is reported because a large one is itself the finding: it
    means no single-point prefill claim about this service is safe.
    """
    if len(points) < 2:
        return None
    xs = [float(x) for x, _ in points]
    ys = [y for _, y in points]
    n = len(xs)
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    slope_ms_per_token = sxy / sxx
    intercept_ms = mean_y - slope_ms_per_token * mean_x

    # A two-point fit has no residual degrees of freedom: the line passes through both
    # points, ss_res is zero by construction, and r2 comes out exactly 1.0 whatever the
    # data. Absent rather than 1.0, because the `quick` profile sweeps two prefill
    # lengths and an acceptance check reading r2 >= 0.95 there would pass on any input.
    r2: float | None = None
    if n > 2:
        ss_tot = sum((y - mean_y) ** 2 for y in ys)
        ss_res = sum(
            (y - (intercept_ms + slope_ms_per_token * x)) ** 2
            for x, y in zip(xs, ys, strict=True)
        )
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "prefill_tok_s_slope": (
            1000.0 / slope_ms_per_token if slope_ms_per_token > 0 else None
        ),
        "slope_ms_per_token": slope_ms_per_token,
        "intercept_ms": intercept_ms,
        "r2": r2,
        "points": n,
    }
