"""Train and evaluate a byte-level BPE tokenizer (Phase 0 D3).

For the Phase 3 from-scratch path only; fine-tuning and continual pre-training
reuse the base model's tokenizer. Thin subcommand over training/data/tokenizer.py.

  python -m training train-tokenizer \
      --corpus "/mnt/ai-data/training/datasets/mix/*.jsonl" \
      --vocab-size 49152 --sample-bytes 10_000_000_000 \
      --eval-corpus web=/mnt/ai-data/training/datasets/eval/web.jsonl \
      --eval-corpus code=/mnt/ai-data/training/datasets/eval/code.jsonl \
      --out /mnt/ai-data/training/tokenizers/base-49k

Exit codes: 0 success, 1 training/evaluation failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse
import json
import os

from common.output import Printer
from training.data import tokenizer as tokenizer_mod
from training.data.prepare import iter_jsonl, load_tokenizer

DEFAULT_REFERENCES = ("HuggingFaceTB/SmolLM2-135M", "gpt2")


def positive_int(value: str) -> int:
    n = int(value.replace("_", ""))
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return n


def eval_corpus_pair(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"expected DOMAIN=PATH (e.g. web=/data/web.jsonl), got {value}")
    domain, path = value.split("=", 1)
    return domain, path


def configure(sub) -> None:
    parser = sub.add_parser(
        "train-tokenizer",
        help="train a byte-level BPE tokenizer for from-scratch pretraining (D3)",
        description="Train a byte-level BPE tokenizer for from-scratch pretraining. "
                    "Fine-tuning and continual pre-training reuse the base model's "
                    "tokenizer; do not use this tool for those paths.")
    parser.add_argument("--corpus", required=True, action="append",
                        help="JSONL file or glob (repeatable; quote globs)")
    parser.add_argument("--out", required=True,
                        help="Artifact directory (convention: /mnt/ai-data/training/tokenizers/<name>)")
    parser.add_argument("--vocab-size", type=positive_int,
                        default=tokenizer_mod.DEFAULT_VOCAB_SIZE,
                        help=f"Total vocabulary size including the 19 special ids "
                             f"(default: {tokenizer_mod.DEFAULT_VOCAB_SIZE})")
    parser.add_argument("--sample-bytes", type=positive_int,
                        default=tokenizer_mod.DEFAULT_SAMPLE_BYTES,
                        help="Stop sampling training text after this many UTF-8 bytes "
                             "(underscores allowed: 10_000_000_000)")
    parser.add_argument("--text-field", default="text",
                        help="Text field name in the JSONL records (default: text)")
    parser.add_argument("--eval-corpus", action="append", type=eval_corpus_pair,
                        default=[], metavar="DOMAIN=PATH",
                        help="Held-out evaluation JSONL per domain (repeatable; "
                             "two domains, e.g. web and code, per the D3 design)")
    parser.add_argument("--eval-docs", type=positive_int,
                        default=tokenizer_mod.DEFAULT_EVAL_DOCS,
                        help=f"Documents per evaluation domain "
                             f"(default: {tokenizer_mod.DEFAULT_EVAL_DOCS})")
    parser.add_argument("--reference", action="append", default=None,
                        help=f"Reference tokenizer for metric comparison (repeatable; "
                             f"default: {', '.join(DEFAULT_REFERENCES)})")
    parser.add_argument("--no-references", action="store_true",
                        help="Skip reference comparison (offline runs)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing artifact directory")
    parser.set_defaults(func=cmd)


def metrics_table(out: Printer, metrics: dict, references: dict) -> None:
    """Fertility/compression table for the new tokenizer and each reference."""
    domains = list(metrics)
    header = f"{'tokenizer':<40} {'domain':<12} {'fertility':>10} {'compression':>12}"
    out.line(header)
    out.line("-" * len(header))
    for name, per_domain in [("trained", metrics)] + sorted(references.items()):
        for domain in domains:
            stats = per_domain.get(domain)
            if stats is None:
                continue
            out.line(f"{name:<40} {domain:<12} "
                     f"{stats['fertility']:>10.4f} {stats['compression']:>12.4f}")


def cmd(args: argparse.Namespace) -> int:
    out = Printer()

    out.line("=== Tokenizer training (Phase 0 D3) ===")
    out.line()
    out.info("From-scratch path only: fine-tunes reuse the base model's tokenizer")

    try:
        corpus = tokenizer_mod.expand_corpus(args.corpus)
    except FileNotFoundError as exc:
        out.error(str(exc))
        out.line("  Solution: check the corpus path or quote the glob pattern")
        return 2

    tokenizer_path = os.path.join(args.out, tokenizer_mod.TOKENIZER_NAME)
    if os.path.exists(tokenizer_path) and not args.force:
        out.error(f"{args.out} already contains {tokenizer_mod.TOKENIZER_NAME}")
        out.line("  Solution: choose a new output directory or pass --force")
        return 1

    out.info(f"Corpus files: {len(corpus)}")
    out.info(f"Vocab size: {args.vocab_size:,}")
    out.info(f"Sample budget: {args.sample_bytes:,} bytes")
    out.info(f"Output: {args.out}")
    out.line()

    sampler = tokenizer_mod.CorpusSampler(corpus, args.text_field, args.sample_bytes)
    out.run("Training byte-level BPE")
    try:
        tok = tokenizer_mod.train(sampler.training_documents(), args.vocab_size)
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        out.error(f"Training failed: {exc}")
        return 1
    out.ok(f"Trained: {sampler.sampled_documents:,} documents, "
           f"{sampler.sampled_bytes:,} bytes sampled")

    # Held-out evaluation domains: explicit --eval-corpus files, else the
    # unconsumed tail of the training corpus (unseen only if the corpus
    # exceeded the byte budget)
    domains: dict[str, list[str]] = {}
    for domain, path in args.eval_corpus:
        texts = []
        for text in iter_jsonl(path, args.text_field):
            texts.append(text)
            if len(texts) == args.eval_docs:
                break
        domains[domain] = texts
    if not domains:
        out.warn("No --eval-corpus given; evaluating on the corpus tail "
                 "(two held-out domains, web and code, are the D3 default)")
        tail = sampler.heldout_documents(args.eval_docs)
        if not tail:
            out.error("Corpus exhausted during training; no held-out documents remain")
            out.line("  Solution: pass --eval-corpus DOMAIN=PATH or lower --sample-bytes")
            return 1
        domains["corpus-tail"] = tail
    if len(domains) < 2:
        out.warn(f"Only {len(domains)} evaluation domain(s); D3 recommends at least two")

    out.run("Evaluating on held-out documents")
    metrics = tokenizer_mod.evaluate(tok, domains)
    failures = {d: s["roundtrip_failures"] for d, s in metrics.items()
                if s["roundtrip_failures"]}
    if failures:
        out.error(f"Round-trip losslessness failed: {failures}")
        return 1
    for domain, stats in metrics.items():
        out.ok(f"{domain}: {stats['documents']} documents round-trip lossless", indent=2)

    references: dict[str, dict] = {}
    if args.no_references:
        out.skip("Reference comparison disabled (--no-references)")
    else:
        for ref_source in (args.reference or list(DEFAULT_REFERENCES)):
            out.run(f"Reference metrics: {ref_source}")
            try:
                reference = load_tokenizer(ref_source)
            except Exception as exc:   # hub fetch: network/auth failures degrade to a skip
                out.warn(f"Reference {ref_source} unavailable: {exc}")
                out.line("  Solution: check network/credentials, or pass --no-references")
                continue
            references[ref_source] = tokenizer_mod.evaluate_reference(reference, domains)

    try:
        manifest = tokenizer_mod.write_artifact(
            tok, args.out, corpus=corpus, text_field=args.text_field,
            sampled_bytes=sampler.sampled_bytes,
            sampled_documents=sampler.sampled_documents,
            vocab_size=args.vocab_size, metrics=metrics, references=references,
            force=args.force)
    except FileExistsError as exc:
        out.error(str(exc))
        out.line("  Solution: choose a new output directory or pass --force")
        return 1

    out.line()
    metrics_table(out, metrics, references)
    out.line()
    out.ok(f"Artifact: {tokenizer_path}")
    out.ok(f"Tokenizer hash: {manifest['tokenizer_hash']}")
    return 0
