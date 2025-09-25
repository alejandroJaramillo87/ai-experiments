# Documentation Style Guide

This guide defines the documentation standards for the AI Experiments infrastructure repository. All documentation must follow these conventions to maintain professional quality and consistency.

## Table of Contents

- [Core Philosophy](#core-philosophy)
- [Writing Standards](#writing-standards)
- [Document Structure](#document-structure)
- [Formatting Standards](#formatting-standards)
- [Technical Standards](#technical-standards)
- [Language Conventions](#language-conventions)
- [Code Documentation](#code-documentation)
- [Cross-References](#cross-references)
- [Prohibited Elements](#prohibited-elements)
- [Quality Checklist](#quality-checklist)

## Core Philosophy

### Linux/Unix Documentation Principles

Documentation follows traditional Unix manual page conventions:

- **Accuracy over aesthetics** - Technical correctness is paramount
- **Minimalism** - Include only essential information
- **Clarity** - Unambiguous, precise language
- **Examples** - Practical, working examples over abstract descriptions
- **No decoration** - No emojis, icons, or visual embellishments

### Open Source Standards

- Assume technical competence of readers
- Provide complete information for reproduction
- Document actual behavior, not intended behavior
- Include performance measurements and benchmarks
- Credit sources and references

### Local Inference Focus

- **Local workstation only** - Document for personal AI workstations, not production servers
- **No cloud deployment** - Avoid AWS, Azure, GCP, or other cloud platform references
- **No CI/CD pipelines** - Focus on direct execution, not automated deployment
- **No production scaling** - Document single-machine inference, not distributed systems
- **Development environment** - Target individual developers and researchers, not operations teams

## Writing Standards

### Voice and Tense

- **Active voice** preferred over passive
  - Good: "The system allocates memory"
  - Avoid: "Memory is allocated by the system"

- **Present tense** for current state
  - Good: "The GPU runs at 2.55 GHz"
  - Avoid: "The GPU will run at 2.55 GHz"

- **Imperative mood** for instructions
  - Good: "Run the setup script"
  - Avoid: "You should run the setup script"

### Tone

- **Professional** - Technical documentation, not marketing
- **Direct** - Get to the point immediately
- **Objective** - No subjective quality judgments
- **Concise** - Shortest accurate description

### Clarity Principles

1. One idea per sentence
2. One topic per paragraph
3. Define before use
4. Concrete over abstract
5. Specific over general

## Document Structure

### Required Elements

Every documentation file must include:

```markdown
# Title

Brief description paragraph explaining the document's purpose.

## Table of Contents
(Required for documents > 500 words)

## Main Content
(Organized with logical heading hierarchy)

---
*Last Updated: YYYY-MM-DD*
```

### Heading Hierarchy

```markdown
# Document Title (H1 - one per document)
## Major Section (H2)
### Subsection (H3)
#### Detail Level (H4 - avoid if possible)
```

### Table of Contents

Required for documents over 500 words:

```markdown
## Table of Contents

- [Major Section](#major-section)
  - [Subsection](#subsection)
- [Another Section](#another-section)
```

## Formatting Standards

### Text Formatting

| Element | Format | Example |
|---------|--------|---------|
| Emphasis | Bold | **Important note** |
| Code/Commands | Backticks | `docker-compose up` |
| File names | Backticks | `pyproject.toml` |
| Variables | Backticks | `$CUDA_HOME` |
| URLs | Bare or markdown | https://example.com |

### Code Blocks

#### Shell Commands

User commands (use $ prefix):
```bash
$ cd /home/user
$ make install
```

Root commands (use # prefix):
```bash
# apt-get update
# systemctl restart docker
```

Output (no prefix):
```
Starting services...
Service started successfully
```

#### Programming Languages

Always specify language:
```python
def calculate_tokens(text):
    return len(text.split())
```

### Tables

Use pipe tables with headers:

```markdown
| Component | Version | Purpose |
|-----------|---------|---------|
| CUDA | 13.0.88 | Parallel computing |
| cuDNN | 9.13.0 | Deep learning |
```

### Lists

Unordered lists (use `-`):
```markdown
- First item
- Second item
  - Nested item
  - Another nested item
```

Ordered lists:
```markdown
1. First step
2. Second step
3. Third step
```

## Technical Standards

### Performance Metrics

Always include units and precision:

```markdown
- Performance: 286.85 tokens/second
- Memory: 15.3GB VRAM
- Latency: 45ms
- Throughput: 1.2GB/s
```

### Version Numbers

Use full semantic versioning:

```markdown
- Software: 1.0.0
- Driver: 580.65.06
- CUDA: 13.0.88
```

### File Paths

Always use absolute paths from repository root:

```markdown
- Configuration: `/docker/llama-gpu/entrypoint.sh`
- Documentation: `/docs/inference/README.md`
- Script: `/scripts/benchmark.py`
```

### Commands and Flags

Document full commands with explanations:

```bash
# Run benchmark with GPU service and custom label
scripts/benchmark.py --service llama-gpu --label "test_run"

# Flags:
# --service: Target service to benchmark
# --label: Identifier for benchmark results
```

## Language Conventions

### Capitalization

- **Sentence case** for headings: "Performance optimization guide"
- **Lowercase** for commands and files: `docker-compose.yaml`
- **Proper nouns** capitalized: Docker, NVIDIA, Python
- **Acronyms** uppercase: GPU, CPU, RAM, VRAM

### Abbreviations

Define on first use:
```markdown
The Large Language Model (LLM) processes requests...
Subsequently, the LLM can be referenced...
```

Common abbreviations that need no definition:
- CPU, GPU, RAM, SSD
- API, URL, JSON
- OS, VM, IDE

### Numbers

- Spell out one through nine in body text
- Use numerals for 10 and above
- Always use numerals for:
  - Metrics and measurements
  - Version numbers
  - Port numbers
  - Technical specifications

### Punctuation

- Oxford comma in series: "CPU, GPU, and memory"
- Single space after periods
- No space before colons
- Hyphenate compound modifiers: "production-ready system"

## Code Documentation

### Inline Comments

Keep minimal and essential:
```python
# Calculate tokens per second
tps = token_count / elapsed_time  # Avoid division by zero handled above
```

### Configuration Files

Document purpose and options:
```yaml
# Maximum batch size for inference
# Higher values improve throughput but increase memory usage
# Default: 2048, Range: 512-4096
batch_size: 2048
```

## Cross-References

### Internal Links

Use relative paths:
```markdown
See [GPU optimization guide](../optimizations/gpu/gpu-optimizations.md)
Details in [benchmarking section](#benchmarking)
```

### External Links

Provide context:
```markdown
Based on [NVIDIA's tuning guide](https://docs.nvidia.com/guide.html)
Uses [llama.cpp](https://github.com/ggerganov/llama.cpp) for inference
```

## Prohibited Elements

### Never Use

- ❌ Emojis or emoticons
- ❌ Exclamation marks (except in warnings)
- ❌ Marketing language ("revolutionary", "amazing")
- ❌ Subjective adjectives ("easy", "simple", "powerful")
- ❌ Contractions ("don't", "can't", "won't")
- ❌ Informal language ("lots of", "a bunch of")
- ❌ Rhetorical questions
- ❌ Personal pronouns in technical descriptions

### Exceptions

Exclamation marks permitted only in:
```bash
#!/bin/bash
```

```markdown
**Warning!** This operation is destructive
```

## Quality Checklist

Before submitting documentation:

### Content Review
- [ ] Technical accuracy verified
- [ ] All commands tested
- [ ] Performance numbers validated
- [ ] Examples execute correctly

### Structure Review
- [ ] Table of Contents matches headings
- [ ] Heading hierarchy is logical
- [ ] Sections flow logically

### Format Review
- [ ] Consistent formatting throughout
- [ ] Code blocks have language specified
- [ ] Tables are properly aligned
- [ ] Lists have consistent markers

### Language Review
- [ ] Grammar and spelling checked
- [ ] No contractions used
- [ ] Capitalization consistent
- [ ] Numbers formatted correctly

### Link Review
- [ ] All internal links work
- [ ] External links are current
- [ ] Cross-references are accurate

### Final Review
- [ ] Follows Linux philosophy
- [ ] No prohibited elements
- [ ] Professional tone maintained
- [ ] Last updated date included

---

*Last Updated: 2025-09-23*