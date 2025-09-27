# AI Infrastructure Base Repository (v1.0.0)

## Current Status

**Version**: v1.0.0 (stable release)
**Mode**: Maintenance - updates for compatibility (CUDA 13 for vLLM, etc.)
**Focus**: Supporting downstream projects while team works on llama-zen-turbo and blog content

## Repository Purpose

Production-ready AI infrastructure for RTX 5090 + AMD Ryzen 9950X workstation. Provides containerized LLM inference services optimized for both latency (llama.cpp) and throughput (vLLM).

**Key Achievement**: 26% latency improvement for CPU inference through systematic optimizations.

## Projects Using This Repository

This repo serves as a git submodule for:

1. **llama-zen-turbo** ([github.com/alejandroJaramillo87/llama-zen-turbo](https://github.com/alejandroJaramillo87/llama-zen-turbo))
   - AMD Zen 5 optimizer for llama.cpp
   - Inherits hugepage wrapper and benchmark tools
   - Targets 4-6x performance improvement

2. **curiosity-chronicles** ([github.com/alejandroJaramillo87/curiosity-chronicles](https://github.com/alejandroJaramillo87/curiosity-chronicles))
   - Technical blog documenting optimization journey
   - Uses benchmark data and documentation
   - Monetization through vendor partnerships

## Infrastructure Services

| Service | Port | Purpose | Key Features |
|---------|------|---------|--------------|
| llama-cpu | 8001 | Low-latency CPU inference | 12 cores, 96GB RAM, 35 tok/s |
| llama-gpu | 8004 | GPU llama.cpp inference | RTX 5090, 287 tok/s |
| vllm-gpu | 8005 | High-throughput batch serving | Optimized for concurrent requests |
| open-webui | 3000 | Web interface | Model interaction UI |

## Hardware Configuration

- **GPU**: RTX 5090 (32GB VRAM, Blackwell architecture)
- **CPU**: AMD Ryzen 9950X (16 cores, Zen 5)
- **Memory**: 128GB DDR5-6000
- **Models**: `/mnt/ai-data/models/`

## Key Optimizations Implemented

### Validated Performance Gains
- **Huge pages**: 26% improvement (see `docker/llama-cpu/hugepage_mmap_wrapper.cpp`)
- **BIOS tuning**: FCLK 2100MHz, C-states disabled
- **CPU allocation**: 12 dedicated cores for single model instance

### Documentation
Detailed optimization guides in `docs/optimizations/`:
- `README.md` - Overview and philosophy
- `bios/` - BIOS settings for Zen 5
- `os/` - Operating system optimizations
- `gpu/` - RTX 5090 specific tuning
- `experiments/` - Future optimization roadmap

## Development Standards

### Style Guides
All projects using this repo should follow:
- `.claude/BASH_OUTPUT_STYLE.md` - Bash output formatting
- `.claude/PYTHON_STYLE.md` - Python coding standards
- `.claude/COMMENT_STYLE.md` - Comment conventions
- `.claude/DOCUMENTATION_STYLE.md` - Documentation standards

### Core Principles
- Unix/Linux philosophy: simple, composable, text-based
- No emojis or decorative elements
- Clear technical writing
- Reproducible benchmarks

## Common Workflows

```bash
# Service management
make up              # Start all services
make status          # Check health
make logs            # View logs

# Development
make benchmark       # Run performance tests
make shell-gpu      # Access GPU container
```

## Maintenance Roadmap

### Next 2 Months (While Working on llama-zen-turbo)
- Monitor for vLLM CUDA 13 support → update when available
- Keep services stable for blog benchmarking
- Document any new findings in experiments/

### Long-term
- Serve as reference infrastructure for AI projects
- Maintain compatibility with latest llama.cpp
- Update for new hardware as needed

## Integration Notes

When using as a submodule:
```bash
# Add to your project
git submodule add https://github.com/alejandroJaramillo87/ai-experiments.git external/ai-experiments

# Pin to v1.0.0
cd external/ai-experiments
git checkout v1.0.0
```

## Important Context

- **Bleeding edge**: First to optimize for RTX 5090 + Zen 5
- **Research-grade**: Discoveries documented in `docs/optimizations/experiments/`
- **Production ready**: 24/7 stable operation validated
- **No compromises**: Optimized for this specific hardware, not portability

---

*This infrastructure powers cutting-edge LLM optimization research and real-world deployment.*