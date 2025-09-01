# AI Model Evaluation Framework

A systematic framework for evaluating Large Language Model performance across cognitive domains.

## Overview

This framework provides automated evaluation of AI models through structured tests covering reasoning, creativity, language, social understanding, and domain integration. The system produces quantitative assessments suitable for model comparison and validation.

## Quick Start

Verify framework functionality:
```bash
make test
```

Evaluate a model:
```bash
python benchmark_runner.py --test-type base --mode single \
  --endpoint http://localhost:8004/completion \
  --model "your-model-name"
```

## Core Components

The framework consists of four main components:

### Test Runner
`benchmark_runner.py` orchestrates test execution, API calls, and result aggregation.

### Domain Tests  
`domains/` contains JSON test definitions organized by cognitive domain (reasoning, creativity, language, social, integration, knowledge).

### Evaluation System
`evaluator/` provides modular scoring algorithms for different assessment types.

### Test Framework
`tests/` contains unit, integration, and validation tests for the framework itself.

## Architecture

```
benchmark_tests/
├── domains/           # Test definitions by cognitive domain
├── evaluator/         # Scoring and assessment modules  
├── tests/            # Framework validation tests
├── benchmark_runner.py    # Main execution engine
└── Makefile          # Command automation
```

## Available Commands

Framework testing:
```bash
make test              # Full test suite
make test MODE=quick   # Fast development tests
make clean             # Remove artifacts
```

Prerequisites and debugging:
```bash
make check-prerequisites  # Verify system readiness
make test-api-suite      # Test LLM connectivity
make debug-help          # Show debug options
```

Domain conversion:
```bash
make convert-core-domains      # Convert domains to instruct format
make test-enhanced-evaluator   # Test enhanced evaluation system
```

## Configuration

Basic setup requires no configuration. For model evaluation, specify your API endpoint:

```bash
export MODEL_ENDPOINT="http://localhost:8004/completion"
```

## Documentation

### Engineering Documentation
- [Installation](docs/engineering/installation.md) - Setup and prerequisites
- [Basic Usage](docs/engineering/basic-usage.md) - Common commands and workflows
- [Architecture](docs/engineering/architecture.md) - System design overview
- [API Reference](docs/engineering/api-reference.md) - Key interfaces and classes
- [Troubleshooting](docs/engineering/troubleshooting.md) - Common issues and solutions

### Domain Documentation  
- [Domain Overview](docs/domains/overview.md) - Cognitive domains and test structure
- [Production Domains](docs/domains/production-domains.md) - Core domain descriptions
- [Base vs Instruct](docs/domains/base-vs-instruct.md) - Model type differences

### Research Documentation
- [Theoretical Foundations](docs/research/theoretical-foundations.md) - Mathematical basis
- [Evaluation Algorithms](docs/research/evaluation-algorithms.md) - Assessment methodologies

### Operations Documentation
- [Deployment](docs/operations/deployment.md) - Production setup
- [Monitoring](docs/operations/monitoring.md) - System health and performance

## License

MIT License - see LICENSE file for details.