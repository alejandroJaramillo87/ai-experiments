# Architecture

High-level system design of the AI Model Evaluation Framework.

## System Overview

The framework follows a modular plugin architecture with clear separation between test execution, evaluation logic, and domain knowledge.

## Core Components

### Test Runner
`benchmark_runner.py` serves as the main orchestration engine:

- Loads test definitions from JSON files
- Makes HTTP API calls to language models  
- Routes responses to appropriate evaluators
- Aggregates results and handles output
- Manages performance monitoring and error handling

### Domain System
`domains/` directory contains test definitions organized by cognitive type:

- JSON files define prompts, expected patterns, and scoring criteria
- Hierarchical organization: domain → model_type → difficulty
- Metadata includes cultural context and evaluation requirements
- Support for both base and instruct model formats

### Evaluation System
`evaluator/` directory provides modular assessment capabilities:

- Subject evaluators for each cognitive domain (reasoning, creativity, etc.)
- Advanced analytics modules (entropy calculation, semantic analysis)
- Cultural authenticity assessment components
- Configurable scoring algorithms with fallback strategies

### Test Framework
`tests/` directory contains framework validation:

- Unit tests for individual components
- Integration tests for end-to-end workflows  
- Shared test infrastructure for consistent setup
- Calibration tests for evaluation accuracy

## Data Flow

```
1. Test Runner loads domain definitions
2. Test Runner sends prompts to model API
3. Model responses routed to appropriate evaluators  
4. Evaluators compute multidimensional scores
5. Results aggregated and saved to output directory
```

## Key Design Principles

### Modularity
Each evaluator is independent and can be used standalone. Advanced analytics modules are optional and degrade gracefully when unavailable.

### Configurability  
Evaluation strategies can be configured via environment variables. Fallback methods ensure operation even without optional dependencies.

### Extensibility
New domains can be added by creating JSON test definitions. New evaluators can be added by implementing the base evaluation interface.

### Reliability
Comprehensive error handling and graceful degradation ensure robust operation across different environments.

## Directory Structure

```
benchmark_tests/
├── benchmark_runner.py           # Main execution engine
├── domains/                      # Test definitions
│   ├── reasoning/               # Logic and analysis tests
│   ├── creativity/              # Creative expression tests  
│   ├── language/                # Linguistic competency tests
│   ├── social/                  # Social understanding tests
│   ├── integration/             # Multi-domain synthesis tests
│   └── knowledge/               # Factual knowledge tests
├── evaluator/                   # Assessment modules
│   ├── subjects/                # Domain-specific evaluators
│   ├── advanced/                # Analytics modules
│   ├── cultural/                # Cultural assessment
│   ├── core/                    # Base classes and orchestration
│   └── validation/              # Fact-checking and validation
├── tests/                       # Framework tests
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   ├── shared/                  # Common test infrastructure
│   ├── functional/              # End-to-end tests
│   └── calibration/             # Evaluation calibration
└── Makefile                     # Command automation
```

## Integration Points

### Model APIs
The framework supports HTTP APIs with JSON payloads. Common formats include OpenAI-compatible and llama.cpp endpoints.

### Evaluation Pipeline
Results flow through a standardized pipeline that supports both individual test execution and batch processing.

### Extension Interface
New evaluators implement the `MultiDimensionalEvaluator` base class. New domains follow the established JSON schema.

## Performance Characteristics

The system is designed for reliability over speed. Evaluation typically processes 1-10 tests per minute depending on model response time and evaluation complexity.

## References

- [Basic Usage](./basic-usage.md)
- [API Reference](./api-reference.md)
- [Domain Overview](../domains/overview.md)