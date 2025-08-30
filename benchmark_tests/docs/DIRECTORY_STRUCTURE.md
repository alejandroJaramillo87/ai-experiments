# Directory Structure Guide

This document explains the organization and purpose of every directory and key file in the benchmark test system. Think of this as the **project's file system architecture documentation**.

## 🏗️ **Top-Level Organization**

```
benchmark_tests/
├── 📁 data/              # Reference data and datasets (like database fixtures)
├── 📁 domains/           # Test definitions (like API route definitions)  
├── 📁 evaluator/         # Business logic modules (like service layer)
├── 📁 tests/            # Framework tests (like typical test directories)
├── 📁 docs/             # Documentation (like any well-documented project)
├── 📁 scripts/          # Utility scripts (like build/deployment scripts)
├── 🎯 Makefile          # Automation commands (like npm scripts)
├── 🤖 benchmark_runner.py       # Main application entry point
├── 📋 README.md         # Project overview (standard in any repo)
└── 🧹 Cleanup utilities  # Maintenance scripts
```

## 📊 **Data Directory** (`data/`)

**Purpose**: Reference datasets and cultural knowledge base
**Like**: Database fixtures, lookup tables, static assets in web apps

```
data/
├── community/           # Community-contributed datasets
│   └── exports/        # Exported community data files
└── cultural/           # Cultural reference data
    ├── traditions.json    # Cultural tradition definitions  
    ├── languages.json     # Language metadata
    └── regions.json       # Geographic and cultural regions
```

### **Why This Structure?**
- **Read-only data**: These files are reference material, not modified by tests
- **Versioning**: Cultural data can be versioned and updated independently
- **Separation**: Community vs. built-in data clearly separated
- **Format**: JSON files for easy parsing and human readability

**Software Engineering Analogy**: Like having a `fixtures/` or `seed-data/` directory in a web application.

## 🎯 **Domains Directory** (`domains/`)

**Purpose**: Test definitions organized by thinking type
**Like**: API route definitions, test specifications, configuration schemas

```
domains/
├── reasoning/           # Logic and analysis tests
│   ├── base_models/    # Tests for base/foundation models
│   └── instruct_models/ # Tests for instruction-tuned models
├── creativity/          # Creative and artistic tasks
│   └── base_models/    # Creative tests for base models
├── language/            # Linguistic and grammar tests
│   └── base_models/    # Language tests for base models
├── social/              # Cultural and interpersonal tests
│   └── base_models/    # Social understanding tests
├── knowledge/           # Factual knowledge tests
│   └── base_models/    # Knowledge tests for base models
└── integration/         # Multi-domain complex tests
    └── base_models/    # Integrated capability tests
```

### **Test Definition Structure**
Each domain contains JSON files like:
```json
{
  "test_id": "reasoning_logic_01",
  "category": "reasoning_general", 
  "difficulty": "medium",
  "prompt": "Analyze the logical consistency...",
  "metadata": {
    "estimated_time": 30,
    "requires_cultural_context": false
  }
}
```

**Software Engineering Analogy**: Like having separate route files for different API modules (`/users`, `/products`, `/orders`) in Express.js or Django.

## 🧠 **Evaluator Directory** (`evaluator/`)

**Purpose**: Business logic for scoring and analysis
**Like**: Service layer, middleware, plugin architecture in web frameworks

```
evaluator/
├── core/                # Base classes and shared utilities
│   ├── base_evaluator.py       # Interface definition (like abstract base class)
│   ├── evaluation_result.py    # Result data structures  
│   └── scoring_utils.py        # Shared scoring functions
├── subjects/            # Domain-specific evaluators
│   ├── reasoning_evaluator.py  # Logic and analysis scoring
│   ├── creativity_evaluator.py # Creative task evaluation
│   ├── language_evaluator.py   # Linguistic analysis
│   └── social_evaluator.py     # Cultural understanding
├── advanced/            # Sophisticated analysis tools
│   ├── ensemble_evaluator.py   # Multiple evaluator coordination
│   ├── disagreement_detector.py # Quality assurance checks
│   └── pattern_analyzer.py     # Statistical analysis tools
├── cultural/            # Cultural context integration
│   ├── cultural_validator.py   # Cultural appropriateness checks
│   ├── tradition_analyzer.py   # Cultural tradition understanding
│   └── bias_detector.py        # Cultural bias detection
├── validation/          # Quality assurance systems  
│   ├── cross_validator.py      # Inter-evaluator consistency checks
│   ├── edge_case_detector.py   # Unusual response handling
│   └── confidence_calculator.py # Scoring confidence estimation
├── linguistics/         # Language-specific analysis
│   ├── grammar_analyzer.py     # Grammar and syntax checking
│   ├── semantic_analyzer.py    # Meaning and context analysis
│   └── multilingual_support.py # Multi-language capabilities
└── data/               # Evaluator-specific data files
    ├── scoring_rubrics.json    # Standardized scoring criteria
    ├── cultural_patterns.json  # Cultural evaluation patterns
    └── linguistic_rules.json   # Language analysis rules
```

### **Plugin Architecture Benefits**
- **Modularity**: Each evaluator handles one concern
- **Extensibility**: Easy to add new evaluation types
- **Testability**: Each component can be unit tested independently
- **Reusability**: Evaluators can be mixed and matched

**Software Engineering Analogy**: Like having separate service classes in a layered architecture, or middleware components in Express.js.

## ✅ **Tests Directory** (`tests/`)

**Purpose**: Framework quality assurance and validation
**Like**: Standard test directory structure in any software project

```
tests/
├── unit/                # Component-level tests
│   ├── test_evaluator.py          # Individual evaluator tests
│   ├── test_scoring_utils.py      # Utility function tests
│   ├── test_result_processing.py  # Result handling tests
│   └── test_config_loading.py     # Configuration tests
├── integration/         # Multi-component tests
│   ├── test_full_evaluation.py    # End-to-end evaluation tests
│   ├── test_concurrent_execution.py # Parallel processing tests
│   └── test_api_integration.py    # External API integration tests
├── functional/          # User workflow tests
│   ├── test_cli_interface.py      # Command-line interface tests
│   ├── test_result_reporting.py   # Output format tests
│   └── test_error_handling.py     # Error scenario tests
├── analysis/            # Performance and quality analysis
│   ├── analyze_scoring_patterns.py # Score distribution analysis
│   ├── benchmark_performance.py   # Speed and resource usage
│   └── validate_consistency.py    # Cross-evaluator consistency
└── results/             # Test output storage
    ├── unit_test_results/         # Unit test outputs
    ├── integration_results/       # Integration test outputs
    └── performance_metrics/       # Performance test data
```

### **Test Strategy**
- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **Functional Tests**: Test user-facing features
- **Analysis Tests**: Validate system behavior and performance

**Software Engineering Analogy**: Standard test pyramid structure used in any mature software project.

## 📚 **Documentation Directory** (`docs/`)

**Purpose**: Comprehensive project documentation
**Like**: Documentation site structure (Gitbook, Docusaurus, etc.)

```
docs/
├── ARCHITECTURE.md                 # System design and patterns
├── CONCEPTS.md                    # Core concepts explained
├── DIRECTORY_STRUCTURE.md         # This file
├── benchmark_test_runner_overview.md # Runner implementation details
├── evaluation_system_overview.md  # Evaluation system deep dive
├── integration_test_framework_plan.md # Test framework design
├── TESTING_GUIDE.md               # How to run and write tests
├── test_runner_interface.md       # API reference documentation
├── test_schema_design.md          # Test definition schema
└── validation_system_overview.py  # Validation system analysis
```

### **Documentation Strategy**
- **Progressive disclosure**: Start with concepts, then architecture, then details
- **Audience-specific**: Different docs for users vs. developers vs. operators
- **Living documentation**: Updated alongside code changes
- **Practical focus**: Examples and usage patterns, not just theory

## 🔧 **Scripts Directory** (`scripts/`)

**Purpose**: Utility and automation scripts
**Like**: `scripts/` directory in Node.js projects, or `bin/` in Unix projects

```
scripts/
├── setup_environment.py      # Development environment setup
├── run_benchmarks.py         # Automated benchmark execution  
├── export_results.py         # Result data export utilities
├── validate_test_data.py     # Test data validation
└── performance_profiling.py  # Performance analysis tools
```

**Software Engineering Analogy**: Like npm scripts, Makefile targets, or DevOps automation scripts.

## 📋 **Root-Level Files**

### **Core Application Files**
- **`benchmark_runner.py`**: Main application entry point (like `app.py` in Flask, `server.js` in Node.js)
- **`run_tests_with_cleanup.py`**: Test execution wrapper with cleanup
- **`cleanup_test_artifacts.py`**: Maintenance utility for cleaning test artifacts

### **Project Management Files** 
- **`README.md`**: Project overview and quick start guide
- **`Makefile`**: Automation commands and workflows
- **`CLEANUP_GUIDE.md`**: Guide for managing test artifacts

### **Configuration Files** (when present)
- **`config.json`**: Application configuration
- **`requirements.txt`**: Python dependencies  
- **`.gitignore`**: Version control exclusions

## 🔄 **Data Flow Through Directories**

### **Test Execution Flow**
```
1. domains/          → Load test definitions
2. benchmark_runner.py → Orchestrate execution
3. evaluator/        → Process and score responses  
4. data/            → Reference cultural/linguistic data
5. test_results/     → Save scored results
```

### **Development Workflow**
```
1. tests/           → Validate system functionality
2. docs/            → Document changes and usage
3. scripts/         → Automate common tasks
4. evaluator/       → Implement new evaluation logic
5. domains/         → Add new test cases
```

## 🛠️ **Extension Points**

### **Adding New Domain Types**
1. Create new directory under `domains/new_domain/`
2. Add JSON test definitions following existing schema
3. No code changes needed - system auto-discovers

### **Adding New Evaluators**
1. Create new evaluator in appropriate `evaluator/` subdirectory
2. Implement `BaseEvaluator` interface
3. Add configuration mapping
4. Add unit tests in `tests/unit/`

### **Adding New Analysis Tools**
1. Create analysis script in `evaluator/advanced/`
2. Add integration tests in `tests/integration/`
3. Document usage in `docs/`
4. Add automation command to `Makefile`

## 🔐 **Security and Isolation**

### **Read-Only Data**
- `data/` directory contains reference data that shouldn't be modified by tests
- Cultural and linguistic data versioned separately from code

### **Test Isolation**
- Each test execution creates isolated temporary artifacts
- Cleanup utilities ensure no test pollution between runs
- Results stored in timestamped directories

### **Configuration Security** 
- API endpoints and credentials configured via environment variables
- No hardcoded secrets in any configuration files
- Sensitive data excluded from version control

## 📈 **Scalability Considerations**

### **Horizontal Scaling**
- Test definitions can be distributed across multiple `domains/` 
- Evaluators can run independently and concurrently
- Results can be aggregated from distributed execution

### **Vertical Scaling**
- Large test suites can be partitioned by domain or difficulty
- Memory-intensive evaluations isolated in separate processes
- Cultural data can be cached and shared across evaluations

---

This directory structure follows **software engineering best practices**:
- **Single Responsibility**: Each directory has one clear purpose
- **Separation of Concerns**: Data, logic, tests, and docs are separate
- **Plugin Architecture**: Easy to extend without modifying existing code  
- **Configuration-Driven**: Behavior controlled by data, not hardcoded logic
- **Maintainable**: Clear organization makes the system easy to understand and modify

The structure scales from simple single evaluations to complex distributed benchmarking systems while maintaining clarity and maintainability.