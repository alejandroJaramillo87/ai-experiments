# Benchmark Tests - Core Requirements & Testing Documentation

## PROJECT STATUS: Core Calibration Focus

**Primary Goal:** Calibrate @benchmark_tests/evaluator/ for production deployment
**Current Priority:** Systematic evaluator calibration (base easy→hard, then instruct easy→hard)
**Status:** Scripts organized, core modules created, testing required

---

## CRITICAL REQUIREMENT: Testing for New Functionality

### **MUST CREATE TESTS FOR:**

#### 1. Core Modules (Priority 1)
- `core/test_results_manager.py` - Test result storage and cognitive pattern detection
- `core/cognitive_evaluation_pipeline.py` - Sophisticated evaluator integration

**Testing Requirements:**
- Unit tests for TestResultsManager class
- Unit tests for CognitiveEvaluationPipeline class  
- Integration tests with actual evaluator framework
- Performance tests to prevent timeout issues
- JSON serialization tests for complex objects

#### 2. Scripts Organization (Priority 2) 
**New Structure Created:**
```
scripts/
├── calibration/     - calibration_success_analysis.py, production_calibration_framework.py
├── optimization/    - token_optimization.py, refined_token_optimization.py, scale_token_optimization.py  
├── validation/      - easy_domain_validation.py, validate_token_optimization.py, validate_new_tests.py
├── benchmarking/    - multi_model_benchmarking.py, enhanced_cognitive_validation.py, comprehensive_easy_domain_testing.py
└── conversion/      - convert_base_to_instruct_creativity.py, convert_core_domains_to_instruct.py
```

**Testing Requirements:**
- Verify import paths still work after reorganization
- Test each category of scripts functions correctly
- Integration tests across script categories

#### 3. Evaluator Calibration Framework (Priority 3)
**Systematic Calibration Path:**
1. **Base Models:** easy → medium → hard (using sophisticated evaluators)
2. **Instruct Models:** easy → medium → hard (using sophisticated evaluators)

**Testing Requirements:**
- Statistical validation of calibration accuracy
- Confidence interval testing for pattern detection
- Cross-domain consistency testing
- Performance benchmarking across difficulty levels

---

## EVALUATOR INTEGRATION STATUS

### Currently Available Sophisticated Evaluators:
- ✅ PatternBasedEvaluator - Working, integrated
- ❓ CulturalAuthenticityAnalyzer - Partially integrated  
- ❓ EnhancedUniversalEvaluator - Framework ready, needs testing
- ❓ BiasAnalysis/EvaluationAggregator - Framework ready, needs testing

### Integration Requirements:
- Replace basic heuristic scoring with sophisticated evaluator results
- Ensure evaluator results serialize properly for storage
- Handle evaluator failures gracefully with fallback scoring
- Statistical validation of evaluator consistency

---

## TEST STRUCTURE REQUIREMENTS

### Test Directory Structure:
```
tests/
├── unit/                    - Unit tests for individual components
│   ├── test_core_modules/   - Core functionality tests (REQUIRED)
│   ├── test_evaluator_integration/ - Evaluator integration tests  
│   └── test_scripts_organization/  - Scripts organization tests
├── integration/             - Integration and end-to-end tests
├── analysis/                - Analysis and validation scripts
├── calibration/             - Calibration framework tests (functional)
└── functional/              - Functional/live server tests (EXCLUDED from regression)
```

### CRITICAL REGRESSION TESTING REQUIREMENT

**⚠️ MANDATORY: Always run regression tests before major changes**

When making wide changes to the codebase, ALWAYS run regression tests to ensure previous functionality is not broken. Failure to do this will prevent progress.

### Regression Test Commands:

#### Primary Regression Test (Recommended):
```bash
# Run all tests EXCEPT functional/ and calibration/ (which require live server)
make test-regression
```

#### Alternative Method (Direct pytest):
```bash
# Run all tests excluding functional and calibration directories
pytest tests/ --ignore=tests/functional/ --ignore=tests/calibration/ -v
```

#### Core Module Safety Check:
```bash  
# Quick test of critical core functionality only
make test-core-safety
```

#### Full Test Suite (Use with caution):
```bash
# Only when server is running and time permits
make test SUITE=all
```

### Test Coverage Requirements:
- **Minimum 80% code coverage** for core/ modules
- **Error handling tests** for evaluator failures
- **Performance tests** to prevent timeout issues
- **Statistical validation tests** for pattern detection accuracy
- **End-to-end tests** for full calibration pipeline
- **Regression tests** MUST pass before any major implementation changes

---

## CALIBRATION METHODOLOGY

### Statistical Requirements:
- **Multi-sample validation:** 3-5 runs per test for statistical significance
- **Confidence intervals:** ±95% confidence for pattern detection
- **Effect size calculation:** Cohen's d for practical significance
- **Cross-validation:** Ensure patterns are robust across test sets

### Quality Thresholds:
- **✅ Excellent Calibration:** ±2 points from target (within statistical variance)
- **🟡 Good Calibration:** ±5 points from target (production acceptable)
- **🟠 Needs Calibration:** ±10 points from target (requires adjustment)
- **❌ Calibration Broken:** >10 points from target (system failure)

### Production Readiness Criteria:
- **Base Models:** 70%+ tests achieve Good Calibration or better
- **Instruct Models:** 70%+ tests achieve Good Calibration or better  
- **Cross-Domain Consistency:** <10% variance between domains
- **Statistical Significance:** p<0.05 for detected patterns

---

## IMPLEMENTATION PRIORITIES

### Phase 1: Core Testing (CURRENT)
1. Create unit tests for core/ modules
2. Fix timeout and performance issues
3. Validate evaluator integration works correctly

### Phase 2: Base Model Calibration ✅ IMPLEMENTED
**Status: Ready for execution - systematic calibration framework complete**

#### Systematic Calibration Implementation:
- **Script:** `scripts/calibration/systematic_base_calibration.py` 
- **Makefile:** `make systematic-base-calibration`
- **Progression:** Automatic easy → medium → hard with halt-on-failure
- **Core Domains:** reasoning, creativity, language, social, knowledge, integration

#### Calibration Process:
1. **Easy domain calibration** with statistical validation (3 samples per test)
2. **Medium domain calibration** with pattern consistency (halt if easy fails)  
3. **Hard domain calibration** with performance optimization (halt if medium fails)
4. **Statistical validation** and production readiness assessment
5. **Comprehensive reporting** with domain-by-domain analysis

#### Quality Thresholds (Implemented):
- **✅ Excellent:** ≥80 score (continue progression)
- **🟡 Good:** ≥70 score (continue with monitoring)  
- **🟠 Needs Calibration:** ≥60 score (halt and fix)
- **❌ Broken:** <60 score (system failure)

### Phase 3: Instruct Model Calibration  
1. Easy instruct calibration building on base model lessons
2. Medium instruct calibration with cross-domain validation
3. Hard instruct calibration with production readiness assessment

### Phase 4: Production Deployment
1. Full statistical validation report
2. Performance benchmarking across hardware configurations
3. Documentation for production deployment procedures

---

## PERFORMANCE REQUIREMENTS

### Hardware Optimization (RTX 5090 + AMD 9950X + 128GB DDR5):
- **Response Time:** <30 seconds per domain evaluation
- **Memory Usage:** <16GB peak usage during evaluation
- **GPU Utilization:** Efficient GGUF model loading and inference
- **Concurrent Processing:** Support for multiple domain evaluation

### Scalability Requirements:
- **26,000+ Test Suite:** Full suite completion within reasonable time
- **Cross-Model Comparison:** Support for multiple model endpoints
- **Result Storage:** Efficient storage and retrieval of evaluation results
- **Pattern Analysis:** Real-time cognitive pattern detection

---

## DOCUMENTATION REQUIREMENTS

### Code Documentation:
- **Docstrings:** All public methods with parameter and return type documentation
- **Type Hints:** Complete type annotations for all functions and classes
- **Examples:** Working code examples for each major component
- **Error Handling:** Documentation of all exception cases

### User Documentation:
- **Calibration Guide:** Step-by-step calibration procedures
- **API Reference:** Complete reference for all evaluator interfaces  
- **Troubleshooting:** Common issues and solutions
- **Performance Tuning:** Optimization guidelines for different hardware

---

**CRITICAL REMINDER:** Focus on systematic evaluator calibration (easy→hard for base, then instruct) before implementing advanced features like dashboards or cross-model analysis.