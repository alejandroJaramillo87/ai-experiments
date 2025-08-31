     Objective

     Create comprehensive technical documentation mapping the architectural flow of /benchmark_tests/evaluator/ when running benchmarks on the 6 production-ready 
     domains with full difficulty parity.

     Research Areas

     1. Evaluator Directory Architecture Analysis

     - Map complete directory structure and component relationships
     - Identify all classes, modules, and their purposes
     - Understand the inheritance and composition patterns

     2. Integration Point Analysis

     - How benchmark_runner.py integrates with evaluator system
     - Entry points and initialization flows
     - Configuration and routing mechanisms

     3. Domain-Specific Evaluation Flows

     - Request flow for each domain (reasoning, creativity, language, social, integration, knowledge)
     - Base model vs instruct model evaluation differences
     - Easy/medium/hard difficulty handling

     4. Component Usage Mapping

     - Create matrix showing which files/classes are used for each domain scenario
     - Identify the evaluation pipeline steps for each test type
     - Map cultural, advanced, and validation component usage

     5. Unused Component Identification

     - Identify evaluator components that are NOT used in the 6 production domains
     - Understand which are for specialized research domains vs experimental
     - Document architectural components that may be dormant

     6. Data Flow and Processing Pipeline

     - Step-by-step evaluation process from test input to final scores
     - How different evaluator types collaborate
     - Error handling and fallback mechanisms

     Deliverable

     Technical markdown document similar to domain_coverage_audit_report.md that provides:
     - Complete architectural flow diagrams (in text/markdown format)
     - Component usage matrix for all 6 domains × 2 model types × 3 difficulties
     - Identification of unused vs active code paths
     - Integration points and data flow analysis
     - Clear understanding of what happens when benchmarks are executed

     This will give you complete visibility into the evaluator system's architecture and usage patterns.
  ⎿  Referenced file benchmark_tests/benchmark_runner.py
  ⎿  Read benchmark_tests/docs/domain_coverage_audit_report.md (160 lines)
  ⎿  Referenced file benchmark_tests/evaluator/subjects/reasoning_evaluator.py
  ⎿  Read benchmark_tests/domains/epistemological_collapse/base_models/hard.json (199 lines)
  ⎿  Read benchmark_tests/tests/validation/test_enhanced_universal_evaluator.py (295 lines)