# Domain Overview

Comprehensive overview of cognitive domains and test structure in the AI Model Evaluation Framework.

## Domain Architecture

The framework evaluates language models across cognitive domains that represent different types of human thinking. Tests are organized in a three-tier hierarchy based on sophistication and evaluation requirements.

## Domain Tiers

### Tier 1: Production Domains
Six core domains with comprehensive test coverage and production-ready evaluation systems.

**Characteristics:**
- 150-500 tests per domain
- Complete base and instruct model coverage
- Culturally diverse content
- Multi-difficulty progression (easy, medium, hard)
- Mature evaluation algorithms

**Domains:**
- reasoning: Logic, analysis, deduction, problem-solving
- creativity: Original content, artistic expression, storytelling  
- language: Grammar, linguistics, translation, communication
- social: Interpersonal understanding, cultural navigation
- integration: Multi-domain synthesis, complex reasoning
- knowledge: Factual accuracy, domain expertise

### Tier 2: Research Domains
Specialized domains for advanced cognitive evaluation requiring sophisticated assessment methods.

**Characteristics:**
- 5-20 tests per domain
- Highly sophisticated content
- Advanced reasoning requirements
- Specialized evaluation needs

**Examples:**
- epistemological_collapse: Quantum philosophy evaluation
- competing_infinities: Mathematical paradox resolution
- synthesis_singularities: Emergence theory testing

### Tier 3: Experimental Domains
Early-stage domains with minimal test coverage, used for exploration and development.

**Characteristics:**
- 1-5 tests per domain
- Placeholder or prototype content
- Unclear evaluation requirements

## Test Organization Structure

### Directory Hierarchy
```
domains/
├── domain_name/
│   ├── base_models/
│   │   ├── easy.json
│   │   ├── medium.json
│   │   └── hard.json
│   ├── instruct_models/
│   │   ├── easy.json
│   │   ├── medium.json
│   │   └── hard.json
│   └── categories.json
```

### Model Types

**base_models/**: Completion-style tests for base language models
- Raw text completion tasks
- No explicit instructions
- Tests fundamental language understanding

**instruct_models/**: Instruction-following tests for instruction-tuned models  
- Explicit task instructions
- System messages and constraints
- Tests instruction adherence and task completion

### Difficulty Levels

**easy**: Straightforward tasks within domain
- Clear expectations
- Minimal ambiguity
- Basic domain knowledge required

**medium**: Moderate complexity requiring deeper understanding
- Some ambiguity or complexity
- Intermediate domain knowledge
- Multiple valid approaches possible

**hard**: Complex tasks requiring sophisticated reasoning
- High ambiguity or nuance
- Advanced domain expertise needed
- Creative or novel solutions required

## Cognitive Mapping

### Primary Cognitive Functions

**Analytical Thinking** (reasoning domain)
- Logical deduction and inference
- Pattern recognition and analysis
- Problem decomposition and solution

**Creative Expression** (creativity domain)
- Original content generation
- Artistic and aesthetic judgment
- Imaginative synthesis

**Language Processing** (language domain)
- Grammatical and syntactic understanding
- Semantic and pragmatic competence
- Cross-linguistic knowledge

**Social Cognition** (social domain)  
- Theory of mind and perspective-taking
- Cultural understanding and sensitivity
- Interpersonal communication skills

**Knowledge Integration** (integration domain)
- Cross-domain synthesis
- Complex reasoning chains
- Multi-modal thinking

**Factual Knowledge** (knowledge domain)
- Domain-specific expertise
- Factual accuracy and recall
- Knowledge application

### Implicit Difficulty Dimensions

Beyond explicit easy/medium/hard levels, tests vary across implicit dimensions:

**Cultural Specificity**: From universal concepts to culture-specific knowledge
**Temporal Complexity**: From simple sequences to complex temporal reasoning  
**Abstraction Level**: From concrete facts to abstract principles
**Ambiguity Tolerance**: From clear problems to ambiguous situations
**Context Dependency**: From self-contained to context-requiring tasks

## Evaluation Approach

### Multi-Dimensional Scoring
Each domain uses specialized evaluators that assess multiple dimensions:
- **Accuracy**: Correctness of facts and logic
- **Completeness**: Coverage of task requirements  
- **Organization**: Structure and coherence
- **Cultural Sensitivity**: Appropriate cultural awareness
- **Creativity**: Originality and innovation (where relevant)

### Scoring Integration
Domain-specific scores combine into overall assessments:
- Individual dimension scores (0.0-1.0)
- Confidence weighting based on evaluator certainty
- Cultural relevance adjustments
- Final aggregation into 0-100 scale

## References

- [Production Domains](./production-domains.md)
- [Research Domains](./research-domains.md)  
- [Base vs Instruct](./base-vs-instruct.md)
- [Cognitive Mapping](./cognitive-mapping.md)