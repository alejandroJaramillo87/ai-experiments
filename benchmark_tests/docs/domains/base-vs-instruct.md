# Base vs Instruct Models

Differences between base model and instruction-tuned model testing approaches.

## Model Type Overview

The framework supports two fundamental categories of language models with different capabilities and testing requirements.

## Base Models

**Definition**: Raw language models trained on next-token prediction without instruction following training.

**Characteristics:**
- Complete text based on patterns learned during pre-training
- No explicit instruction following capability
- Rely on context and prompting for task specification
- Generate continuations rather than directed responses

**Testing Approach:**
Tests use completion-style prompts that rely on pattern recognition:

```json
{
  "prompt": "Cherry blossoms fall\nGentle spring breeze carries them\n",
  "expected_patterns": ["to the ground", "softly down", "away gently"],
  "evaluation": "pattern_matching"
}
```

**Evaluation Focus:**
- Pattern completion accuracy
- Contextual understanding from examples
- Implicit task recognition
- Natural language generation quality

## Instruct Models

**Definition**: Models fine-tuned to follow explicit instructions and complete specified tasks.

**Characteristics:**
- Trained to interpret and execute explicit instructions
- Respond to system messages and task specifications
- Follow formatting requirements and constraints
- Optimized for helpfulness and task completion

**Testing Approach:**  
Tests use explicit instruction formats with clear task specifications:

```json
{
  "system": "You are a poetry expert specializing in traditional Japanese haiku.",
  "instruction": "Complete this traditional Japanese haiku following the 5-7-5 syllable pattern and incorporating seasonal spring imagery.",
  "input": "Cherry blossoms fall\nGentle spring breeze carries them\n[Complete the haiku]",
  "constraints": ["Exactly 5 syllables", "Spring theme", "Poetic closure"],
  "evaluation_criteria": {
    "syllable_accuracy": 0.4,
    "thematic_coherence": 0.3, 
    "cultural_authenticity": 0.3
  }
}
```

**Evaluation Focus:**
- Instruction following accuracy
- Task completion effectiveness
- Constraint adherence
- Response formatting and structure

## Key Differences

### Task Presentation

**Base Model Example:**
```
"Traditional Japanese reasoning follows specific patterns. Consider this scenario: A student must choose between immediate benefit and long-term honor."
```

**Instruct Model Example:**
```
System: "You are an expert in traditional Japanese ethics and decision-making."
Instruction: "Analyze this ethical dilemma from a traditional Japanese perspective, considering concepts of honor, duty, and long-term consequences."
Input: "A student must choose between immediate benefit and long-term honor."
```

### Response Expectations

**Base Models:**
- Natural continuation of the prompt
- Implicit understanding of task requirements  
- Pattern-based completion
- Context-driven behavior

**Instruct Models:**
- Direct response to specified task
- Explicit adherence to given instructions
- Structured output following requirements
- System message compliance

### Evaluation Adaptations

**Base Model Evaluation:**
- Pattern matching against expected completions
- Contextual appropriateness assessment
- Implicit task recognition scoring
- Natural generation quality

**Instruct Model Evaluation:**
- Instruction adherence measurement
- Task completion verification
- Constraint satisfaction checking
- Structured response quality

## Domain-Specific Adaptations

### Reasoning Domain

**Base**: Logic puzzle completions, pattern recognition
**Instruct**: Explicit problem-solving instructions with step requirements

### Creativity Domain

**Base**: Creative continuations of started works
**Instruct**: Specific creative tasks with style, format, and content requirements

### Language Domain

**Base**: Grammar pattern completions, translation by example  
**Instruct**: Explicit linguistic analysis tasks with specific output formats

### Social Domain

**Base**: Social scenario continuations
**Instruct**: Advice-giving with specific cultural context requirements

### Integration Domain

**Base**: Complex scenario completions requiring multi-domain thinking
**Instruct**: Structured analysis tasks requiring integration of multiple perspectives

### Knowledge Domain

**Base**: Factual completion and knowledge demonstration through context
**Instruct**: Explicit knowledge queries with accuracy and format requirements

## Comparative Evaluation

### Scoring Adjustments

**Base Models:**
- Higher tolerance for implicit task interpretation
- Pattern quality weighted more heavily
- Context utilization assessment
- Natural flow and coherence emphasis

**Instruct Models:**
- Strict instruction adherence requirements
- Task completion verification priority
- Format compliance checking
- Explicit constraint satisfaction

### Performance Expectations

**Base Models** typically excel at:
- Natural language continuation
- Creative and open-ended generation
- Implicit pattern recognition
- Contextual adaptation

**Instruct Models** typically excel at:
- Structured task completion
- Following specific requirements
- Maintaining consistent formatting
- Direct question answering

## Implementation Status

All production domains have achieved base/instruct parity:
- Complete test coverage for both model types
- Adapted evaluation criteria for each approach
- Consistent difficulty progressions
- Culturally authentic content in both formats

## Testing Strategy

### Model Type Selection

Choose model type based on intended evaluation:
- **Base tests**: For evaluating fundamental language understanding and generation
- **Instruct tests**: For evaluating task-following and structured response capabilities

### Evaluation Interpretation

Results should be interpreted in context of model type:
- Base model scores reflect pattern completion and implicit understanding
- Instruct model scores reflect instruction following and task completion
- Direct comparison requires consideration of different capabilities

## References

- [Production Domains](./production-domains.md)
- [Difficulty Levels](./difficulty-levels.md)
- [Cognitive Mapping](./cognitive-mapping.md)