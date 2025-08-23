# Long-Context Benchmark Test Suite

This directory contains a comprehensive benchmark suite designed to rigorously test base language models on tasks requiring deep understanding of ~32,000 token context windows.

## Overview

The benchmark suite contains **10 challenging scenarios** across **5 categories** that test different aspects of long-context reasoning capabilities:

### Categories Covered

1. **Multi-Document Synthesis (3 scenarios)**
   - Combine multiple lengthy documents for nuanced analysis
   - Test ability to synthesize conflicting information
   - Evaluate cross-document reasoning capabilities

2. **Large Codebase Comprehension (2 scenarios)**
   - Analyze thousands of lines of code across multiple files
   - Identify subtle bugs and architectural issues
   - Understand complex data flow and system interactions

3. **Needle in a Haystack (2 scenarios)**
   - Find specific information embedded in massive text
   - Test precise information retrieval capabilities
   - Evaluate attention mechanisms across long contexts

4. **Complex Chain-of-Thought Reasoning (2 scenarios)**
   - Multi-step analysis with interdependent variables
   - Apply complex business rules to large datasets
   - Test logical reasoning across extended contexts

5. **State Tracking Across a Narrative (1 scenario)**
   - Track character relationships and plot developments
   - Maintain state consistency across long narratives
   - Test temporal reasoning and character evolution

## File Structure

- `test_long_context_benchmarks.py` - Core benchmark scenarios
- `run_long_context_benchmarks.py` - Test runner and validation framework
- `README.md` - This documentation

## Benchmark Scenarios

| ID | Name | Category | Est. Tokens | Description |
|----|------|----------|-------------|-------------|
| 1 | Financial Report Synthesis | Multi-Document | ~4,891 | Analyze 3 financial reports to provide investment recommendations |
| 2 | Medical Research Synthesis | Multi-Document | ~3,945 | Synthesize clinical studies for treatment protocols |
| 3 | Climate Policy Analysis | Multi-Document | ~4,301 | Integrate government reports for policy recommendations |
| 4 | E-commerce Memory Leak Investigation | Large Codebase | ~8,657 | Debug memory leaks in Java microservices |
| 5 | Distributed System Data Flow | Large Codebase | ~736 | Analyze data consistency in microservices |
| 6 | Security Log Analysis | Needle in Haystack | ~4,112 | Find security breach evidence in extensive logs |
| 7 | Citation Network Analysis | Needle in Haystack | ~533 | Locate specific research breakthrough identifier |
| 8 | Supply Chain Optimization | Chain-of-Thought | ~514 | Optimize global supply chain with constraints |
| 9 | Portfolio Optimization | Chain-of-Thought | ~499 | Rebalance investment portfolio under stress |
| 10 | Political Thriller Analysis | State Tracking | ~1,337 | Track character relationships in complex narrative |

## Usage

### List Available Scenarios
```bash
python run_long_context_benchmarks.py --list
```

### Validate Scenario Structure
```bash
python run_long_context_benchmarks.py --scenario_id 1 --validate_only
```

### Run Specific Scenario (Demo Mode)
```bash
python run_long_context_benchmarks.py --scenario_id 1
```

## Benchmark Format

Each scenario is a Python dictionary with exactly three keys:

```python
{
    "name": "Descriptive scenario name",
    "prompt": "Extensive context (~32,000 tokens) with specific task at the end",
    "validator": "Clear success criteria for evaluating model response"
}
```

### Validator Examples

- `"Response must identify the exact timestamp when the credential 'BREACH_ACTOR_ALPHA' was first used"`
- `"Response must provide concrete fixes for each identified memory leak source"`
- `"Response must correctly track all character identity changes and final fates"`

## Integration with Testing Infrastructure

### For Base Model APIs
```python
from test_long_context_benchmarks import LONG_CONTEXT_BENCHMARKS

# Example integration
def run_base_model_benchmark(scenario_id, model_endpoint):
    scenario = LONG_CONTEXT_BENCHMARKS[scenario_id - 1]
    
    # Send to your base model API
    response = send_to_model(
        prompt=scenario["prompt"],
        max_tokens=15000,
        temperature=0.1
    )
    
    # Validate response
    is_valid = validate_response(response, scenario["validator"])
    return is_valid, response
```

### Performance Metrics

When implementing with real models, track:
- **Response Accuracy**: Does the response meet validator criteria?
- **Processing Time**: How long does the model take to process?
- **Token Throughput**: Tokens processed per second
- **Memory Usage**: Peak memory during processing
- **Context Utilization**: Does the model use the full context effectively?

## Quality Assurance

### Scenario Validation Checklist
- ✅ All scenarios have required keys (name, prompt, validator)
- ✅ Prompts contain extensive content before final task
- ✅ Validators specify clear, measurable success criteria
- ✅ Categories are well-distributed across different reasoning types
- ✅ Content is realistic and professionally relevant

### Token Estimates
Token counts are estimated using the approximation of 1 token ≈ 4 characters for English text. Actual token counts may vary depending on the tokenizer used by specific models.

## Contributing

When adding new scenarios:

1. Follow the exact format: `{"name", "prompt", "validator"}`
2. Ensure prompts approach ~32,000 tokens when possible
3. Place the specific task/question at the end of extensive context
4. Write clear, measurable validators
5. Test scenarios before submission

## License

This benchmark suite is provided for research and evaluation purposes. See the main repository license for full terms.