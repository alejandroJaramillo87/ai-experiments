# Test Definition JSON Schema Design

## Analysis of Current Structure

### Test Case Components Identified:
1. **Core Fields**: name, prompt, params
2. **Parameters**: max_tokens, temperature, top_p, stream, stop (optional)
3. **Implicit Metadata**: reasoning type (inferred from name), category, expected behavior

### Reasoning Categories Identified:
- Tests 1-15: Complex multi-document synthesis (mixed)
- Tests 16-25: Chain-of-thought and multi-step reasoning
- Tests 26-30: Self-verification loops
- Tests 31-35: Mathematical reasoning
- Tests 36-40: Multi-hop inference
- Tests 41-45: Scaffolded reasoning
- Tests 46-50: Backward reasoning

## Proposed JSON Schema

### Single Test Definition Structure:
```json
{
  "id": "complex_test_1",
  "name": "Complex Test 1: Multi-Document Synthesis from Scientific Abstracts",
  "category": "synthesis",
  "reasoning_type": "general",
  "description": "Tests multi-document synthesis capabilities with scientific abstracts",
  "prompt": "...",
  "parameters": {
    "max_tokens": 1500,
    "temperature": 0.4,
    "top_p": 0.95,
    "stream": false,
    "stop": ["\n\n\n"]
  },
  "metadata": {
    "expected_length": "medium",
    "complexity": "high",
    "context_size": "large",
    "domain": "scientific",
    "timeout_seconds": 600,
    "reasoning_patterns": ["synthesis", "comparison", "analysis"]
  },
  "evaluation_config": {
    "reasoning_type": "general",
    "custom_weights": null,
    "expected_indicators": []
  }
}
```

### Test Suite Structure:
```json
{
  "suite_name": "Advanced Long-Context Reasoning Test Suite",
  "suite_id": "reasoning_comprehensive_v1",
  "description": "Comprehensive 50-test suite for evaluating long-context reasoning capabilities",
  "version": "1.0.0",
  "api_config": {
    "endpoint": "http://127.0.0.1:8004/v1/completions",
    "model": "your-base-model-name",
    "headers": {
      "Content-Type": "application/json"
    }
  },
  "global_settings": {
    "timeout_seconds": 600,
    "retry_attempts": 3,
    "delay_between_tests": 1,
    "output_directory": "test_results"
  },
  "test_categories": {
    "synthesis": {
      "description": "Multi-document synthesis and integration",
      "test_ids": ["complex_test_1", "complex_test_2", ...]
    },
    "chain_of_thought": {
      "description": "Step-by-step reasoning chains", 
      "test_ids": ["complex_test_16", "complex_test_17", ...]
    },
    "verification": {
      "description": "Self-checking and verification loops",
      "test_ids": ["complex_test_26", "complex_test_27", ...]
    },
    "mathematical": {
      "description": "Mathematical and logical reasoning",
      "test_ids": ["complex_test_31", "complex_test_32", ...]
    },
    "multi_hop": {
      "description": "Multi-source inference and connection",
      "test_ids": ["complex_test_36", "complex_test_37", ...]
    },
    "scaffolded": {
      "description": "Structured reasoning frameworks",
      "test_ids": ["complex_test_41", "complex_test_42", ...]
    },
    "backward": {
      "description": "Reverse reasoning from conclusions",
      "test_ids": ["complex_test_46", "complex_test_47", ...]
    }
  },
  "tests": [
    {
      "id": "complex_test_1",
      "name": "...",
      ...
    }
  ]
}
```

## File Organization Options

### Option A: Single File Architecture
- `reasoning_test_suite.json` - Contains everything
- **Pros**: Simple, atomic, easy to version
- **Cons**: Large file, harder to edit individual tests

### Option B: Modular File Architecture  
```
test_definitions/
├── suite_config.json       # Suite metadata and configuration
├── categories/              # Category definitions
│   ├── synthesis.json
│   ├── chain_of_thought.json
│   ├── verification.json
│   ├── mathematical.json
│   ├── multi_hop.json
│   ├── scaffolded.json
│   └── backward.json
└── tests/                   # Individual test definitions
    ├── complex_test_01.json
    ├── complex_test_02.json
    └── ...
```
- **Pros**: Modular, easy to edit, clear organization
- **Cons**: More complex loading, potential for inconsistencies

### Option C: Hybrid Architecture (RECOMMENDED)
```
test_definitions/
├── test_suite_metadata.json       # Suite config and API settings
├── reasoning_tests_complete.json  # All 50 tests in structured format
└── categories.json                # Category definitions and mappings
```
- **Pros**: Balance of simplicity and organization
- **Cons**: Still manageable size, good for this specific use case

## Key Design Decisions

1. **Reasoning Type Detection**: Make explicit instead of inferred from name
2. **Parameter Validation**: JSON schema can validate required fields
3. **Extensibility**: Easy to add new fields without breaking existing tests
4. **Backward Compatibility**: Can generate old format from new format
5. **Tool Integration**: Designed to work seamlessly with existing ReasoningEvaluator