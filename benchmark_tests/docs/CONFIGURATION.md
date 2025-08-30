# Configuration Guide

This guide covers all configuration options for customizing AI model evaluations. Think of this as your **settings and preferences documentation** - everything you need to tailor the system to your specific needs.

## 🎛️ **Configuration Hierarchy**

The system uses a **layered configuration approach** (like most web frameworks):

```
1. Built-in Defaults     # Reasonable defaults for most use cases
         ↓
2. Configuration Files   # Project-specific settings  
         ↓
3. Environment Variables # Deployment-specific settings
         ↓
4. Command Line Args     # Runtime-specific overrides
```

**Higher layers override lower layers** - command line arguments have the highest priority.

## 🔧 **Command Line Configuration**

### **Core Options**
```bash
python benchmark_runner.py \
  --test-type base \           # Test type: base, instruct, custom
  --mode single \              # Execution mode: single, category, concurrent  
  --model "your-model" \       # Model identifier
  --endpoint "http://..." \    # API endpoint URL
  --output-dir "results/" \    # Where to save results
  --verbose                    # Detailed logging
```

### **Test Selection**
```bash
# Run specific test
--test-id "reasoning_logic_01"

# Run test category 
--category "reasoning_general"

# Filter by difficulty
--difficulty easy              # Options: easy, medium, hard

# Limit number of tests
--limit 5                      # Run only first 5 matching tests
```

### **Execution Control**
```bash
# Concurrent processing
--workers 4                    # Number of parallel workers

# Performance monitoring
--performance-monitoring       # Track execution metrics

# Timeout settings
--timeout 30                   # Request timeout in seconds

# Retry behavior
--max-retries 3               # Retry failed requests
--retry-delay 2               # Delay between retries (seconds)
```

### **Output Customization**
```bash
# Result format
--format json                  # Options: json, csv, yaml

# Output verbosity  
--quiet                       # Minimal output
--verbose                     # Detailed output  
--debug                       # Full debug information

# File naming
--prefix "experiment_1_"      # Add prefix to result files
--timestamp                   # Add timestamp to result files
```

## 📁 **Configuration Files**

### **Main Configuration** (`config.json`)
Create a configuration file for consistent settings:

```json
{
  "model": {
    "endpoint": "http://localhost:8004/v1/completions",
    "name": "my-model-v1",
    "timeout": 30,
    "max_retries": 3,
    "parameters": {
      "temperature": 0.7,
      "max_tokens": 500,
      "top_p": 0.9
    }
  },
  "evaluation": {
    "default_test_type": "base",
    "concurrent_workers": 4,
    "performance_monitoring": true,
    "output_directory": "results"
  },
  "scoring": {
    "confidence_threshold": 0.7,
    "require_cultural_context": false,
    "enable_advanced_analysis": true
  },
  "logging": {
    "level": "INFO",
    "file": "benchmark.log",
    "format": "detailed"
  }
}
```

### **Domain-Specific Configuration** (`domain_config.json`)
Customize evaluation criteria for each domain:

```json
{
  "reasoning": {
    "weights": {
      "organization_quality": 0.25,
      "technical_accuracy": 0.40,
      "completeness": 0.20,
      "reliability": 0.15
    },
    "strict_mode": true,
    "require_evidence": true,
    "min_response_length": 100
  },
  "creativity": {
    "weights": {
      "organization_quality": 0.20,
      "technical_accuracy": 0.15,
      "completeness": 0.25,
      "reliability": 0.40
    },
    "originality_weight": 0.3,
    "style_importance": 0.7,
    "allow_abstract": true
  },
  "language": {
    "grammar_weight": 0.4,
    "clarity_weight": 0.3,
    "appropriateness_weight": 0.3,
    "check_spelling": true,
    "check_grammar": true
  }
}
```

### **Cultural Configuration** (`cultural_config.json`)
Settings for cultural context integration:

```json
{
  "enabled": true,
  "default_contexts": ["general"],
  "validation_strictness": "medium",
  "bias_detection": {
    "enabled": true,
    "sensitivity": "high",
    "categories": ["gender", "racial", "religious", "cultural"]
  },
  "cultural_groups": {
    "include": ["west_african", "east_asian", "latin_american"],
    "exclude": [],
    "require_validation": true
  },
  "linguistic_diversity": {
    "multilingual_support": true,
    "primary_languages": ["english", "spanish", "mandarin"],
    "translation_quality_check": true
  }
}
```

## 🌍 **Environment Variables**

### **Model Configuration**
```bash
# Model API settings
export MODEL_ENDPOINT="http://localhost:8004/v1/completions"
export MODEL_NAME="my-awesome-model"
export MODEL_API_KEY="your-api-key-here"          # If needed
export MODEL_TIMEOUT="30"

# Model parameters
export MODEL_TEMPERATURE="0.7"
export MODEL_MAX_TOKENS="500"
export MODEL_TOP_P="0.9"
```

### **System Configuration**
```bash
# Performance settings
export BENCHMARK_WORKERS="4"
export BENCHMARK_TIMEOUT="30"
export BENCHMARK_RETRIES="3"

# Output settings
export RESULTS_DIR="/path/to/results"
export LOG_LEVEL="INFO" 
export LOG_FILE="/path/to/benchmark.log"

# Feature flags
export ENABLE_PERFORMANCE_MONITORING="true"
export ENABLE_CULTURAL_VALIDATION="true"
export ENABLE_ADVANCED_ANALYSIS="false"
```

### **Development Settings**
```bash
# Development mode
export DEBUG_MODE="true"
export VERBOSE_LOGGING="true"
export SAVE_RAW_RESPONSES="true"

# Testing settings
export TEST_DATA_PATH="/path/to/test/data"
export MOCK_API_RESPONSES="false"
export SKIP_SLOW_TESTS="true"
```

## ⚙️ **Model API Configuration**

### **OpenAI-Compatible APIs**
```bash
python benchmark_runner.py \
  --endpoint "https://api.openai.com/v1/completions" \
  --model "gpt-3.5-turbo" \
  --api-key-env "OPENAI_API_KEY"
```

**Environment Setup**:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export MODEL_ENDPOINT="https://api.openai.com/v1/completions"
```

### **Local Model Servers**

**Ollama**:
```bash
python benchmark_runner.py \
  --endpoint "http://localhost:11434/api/generate" \
  --model "llama2:7b" \
  --api-format "ollama"
```

**Text Generation WebUI**:
```bash
python benchmark_runner.py \
  --endpoint "http://localhost:5000/api/v1/generate" \
  --model "your-local-model" \
  --api-format "textgen"
```

**Custom API Format**:
```json
{
  "api_format": "custom",
  "request_template": {
    "model": "{model_name}",
    "input": "{prompt}",
    "parameters": {
      "max_length": "{max_tokens}",
      "temp": "{temperature}"
    }
  },
  "response_path": "choices[0].text"
}
```

## 🎯 **Evaluation Customization**

### **Scoring Weights**
Adjust importance of different quality dimensions:

```json
{
  "scoring_weights": {
    "reasoning": {
      "organization_quality": 0.25,  # How well-structured is the response?
      "technical_accuracy": 0.40,   # Are facts and logic correct?
      "completeness": 0.20,         # Does it address all requirements?
      "reliability": 0.15           # Is it trustworthy for production?
    },
    "creativity": {
      "organization_quality": 0.15,
      "technical_accuracy": 0.10,
      "completeness": 0.25,
      "reliability": 0.50
    }
  }
}
```

### **Difficulty-Specific Settings**
Different standards for different difficulty levels:

```json
{
  "difficulty_settings": {
    "easy": {
      "min_score_threshold": 60,
      "expected_response_time": 10,
      "allow_simple_responses": true
    },
    "medium": {
      "min_score_threshold": 70,
      "expected_response_time": 20,
      "require_detailed_reasoning": true
    },
    "hard": {
      "min_score_threshold": 75,
      "expected_response_time": 45,
      "require_comprehensive_analysis": true
    }
  }
}
```

### **Category-Specific Evaluators**
Choose which evaluators to use for each category:

```json
{
  "category_evaluators": {
    "reasoning_general": ["reasoning", "logic", "evidence"],
    "reasoning_complex": ["reasoning", "logic", "evidence", "advanced_analysis"],
    "creativity_writing": ["creativity", "style", "originality"],
    "language_grammar": ["language", "grammar", "syntax"],
    "social_cultural": ["social", "cultural", "bias_detection"]
  }
}
```

## 📊 **Performance Configuration**

### **Concurrency Settings**
```json
{
  "concurrency": {
    "max_workers": 4,              # Number of parallel evaluations
    "batch_size": 10,              # Tests per batch
    "rate_limiting": {
      "requests_per_second": 2,    # API rate limiting
      "burst_size": 5              # Allow short bursts
    },
    "timeout_settings": {
      "request_timeout": 30,       # Individual request timeout
      "evaluation_timeout": 60     # Total evaluation timeout
    }
  }
}
```

### **Memory Management**
```json
{
  "memory": {
    "max_cache_size": "1GB",       # Cache size for cultural data
    "batch_processing": true,      # Process results in batches
    "cleanup_frequency": 100,      # Clean up every N evaluations
    "stream_large_results": true   # Stream large result sets to disk
  }
}
```

### **Monitoring and Profiling**
```json
{
  "monitoring": {
    "performance_tracking": true,
    "resource_monitoring": true,
    "detailed_timing": false,
    "memory_profiling": false,
    "metrics_export": {
      "format": "json",
      "file": "performance_metrics.json",
      "include_system_stats": true
    }
  }
}
```

## 🔍 **Debugging Configuration**

### **Logging Levels**
```json
{
  "logging": {
    "level": "INFO",               # DEBUG, INFO, WARNING, ERROR
    "format": "detailed",          # simple, detailed, json
    "output": {
      "console": true,
      "file": "benchmark.log",
      "structured": false
    },
    "component_levels": {
      "evaluator": "DEBUG",
      "api_client": "INFO",
      "scoring": "WARNING"
    }
  }
}
```

### **Debug Options**
```bash
# Save all intermediate data
python benchmark_runner.py --save-intermediate --debug

# Verbose API communication
python benchmark_runner.py --debug-api --verbose

# Profile performance
python benchmark_runner.py --profile --timing-details
```

### **Development Mode Settings**
```json
{
  "development": {
    "mock_api_calls": false,       # Use mock responses for testing
    "save_raw_responses": true,    # Keep raw model responses
    "validate_configs": true,      # Strict config validation
    "detailed_errors": true,       # Full error stack traces
    "timing_breakdown": true       # Detailed timing information
  }
}
```

## 🛡️ **Security Configuration**

### **API Security**
```json
{
  "security": {
    "api_authentication": {
      "type": "bearer_token",      # bearer_token, api_key, oauth
      "token_env_var": "MODEL_API_TOKEN",
      "header_name": "Authorization"
    },
    "rate_limiting": {
      "enabled": true,
      "max_requests_per_minute": 60
    },
    "request_validation": {
      "validate_responses": true,
      "sanitize_inputs": true,
      "log_api_calls": false       # Don't log sensitive data
    }
  }
}
```

### **Data Privacy**
```json
{
  "privacy": {
    "anonymize_responses": false,  # Remove identifying information
    "encrypt_stored_data": false,  # Encrypt result files
    "data_retention_days": 30,     # Auto-delete old results
    "exclude_sensitive_tests": [], # Skip tests with sensitive content
    "audit_logging": true          # Log all access to results
  }
}
```

## 🎨 **Custom Test Types**

### **Creating Custom Test Types**
```json
{
  "custom_test_types": {
    "my_domain_specific": {
      "base_directory": "domains/custom/my_domain",
      "evaluators": ["custom_evaluator", "reasoning"],
      "default_difficulty": "medium",
      "scoring_weights": {
        "domain_expertise": 0.5,
        "general_quality": 0.3,
        "practical_applicability": 0.2
      }
    }
  }
}
```

### **Test Selection Rules**
```json
{
  "test_selection": {
    "include_patterns": ["reasoning_*", "creativity_basic_*"],
    "exclude_patterns": ["*_deprecated", "*_experimental"],
    "difficulty_filter": ["easy", "medium"],
    "category_weights": {
      "reasoning": 0.4,
      "creativity": 0.3,
      "language": 0.2,
      "social": 0.1
    }
  }
}
```

## 📋 **Configuration Examples**

### **Development Configuration**
```json
{
  "model": {
    "endpoint": "http://localhost:8004/v1/completions",
    "name": "dev-model",
    "timeout": 10
  },
  "evaluation": {
    "concurrent_workers": 1,
    "limit_tests": 5
  },
  "logging": {
    "level": "DEBUG",
    "verbose": true
  }
}
```

### **Production Configuration**  
```json
{
  "model": {
    "endpoint": "https://api.production-model.com/v1/completions",
    "timeout": 60,
    "max_retries": 5
  },
  "evaluation": {
    "concurrent_workers": 8,
    "performance_monitoring": true
  },
  "logging": {
    "level": "INFO",
    "file": "/var/log/benchmark.log"
  },
  "security": {
    "rate_limiting": {"max_requests_per_minute": 120},
    "audit_logging": true
  }
}
```

### **Research Configuration**
```json
{
  "evaluation": {
    "enable_advanced_analysis": true,
    "detailed_cultural_analysis": true,
    "comprehensive_scoring": true
  },
  "scoring": {
    "confidence_analysis": true,
    "cross_evaluator_validation": true,
    "statistical_significance": true
  },
  "output": {
    "save_intermediate_results": true,
    "detailed_reasoning": true,
    "export_raw_data": true
  }
}
```

## 🔄 **Configuration Management**

### **Loading Configuration**
```bash
# Load from specific config file
python benchmark_runner.py --config config.json

# Load multiple config files (later files override earlier ones)
python benchmark_runner.py --config base_config.json --config env_config.json

# Validate configuration without running
python benchmark_runner.py --validate-config config.json
```

### **Configuration Templates**
```bash
# Generate default configuration template
python benchmark_runner.py --generate-config > my_config.json

# Generate configuration for specific use case
python benchmark_runner.py --generate-config --template production

# Validate existing configuration
python benchmark_runner.py --validate-config my_config.json
```

---

This configuration system provides **maximum flexibility** while maintaining **sensible defaults**. You can start simple and add complexity as your needs grow, following the same patterns used in modern web frameworks and cloud applications.