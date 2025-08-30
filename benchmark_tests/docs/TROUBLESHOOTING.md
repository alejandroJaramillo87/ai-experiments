# Troubleshooting Guide

This guide covers common issues, debugging techniques, and solutions for the AI Model Evaluation Framework. Think of this as your **technical support manual** - systematic approaches to diagnosing and fixing problems.

## 🎯 **Quick Diagnostic Steps**

### **5-Minute Health Check**
```bash
# 1. Verify basic functionality
python benchmark_runner.py --help

# 2. Check dependencies
python -c "import evaluator; print('Evaluator module OK')"

# 3. Test simple evaluation
python benchmark_runner.py --test-type base --mode single \
  --test-id easy_reasoning_01 --endpoint http://localhost:8004 \
  --model test-model --timeout 30

# 4. Check system resources
free -h && df -h && ps aux | grep python

# 5. Review recent logs
tail -n 50 benchmark.log
```

### **Common Issue Categories**
1. **🔌 Connection Issues**: API endpoints, network connectivity
2. **⚙️ Configuration Problems**: Invalid settings, missing parameters
3. **🧠 Evaluation Errors**: Evaluator failures, scoring inconsistencies  
4. **📊 Performance Issues**: Slow execution, high resource usage
5. **📁 File/Directory Issues**: Permissions, missing files, storage
6. **🐛 Code Issues**: Import errors, version conflicts

## 🔌 **Connection and API Issues**

### **Problem: "Connection refused" or API timeouts**

**Symptoms**:
```
requests.exceptions.ConnectionError: HTTPSConnectionPool(host='localhost', port=8004): 
Max retries exceeded with url: /v1/completions
```

**Diagnosis Steps**:
```bash
# 1. Check if model server is running
curl -I http://localhost:8004/health
# or
telnet localhost 8004

# 2. Check network connectivity
ping localhost
nslookup your-model-server.com

# 3. Verify endpoint format
echo $MODEL_ENDPOINT
# Should be: http://host:port/v1/completions
```

**Solutions**:

**Local Model Server**:
```bash
# Start your model server
# For Ollama:
ollama serve

# For Text Generation WebUI:
python server.py --listen --api

# For custom server:
python your_model_server.py --port 8004
```

**Configuration Fix**:
```bash
# Update endpoint configuration
export MODEL_ENDPOINT="http://localhost:11434/api/generate"  # Ollama
export MODEL_ENDPOINT="http://localhost:5000/api/v1/generate"  # TextGen

# Or use command line override
python benchmark_runner.py --endpoint "http://correct-host:port/path"
```

**Firewall/Network**:
```bash
# Check firewall (Linux)
sudo ufw status
sudo iptables -L

# Test with curl
curl -X POST http://localhost:8004/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "prompt": "Hello", "max_tokens": 10}'
```

---

### **Problem: Authentication/Authorization Errors**

**Symptoms**:
```
401 Unauthorized
403 Forbidden
Invalid API key
```

**Diagnosis**:
```bash
# Check API key configuration
echo $MODEL_API_KEY
# Should show your API key (if required)

# Test authentication
curl -H "Authorization: Bearer $MODEL_API_KEY" \
  http://your-api-endpoint/v1/models
```

**Solutions**:
```bash
# Set API key environment variable
export MODEL_API_KEY="your-actual-api-key-here"

# Or use config file
cat > config.json << EOF
{
  "model": {
    "endpoint": "https://api.openai.com/v1/completions",
    "api_key": "your-api-key",
    "headers": {
      "Authorization": "Bearer your-api-key"
    }
  }
}
EOF

# Use configuration
python benchmark_runner.py --config config.json
```

---

### **Problem: Model responses are empty or malformed**

**Symptoms**:
```
Model response is empty
JSON decode error
Unexpected response format
```

**Diagnosis**:
```bash
# Check raw API response
curl -X POST http://localhost:8004/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "your-model", "prompt": "Test prompt", "max_tokens": 50}' \
  | jq '.'

# Enable debug logging
python benchmark_runner.py --debug --verbose \
  --test-type base --mode single --test-id easy_reasoning_01
```

**Solutions**:

**API Format Mismatch**:
```python
# Check and fix API format in config
{
  "model": {
    "api_format": "openai",  # or "ollama", "textgen", "custom"
    "response_path": "choices[0].text"  # Path to extract response
  }
}
```

**Model Parameters**:
```python
# Adjust model parameters
{
  "model": {
    "parameters": {
      "max_tokens": 500,    # Increase if responses are cut off
      "temperature": 0.7,   # Adjust creativity/randomness
      "top_p": 0.9,        # Nucleus sampling
      "stop": ["\n\n"]     # Stop sequences
    }
  }
}
```

## ⚙️ **Configuration Issues**

### **Problem: "Configuration validation failed"**

**Symptoms**:
```
ConfigurationError: Invalid configuration
KeyError: 'required_field'
ValueError: Configuration field out of range
```

**Diagnosis**:
```bash
# Validate configuration file
python -c "
import json
with open('config.json') as f:
    config = json.load(f)
    print('Config loaded successfully')
    print(json.dumps(config, indent=2))
"

# Check for missing required fields
python benchmark_runner.py --validate-config config.json
```

**Solutions**:

**Generate Valid Configuration Template**:
```bash
# Generate default configuration
python benchmark_runner.py --generate-config > valid_config.json

# Edit and use the template
cp valid_config.json config.json
# Edit config.json with your settings
```

**Fix Common Configuration Errors**:
```json
{
  "model": {
    "endpoint": "http://localhost:8004/v1/completions",  // Required
    "name": "model-name",                                // Required
    "timeout": 60,                                       // Must be > 0
    "max_retries": 3                                     // Must be >= 0
  },
  "evaluation": {
    "concurrent_workers": 4,                             // Must be > 0
    "weights": {                                         // Must sum to 1.0
      "organization_quality": 0.25,
      "technical_accuracy": 0.4,
      "completeness": 0.2,
      "reliability": 0.15
    }
  }
}
```

---

### **Problem: Test files not found**

**Symptoms**:
```
TestLoadError: Test file not found: reasoning_logic_01.json
FileNotFoundError: No such file or directory
```

**Diagnosis**:
```bash
# Check available test files
find domains/ -name "*.json" | sort

# Verify specific test exists
ls -la domains/reasoning/base_models/easy_reasoning_01.json

# Check test ID format
cat domains/reasoning/base_models/easy_reasoning_01.json | jq '.test_id'
```

**Solutions**:

**Use Correct Test ID**:
```bash
# List available test IDs
find domains/ -name "*.json" -exec basename {} .json \;

# Use exact test ID
python benchmark_runner.py --mode single --test-id "easy_reasoning_01"
```

**Fix File Paths**:
```bash
# Ensure you're in the correct directory
cd benchmark_tests
pwd  # Should end with /benchmark_tests

# Fix permissions if needed
chmod +r domains/**/*.json
```

## 🧠 **Evaluation Issues**

### **Problem: Inconsistent or unrealistic scores**

**Symptoms**:
```
Scores vary wildly between runs
All scores are 0 or 100
Dimension scores don't make sense
```

**Diagnosis**:
```bash
# Run same test multiple times
for i in {1..5}; do
  python benchmark_runner.py --mode single --test-id easy_reasoning_01 \
    --output-dir "debug/run_$i"
done

# Compare results
python scripts/compare_consistency.py debug/run_*

# Enable detailed evaluation logging
python benchmark_runner.py --debug --detailed-scoring \
  --mode single --test-id easy_reasoning_01
```

**Solutions**:

**Fix Evaluator Configuration**:
```json
{
  "reasoning": {
    "weights": {
      "organization_quality": 0.25,
      "technical_accuracy": 0.4,     // Ensure weights sum to 1.0
      "completeness": 0.2,
      "reliability": 0.15
    },
    "strict_mode": false,             // Try disabling strict mode
    "confidence_threshold": 0.7       // Adjust threshold
  }
}
```

**Debug Specific Evaluator**:
```python
# Create debug script: debug_evaluator.py
from evaluator.subjects.reasoning_evaluator import ReasoningEvaluator

evaluator = ReasoningEvaluator()
test_text = "Your test response here"
context = {"test_id": "debug", "difficulty": "medium"}

result = evaluator.evaluate(test_text, context)
print(f"Overall score: {result.overall_score}")
for dim in result.dimensions:
    print(f"{dim.name}: {dim.score} (confidence: {dim.confidence})")
    print(f"  Evidence: {dim.evidence}")
```

**Calibrate Evaluation Weights**:
```bash
# Run calibration against known good/bad responses
python scripts/calibrate_evaluators.py \
  --good-examples examples/high_quality/ \
  --bad-examples examples/low_quality/
```

---

### **Problem: Cultural validation errors**

**Symptoms**:
```
CulturalValidationError: Bias detected
Cultural context conflict
Inappropriate cultural reference
```

**Diagnosis**:
```bash
# Check cultural data loading
python -c "
from evaluator.cultural.cultural_validator import CulturalValidator
validator = CulturalValidator()
print('Cultural validator loaded successfully')
"

# Debug specific cultural context
python benchmark_runner.py --debug --cultural-analysis \
  --mode single --test-id social_cultural_01
```

**Solutions**:

**Update Cultural Configuration**:
```json
{
  "cultural": {
    "validation_strictness": "medium",    // Try "low" for debugging
    "bias_detection": {
      "enabled": true,
      "sensitivity": "medium"             // Reduce from "high"
    },
    "excluded_checks": ["gender_bias"]    // Temporarily disable specific checks
  }
}
```

**Disable Cultural Validation Temporarily**:
```bash
# Disable cultural validation for debugging
python benchmark_runner.py --no-cultural-validation \
  --mode single --test-id problematic_test
```

## 📊 **Performance Issues**

### **Problem: Slow execution or timeouts**

**Symptoms**:
```
Request timeout after 30 seconds
Very slow evaluation (> 5 minutes per test)
High CPU/memory usage
```

**Diagnosis**:
```bash
# Monitor resource usage during execution
top -p $(pgrep -f benchmark_runner)

# Profile execution
python -m cProfile -o profile.stats benchmark_runner.py \
  --mode single --test-id easy_reasoning_01

# Analyze profile
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"

# Check model API response times
time curl -X POST http://localhost:8004/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "prompt": "Quick test", "max_tokens": 10}'
```

**Solutions**:

**Optimize Configuration**:
```json
{
  "model": {
    "timeout": 60,              // Increase timeout
    "max_tokens": 200,          // Reduce if appropriate
    "temperature": 0.5          // Lower temperature = faster
  },
  "evaluation": {
    "concurrent_workers": 2,    // Reduce concurrent workers
    "enable_caching": true,     // Enable result caching
    "batch_size": 10           // Process in smaller batches
  }
}
```

**Optimize Model Parameters**:
```bash
# Use faster model settings
python benchmark_runner.py \
  --model-param "max_tokens=100" \
  --model-param "temperature=0.3" \
  --timeout 30
```

**Scale Resources**:
```bash
# Increase system limits
ulimit -n 4096  # Increase file descriptors
ulimit -u 2048  # Increase process limit

# Monitor and adjust
htop
iotop
```

---

### **Problem: Memory usage keeps growing**

**Symptoms**:
```
Python process using > 8GB RAM
"Out of memory" errors
System becomes unresponsive
```

**Diagnosis**:
```bash
# Monitor memory usage over time
python -c "
import psutil
import time
import os

pid = os.getpid()
process = psutil.Process(pid)

for i in range(10):
    mem = process.memory_info()
    print(f'RSS: {mem.rss / 1024 / 1024:.1f} MB, VMS: {mem.vms / 1024 / 1024:.1f} MB')
    # Run your benchmark here
    time.sleep(30)
"

# Check for memory leaks
python -m memory_profiler benchmark_runner.py --mode single --test-id test
```

**Solutions**:

**Enable Memory Management**:
```json
{
  "evaluation": {
    "memory_management": {
      "max_cache_size_mb": 1000,    // Limit cache size
      "cleanup_frequency": 100,     // Clean up every N evaluations
      "batch_processing": true      // Process in batches
    }
  }
}
```

**Process in Smaller Batches**:
```bash
# Process tests in smaller batches
python benchmark_runner.py --mode category \
  --category reasoning_general \
  --batch-size 10 \
  --cleanup-after-batch
```

## 📁 **File and Permission Issues**

### **Problem: Permission denied errors**

**Symptoms**:
```
PermissionError: [Errno 13] Permission denied
OSError: [Errno 1] Operation not permitted
```

**Diagnosis**:
```bash
# Check file permissions
ls -la test_results/
ls -la domains/
ls -la evaluator/

# Check directory permissions
stat test_results/

# Check user/group ownership
id
groups
```

**Solutions**:

**Fix File Permissions**:
```bash
# Fix permissions for result directory
chmod -R 755 test_results/

# Fix ownership if needed
sudo chown -R $(whoami):$(whoami) benchmark_tests/

# Create missing directories with correct permissions
mkdir -p test_results/$(date +%Y%m%d)
chmod 755 test_results/$(date +%Y%m%d)
```

**Run with Appropriate User**:
```bash
# Don't run as root unless necessary
# Create dedicated user for evaluation service
sudo useradd -m -s /bin/bash evaluator
sudo su - evaluator
```

---

### **Problem: Disk space issues**

**Symptoms**:
```
OSError: [Errno 28] No space left on device
Disk usage 100%
```

**Diagnosis**:
```bash
# Check disk usage
df -h
du -sh test_results/
du -sh logs/

# Find large files
find . -type f -size +100M -exec ls -lh {} \;

# Check log file sizes
ls -lh *.log
```

**Solutions**:

**Clean Up Old Results**:
```bash
# Remove results older than 7 days
find test_results/ -type f -mtime +7 -delete

# Clean up log files
truncate -s 0 benchmark.log
logrotate /etc/logrotate.conf

# Use automated cleanup
python cleanup_test_artifacts.py --older-than 7
```

**Configure Log Rotation**:
```bash
# Add to /etc/logrotate.d/benchmark-tests
/path/to/benchmark_tests/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    copytruncate
}
```

## 🐛 **Code and Import Issues**

### **Problem: Import errors or module not found**

**Symptoms**:
```
ImportError: No module named 'evaluator'
ModuleNotFoundError: No module named 'requests'
```

**Diagnosis**:
```bash
# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Check if modules are installed
python -c "import evaluator; print('OK')"
pip list | grep requests

# Check current directory
pwd
ls -la evaluator/
```

**Solutions**:

**Fix Python Path**:
```bash
# Add current directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use -m flag
python -m benchmark_runner --help
```

**Install Missing Dependencies**:
```bash
# Install from requirements file
pip install -r requirements.txt

# Install specific package
pip install requests numpy

# Upgrade packages
pip install --upgrade requests
```

**Virtual Environment Issues**:
```bash
# Create new virtual environment
python -m venv fresh_venv
source fresh_venv/bin/activate  # Linux/Mac
# or
fresh_venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

---

### **Problem: Version conflicts**

**Symptoms**:
```
VersionConflict: package 1.0.0 conflicts with requirement package>=2.0.0
AttributeError: module has no attribute 'new_function'
```

**Diagnosis**:
```bash
# Check installed versions
pip list
pip show numpy requests

# Check for conflicts
pip check

# See dependency tree
pip install pipdeptree
pipdeptree
```

**Solutions**:

**Update Dependencies**:
```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade numpy==1.21.0

# Force reinstall
pip install --force-reinstall numpy
```

**Use Version Pinning**:
```bash
# Pin exact versions in requirements.txt
numpy==1.21.0
requests==2.25.1
pandas==1.3.0

# Install pinned versions
pip install -r requirements.txt
```

## 🔧 **Advanced Debugging Techniques**

### **Enable Comprehensive Logging**

**debug_config.json**:
```json
{
  "logging": {
    "level": "DEBUG",
    "format": "detailed",
    "components": {
      "evaluator": "DEBUG",
      "api_client": "DEBUG", 
      "cultural_validator": "DEBUG",
      "performance": "DEBUG"
    },
    "output": {
      "console": true,
      "file": "debug.log",
      "structured": true
    }
  }
}
```

### **Performance Profiling**

**profile_evaluation.py**:
```python
import cProfile
import pstats
from benchmark_runner import BenchmarkTestRunner

def profile_evaluation():
    runner = BenchmarkTestRunner(
        model_endpoint="http://localhost:8004/v1/completions",
        model_name="test-model"
    )
    
    result = runner.execute_single_test("easy_reasoning_01", "base")
    return result

if __name__ == "__main__":
    # Profile the evaluation
    cProfile.run('profile_evaluation()', 'evaluation_profile.stats')
    
    # Analyze results
    p = pstats.Stats('evaluation_profile.stats')
    p.sort_stats('cumulative')
    p.print_stats(20)
    
    # Find bottlenecks
    p.print_stats('evaluate')
```

### **Memory Debugging**

```bash
# Install memory profiler
pip install memory-profiler

# Profile memory usage
python -m memory_profiler benchmark_runner.py --mode single --test-id test

# Line-by-line memory profiling
@profile
def memory_intensive_function():
    # Your code here
    pass
```

### **Network Debugging**

```bash
# Capture network traffic
sudo tcpdump -i any -w network_capture.pcap host localhost and port 8004

# Analyze with Wireshark
wireshark network_capture.pcap

# Monitor HTTP requests
mitmproxy -s capture_script.py
```

## 📞 **Getting Help**

### **When to Seek Community Help**
- Configuration issues persist after following this guide
- Behavior differs significantly from documentation
- Performance issues that don't respond to tuning
- Suspected bugs in evaluation logic

### **How to Report Issues Effectively**

**Include This Information**:
```bash
# System information
python --version
pip list
uname -a  # Linux/Mac
systeminfo  # Windows

# Configuration (remove sensitive data)
cat config.json | jq 'del(.api_key, .secrets)'

# Error logs (last 50 lines)
tail -n 50 benchmark.log

# Steps to reproduce
echo "Exact commands that cause the issue"
```

### **Create Minimal Reproduction Case**
```bash
# Minimal test case
python benchmark_runner.py --debug \
  --test-type base --mode single \
  --test-id easy_reasoning_01 \
  --endpoint http://localhost:8004 \
  --model test-model \
  --timeout 30 \
  --output-dir minimal_repro
  
# Include the output and any error messages
```

---

This troubleshooting guide covers the most common issues encountered in production deployments. For complex issues not covered here, enable debug logging and use the diagnostic techniques provided to gather detailed information for further analysis.