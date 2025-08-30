# Common Workflows Guide

This guide shows typical usage patterns and best practices for AI model evaluation. Think of this as your **cookbook of proven workflows** - real-world scenarios with step-by-step instructions.

## 🎯 **Quick Reference**

| Workflow | Time | Use Case | Command Pattern |
|----------|------|----------|------------------|
| [Quick Check](#quick-model-check) | 2 mins | Development testing | `--mode single --test-id easy_*` |
| [Model Comparison](#model-comparison) | 10 mins | Choose between models | `--output-dir model_a/` vs `model_b/` |
| [Production Readiness](#production-readiness-assessment) | 30 mins | Pre-deployment testing | `--mode concurrent --category all` |
| [Performance Monitoring](#performance-monitoring) | 5 mins | Regular health checks | `--performance-monitoring --limit 10` |
| [Domain Analysis](#domain-specific-analysis) | 15 mins | Specific capability assessment | `--category domain_name` |

## 🚀 **Development Workflows**

### **Quick Model Check** (2 minutes)
**When**: During model development, after changes, quick sanity checks

**Goal**: Verify basic functionality without running full evaluation

```bash
# Single easy test to verify basic functionality
python benchmark_runner.py --test-type base --mode single \
  --test-id easy_reasoning_01 --model "dev-model-v1"

# Quick category check (3-5 tests)
python benchmark_runner.py --test-type base --mode category \
  --category reasoning_general --limit 5 --model "dev-model-v1"
```

**Expected Output**:
```
✅ Test completed: easy_reasoning_01
📊 Overall Score: 78.5
⚡ Execution Time: 3.2s
💾 Results saved to: test_results/
```

**Success Criteria**: 
- Test completes without errors
- Score above 60 (basic functionality)
- Response makes sense when you read it

**Next Steps**:
- ✅ **Success**: Continue with more comprehensive testing
- ❌ **Failure**: Check model configuration, API connectivity, prompt format

---

### **Iterative Development Testing** (5-10 minutes)
**When**: Making incremental improvements, comparing configurations

**Goal**: Test specific changes with consistent baseline

```bash
# Create baseline results
python benchmark_runner.py --test-type base --mode category \
  --category reasoning_general --model "baseline" \
  --output-dir "baseline_results" --limit 10

# Test with temperature 0.5
python benchmark_runner.py --test-type base --mode category \
  --category reasoning_general --model "test-temp-05" \
  --output-dir "temp_05_results" --limit 10 \
  --model-param "temperature=0.5"

# Test with temperature 0.9  
python benchmark_runner.py --test-type base --mode category \
  --category reasoning_general --model "test-temp-09" \
  --output-dir "temp_09_results" --limit 10 \
  --model-param "temperature=0.9"
```

**Analysis Pattern**:
1. **Compare overall scores**: Which configuration performs better?
2. **Check consistency**: Lower standard deviation = more predictable
3. **Review specific failures**: What types of questions cause problems?

**Automation Tip**:
```bash
# Create a development test script
cat > dev_test.sh << EOF
#!/bin/bash
MODEL_NAME=\${1:-"dev-model"}
echo "Testing \$MODEL_NAME..."
python benchmark_runner.py --test-type base --mode category \\
  --category reasoning_general --limit 5 --model "\$MODEL_NAME" \\
  --quiet --output-dir "dev_test_\$(date +%Y%m%d_%H%M%S)"
EOF
chmod +x dev_test.sh

# Usage: ./dev_test.sh "my-model-v2"
```

## 🔄 **Model Comparison Workflows**

### **Model Comparison** (10-15 minutes)
**When**: Choosing between different models, versions, or configurations

**Goal**: Objective comparison with identical test conditions

```bash
# Define common parameters
COMMON_ARGS="--test-type base --mode category --category reasoning_general --workers 4"

# Test Model A
python benchmark_runner.py $COMMON_ARGS \
  --model "model-a" --output-dir "comparison/model_a"

# Test Model B  
python benchmark_runner.py $COMMON_ARGS \
  --model "model-b" --output-dir "comparison/model_b"

# Test Model C
python benchmark_runner.py $COMMON_ARGS \
  --model "model-c" --output-dir "comparison/model_c"
```

**Analysis Workflow**:
```bash
# Compare overall performance
echo "Model A:" && cat comparison/model_a/batch_summary.json | grep overall_score
echo "Model B:" && cat comparison/model_b/batch_summary.json | grep overall_score  
echo "Model C:" && cat comparison/model_c/batch_summary.json | grep overall_score

# Compare by category
python scripts/compare_results.py comparison/model_a comparison/model_b comparison/model_c
```

**Decision Matrix Example**:
| Model | Overall Score | Technical Accuracy | Reliability | Speed | Cost |
|-------|---------------|-------------------|-------------|-------|------|
| Model A | 87.2 | 92 | 84 | Fast | High |
| Model B | 84.1 | 89 | 88 | Medium | Medium |
| Model C | 79.8 | 85 | 82 | Slow | Low |

**Recommendation**: Choose based on priorities:
- **Quality-first**: Model A
- **Balanced**: Model B  
- **Cost-conscious**: Model C

---

### **Multi-Domain Comparison** (20-30 minutes)
**When**: Need comprehensive understanding across all capabilities

**Goal**: Identify model strengths and weaknesses across domains

```bash
# Comprehensive multi-domain evaluation
DOMAINS=("reasoning_general" "creativity_writing" "language_grammar" "social_cultural")
MODEL="comprehensive-test-model"

for domain in "${DOMAINS[@]}"; do
  echo "Testing domain: $domain"
  python benchmark_runner.py --test-type base --mode category \
    --category "$domain" --model "$MODEL" \
    --output-dir "comprehensive/${MODEL}/${domain}" \
    --workers 2
done

# Generate comprehensive report
python scripts/generate_comprehensive_report.py "comprehensive/${MODEL}/"
```

**Result Analysis Pattern**:
```json
{
  "model_profile": {
    "reasoning_general": 89.2,     // Excellent reasoning
    "creativity_writing": 76.4,    // Fair creativity
    "language_grammar": 91.8,      // Excellent language
    "social_cultural": 68.9        // Weak social understanding  
  },
  "recommendations": {
    "best_for": ["technical analysis", "documentation", "language tasks"],
    "avoid_for": ["creative writing", "social situations", "cultural content"],
    "with_supervision": ["creative tasks", "social interactions"]
  }
}
```

## 🏭 **Production Workflows**

### **Production Readiness Assessment** (30-45 minutes)
**When**: Before deploying model to production

**Goal**: Comprehensive evaluation to ensure quality and reliability

```bash
# Phase 1: Quick smoke test
python benchmark_runner.py --test-type base --mode single \
  --test-id easy_reasoning_01 --model "production-candidate" \
  || exit 1

# Phase 2: Category evaluation  
python benchmark_runner.py --test-type base --mode concurrent \
  --workers 4 --category reasoning_general \
  --model "production-candidate" \
  --output-dir "production_readiness/reasoning" \
  --performance-monitoring

# Phase 3: Multi-domain evaluation
python benchmark_runner.py --test-type base --mode concurrent \
  --workers 4 --category all \
  --model "production-candidate" \
  --output-dir "production_readiness/comprehensive" \
  --performance-monitoring

# Phase 4: Stress testing
python benchmark_runner.py --test-type instruct --mode concurrent \
  --workers 8 --category reasoning_complex \
  --model "production-candidate" \
  --output-dir "production_readiness/stress_test"
```

**Production Readiness Checklist**:
- [ ] **Overall score ≥ 80**: Minimum quality threshold
- [ ] **Reliability score ≥ 75**: Suitable for autonomous operation
- [ ] **Standard deviation ≤ 10**: Consistent performance
- [ ] **No failed test executions**: Technical stability
- [ ] **Response time ≤ 30s**: Acceptable performance
- [ ] **All critical categories ≥ 70**: Domain coverage

**Go/No-Go Decision**:
```bash
# Automated production readiness check
python scripts/production_readiness_check.py production_readiness/comprehensive/
```

**Expected Output**:
```
🔍 PRODUCTION READINESS ASSESSMENT
================================

✅ Overall Score: 84.2 (≥80 required)
✅ Reliability: 78.9 (≥75 required)  
✅ Consistency: σ=8.3 (≤10 required)
✅ Technical Stability: 0 failures
⚠️  Response Time: 45.2s (>30s warning)
✅ Domain Coverage: All ≥70

RECOMMENDATION: ✅ APPROVED FOR PRODUCTION
Note: Monitor response times in production
```

---

### **Performance Monitoring** (5 minutes)
**When**: Regular health checks of deployed models

**Goal**: Detect performance degradation early

```bash
# Daily health check (automated via cron)
python benchmark_runner.py --test-type base --mode category \
  --category reasoning_general --limit 10 \
  --model "production-model" \
  --output-dir "monitoring/$(date +%Y%m%d)" \
  --performance-monitoring --quiet

# Weekly comprehensive check
python benchmark_runner.py --test-type base --mode concurrent \
  --workers 4 --category all --limit 20 \
  --model "production-model" \
  --output-dir "monitoring/weekly/$(date +%Y%m%d)" \
  --performance-monitoring
```

**Monitoring Automation**:
```bash
# Create monitoring cron job
cat > monitor_model.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d)
RESULTS_DIR="monitoring/$DATE"

python benchmark_runner.py --test-type base --mode category \
  --category reasoning_general --limit 5 \
  --model "production-model" \
  --output-dir "$RESULTS_DIR" --quiet

# Check for performance degradation
SCORE=$(cat "$RESULTS_DIR/batch_summary.json" | jq '.summary.average_score')
if (( $(echo "$SCORE < 75" | bc -l) )); then
  echo "ALERT: Model performance degraded to $SCORE" | mail -s "Model Alert" admin@company.com
fi
EOF

# Add to cron (daily at 2 AM)
# 0 2 * * * /path/to/monitor_model.sh
```

## 🎯 **Domain-Specific Workflows**

### **Domain-Specific Analysis** (15-20 minutes)
**When**: Focusing on specific capabilities or use cases

**Goal**: Deep understanding of performance in particular domain

```bash
# Reasoning-focused analysis
python benchmark_runner.py --test-type base --mode category \
  --category reasoning_general --model "reasoning-specialist" \
  --output-dir "domain_analysis/reasoning/general" \
  --workers 4

python benchmark_runner.py --test-type base --mode category \
  --category reasoning_complex --model "reasoning-specialist" \
  --output-dir "domain_analysis/reasoning/complex" \
  --workers 4

# Analyze reasoning patterns
python scripts/analyze_reasoning_patterns.py domain_analysis/reasoning/
```

**Creative Task Analysis**:
```bash
# Creative capabilities evaluation
CREATIVE_CATEGORIES=("creativity_writing" "creativity_storytelling" "creativity_brainstorming")

for category in "${CREATIVE_CATEGORIES[@]}"; do
  python benchmark_runner.py --test-type base --mode category \
    --category "$category" --model "creative-model" \
    --output-dir "domain_analysis/creative/$category"
done

# Generate creativity report
python scripts/analyze_creativity_patterns.py domain_analysis/creative/
```

### **Cultural and Social Analysis** (20-25 minutes)
**When**: Deploying globally or handling diverse user interactions

**Goal**: Assess cultural sensitivity and social understanding

```bash
# Cultural awareness testing
python benchmark_runner.py --test-type base --mode category \
  --category social_cultural --model "global-model" \
  --output-dir "cultural_analysis/general" \
  --cultural-validation --workers 2

# Multi-cultural context testing
CULTURAL_CONTEXTS=("west_african" "east_asian" "latin_american" "european")

for context in "${CULTURAL_CONTEXTS[@]}"; do
  python benchmark_runner.py --test-type base --mode category \
    --category social_cultural --model "global-model" \
    --cultural-context "$context" \
    --output-dir "cultural_analysis/$context"
done

# Generate cultural sensitivity report
python scripts/cultural_analysis_report.py cultural_analysis/
```

## 🔧 **Troubleshooting Workflows**

### **Debugging Poor Performance** (10-15 minutes)
**When**: Model scores are lower than expected

**Goal**: Identify root causes and improvement opportunities

```bash
# Step 1: Isolate the problem
python benchmark_runner.py --test-type base --mode single \
  --test-id easy_reasoning_01 --model "problem-model" \
  --debug --save-intermediate

# Step 2: Check specific dimensions
python benchmark_runner.py --test-type base --mode category \
  --category reasoning_general --limit 10 \
  --model "problem-model" \
  --detailed-scoring --output-dir "debug/reasoning"

# Step 3: Compare with known good model
python benchmark_runner.py --test-type base --mode category \
  --category reasoning_general --limit 10 \
  --model "baseline-model" \
  --output-dir "debug/baseline"

# Step 4: Analyze differences
python scripts/debug_performance_diff.py debug/reasoning debug/baseline
```

**Common Issues and Solutions**:

**Low Organization Quality**:
```bash
# Test with structure prompts
python benchmark_runner.py --test-type base --mode single \
  --test-id easy_reasoning_01 --model "problem-model" \
  --prompt-prefix "Please structure your response with clear headings: "
```

**Low Technical Accuracy**:
```bash
# Test knowledge boundaries
python benchmark_runner.py --test-type base --mode category \
  --category knowledge_facts --model "problem-model" \
  --output-dir "debug/knowledge_check"
```

**Inconsistent Performance**:
```bash
# Test same question multiple times
for i in {1..5}; do
  python benchmark_runner.py --test-type base --mode single \
    --test-id reasoning_logic_01 --model "problem-model" \
    --output-dir "debug/consistency/run_$i"
done

python scripts/analyze_consistency.py debug/consistency/
```

---

### **API Integration Debugging** (5-10 minutes)
**When**: Connection issues or unexpected API responses

**Goal**: Isolate and fix integration problems

```bash
# Step 1: Test API directly
curl -X POST http://localhost:8004/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "prompt": "Test prompt",
    "max_tokens": 100
  }'

# Step 2: Test with minimal benchmark call
python benchmark_runner.py --test-type base --mode single \
  --test-id easy_reasoning_01 --model "your-model" \
  --debug-api --verbose

# Step 3: Check logs
tail -f benchmark.log
```

**Common API Issues**:
- **Connection refused**: Check endpoint URL and model server status
- **Timeout errors**: Increase timeout or reduce max_tokens  
- **Authentication errors**: Verify API keys and authentication headers
- **Rate limiting**: Add delays between requests or reduce concurrency

## 📊 **Analysis and Reporting Workflows**

### **Comprehensive Model Report** (30-45 minutes)
**When**: Need detailed analysis for stakeholders

**Goal**: Generate comprehensive model assessment report

```bash
# Full evaluation across all domains
python benchmark_runner.py --test-type base --mode concurrent \
  --workers 6 --category all \
  --model "report-model" \
  --output-dir "comprehensive_report/raw_results" \
  --performance-monitoring

# Generate detailed report
python scripts/generate_comprehensive_report.py \
  comprehensive_report/raw_results/ \
  --output comprehensive_report/model_assessment_report.pdf \
  --include-charts --include-recommendations
```

**Report Contents**:
1. **Executive Summary**: Overall performance and recommendations
2. **Detailed Scores**: All dimensions and categories  
3. **Strengths and Weaknesses**: What the model does well/poorly
4. **Use Case Recommendations**: Best applications for this model
5. **Comparison with Baselines**: How it compares to other models
6. **Performance Metrics**: Speed, consistency, reliability stats
7. **Sample Responses**: Examples of good and poor responses

---

### **Trend Analysis** (15-20 minutes)  
**When**: Tracking model improvement over time

**Goal**: Understand development progress and identify patterns

```bash
# Collect historical data
DATES=("2024-01-15" "2024-02-01" "2024-02-15" "2024-03-01")
MODEL="evolving-model"

for date in "${DATES[@]}"; do
  # Simulate historical testing (use actual historical results in practice)
  python benchmark_runner.py --test-type base --mode category \
    --category reasoning_general --model "$MODEL" \
    --output-dir "trends/data/$date" \
    --timestamp-override "$date"
done

# Generate trend analysis
python scripts/trend_analysis.py trends/data/ \
  --output trends/improvement_trends.html
```

This guide provides proven workflows for every stage of AI model development and deployment. Use these patterns as starting points and adapt them to your specific needs and constraints.