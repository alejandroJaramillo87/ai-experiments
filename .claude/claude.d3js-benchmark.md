# D3.js Visualization Architecture

## Directory Structure

benchmark_results/
├── data/
│   ├── runs/
│   │   ├── 2024-01-20-llama3-8b/
│   │   │   ├── summary.json
│   │   │   ├── detailed_results.json
│   │   │   └── raw_responses.jsonl
│   │   └── [timestamp-model-name]/
│   └── aggregated/
│       ├── model_comparisons.json
│       └── historical_trends.json
├── web/
│   ├── index.html
│   ├── css/
│   │   └── dashboard.css
│   ├── js/
│   │   ├── visualizations.js
│   │   ├── charts/
│   │   │   ├── radar_chart.js
│   │   │   ├── heatmap.js
│   │   │   ├── progression_chart.js
│   │   │   └── failure_analysis.js
│   │   └── data_loader.js
│   └── templates/
│       └── report_template.html
└── scripts/
    ├── generate_report.py
    └── aggregate_results.py

## Data Schema for Visualization

### Run Summary Format

{
  "run_metadata": {
    "id": "2024-01-20-llama3-8b",
    "timestamp": "2024-01-20T10:30:00Z",
    "model": {
      "name": "llama3-8b",
      "parameters": "8B",
      "quantization": "fp4"
    },
    "hardware": {
      "gpu": "RTX 5090",
      "vram_used": "18GB",
      "inference_speed": "45 tokens/sec"
    }
  },
  "aggregate_scores": {
    "overall": 0.67,
    "by_difficulty": {
      "easy": 0.85,
      "medium": 0.62,
      "hard": 0.31
    },
    "by_domain": {
      "speculative_worlds": 0.72,
      "paradox_resolution": 0.58,
      "system_architecture": 0.69
    }
  },
  "capability_scores": {
    "logical_consistency": 0.75,
    "creativity": 0.68,
    "paradox_handling": 0.45,
    "meta_reasoning": 0.32
  },
  "failure_analysis": {
    "total_tests": 180,
    "failures": {
      "repetition": 12,
      "incoherence": 8,
      "refusal": 5,
      "timeout": 3
    }
  }
}

## Visualization Components

### 1. Model Capability Radar Chart

// js/charts/radar_chart.js
function renderCapabilityRadar(data) {
    const capabilities = [
        'logical_consistency',
        'creativity', 
        'paradox_handling',
        'meta_reasoning',
        'pattern_recognition',
        'synthesis'
    ];
    
    // D3.js radar chart implementation
    // Shows model strengths/weaknesses
}

### 2. Domain-Difficulty Heatmap

// js/charts/heatmap.js
function renderDomainHeatmap(data) {
    // Domains on Y-axis
    // Difficulties on X-axis  
    // Color intensity = score
    // Click for detailed breakdown
}

### 3. Historical Progression Chart

// js/charts/progression_chart.js
function renderProgressionChart(historicalData) {
    // Time series of model performance
    // Track improvements over optimizations
    // Multiple models on same chart
}

### 4. Failure Pattern Analysis

// js/charts/failure_analysis.js
function renderFailurePatterns(data) {
    // Sankey diagram showing:
    // Test type → Failure mode → Frequency
}

## GitHub Pages Integration

### Automated Report Generation

# scripts/generate_report.py
def generate_report(run_id):
    # 1. Load test results
    results = load_results(run_id)
    
    # 2. Generate visualizations
    charts = {
        'radar': generate_radar_data(results),
        'heatmap': generate_heatmap_data(results),
        'progression': generate_progression_data(results)
    }
    
    # 3. Create HTML report
    html = render_template('report_template.html', {
        'run_id': run_id,
        'results': results,
        'charts': charts
    })
    
    # 4. Save to GitHub Pages directory
    save_report(html, f'docs/reports/{run_id}.html')
    
    # 5. Update index
    update_report_index(run_id)

### Interactive Dashboard Features

1. Model Comparison View
   - Select multiple models
   - Side-by-side visualization
   - Relative strengths/weaknesses

2. Drill-Down Navigation
   - Click domain → see all tests
   - Click test → see actual response
   - Click score → see evaluation details

3. Filter Controls
   - By difficulty
   - By domain
   - By capability
   - By failure type

4. Export Options
   - Download raw data (JSON)
   - Export charts (PNG/SVG)
   - Generate PDF report

## Implementation Timeline

1. Week 1: Basic data pipeline
   - Test result storage
   - JSON generation
   - Basic aggregation

2. Week 2: Core visualizations
   - Radar chart
   - Heatmap
   - Basic HTML template

3. Week 3: Interactive features
   - Filtering
   - Drill-down
   - Model comparison

4. Week 4: Polish & deployment
   - GitHub Pages setup
   - Automated generation
   - Documentation

## Example Usage

# Run benchmark
python run_benchmark.py --model llama3-8b --domains all

# Generate report
python scripts/generate_report.py --run-id 2024-01-20-llama3-8b

# Aggregate historical data
python scripts/aggregate_results.py --last-n-runs 10

# Serve locally for development
cd benchmark_results/web && python -m http.server 8000