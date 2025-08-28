#!/usr/bin/env python3
"""
Quick Benchmark Summary Script

Provides rapid performance insights from benchmark test results.
Focuses on actionable metrics for AI workstation optimization.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

def load_latest_analysis() -> Dict:
    """Load the most recent benchmark analysis results"""
    base_dir = Path("/home/alejandro/workspace/ai-workstation")
    
    # Find the most recent analysis directory
    analysis_dirs = list(base_dir.glob("benchmark_analysis_*"))
    if not analysis_dirs:
        print("No benchmark analysis found. Run benchmark_results_analyzer.py first.")
        return {}
    
    latest_dir = max(analysis_dirs, key=lambda p: p.name)
    analysis_file = latest_dir / "comprehensive_analysis.json"
    
    if not analysis_file.exists():
        print(f"Analysis file not found in {latest_dir}")
        return {}
    
    with open(analysis_file, 'r') as f:
        return json.load(f)

def print_performance_summary(data: Dict):
    """Print key performance metrics"""
    print("=" * 60)
    print("🚀 AI WORKSTATION PERFORMANCE SUMMARY")
    print("=" * 60)
    
    overview = data['dataset_summary']['dataset_overview']
    print(f"📊 Tests Analyzed: {overview['total_tests']}")
    print(f"✅ Success Rate: {overview['success_rate']:.1f}%")
    print(f"🤖 Models Compared: {overview['models_tested']}")
    print()
    
    # Speed rankings
    print("🏆 SPEED RANKINGS")
    print("-" * 30)
    speed_data = data['dataset_summary']['speed_ranking']
    for i, model in enumerate(sorted(speed_data, key=lambda x: x['speed_rank']), 1):
        print(f"{i}. {model['model_name']:12} | {model['mean']:6.1f} ± {model['std']:4.1f} tokens/sec")
    print()

def print_detailed_comparison(data: Dict):
    """Print detailed model comparison"""
    print("🔬 DETAILED MODEL COMPARISON")
    print("-" * 40)
    
    model_perf = data['dataset_summary']['model_performance']
    
    print("THROUGHPUT ANALYSIS:")
    for model_name, stats in model_perf.items():
        print(f"\n📈 {model_name}:")
        print(f"   Speed: {stats['tokens_per_second_mean']:6.1f} tokens/sec")
        print(f"   Range: {stats['tokens_per_second_min']:6.1f} - {stats['tokens_per_second_max']:6.1f}")
        print(f"   Exec Time: {stats['execution_time_mean']:5.1f} ± {stats['execution_time_std']:4.1f} sec")
        print(f"   Tokens: {stats['total_tokens_mean']:6.0f} avg tokens")
        print(f"   Response: {stats['response_length_mean']:7.0f} avg characters")
    print()

def print_optimization_insights(data: Dict):
    """Print hardware optimization insights"""
    print("⚡ HARDWARE OPTIMIZATION INSIGHTS")
    print("-" * 35)
    
    throughput_data = data['patterns_and_anomalies']['throughput_analysis']
    
    print("CONSISTENCY ANALYSIS:")
    for model, stats in throughput_data.items():
        consistency = "Excellent" if stats['consistency'] < 0.05 else "Good" if stats['consistency'] < 0.1 else "Fair"
        print(f"• {model:12} | {consistency:9} (CV: {stats['consistency']:.3f})")
        print(f"                | Peak: {stats['peak_performance']:6.1f} | Min: {stats['min_performance']:6.1f}")
    print()
    
    print("RTX 5090 UTILIZATION RECOMMENDATIONS:")
    model_perf = data['dataset_summary']['model_performance']
    
    for model_name, stats in model_perf.items():
        avg_speed = stats['tokens_per_second_mean']
        efficiency = stats['tokens_per_second_mean'] / stats['execution_time_mean']
        
        if avg_speed > 200:
            gpu_rec = "🟢 Optimal for GPU deployment"
        elif avg_speed > 150:
            gpu_rec = "🟡 Good GPU utilization"
        else:
            gpu_rec = "🔴 Consider CPU offloading"
        
        print(f"• {model_name:12} | {gpu_rec}")
        print(f"                | Efficiency: {efficiency:5.1f} tokens/sec²")
    print()

def print_production_recommendations(data: Dict):
    """Print production deployment recommendations"""
    print("🏭 PRODUCTION DEPLOYMENT GUIDE")
    print("-" * 35)
    
    speed_ranking = data['dataset_summary']['speed_ranking']
    fastest_model = min(speed_ranking, key=lambda x: x['speed_rank'])
    
    print("WORKLOAD RECOMMENDATIONS:")
    print(f"• High-Throughput Tasks: {fastest_model['model_name']}")
    print(f"  └─ Target SLA: <20 sec response time")
    print(f"  └─ Expected: {fastest_model['mean']:.0f} tokens/sec")
    
    # Find most comprehensive model (highest token count)
    model_perf = data['dataset_summary']['model_performance']
    most_comprehensive = max(model_perf.items(), key=lambda x: x[1]['total_tokens_mean'])
    
    print(f"• Detailed Analysis Tasks: {most_comprehensive[0]}")
    print(f"  └─ Target SLA: <60 sec response time")
    print(f"  └─ Response Quality: {most_comprehensive[1]['response_length_mean']:.0f} avg chars")
    print()
    
    print("RESOURCE ALLOCATION:")
    print("• RTX 5090 (32GB VRAM)")
    print(f"  └─ Primary: {fastest_model['model_name']} (20GB)")
    print(f"  └─ Secondary: Reserved (12GB)")
    print("• AMD 9950X (16 cores)")
    print("  └─ Inference Support: 4-6 cores")
    print("  └─ Preprocessing: 2-4 cores")
    print("  └─ System/Monitoring: 2 cores")
    print("• 128GB DDR5 RAM")
    print("  └─ Model Loading: 32GB")
    print("  └─ Batch Processing: 64GB")
    print("  └─ System/Cache: 32GB")
    print()

def calculate_cost_efficiency(data: Dict):
    """Calculate cost-efficiency metrics"""
    print("💰 COST-EFFICIENCY ANALYSIS")
    print("-" * 30)
    
    # Estimated hardware cost
    hardware_cost = 8000  # RTX 5090 + AMD 9950X + RAM + System
    
    speed_ranking = data['dataset_summary']['speed_ranking']
    total_throughput = sum(model['mean'] for model in speed_ranking)
    
    cost_per_token_sec = hardware_cost / total_throughput
    
    print(f"Hardware Investment: ${hardware_cost:,}")
    print(f"Combined Throughput: {total_throughput:.1f} tokens/sec")
    print(f"Cost per tokens/sec: ${cost_per_token_sec:.2f}")
    print()
    
    print("CLOUD COMPARISON (Estimated):")
    cloud_cost_per_hour = 15  # Approximate cost for equivalent GPU instance
    hours_per_month = 730
    monthly_cloud_cost = cloud_cost_per_hour * hours_per_month
    
    roi_months = hardware_cost / monthly_cloud_cost
    
    print(f"Equivalent Cloud Cost: ${monthly_cloud_cost:,}/month")
    print(f"Hardware ROI Period: {roi_months:.1f} months")
    print(f"Annual Savings: ${(monthly_cloud_cost * 12) - hardware_cost:,}")
    print()

def main():
    """Main execution function"""
    data = load_latest_analysis()
    
    if not data:
        return
    
    print()
    print_performance_summary(data)
    print_detailed_comparison(data)
    print_optimization_insights(data)
    print_production_recommendations(data)
    calculate_cost_efficiency(data)
    
    print("=" * 60)
    print("📁 GENERATED FILES:")
    print("• comprehensive_analysis.json - Complete raw data")
    print("• raw_performance_data.csv - Spreadsheet-ready data")  
    print("• performance_summary_report.md - Technical summary")
    print("• detailed_benchmark_analysis_report.md - Full analysis")
    print("• visualizations/ - Performance charts and graphs")
    print("=" * 60)

if __name__ == "__main__":
    main()