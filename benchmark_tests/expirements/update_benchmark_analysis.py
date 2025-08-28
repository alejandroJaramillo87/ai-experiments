#!/usr/bin/env python3
"""
Update Benchmark Analysis Script

Automatically updates benchmark analysis when new test results are available.
Maintains historical comparison and tracks performance trends over time.
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from benchmark_results_analyzer import BenchmarkAnalyzer

def check_for_new_results(base_dir: Path) -> bool:
    """Check if new test results are available since last analysis"""
    results_dirs = {
        "GPT-OSS-20B": base_dir / "tests/benchmark_tests/test_results_gpt-oss-20b",
        "Qwen3": base_dir / "tests/benchmark_tests/test_results_qwen3"
    }
    
    # Find most recent analysis
    analysis_dirs = list(base_dir.glob("benchmark_analysis_*"))
    if not analysis_dirs:
        print("No previous analysis found - will run complete analysis")
        return True
    
    latest_analysis = max(analysis_dirs, key=lambda p: p.name)
    latest_analysis_time = latest_analysis.stat().st_mtime
    
    # Check if any result files are newer than the last analysis
    for model_name, results_dir in results_dirs.items():
        if results_dir.exists():
            for result_file in results_dir.glob("*_result.json"):
                if result_file.stat().st_mtime > latest_analysis_time:
                    print(f"New results detected: {result_file.name}")
                    return True
    
    return False

def create_analysis_archive():
    """Archive older analysis directories to keep workspace clean"""
    base_dir = Path("/home/alejandro/workspace/ai-workstation")
    analysis_dirs = list(base_dir.glob("benchmark_analysis_*"))
    
    if len(analysis_dirs) <= 3:  # Keep 3 most recent
        return
    
    # Create archive directory
    archive_dir = base_dir / "benchmark_analysis_archive"
    archive_dir.mkdir(exist_ok=True)
    
    # Sort by creation time and move older ones
    analysis_dirs.sort(key=lambda p: p.stat().st_mtime)
    for old_dir in analysis_dirs[:-3]:
        archive_path = archive_dir / old_dir.name
        if not archive_path.exists():
            shutil.move(str(old_dir), str(archive_path))
            print(f"Archived old analysis: {old_dir.name}")

def create_trend_analysis(base_dir: Path):
    """Create trend analysis comparing multiple benchmark runs"""
    analysis_dirs = sorted(base_dir.glob("benchmark_analysis_*"), key=lambda p: p.name)
    
    if len(analysis_dirs) < 2:
        print("Need at least 2 analysis runs for trend analysis")
        return
    
    trend_data = []
    
    for analysis_dir in analysis_dirs[-5:]:  # Last 5 runs
        analysis_file = analysis_dir / "comprehensive_analysis.json"
        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                data = json.load(f)
            
            # Extract key metrics
            speed_ranking = data['dataset_summary']['speed_ranking']
            timestamp = data['generation_timestamp']
            
            trend_entry = {
                'timestamp': timestamp,
                'analysis_dir': analysis_dir.name,
                'model_speeds': {model['model_name']: model['mean'] for model in speed_ranking}
            }
            trend_data.append(trend_entry)
    
    # Save trend analysis
    latest_dir = analysis_dirs[-1]
    trend_file = latest_dir / "performance_trends.json"
    
    with open(trend_file, 'w') as f:
        json.dump(trend_data, f, indent=2)
    
    print(f"Trend analysis saved to {trend_file}")
    
    # Create trend summary
    if len(trend_data) >= 2:
        create_trend_summary(trend_data, latest_dir)

def create_trend_summary(trend_data: list, output_dir: Path):
    """Create human-readable trend summary"""
    summary = []
    summary.append("# Performance Trend Analysis\n")
    summary.append(f"**Analysis Period:** {trend_data[0]['timestamp']} to {trend_data[-1]['timestamp']}\n")
    summary.append(f"**Benchmark Runs:** {len(trend_data)}\n\n")
    
    # Calculate trends for each model
    summary.append("## Speed Trends\n")
    
    # Get all models across all runs
    all_models = set()
    for run in trend_data:
        all_models.update(run['model_speeds'].keys())
    
    for model in sorted(all_models):
        speeds = []
        for run in trend_data:
            if model in run['model_speeds']:
                speeds.append(run['model_speeds'][model])
        
        if len(speeds) >= 2:
            speed_change = speeds[-1] - speeds[0]
            speed_change_pct = (speed_change / speeds[0]) * 100
            trend_direction = "📈" if speed_change > 0 else "📉" if speed_change < 0 else "➡️"
            
            summary.append(f"### {model}\n")
            summary.append(f"- **Initial Speed:** {speeds[0]:.1f} tokens/sec")
            summary.append(f"- **Current Speed:** {speeds[-1]:.1f} tokens/sec")
            summary.append(f"- **Change:** {trend_direction} {speed_change:+.1f} tokens/sec ({speed_change_pct:+.1f}%)")
            summary.append("")
    
    # Performance stability analysis
    summary.append("## Performance Stability\n")
    for model in sorted(all_models):
        speeds = [run['model_speeds'].get(model) for run in trend_data if model in run['model_speeds']]
        if len(speeds) >= 3:
            avg_speed = sum(speeds) / len(speeds)
            std_dev = (sum((s - avg_speed) ** 2 for s in speeds) / len(speeds)) ** 0.5
            cv = std_dev / avg_speed
            
            stability = "High" if cv < 0.05 else "Medium" if cv < 0.1 else "Low"
            summary.append(f"- **{model}:** {stability} stability (CV: {cv:.3f})")
    
    summary.append("\n")
    
    # Save trend summary
    trend_summary_file = output_dir / "performance_trends_summary.md"
    with open(trend_summary_file, 'w') as f:
        f.write('\n'.join(summary))
    
    print(f"Trend summary saved to {trend_summary_file}")

def main():
    """Main execution function"""
    base_dir = Path("/home/alejandro/workspace/ai-workstation")
    
    print("🔍 Checking for new benchmark results...")
    
    # Check if update is needed
    if not check_for_new_results(base_dir):
        print("✅ No new results detected. Analysis is up to date.")
        
        # Still create trend analysis if multiple runs exist
        print("📊 Checking for trend analysis opportunities...")
        create_trend_analysis(base_dir)
        return
    
    print("🚀 New results detected! Running updated analysis...")
    
    # Run new analysis
    analyzer = BenchmarkAnalyzer()
    results = analyzer.run_complete_analysis()
    
    if results:
        print("\n✅ Analysis update completed successfully!")
        
        # Create trend analysis
        print("📊 Generating trend analysis...")
        create_trend_analysis(base_dir)
        
        # Clean up old analyses
        print("🧹 Archiving old analysis files...")
        create_analysis_archive()
        
        # Show quick summary
        print("\n" + "="*50)
        print("📋 QUICK RESULTS SUMMARY")
        print("="*50)
        
        overview = results['dataset_summary']['dataset_overview']
        print(f"Tests Analyzed: {overview['total_tests']}")
        print(f"Success Rate: {overview['success_rate']:.1f}%")
        
        speed_ranking = results['dataset_summary']['speed_ranking']
        if speed_ranking:
            fastest = min(speed_ranking, key=lambda x: x['speed_rank'])
            print(f"Fastest Model: {fastest['model_name']} ({fastest['mean']:.1f} tokens/sec)")
        
        print(f"\n📁 Results saved to latest analysis directory")
        print("💡 Run 'python quick_benchmark_summary.py' for detailed insights")
        
    else:
        print("❌ Analysis update failed. Check error messages above.")

if __name__ == "__main__":
    main()