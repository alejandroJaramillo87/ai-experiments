#!/usr/bin/env python3
"""
Command-line interface for statistical pattern analysis.

Usage:
    python run_pattern_analysis.py [results_directory] [--output output_file] [--detailed]
    
Examples:
    python run_pattern_analysis.py                           # Analyze test_results/
    python run_pattern_analysis.py test_results/             # Analyze test_results/
    python run_pattern_analysis.py my_results/ --detailed    # Detailed analysis
    python run_pattern_analysis.py test_results/ --output my_analysis.json
"""

import argparse
import sys
import os
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from evaluator.analysis.statistical_pattern_detector import run_pattern_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Run statistical pattern analysis on LLM evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_pattern_analysis.py                           # Analyze test_results/
    python run_pattern_analysis.py test_results/             # Analyze test_results/
    python run_pattern_analysis.py my_results/ --detailed    # Detailed analysis with stats
    python run_pattern_analysis.py test_results/ --output my_analysis.json
        """
    )
    
    parser.add_argument(
        'results_directory', 
        nargs='?', 
        default='test_results/',
        help='Directory containing JSON result files (default: test_results/)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file for detailed analysis results (default: pattern_analysis.json)'
    )
    
    parser.add_argument(
        '--detailed', '-d',
        action='store_true',
        help='Show detailed statistical analysis with p-values and effect sizes'
    )
    
    args = parser.parse_args()
    
    # Check if results directory exists
    if not os.path.exists(args.results_directory):
        print(f"❌ Error: Results directory '{args.results_directory}' not found")
        print("\nTo generate test results, run:")
        print("  make test-easy-reasoning-all")
        print("  # or")
        print("  python benchmark_runner.py --test-definitions domains/reasoning/base_models/easy.json")
        return 1
    
    # Set default output file based on directory name
    if args.output is None:
        dir_name = os.path.basename(args.results_directory.rstrip('/'))
        args.output = f'pattern_analysis_{dir_name}.json'
    
    print(f"📊 Running Statistical Pattern Analysis")
    print(f"======================================")
    print(f"📁 Input directory: {args.results_directory}")
    print(f"📄 Output file: {args.output}")
    print(f"🔍 Analysis mode: {'Detailed' if args.detailed else 'Standard'}")
    print()
    
    try:
        # Run the analysis
        analysis = run_pattern_analysis(args.results_directory, args.output)
        
        # Display results
        print(analysis.pattern_summary)
        
        if args.detailed:
            print(f"\n📈 Statistical Significance:")
            for metric, p_val in analysis.statistical_significance.items():
                sig = "✅ SIGNIFICANT" if p_val < 0.05 else "⚠️ Not significant"
                print(f"  • {metric}: p={p_val:.4f} {sig}")
            
            print(f"\n🎯 Effect Sizes:")
            for metric, d_val in analysis.effect_sizes.items():
                size = "🔥 Large" if d_val > 0.8 else "📊 Medium" if d_val > 0.5 else "📉 Small"
                print(f"  • {metric}: d={d_val:.3f} {size}")
            
            print(f"\n🎯 Classification Accuracy: {analysis.classification_accuracy:.1%}")
            
            print(f"\n📋 Recommendations:")
            for rec in analysis.recommendations:
                print(f"  • {rec}")
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📄 Detailed results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return 1


if __name__ == "__main__":
    exit(main())