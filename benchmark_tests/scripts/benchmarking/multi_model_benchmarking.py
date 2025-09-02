#!/usr/bin/env python3
"""
Multi-Model Benchmarking CLI Script

This script provides a command-line interface to the MultiModelBenchmarking
system for comparative analysis across different model endpoints.

Usage:
    python scripts/benchmarking/multi_model_benchmarking.py [options]
    
Features:
- Pattern-based model comparison (not absolute truth validation)
- Systematic calibration across model endpoints
- Statistical validation with multi-sample testing
- Behavioral signature analysis and ranking
"""

import sys
import argparse
from pathlib import Path

# Add core modules to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core"))

try:
    from core.benchmarking_engine import MultiModelBenchmarking, ModelEndpoint
except ImportError as e:
    print(f"Error importing benchmarking engine: {e}")
    print("Please ensure core/benchmarking_engine.py is available")
    sys.exit(1)

def create_parser():
    """Create argument parser for CLI"""
    parser = argparse.ArgumentParser(
        description="Multi-Model Comparative Benchmarking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run quick benchmark with default domains
    python scripts/benchmarking/multi_model_benchmarking.py
    
    # Test specific domains
    python scripts/benchmarking/multi_model_benchmarking.py --domains reasoning creativity
    
    # Extended test with more samples per domain
    python scripts/benchmarking/multi_model_benchmarking.py --tests-per-domain 5
    
    # Add custom model endpoint
    python scripts/benchmarking/multi_model_benchmarking.py --add-model "Custom-7B" "http://localhost:8005/v1/completions" "/app/models/custom-7b.gguf"
        """
    )
    
    parser.add_argument(
        '--domains',
        nargs='*',
        default=['abstract_reasoning', 'liminal_concepts', 'synthetic_knowledge', 'emergent_systems'],
        help='Test domains to benchmark (default: optimized domains)'
    )
    
    parser.add_argument(
        '--tests-per-domain',
        type=int,
        default=2,
        help='Number of tests per domain (default: 2)'
    )
    
    parser.add_argument(
        '--models',
        nargs='*',
        help='Specific model keys to test (default: all configured models)'
    )
    
    parser.add_argument(
        '--add-model',
        nargs=3,
        metavar=('NAME', 'ENDPOINT', 'MODEL_PATH'),
        action='append',
        help='Add custom model endpoint (can be used multiple times)'
    )
    
    parser.add_argument(
        '--output-file',
        help='Custom output file path (default: test_results/multi_model_benchmark_results.json)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick benchmark with minimal domains and tests'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser

def main():
    """Main entry point for multi-model benchmarking CLI"""
    
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure logging level
    import logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    try:
        # Initialize benchmarker
        print("🚀 Initializing Multi-Model Benchmarking System")
        benchmarker = MultiModelBenchmarking()
        
        # Add custom model endpoints if specified
        if args.add_model:
            for model_info in args.add_model:
                name, endpoint, model_path = model_info
                custom_model = ModelEndpoint(
                    name=name,
                    endpoint_url=endpoint,
                    model_path=model_path
                )
                benchmarker.add_model_endpoint(custom_model)
                print(f"✅ Added custom model: {name}")
        
        # Configure test parameters
        if args.quick:
            domains = ['reasoning']  # Single domain for quick test
            tests_per_domain = 1
            print("⚡ Quick benchmark mode enabled")
        else:
            domains = args.domains
            tests_per_domain = args.tests_per_domain
        
        print(f"🎯 Configuration:")
        print(f"   Domains: {domains}")
        print(f"   Tests per domain: {tests_per_domain}")
        
        if args.models:
            print(f"   Models to test: {args.models}")
        else:
            print(f"   Testing all configured models")
        
        # Run comparative benchmark
        print("\n🔬 Starting comparative benchmark...")
        summary = benchmarker.run_comparative_benchmark(
            test_domains=domains,
            tests_per_domain=tests_per_domain,
            models_to_test=args.models
        )
        
        # Print comprehensive report
        benchmarker.print_benchmark_report(summary)
        
        # Success summary
        print(f"\n✅ Benchmark completed successfully!")
        print(f"   Models tested: {summary.total_models}")
        print(f"   Total tests: {summary.total_tests}")
        
        if summary.model_rankings:
            best_model = max(summary.model_rankings, key=summary.model_rankings.get)
            best_score = summary.model_rankings[best_model]
            print(f"   Top performer: {best_model} ({best_score:.1f})")
        
        # Results location
        results_file = args.output_file or "test_results/multi_model_benchmark_results.json"
        print(f"   Results saved: {results_file}")
        
        # Return appropriate exit code based on results
        if summary.total_models > 0:
            avg_score = sum(summary.model_rankings.values()) / len(summary.model_rankings)
            if avg_score >= 70:
                return 0  # Excellent results
            elif avg_score >= 60:
                return 0  # Good results
            else:
                print("\n⚠️  Average performance below 60 - framework may need calibration")
                return 1  # Below threshold
        else:
            print("\n❌ No models were successfully tested")
            return 1
        
    except KeyboardInterrupt:
        print("\n⏸️  Benchmark interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())