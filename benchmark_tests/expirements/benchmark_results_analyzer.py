#!/usr/bin/env python3
"""
Benchmark Results Analysis Script

This script systematically analyzes benchmark test results from multiple model directories,
extracting raw performance metrics and generating comparative analysis reports.

Focus: Raw performance data (execution time, tokens/second, success rates) 
rather than pre-computed evaluation scores.

Created for AI Workstation - RTX 5090 + AMD 9950X Performance Analysis
"""

import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PerformanceMetrics:
    """Raw performance metrics extracted from test results"""
    test_id: str
    test_name: str
    model_name: str
    success: bool
    execution_time: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tokens_per_second: float
    prompt_processing_time: Optional[float] = None
    generation_time: Optional[float] = None
    prompt_tokens_per_second: Optional[float] = None
    generation_tokens_per_second: Optional[float] = None
    timestamp: Optional[str] = None
    error_message: Optional[str] = None
    response_length: int = 0
    
class BenchmarkAnalyzer:
    """Comprehensive benchmark results analyzer"""
    
    def __init__(self, base_dir: str = "/home/alejandro/workspace/ai-workstation/tests/benchmark_tests"):
        self.base_dir = Path(base_dir)
        self.results_dirs = {
            "GPT-OSS-20B": self.base_dir / "test_results_gpt-oss-20b",
            "Qwen3": self.base_dir / "test_results_qwen3"
        }
        self.raw_data: List[PerformanceMetrics] = []
        self.analysis_results: Dict = {}
        
    def discover_result_files(self) -> Dict[str, List[Path]]:
        """Discover all JSON result files in the directories"""
        discovered_files = {}
        
        for model_name, results_dir in self.results_dirs.items():
            if not results_dir.exists():
                print(f"Warning: Directory {results_dir} does not exist")
                continue
                
            # Find individual test result files (not batch files)
            json_files = list(results_dir.glob("*_result.json"))
            batch_files = list(results_dir.glob("batch_results_*.json"))
            
            discovered_files[model_name] = {
                "individual_tests": json_files,
                "batch_results": batch_files
            }
            
            print(f"Found {len(json_files)} individual test files and {len(batch_files)} batch files for {model_name}")
            
        return discovered_files
    
    def extract_performance_metrics(self, result_file: Path, model_name: str) -> Optional[PerformanceMetrics]:
        """Extract raw performance metrics from a single test result file"""
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract basic metrics
            metrics = PerformanceMetrics(
                test_id=data.get('test_id', ''),
                test_name=data.get('test_name', ''),
                model_name=model_name,
                success=data.get('success', False),
                execution_time=data.get('execution_time', 0.0),
                prompt_tokens=data.get('prompt_tokens', 0),
                completion_tokens=data.get('completion_tokens', 0),
                total_tokens=data.get('prompt_tokens', 0) + data.get('completion_tokens', 0),
                tokens_per_second=data.get('tokens_per_second', 0.0),
                timestamp=data.get('timestamp'),
                error_message=data.get('error_message'),
                response_length=len(data.get('response_text', ''))
            )
            
            # Extract detailed timing information if available
            api_response = data.get('api_response', {})
            timings = api_response.get('timings', {})
            
            if timings:
                metrics.prompt_processing_time = timings.get('prompt_ms', 0.0) / 1000.0
                metrics.generation_time = timings.get('predicted_ms', 0.0) / 1000.0
                metrics.prompt_tokens_per_second = timings.get('prompt_per_second', 0.0)
                metrics.generation_tokens_per_second = timings.get('predicted_per_second', 0.0)
            
            return metrics
            
        except Exception as e:
            print(f"Error processing {result_file}: {e}")
            return None
    
    def load_all_results(self):
        """Load and process all result files"""
        print("Loading benchmark results...")
        discovered_files = self.discover_result_files()
        
        for model_name, file_groups in discovered_files.items():
            print(f"\nProcessing {model_name} results...")
            
            # Process individual test files
            for result_file in file_groups["individual_tests"]:
                metrics = self.extract_performance_metrics(result_file, model_name)
                if metrics:
                    self.raw_data.append(metrics)
        
        print(f"\nLoaded {len(self.raw_data)} test results across all models")
    
    def create_dataframe(self) -> pd.DataFrame:
        """Convert raw data to pandas DataFrame for analysis"""
        data_dicts = [asdict(metric) for metric in self.raw_data]
        df = pd.DataFrame(data_dicts)
        
        # Add derived metrics
        df['efficiency_score'] = df['tokens_per_second'] / df['execution_time']
        df['token_utilization'] = df['completion_tokens'] / df['total_tokens']
        df['response_efficiency'] = df['response_length'] / df['execution_time']
        
        return df
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive summary statistics"""
        summary = {}
        
        # Overall statistics
        summary['dataset_overview'] = {
            'total_tests': len(df),
            'successful_tests': df['success'].sum(),
            'success_rate': df['success'].mean() * 100,
            'models_tested': df['model_name'].nunique(),
            'unique_test_types': df['test_id'].nunique()
        }
        
        # Performance metrics by model
        model_stats = df.groupby('model_name').agg({
            'execution_time': ['mean', 'std', 'min', 'max', 'median'],
            'tokens_per_second': ['mean', 'std', 'min', 'max', 'median'],
            'total_tokens': ['mean', 'std', 'min', 'max', 'median'],
            'success': ['sum', 'count', 'mean'],
            'response_length': ['mean', 'std', 'min', 'max', 'median']
        }).round(3)
        
        # Convert multi-level columns to flat structure for JSON serialization
        model_stats.columns = ['_'.join(col).strip() for col in model_stats.columns.values]
        summary['model_performance'] = model_stats.to_dict('index')
        
        # Speed comparison
        speed_comparison = df.groupby('model_name')['tokens_per_second'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        speed_comparison['speed_rank'] = speed_comparison['mean'].rank(ascending=False)
        summary['speed_ranking'] = speed_comparison.to_dict('records')
        
        # Efficiency analysis
        efficiency_stats = df.groupby('model_name').agg({
            'efficiency_score': ['mean', 'std'],
            'token_utilization': ['mean', 'std'],
            'response_efficiency': ['mean', 'std']
        }).round(3)
        
        # Convert multi-level columns to flat structure for JSON serialization
        efficiency_stats.columns = ['_'.join(col).strip() for col in efficiency_stats.columns.values]
        summary['efficiency_metrics'] = efficiency_stats.to_dict('index')
        
        return summary
    
    def identify_patterns_and_anomalies(self, df: pd.DataFrame) -> Dict:
        """Identify performance patterns and anomalies"""
        patterns = {}
        
        # Execution time patterns
        patterns['execution_time_analysis'] = {}
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]['execution_time']
            q1, q3 = model_data.quantile([0.25, 0.75])
            iqr = q3 - q1
            
            # Identify outliers
            outliers = model_data[(model_data < q1 - 1.5 * iqr) | (model_data > q3 + 1.5 * iqr)]
            
            patterns['execution_time_analysis'][model] = {
                'mean': model_data.mean(),
                'std': model_data.std(),
                'outliers_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(model_data)) * 100
            }
        
        # Token throughput patterns
        patterns['throughput_analysis'] = {}
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]['tokens_per_second']
            
            patterns['throughput_analysis'][model] = {
                'mean_throughput': model_data.mean(),
                'consistency': model_data.std() / model_data.mean(),  # Coefficient of variation
                'peak_performance': model_data.max(),
                'min_performance': model_data.min()
            }
        
        # Success rate patterns by test complexity
        # Analyze if certain test types have different success rates
        test_complexity_analysis = df.groupby(['model_name', 'test_id']).agg({
            'success': 'mean',
            'execution_time': 'mean',
            'tokens_per_second': 'mean'
        }).reset_index()
        
        patterns['test_complexity_impact'] = test_complexity_analysis.to_dict('records')
        
        return patterns
    
    def generate_comparative_analysis(self, df: pd.DataFrame) -> Dict:
        """Generate detailed comparative analysis between models"""
        comparison = {}
        
        models = df['model_name'].unique()
        if len(models) < 2:
            print("Warning: Need at least 2 models for comparative analysis")
            return comparison
        
        # Head-to-head comparison
        comparison['head_to_head'] = {}
        
        # Speed comparison
        speed_stats = df.groupby('model_name')['tokens_per_second'].agg(['mean', 'std'])
        fastest_model = speed_stats['mean'].idxmax()
        slowest_model = speed_stats['mean'].idxmin()
        
        speed_difference = speed_stats.loc[fastest_model, 'mean'] - speed_stats.loc[slowest_model, 'mean']
        speed_improvement_pct = (speed_difference / speed_stats.loc[slowest_model, 'mean']) * 100
        
        comparison['speed_analysis'] = {
            'fastest_model': fastest_model,
            'slowest_model': slowest_model,
            'speed_difference_tokens_per_sec': speed_difference,
            'improvement_percentage': speed_improvement_pct,
            'detailed_speeds': speed_stats.to_dict('index')
        }
        
        # Reliability comparison
        reliability_stats = df.groupby('model_name')['success'].agg(['mean', 'count'])
        most_reliable = reliability_stats['mean'].idxmax()
        
        comparison['reliability_analysis'] = {
            'most_reliable_model': most_reliable,
            'success_rates': reliability_stats.to_dict('index')
        }
        
        # Efficiency comparison
        efficiency_by_model = df.groupby('model_name').agg({
            'execution_time': 'mean',
            'total_tokens': 'mean',
            'response_length': 'mean'
        })
        
        # Calculate tokens per minute for easier interpretation
        efficiency_by_model['tokens_per_minute'] = (
            efficiency_by_model['total_tokens'] / efficiency_by_model['execution_time'] * 60
        )
        
        comparison['efficiency_analysis'] = efficiency_by_model.to_dict('index')
        
        return comparison
    
    def create_visualizations(self, df: pd.DataFrame, output_dir: Path):
        """Generate visualization plots"""
        output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Speed comparison box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='model_name', y='tokens_per_second')
        plt.title('Tokens per Second Distribution by Model')
        plt.ylabel('Tokens per Second')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'speed_comparison_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Execution time comparison
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df, x='model_name', y='execution_time')
        plt.title('Execution Time Distribution by Model')
        plt.ylabel('Execution Time (seconds)')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'execution_time_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Success rate comparison
        success_rates = df.groupby('model_name')['success'].mean() * 100
        plt.figure(figsize=(8, 6))
        success_rates.plot(kind='bar')
        plt.title('Success Rate by Model')
        plt.ylabel('Success Rate (%)')
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'success_rate_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Performance correlation heatmap
        numeric_cols = ['execution_time', 'tokens_per_second', 'total_tokens', 'response_length']
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Performance Metrics Correlation Matrix')
        plt.tight_layout()
        plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Speed vs Token Count scatter plot
        plt.figure(figsize=(10, 6))
        for model in df['model_name'].unique():
            model_data = df[df['model_name'] == model]
            plt.scatter(model_data['total_tokens'], model_data['tokens_per_second'], 
                       alpha=0.6, label=model, s=50)
        
        plt.xlabel('Total Tokens')
        plt.ylabel('Tokens per Second')
        plt.title('Throughput vs Token Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'throughput_vs_tokens.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}")
    
    def save_results(self, output_dir: Path):
        """Save all analysis results to files"""
        output_dir.mkdir(exist_ok=True)
        
        # Create DataFrame
        df = self.create_dataframe()
        
        # Generate analyses
        summary_stats = self.generate_summary_statistics(df)
        patterns_anomalies = self.identify_patterns_and_anomalies(df)
        comparative_analysis = self.generate_comparative_analysis(df)
        
        # Combine all results
        complete_analysis = {
            'generation_timestamp': datetime.now().isoformat(),
            'dataset_summary': summary_stats,
            'patterns_and_anomalies': patterns_anomalies,
            'comparative_analysis': comparative_analysis,
            'raw_data_summary': {
                'total_records': len(df),
                'columns': list(df.columns),
                'models_analyzed': list(df['model_name'].unique())
            }
        }
        
        # Save comprehensive analysis
        with open(output_dir / 'comprehensive_analysis.json', 'w') as f:
            json.dump(complete_analysis, f, indent=2, default=str)
        
        # Save raw data CSV
        df.to_csv(output_dir / 'raw_performance_data.csv', index=False)
        
        # Save summary report
        self.generate_summary_report(complete_analysis, output_dir / 'performance_summary_report.md')
        
        # Create visualizations
        self.create_visualizations(df, output_dir / 'visualizations')
        
        print(f"All results saved to {output_dir}")
        return complete_analysis
    
    def generate_summary_report(self, analysis_data: Dict, output_file: Path):
        """Generate a human-readable summary report"""
        report = []
        report.append("# Benchmark Performance Analysis Report\n")
        report.append(f"**Generated:** {analysis_data['generation_timestamp']}\n")
        report.append("**Focus:** Raw performance metrics (execution time, throughput, success rates)\n\n")
        
        # Dataset Overview
        overview = analysis_data['dataset_summary']['dataset_overview']
        report.append("## Dataset Overview\n")
        report.append(f"- **Total Tests:** {overview['total_tests']}")
        report.append(f"- **Successful Tests:** {overview['successful_tests']}")
        report.append(f"- **Success Rate:** {overview['success_rate']:.1f}%")
        report.append(f"- **Models Tested:** {overview['models_tested']}")
        report.append(f"- **Unique Test Types:** {overview['unique_test_types']}\n")
        
        # Speed Rankings
        speed_ranking = analysis_data['dataset_summary']['speed_ranking']
        report.append("## Speed Performance Rankings\n")
        for rank_data in sorted(speed_ranking, key=lambda x: x['speed_rank']):
            report.append(f"**{int(rank_data['speed_rank'])}.** {rank_data['model_name']}: "
                         f"{rank_data['mean']:.1f} ± {rank_data['std']:.1f} tokens/sec "
                         f"({rank_data['count']} tests)")
        report.append("")
        
        # Comparative Analysis
        comp_analysis = analysis_data['comparative_analysis']
        if 'speed_analysis' in comp_analysis:
            speed_analysis = comp_analysis['speed_analysis']
            report.append("## Speed Comparison\n")
            report.append(f"- **Fastest Model:** {speed_analysis['fastest_model']}")
            report.append(f"- **Slowest Model:** {speed_analysis['slowest_model']}")
            report.append(f"- **Speed Difference:** {speed_analysis['speed_difference_tokens_per_sec']:.1f} tokens/sec")
            report.append(f"- **Performance Improvement:** {speed_analysis['improvement_percentage']:.1f}%\n")
        
        # Key Findings
        report.append("## Key Performance Insights\n")
        
        # Analyze throughput consistency
        if 'throughput_analysis' in analysis_data['patterns_and_anomalies']:
            throughput_data = analysis_data['patterns_and_anomalies']['throughput_analysis']
            report.append("### Throughput Consistency\n")
            for model, data in throughput_data.items():
                consistency_rating = "High" if data['consistency'] < 0.2 else "Medium" if data['consistency'] < 0.5 else "Low"
                report.append(f"- **{model}:** {consistency_rating} consistency "
                             f"(CV: {data['consistency']:.3f}, Peak: {data['peak_performance']:.1f} tokens/sec)")
            report.append("")
        
        # Hardware Utilization Insights
        report.append("## Hardware Utilization Analysis\n")
        report.append("*Analysis based on RTX 5090 + AMD 9950X performance characteristics*\n")
        
        # Add model-specific insights
        for model_name in analysis_data['raw_data_summary']['models_analyzed']:
            model_perf = analysis_data['dataset_summary']['model_performance']
            if model_name in [key for key in model_perf.keys() if model_name in key]:
                report.append(f"### {model_name}\n")
                # Extract performance data and provide insights
                report.append("- Performance characteristics and optimization recommendations")
                report.append("- Memory utilization patterns")
                report.append("- Bottleneck identification\n")
        
        # Recommendations
        report.append("## Recommendations\n")
        report.append("1. **Performance Optimization:**")
        report.append("   - Focus on models with highest tokens/second for production workloads")
        report.append("   - Investigate outliers in execution time for optimization opportunities")
        report.append("\n2. **Resource Allocation:**")
        report.append("   - Allocate GPU memory based on model throughput characteristics")
        report.append("   - Consider parallel processing for consistent high-throughput models")
        report.append("\n3. **Monitoring and Scaling:**")
        report.append("   - Implement real-time performance monitoring")
        report.append("   - Set up alerts for performance degradation")
        
        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Summary report saved to {output_file}")
    
    def run_complete_analysis(self, output_dir: Optional[str] = None) -> Dict:
        """Run the complete analysis pipeline"""
        if output_dir is None:
            output_dir = f"/home/alejandro/workspace/ai-workstation/benchmark_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_path = Path(output_dir)
        
        print("=" * 60)
        print("BENCHMARK RESULTS ANALYZER")
        print("=" * 60)
        print(f"Analysis output directory: {output_path}")
        print()
        
        # Step 1: Load all results
        self.load_all_results()
        
        if not self.raw_data:
            print("No data loaded. Check directory paths and file permissions.")
            return {}
        
        # Step 2: Generate analysis and save results
        results = self.save_results(output_path)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Results saved to: {output_path}")
        print(f"Total tests analyzed: {len(self.raw_data)}")
        print(f"Models compared: {len(set(m.model_name for m in self.raw_data))}")
        
        return results


def main():
    """Main execution function"""
    print("Initializing Benchmark Results Analyzer...")
    
    # Initialize analyzer
    analyzer = BenchmarkAnalyzer()
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    if results:
        print("\n✓ Analysis completed successfully!")
        print("\nQuick Summary:")
        overview = results['dataset_summary']['dataset_overview']
        print(f"  - Analyzed {overview['total_tests']} tests")
        print(f"  - Success rate: {overview['success_rate']:.1f}%")
        print(f"  - Models compared: {overview['models_tested']}")
        
        # Show top performer
        speed_ranking = results['dataset_summary']['speed_ranking']
        if speed_ranking:
            fastest = min(speed_ranking, key=lambda x: x['speed_rank'])
            print(f"  - Fastest model: {fastest['model_name']} ({fastest['mean']:.1f} tokens/sec)")
    else:
        print("\n✗ Analysis failed. Check error messages above.")


if __name__ == "__main__":
    main()