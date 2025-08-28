#!/usr/bin/env python3
"""
Independent Benchmark Analysis Script
====================================

Analyzes LLM benchmark test results comprehensively, focusing on raw performance 
data rather than pre-computed evaluation scores. Compares gpt-oss-20b vs qwen3
models across multiple performance dimensions.

Author: Claude Code Analysis
"""

import json
import os
import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import statistics
import csv
from datetime import datetime

@dataclass
class TestResult:
    """Raw test result data structure"""
    test_id: str
    test_name: str
    model_name: str
    success: bool
    execution_time: float
    response_text: str
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float
    response_length_words: int
    response_length_chars: int
    finish_reason: str = ""
    
@dataclass
class PerformanceMetrics:
    """Independent performance metrics calculated from raw data"""
    model_name: str
    total_tests: int
    success_rate: float
    avg_execution_time: float
    avg_tokens_per_second: float
    avg_response_length: int
    response_completeness_score: float
    consistency_score: float
    efficiency_score: float
    
@dataclass
class QualityAssessment:
    """Independent quality assessment based on response content analysis"""
    structural_quality: float
    content_depth: float  
    task_completion: float
    coherence_score: float
    professional_formatting: float

class IndependentBenchmarkAnalyzer:
    """
    Comprehensive analyzer focusing on raw performance data
    """
    
    def __init__(self):
        self.gpt_oss_results: List[TestResult] = []
        self.qwen3_results: List[TestResult] = []
        self.test_categories = defaultdict(list)
        
    def load_results(self, gpt_oss_dir: str, qwen3_dir: str) -> None:
        """Load and parse all result files from both directories"""
        print("🔍 Loading test results...")
        
        # Load GPT-OSS-20B results
        self._load_model_results(gpt_oss_dir, "gpt-oss-20b", self.gpt_oss_results)
        
        # Load Qwen3 results
        self._load_model_results(qwen3_dir, "qwen3", self.qwen3_results)
        
        print(f"✓ Loaded {len(self.gpt_oss_results)} GPT-OSS-20B results")
        print(f"✓ Loaded {len(self.qwen3_results)} Qwen3 results")
        
    def _load_model_results(self, results_dir: str, model_name: str, results_list: List[TestResult]) -> None:
        """Load results for a specific model"""
        batch_file = None
        for file in os.listdir(results_dir):
            if file.startswith("batch_results_") and file.endswith(".json"):
                batch_file = file
                break
                
        if not batch_file:
            print(f"⚠️  No batch results file found in {results_dir}")
            return
            
        batch_path = os.path.join(results_dir, batch_file)
        
        try:
            with open(batch_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for result in data.get('individual_results', []):
                test_result = TestResult(
                    test_id=result.get('test_id', ''),
                    test_name=result.get('test_name', ''),
                    model_name=model_name,
                    success=result.get('success', False),
                    execution_time=result.get('execution_time', 0.0),
                    response_text=result.get('response_text', ''),
                    prompt_tokens=result.get('prompt_tokens', 0),
                    completion_tokens=result.get('completion_tokens', 0),
                    tokens_per_second=result.get('tokens_per_second', 0.0),
                    response_length_words=len(result.get('response_text', '').split()),
                    response_length_chars=len(result.get('response_text', '')),
                    finish_reason=result.get('api_response', {}).get('choices', [{}])[0].get('finish_reason', '')
                )
                results_list.append(test_result)
                
                # Categorize tests
                category = self._extract_category(test_result.test_name)
                self.test_categories[category].append(test_result)
                
        except Exception as e:
            print(f"❌ Error loading {batch_path}: {e}")
    
    def _extract_category(self, test_name: str) -> str:
        """Extract test category from test name"""
        test_name_lower = test_name.lower()
        
        # Category mapping based on test names
        if "synthesis" in test_name_lower or "document" in test_name_lower:
            return "synthesis"
        elif "chain" in test_name_lower or "thought" in test_name_lower:
            return "chain_of_thought"
        elif "multi-hop" in test_name_lower or "inference" in test_name_lower:
            return "multi_hop"
        elif "mathematical" in test_name_lower or "probability" in test_name_lower:
            return "mathematical"
        elif "verification" in test_name_lower or "self-check" in test_name_lower:
            return "verification"
        elif "backward" in test_name_lower or "reverse" in test_name_lower:
            return "backward_reasoning"
        elif "scaffold" in test_name_lower or "structured" in test_name_lower:
            return "scaffolded_reasoning"
        elif "creative" in test_name_lower or "generation" in test_name_lower:
            return "creative"
        elif "analysis" in test_name_lower or "diagnosis" in test_name_lower:
            return "analysis"
        else:
            return "general"
    
    def calculate_performance_metrics(self) -> Tuple[PerformanceMetrics, PerformanceMetrics]:
        """Calculate independent performance metrics for both models"""
        print("📊 Calculating independent performance metrics...")
        
        gpt_metrics = self._calculate_model_metrics(self.gpt_oss_results, "gpt-oss-20b")
        qwen3_metrics = self._calculate_model_metrics(self.qwen3_results, "qwen3")
        
        return gpt_metrics, qwen3_metrics
    
    def _calculate_model_metrics(self, results: List[TestResult], model_name: str) -> PerformanceMetrics:
        """Calculate metrics for a specific model"""
        if not results:
            return PerformanceMetrics(model_name, 0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0)
        
        successful_results = [r for r in results if r.success]
        
        # Basic metrics
        success_rate = len(successful_results) / len(results) * 100
        avg_execution_time = statistics.mean([r.execution_time for r in successful_results])
        avg_tokens_per_second = statistics.mean([r.tokens_per_second for r in successful_results])
        avg_response_length = statistics.mean([r.response_length_words for r in successful_results])
        
        # Advanced metrics
        response_completeness = self._calculate_response_completeness(successful_results)
        consistency_score = self._calculate_consistency_score(successful_results)  
        efficiency_score = self._calculate_efficiency_score(successful_results)
        
        return PerformanceMetrics(
            model_name=model_name,
            total_tests=len(results),
            success_rate=success_rate,
            avg_execution_time=avg_execution_time,
            avg_tokens_per_second=avg_tokens_per_second,
            avg_response_length=avg_response_length,
            response_completeness_score=response_completeness,
            consistency_score=consistency_score,
            efficiency_score=efficiency_score
        )
    
    def _calculate_response_completeness(self, results: List[TestResult]) -> float:
        """Calculate how complete responses appear to be based on content analysis"""
        completeness_scores = []
        
        for result in results:
            score = 0.0
            text = result.response_text.lower()
            
            # Check for completion indicators
            if any(indicator in text for indicator in ["conclusion", "summary", "in summary", "final", "therefore"]):
                score += 30
                
            # Check for structural completeness
            if result.response_length_words >= 100:  # Substantial response
                score += 25
                
            # Check against incompleteness indicators
            if not any(incomplete in text for incomplete in ["...", "continues", "more to", "incomplete"]):
                score += 25
                
            # Check for proper ending (not cut off)
            if result.finish_reason in ["stop", "length"] or result.finish_reason == "":
                score += 20
                
            completeness_scores.append(min(score, 100))
            
        return statistics.mean(completeness_scores) if completeness_scores else 0.0
    
    def _calculate_consistency_score(self, results: List[TestResult]) -> float:
        """Calculate consistency in performance metrics"""
        if len(results) < 2:
            return 100.0
            
        # Calculate coefficient of variation for key metrics
        exec_times = [r.execution_time for r in results]
        token_speeds = [r.tokens_per_second for r in results]
        response_lengths = [r.response_length_words for r in results]
        
        # Lower CV = higher consistency (inverted for scoring)
        cv_exec = statistics.stdev(exec_times) / statistics.mean(exec_times) if statistics.mean(exec_times) > 0 else 0
        cv_speed = statistics.stdev(token_speeds) / statistics.mean(token_speeds) if statistics.mean(token_speeds) > 0 else 0
        cv_length = statistics.stdev(response_lengths) / statistics.mean(response_lengths) if statistics.mean(response_lengths) > 0 else 0
        
        # Convert to consistency score (lower CV = higher consistency)
        consistency = (1 - min(cv_exec + cv_speed + cv_length, 1.0)) * 100
        return max(consistency, 0.0)
    
    def _calculate_efficiency_score(self, results: List[TestResult]) -> float:
        """Calculate efficiency as tokens per second per response quality"""
        if not results:
            return 0.0
        
        efficiency_scores = []
        for result in results:
            # Base efficiency: tokens per second
            base_efficiency = result.tokens_per_second
            
            # Quality factor based on response length and completeness
            quality_factor = min(result.response_length_words / 200, 2.0)  # Cap at 2x
            
            # Efficiency score
            efficiency = base_efficiency * quality_factor
            efficiency_scores.append(efficiency)
            
        # Normalize to 0-100 scale
        max_efficiency = max(efficiency_scores) if efficiency_scores else 1
        normalized_scores = [(score / max_efficiency) * 100 for score in efficiency_scores]
        
        return statistics.mean(normalized_scores)
    
    def assess_response_quality(self) -> Dict[str, QualityAssessment]:
        """Assess response quality independently for both models"""
        print("🔬 Assessing response quality independently...")
        
        quality_assessments = {}
        
        for model_name, results in [("gpt-oss-20b", self.gpt_oss_results), ("qwen3", self.qwen3_results)]:
            successful_results = [r for r in results if r.success]
            if not successful_results:
                continue
                
            structural_scores = []
            depth_scores = []
            completion_scores = []
            coherence_scores = []
            formatting_scores = []
            
            for result in successful_results:
                # Analyze each response
                struct_score = self._analyze_structural_quality(result.response_text)
                depth_score = self._analyze_content_depth(result.response_text)
                completion_score = self._analyze_task_completion(result.response_text, result.test_name)
                coherence_score = self._analyze_coherence(result.response_text)
                format_score = self._analyze_formatting(result.response_text)
                
                structural_scores.append(struct_score)
                depth_scores.append(depth_score)
                completion_scores.append(completion_score)
                coherence_scores.append(coherence_score)
                formatting_scores.append(format_score)
            
            quality_assessments[model_name] = QualityAssessment(
                structural_quality=statistics.mean(structural_scores),
                content_depth=statistics.mean(depth_scores),
                task_completion=statistics.mean(completion_scores),
                coherence_score=statistics.mean(coherence_scores),
                professional_formatting=statistics.mean(formatting_scores)
            )
            
        return quality_assessments
    
    def _analyze_structural_quality(self, text: str) -> float:
        """Analyze structural quality of response"""
        score = 0.0
        
        # Headers and organization
        if "**" in text or "##" in text or "###" in text:
            score += 20
            
        # Lists and enumeration
        if re.search(r'^\d+\.', text, re.MULTILINE) or re.search(r'^[-•*]', text, re.MULTILINE):
            score += 15
            
        # Paragraph structure
        paragraphs = text.split('\n\n')
        if len(paragraphs) >= 3:
            score += 15
            
        # Tables
        if '|' in text and text.count('|') >= 6:
            score += 25
            
        # Clear sections
        if any(section in text.lower() for section in ['introduction', 'conclusion', 'summary', 'overview']):
            score += 25
            
        return min(score, 100)
    
    def _analyze_content_depth(self, text: str) -> float:
        """Analyze depth of content"""
        score = 0.0
        text_lower = text.lower()
        
        # Technical terms and complexity
        complex_terms = ['analysis', 'methodology', 'framework', 'comprehensive', 'systematic']
        score += sum(10 for term in complex_terms if term in text_lower)
        
        # Evidence of reasoning
        reasoning_indicators = ['because', 'therefore', 'consequently', 'thus', 'however', 'furthermore']
        score += sum(5 for indicator in reasoning_indicators if indicator in text_lower)
        
        # Length as depth indicator
        word_count = len(text.split())
        if word_count > 500:
            score += 20
        elif word_count > 200:
            score += 10
            
        return min(score, 100)
    
    def _analyze_task_completion(self, text: str, test_name: str) -> float:
        """Analyze how well the task appears to be completed"""
        score = 50.0  # Base score
        text_lower = text.lower()
        test_lower = test_name.lower()
        
        # Task-specific completion indicators
        if "synthesis" in test_lower:
            if any(indicator in text_lower for indicator in ['compare', 'contrast', 'integrate', 'synthesize']):
                score += 25
        elif "analysis" in test_lower:
            if any(indicator in text_lower for indicator in ['analyze', 'examine', 'investigate', 'assess']):
                score += 25
        elif "generation" in test_lower:
            if len(text.split()) > 200:  # Generated substantial content
                score += 25
                
        # General completion indicators
        if any(ending in text_lower for ending in ['conclusion', 'final', 'summary', 'end']):
            score += 15
            
        # Penalty for obvious incompleteness
        if any(incomplete in text_lower for incomplete in ['...', 'incomplete', 'continues', 'to be continued']):
            score -= 30
            
        return max(min(score, 100), 0)
    
    def _analyze_coherence(self, text: str) -> float:
        """Analyze coherence of response"""
        score = 70.0  # Base coherence score
        
        # Check for repetitive patterns (major coherence issue)
        sentences = text.split('.')
        sentence_counts = Counter(s.strip().lower() for s in sentences if len(s.strip()) > 10)
        max_repetition = max(sentence_counts.values()) if sentence_counts else 1
        
        if max_repetition > 3:
            score -= min(max_repetition * 10, 40)
            
        # Check for logical flow
        transitions = ['however', 'therefore', 'furthermore', 'additionally', 'consequently']
        transition_count = sum(1 for trans in transitions if trans in text.lower())
        score += min(transition_count * 5, 20)
        
        # Penalize meta-reasoning loops
        meta_patterns = ['i think', 'let me', 'i should', 'i need to']
        meta_count = sum(text.lower().count(pattern) for pattern in meta_patterns)
        if meta_count > 5:
            score -= min(meta_count * 5, 25)
            
        return max(min(score, 100), 0)
    
    def _analyze_formatting(self, text: str) -> float:
        """Analyze professional formatting"""
        score = 0.0
        
        # Professional elements
        if '**' in text:  # Bold formatting
            score += 15
        if '##' in text or '###' in text:  # Headers
            score += 15
        if '|' in text and text.count('|') >= 6:  # Tables
            score += 25
        if re.search(r'^\d+\.', text, re.MULTILINE):  # Numbered lists
            score += 15
        if re.search(r'^[-•*]', text, re.MULTILINE):  # Bullet lists
            score += 10
        if text.count('\n\n') >= 3:  # Proper paragraphing
            score += 10
        if any(format_indicator in text for format_indicator in ['```', '`', '---', '===']):
            score += 10
            
        return min(score, 100)
    
    def generate_comparative_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive comparative analysis"""
        print("⚖️  Generating comparative analysis...")
        
        gpt_metrics, qwen3_metrics = self.calculate_performance_metrics()
        quality_assessments = self.assess_response_quality()
        
        # Category analysis
        category_analysis = self._analyze_by_category()
        
        # Performance comparison
        comparison = {
            "execution_summary": {
                "gpt_oss_20b": {
                    "total_tests": gpt_metrics.total_tests,
                    "success_rate": gpt_metrics.success_rate,
                    "avg_execution_time": gpt_metrics.avg_execution_time,
                    "avg_tokens_per_second": gpt_metrics.avg_tokens_per_second,
                    "avg_response_length": gpt_metrics.avg_response_length
                },
                "qwen3": {
                    "total_tests": qwen3_metrics.total_tests,
                    "success_rate": qwen3_metrics.success_rate,
                    "avg_execution_time": qwen3_metrics.avg_execution_time,
                    "avg_tokens_per_second": qwen3_metrics.avg_tokens_per_second,
                    "avg_response_length": qwen3_metrics.avg_response_length
                }
            },
            "performance_comparison": {
                "speed_advantage": {
                    "winner": "gpt-oss-20b" if gpt_metrics.avg_tokens_per_second > qwen3_metrics.avg_tokens_per_second else "qwen3",
                    "difference_percent": abs(gpt_metrics.avg_tokens_per_second - qwen3_metrics.avg_tokens_per_second) / max(gpt_metrics.avg_tokens_per_second, qwen3_metrics.avg_tokens_per_second) * 100
                },
                "response_length": {
                    "longer_responses": "gpt-oss-20b" if gpt_metrics.avg_response_length > qwen3_metrics.avg_response_length else "qwen3",
                    "difference_percent": abs(gpt_metrics.avg_response_length - qwen3_metrics.avg_response_length) / max(gpt_metrics.avg_response_length, qwen3_metrics.avg_response_length) * 100
                },
                "consistency": {
                    "more_consistent": "gpt-oss-20b" if gpt_metrics.consistency_score > qwen3_metrics.consistency_score else "qwen3",
                    "gpt_consistency": gpt_metrics.consistency_score,
                    "qwen3_consistency": qwen3_metrics.consistency_score
                }
            },
            "quality_comparison": quality_assessments,
            "category_analysis": category_analysis,
            "patterns_and_anomalies": self._identify_patterns_and_anomalies()
        }
        
        return comparison
    
    def _analyze_by_category(self) -> Dict[str, Any]:
        """Analyze performance by test category"""
        category_analysis = {}
        
        for category, tests in self.test_categories.items():
            gpt_tests = [t for t in tests if t.model_name == "gpt-oss-20b"]
            qwen3_tests = [t for t in tests if t.model_name == "qwen3"]
            
            if not gpt_tests or not qwen3_tests:
                continue
                
            category_analysis[category] = {
                "test_count": len(gpt_tests),
                "gpt_oss_20b": {
                    "avg_tokens_per_second": statistics.mean([t.tokens_per_second for t in gpt_tests if t.success]),
                    "avg_response_length": statistics.mean([t.response_length_words for t in gpt_tests if t.success]),
                    "success_rate": len([t for t in gpt_tests if t.success]) / len(gpt_tests) * 100
                },
                "qwen3": {
                    "avg_tokens_per_second": statistics.mean([t.tokens_per_second for t in qwen3_tests if t.success]),
                    "avg_response_length": statistics.mean([t.response_length_words for t in qwen3_tests if t.success]),
                    "success_rate": len([t for t in qwen3_tests if t.success]) / len(qwen3_tests) * 100
                }
            }
            
        return category_analysis
    
    def _identify_patterns_and_anomalies(self) -> Dict[str, Any]:
        """Identify interesting patterns and anomalies"""
        patterns = {
            "performance_patterns": [],
            "anomalies": [],
            "trends": []
        }
        
        # Speed consistency analysis
        gpt_speeds = [r.tokens_per_second for r in self.gpt_oss_results if r.success]
        qwen3_speeds = [r.tokens_per_second for r in self.qwen3_results if r.success]
        
        if gpt_speeds and qwen3_speeds:
            gpt_cv = statistics.stdev(gpt_speeds) / statistics.mean(gpt_speeds)
            qwen3_cv = statistics.stdev(qwen3_speeds) / statistics.mean(qwen3_speeds)
            
            patterns["performance_patterns"].append({
                "pattern": "speed_consistency",
                "finding": f"GPT-OSS-20B shows {'higher' if gpt_cv < qwen3_cv else 'lower'} speed consistency",
                "gpt_coefficient_variation": gpt_cv,
                "qwen3_coefficient_variation": qwen3_cv
            })
        
        # Response length patterns
        gpt_lengths = [r.response_length_words for r in self.gpt_oss_results if r.success]
        qwen3_lengths = [r.response_length_words for r in self.qwen3_results if r.success]
        
        if gpt_lengths and qwen3_lengths:
            patterns["performance_patterns"].append({
                "pattern": "response_verbosity",
                "finding": f"{'Qwen3' if statistics.mean(qwen3_lengths) > statistics.mean(gpt_lengths) else 'GPT-OSS-20B'} produces more verbose responses",
                "gpt_avg_length": statistics.mean(gpt_lengths),
                "qwen3_avg_length": statistics.mean(qwen3_lengths)
            })
        
        # Identify potential anomalies
        for model_name, results in [("gpt-oss-20b", self.gpt_oss_results), ("qwen3", self.qwen3_results)]:
            speeds = [r.tokens_per_second for r in results if r.success]
            if speeds:
                mean_speed = statistics.mean(speeds)
                stdev_speed = statistics.stdev(speeds) if len(speeds) > 1 else 0
                
                # Find outliers (more than 2 standard deviations from mean)
                outliers = [r for r in results if r.success and abs(r.tokens_per_second - mean_speed) > 2 * stdev_speed]
                if outliers:
                    patterns["anomalies"].append({
                        "model": model_name,
                        "type": "speed_outliers",
                        "count": len(outliers),
                        "tests": [{"test_id": r.test_id, "tokens_per_second": r.tokens_per_second} for r in outliers[:3]]
                    })
        
        return patterns
    
    def save_results(self, analysis_results: Dict[str, Any]) -> None:
        """Save analysis results to files"""
        print("💾 Saving analysis results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert dataclasses to dictionaries for JSON serialization
        serializable_results = self._make_json_serializable(analysis_results)
        
        # Save comprehensive analysis
        with open(f"processed_benchmark_comparison_data.json", 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save CSV summary for spreadsheet analysis
        self._save_csv_summary()
        
        print("✓ Results saved to processed_benchmark_comparison_data.json")
        print("✓ CSV summary saved to raw_performance_comparison.csv")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert dataclasses and other non-serializable objects to dictionaries"""
        if hasattr(obj, '__dataclass_fields__'):
            return {field: self._make_json_serializable(getattr(obj, field)) for field in obj.__dataclass_fields__}
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _save_csv_summary(self) -> None:
        """Save CSV summary for spreadsheet analysis"""
        with open("raw_performance_comparison.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Model", "Test_ID", "Test_Name", "Success", "Execution_Time", 
                "Tokens_Per_Second", "Response_Length_Words", "Prompt_Tokens", 
                "Completion_Tokens", "Category"
            ])
            
            for result in self.gpt_oss_results + self.qwen3_results:
                category = self._extract_category(result.test_name)
                writer.writerow([
                    result.model_name, result.test_id, result.test_name,
                    result.success, result.execution_time, result.tokens_per_second,
                    result.response_length_words, result.prompt_tokens, 
                    result.completion_tokens, category
                ])

def main():
    """Main analysis execution"""
    analyzer = IndependentBenchmarkAnalyzer()
    
    # Paths to result directories
    gpt_oss_dir = "/home/alejandro/workspace/ai-workstation/tests/benchmark_tests/test_results_gpt-oss-20b"
    qwen3_dir = "/home/alejandro/workspace/ai-workstation/tests/benchmark_tests/test_results_qwen3"
    
    try:
        # Load and analyze results
        analyzer.load_results(gpt_oss_dir, qwen3_dir)
        
        # Generate comprehensive analysis
        analysis_results = analyzer.generate_comparative_analysis()
        
        # Save results
        analyzer.save_results(analysis_results)
        
        print("\n" + "="*60)
        print("🎯 INDEPENDENT BENCHMARK ANALYSIS COMPLETE")
        print("="*60)
        
        # Print key findings
        exec_summary = analysis_results["execution_summary"]
        perf_comparison = analysis_results["performance_comparison"]
        
        print(f"\n📊 EXECUTIVE SUMMARY:")
        print(f"GPT-OSS-20B: {exec_summary['gpt_oss_20b']['avg_tokens_per_second']:.1f} tokens/sec")
        print(f"Qwen3:       {exec_summary['qwen3']['avg_tokens_per_second']:.1f} tokens/sec")
        print(f"Speed winner: {perf_comparison['speed_advantage']['winner']} (+{perf_comparison['speed_advantage']['difference_percent']:.1f}%)")
        print(f"Verbosity winner: {perf_comparison['response_length']['longer_responses']} (+{perf_comparison['response_length']['difference_percent']:.1f}%)")
        
        return analysis_results
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

if __name__ == "__main__":
    main()