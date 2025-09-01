#!/usr/bin/env python3
"""
Enhanced Cognitive Validation System

Comprehensive cognitive pattern detection using sophisticated evaluator framework.
Replaces basic scoring with advanced pattern analysis and creates detailed
cognitive profiles showing model strengths/weaknesses across:
- Reasoning abilities
- Memory and knowledge recall  
- Creative thinking
- Social and cultural competency
- Cross-domain integration

Results are stored with complete response data for inspection and analysis.
"""

import json
import time
import requests
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import statistics

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent / "core"))
sys.path.append(str(Path(__file__).parent.parent))

from core.results_manager import TestResultsManager, CognitiveProfile
from core.cognitive_evaluation_pipeline import CognitiveEvaluationPipeline, CognitiveEvaluationResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedCognitiveValidator:
    """Enhanced cognitive validation using sophisticated evaluator framework"""
    
    def __init__(self):
        # Model configuration
        self.model_config = {
            "name": "Qwen3-30B-A3B-UD-Q6_K_XL",
            "endpoint": "http://127.0.0.1:8004/v1/completions",
            "model_path": "/app/models/gguf/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-UD-Q6_K_XL.gguf",
            "max_context": 65536
        }
        
        # Proven token optimization strategy
        self.optimal_tokens = {
            'easy': 400,    # Validated sweet spot
            'medium': 500,
            'hard': 600
        }
        
        # Initialize sophisticated evaluation components
        self.results_manager = TestResultsManager()
        self.evaluation_pipeline = CognitiveEvaluationPipeline()
        
        # Current run directory
        self.current_run_dir = None
    
    def get_easy_domains(self) -> List[str]:
        """Get all domains with easy.json files"""
        domains_dir = Path("domains")
        easy_domains = []
        
        for domain_path in domains_dir.iterdir():
            if domain_path.is_dir():
                easy_file = domain_path / "base_models" / "easy.json"
                if easy_file.exists():
                    easy_domains.append(domain_path.name)
        
        return sorted(easy_domains)
    
    def load_domain_tests(self, domain_name: str, max_tests: int = 10) -> List[Dict[str, Any]]:
        """Load tests from domain's easy.json file"""
        
        easy_file = Path("domains") / domain_name / "base_models" / "easy.json"
        
        if not easy_file.exists():
            logger.warning(f"No easy.json found for domain: {domain_name}")
            return []
        
        try:
            with open(easy_file, 'r', encoding='utf-8') as f:
                domain_data = json.load(f)
            
            tests = domain_data.get('tests', [])[:max_tests]
            logger.info(f"Loaded {len(tests)} tests from {domain_name}")
            return tests
            
        except Exception as e:
            logger.error(f"Error loading {domain_name}: {e}")
            return []
    
    def make_api_request(self, test_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make API request to model with proven token strategy"""
        
        payload = {
            "model": self.model_config["model_path"],
            "prompt": test_data.get('prompt', ''),
            "max_tokens": self.optimal_tokens['easy'],  # Proven 400-token strategy
            "temperature": test_data.get('temperature', 0.0),
            "top_p": test_data.get('top_p', 1.0),
            "stream": False
        }
        
        try:
            start_time = time.time()
            response = requests.post(self.model_config["endpoint"], json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            end_time = time.time()
            
            choice = result['choices'][0]
            return {
                'response_text': choice['text'],
                'finish_reason': choice['finish_reason'],
                'completion_tokens': result['usage']['completion_tokens'],
                'total_tokens': result['usage']['total_tokens'],
                'response_time_seconds': end_time - start_time
            }
            
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def test_domain(self, domain_name: str) -> Dict[str, Any]:
        """Test domain using sophisticated cognitive evaluation"""
        
        print(f"\n🧠 COGNITIVE ANALYSIS: {domain_name.title()}")
        print("-" * 50)
        
        # Load tests
        tests = self.load_domain_tests(domain_name, max_tests=8)  # Focused validation set
        if not tests:
            logger.warning(f"No tests found for {domain_name}")
            return {'domain': domain_name, 'success': False, 'error': 'No tests loaded'}
        
        start_time = time.time()
        evaluation_results = []
        successful_tests = 0
        
        for i, test_data in enumerate(tests):
            test_id = test_data.get('id', f'{domain_name}_test_{i+1}')
            print(f"  🔹 Analyzing {i+1}/{len(tests)}: {test_id[:40]}...")
            
            # Make API request
            response_data = self.make_api_request(test_data)
            if not response_data:
                logger.warning(f"Failed to get response for {test_id}")
                continue
            
            # Enhanced cognitive evaluation
            try:
                cognitive_result = self.evaluation_pipeline.evaluate_response(
                    test_id=test_id,
                    prompt=test_data.get('prompt', ''),
                    response_text=response_data['response_text'],
                    test_metadata={'domain': domain_name, **test_data}
                )
                
                # Save detailed response with cognitive analysis
                self.results_manager.save_test_response(
                    run_dir=self.current_run_dir,
                    test_id=test_id,
                    prompt=test_data.get('prompt', ''),
                    response_text=response_data['response_text'],
                    evaluation_results={
                        'cognitive_evaluation': cognitive_result.__dict__,
                        'api_response': response_data
                    },
                    test_metadata={'domain': domain_name, **test_data}
                )
                
                evaluation_results.append(cognitive_result)
                successful_tests += 1
                
                # Display cognitive insights
                print(f"    ✅ Score: {cognitive_result.overall_score:.1f} | "
                      f"Confidence: {cognitive_result.confidence_score:.2f} | "
                      f"Domain: {cognitive_result.cognitive_domain}")
                
                # Show top cognitive ability
                if cognitive_result.cognitive_subscores:
                    top_ability = max(cognitive_result.cognitive_subscores.items(), key=lambda x: x[1])
                    print(f"    🎯 Strongest: {top_ability[0].replace('_', ' ').title()} ({top_ability[1]:.1f})")
                
            except Exception as e:
                logger.error(f"Cognitive evaluation failed for {test_id}: {e}")
                continue
            
            # Rate limiting
            time.sleep(1.5)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate domain statistics
        if successful_tests > 0:
            overall_scores = [r.overall_score for r in evaluation_results]
            confidence_scores = [r.confidence_score for r in evaluation_results]
            
            # Aggregate cognitive subscores by ability
            all_subscores = {}
            for result in evaluation_results:
                for ability, score in result.cognitive_subscores.items():
                    if ability not in all_subscores:
                        all_subscores[ability] = []
                    all_subscores[ability].append(score)
            
            # Calculate aggregate statistics
            avg_subscores = {ability: statistics.mean(scores) 
                           for ability, scores in all_subscores.items()}
            
            domain_summary = {
                'domain': domain_name,
                'success': True,
                'test_count': len(tests),
                'successful_tests': successful_tests,
                'overall_score': statistics.mean(overall_scores),
                'score_std': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
                'confidence': statistics.mean(confidence_scores),
                'cognitive_subscores': avg_subscores,
                'duration_seconds': duration,
                'cognitive_insights': self._generate_domain_insights(evaluation_results)
            }
            
            print(f"✅ {domain_name.upper()}: {domain_summary['overall_score']:.1f} ± {domain_summary['score_std']:.1f} "
                  f"(confidence: {domain_summary['confidence']:.2f})")
            
            return domain_summary
        else:
            logger.error(f"No successful evaluations for {domain_name}")
            return {'domain': domain_name, 'success': False, 'error': 'No successful evaluations'}
    
    def _generate_domain_insights(self, evaluation_results: List[CognitiveEvaluationResult]) -> Dict[str, Any]:
        """Generate cognitive insights for domain"""
        
        insights = {
            'cognitive_patterns': [],
            'strengths': [],
            'concerns': [],
            'behavioral_consistency': 0.0
        }
        
        if not evaluation_results:
            return insights
        
        # Analyze cognitive patterns across tests
        pattern_strengths = []
        consistency_scores = []
        
        for result in evaluation_results:
            if result.behavioral_patterns:
                pattern_strengths.append(result.pattern_strength)
                consistency_scores.append(result.consistency_measure)
        
        # Calculate behavioral consistency
        if consistency_scores:
            insights['behavioral_consistency'] = statistics.mean(consistency_scores)
        
        # Identify cognitive strengths
        all_subscores = {}
        for result in evaluation_results:
            for ability, score in result.cognitive_subscores.items():
                if ability not in all_subscores:
                    all_subscores[ability] = []
                all_subscores[ability].append(score)
        
        # Find strong and weak cognitive abilities
        for ability, scores in all_subscores.items():
            avg_score = statistics.mean(scores)
            if avg_score >= 75:
                insights['strengths'].append(f"{ability.replace('_', ' ').title()}: {avg_score:.1f}")
            elif avg_score <= 50:
                insights['concerns'].append(f"{ability.replace('_', ' ').title()}: {avg_score:.1f}")
        
        return insights
    
    def run_comprehensive_validation(self) -> CognitiveProfile:
        """Run comprehensive cognitive validation across all domains"""
        
        print(f"🧠 ENHANCED COGNITIVE VALIDATION")
        print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print(f"🎯 Strategy: Sophisticated evaluator integration with proven 400-token optimization")
        print(f"🤖 Model: {self.model_config['name']}")
        print(f"🔍 Analysis: Reasoning | Memory | Creativity | Social | Integration")
        print()
        
        # Create unique run directory
        self.current_run_dir = self.results_manager.create_run_directory(
            model_name=self.model_config['name'],
            model_path=self.model_config['model_path'],
            test_configuration={
                'strategy': 'Enhanced Cognitive Validation',
                'token_limit': 400,
                'evaluation_pipeline': 'Sophisticated Pattern Analysis',
                'domains': 'All Easy Domains'
            }
        )
        
        print(f"📁 Results Directory: {Path(self.current_run_dir).name}")
        print()
        
        # Get all easy domains
        easy_domains = self.get_easy_domains()
        print(f"📋 Testing {len(easy_domains)} cognitive domains:")
        for domain in easy_domains:
            print(f"   • {domain}")
        print()
        
        # Test each domain
        domain_results = []
        successful_domains = 0
        
        for domain in easy_domains:
            result = self.test_domain(domain)
            if result.get('success', False):
                domain_results.append(result)
                successful_domains += 1
            
            # Brief pause between domains
            time.sleep(2)
        
        # Generate comprehensive cognitive profile
        print(f"\n🧠 GENERATING COGNITIVE PROFILE...")
        cognitive_profile = self.results_manager.analyze_cognitive_patterns(self.current_run_dir)
        
        # Display results
        self._display_validation_summary(domain_results, cognitive_profile)
        
        # Save comprehensive results
        self._save_validation_results(domain_results, cognitive_profile)
        
        return cognitive_profile
    
    def _display_validation_summary(self, domain_results: List[Dict], cognitive_profile: CognitiveProfile):
        """Display comprehensive validation summary"""
        
        print(f"\n🏆 COGNITIVE VALIDATION RESULTS")
        print("=" * 60)
        
        if domain_results:
            overall_scores = [r['overall_score'] for r in domain_results if 'overall_score' in r]
            confidence_scores = [r['confidence'] for r in domain_results if 'confidence' in r]
            
            print(f"✅ Successful domains: {len(domain_results)}/{len(self.get_easy_domains())}")
            print(f"🎯 Overall cognitive score: {statistics.mean(overall_scores):.1f} ± {statistics.stdev(overall_scores):.1f}")
            print(f"🔍 Average confidence: {statistics.mean(confidence_scores):.2f}")
            print()
            
            # Cognitive domain breakdown
            print(f"🧠 COGNITIVE DOMAIN PERFORMANCE:")
            sorted_domains = sorted(domain_results, key=lambda x: x.get('overall_score', 0), reverse=True)
            for result in sorted_domains:
                domain_name = result['domain'].upper()
                score = result.get('overall_score', 0)
                confidence = result.get('confidence', 0)
                print(f"   🏅 {domain_name:15} | {score:5.1f} | Confidence: {confidence:.2f}")
            print()
        
        # Display cognitive profile
        print(cognitive_profile.model_name)
        print(f"\n📊 COGNITIVE ABILITY SCORES:")
        print(f"   Reasoning:    {cognitive_profile.reasoning_score:5.1f}/100")
        print(f"   Memory:       {cognitive_profile.memory_score:5.1f}/100")
        print(f"   Creativity:   {cognitive_profile.creativity_score:5.1f}/100")
        print(f"   Social:       {cognitive_profile.social_score:5.1f}/100")
        print(f"   Integration:  {cognitive_profile.integration_score:5.1f}/100")
        print()
        
        if cognitive_profile.strengths:
            print(f"🎯 COGNITIVE STRENGTHS:")
            for strength in cognitive_profile.strengths:
                print(f"   ✅ {strength}")
            print()
        
        if cognitive_profile.weaknesses:
            print(f"⚠️  COGNITIVE WEAKNESSES:")
            for weakness in cognitive_profile.weaknesses:
                print(f"   ❌ {weakness}")
            print()
        
        if cognitive_profile.blind_spots:
            print(f"🚨 CRITICAL BLIND SPOTS:")
            for blind_spot in cognitive_profile.blind_spots:
                print(f"   🔍 {blind_spot}")
            print()
        
        # Framework assessment
        overall_avg = statistics.mean([
            cognitive_profile.reasoning_score,
            cognitive_profile.memory_score,
            cognitive_profile.creativity_score,
            cognitive_profile.social_score,
            cognitive_profile.integration_score
        ])
        
        if overall_avg >= 75:
            status = "🎉 EXCELLENT COGNITIVE PERFORMANCE"
            recommendation = "Model demonstrates strong cognitive abilities across domains"
        elif overall_avg >= 65:
            status = "✅ GOOD COGNITIVE PERFORMANCE"
            recommendation = "Model shows solid cognitive abilities with optimization opportunities"
        else:
            status = "🟡 COGNITIVE PERFORMANCE NEEDS IMPROVEMENT"
            recommendation = "Model requires cognitive enhancement before production deployment"
        
        print(f"🏆 ASSESSMENT: {status}")
        print(f"📋 Recommendation: {recommendation}")
    
    def _save_validation_results(self, domain_results: List[Dict], cognitive_profile: CognitiveProfile):
        """Save comprehensive validation results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = Path(self.current_run_dir) / f"validation_summary_{timestamp}.json"
        
        validation_summary = {
            'timestamp': datetime.now().isoformat(),
            'model_configuration': self.model_config,
            'validation_strategy': 'Enhanced Cognitive Analysis',
            'domain_results': domain_results,
            'cognitive_profile': cognitive_profile.__dict__,
            'framework_status': 'Production Enhanced Cognitive Evaluation'
        }
        
        with open(summary_file, 'w') as f:
            json.dump(validation_summary, f, indent=2, default=str)
        
        print(f"\n📄 Comprehensive results saved to: {summary_file}")
        print(f"📁 Full analysis available in: {self.current_run_dir}")

def main():
    """Run enhanced cognitive validation"""
    
    validator = EnhancedCognitiveValidator()
    cognitive_profile = validator.run_comprehensive_validation()
    
    # Display cognitive summary report
    if cognitive_profile.sample_size > 0:
        report = validator.results_manager.get_cognitive_summary_report(validator.current_run_dir)
        print(report)
        
        print(f"\n🎊 ENHANCED COGNITIVE VALIDATION COMPLETE!")
        print(f"✅ Sophisticated evaluator integration successful")
        print(f"✅ Cognitive pattern detection active")
        print(f"✅ Comprehensive analysis ready for inspection")

if __name__ == "__main__":
    main()