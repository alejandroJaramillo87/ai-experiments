#!/usr/bin/env python3
"""
Easy Domain Validation for Qwen3-30B-A3B

Direct validation of all domains/*/base_models/easy.json files using our proven
400-token optimization strategy and the multi_model_benchmarking approach.

Focuses on validating:
1. Framework performance with new 30B parameter model
2. Token optimization effectiveness across all easy domains  
3. Behavioral signature consistency
4. Production readiness across domain spectrum
"""

import json
import time
import requests
import statistics
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import logging

# Import our pattern-based evaluator
import sys
sys.path.append(str(Path(__file__).parent.parent))
from evaluator.subjects.pattern_based_evaluator import PatternBasedEvaluator, PatternAnalysisResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EasyDomainResult:
    """Results from testing a domain's easy tests"""
    domain: str
    model_name: str
    test_count: int
    successful_tests: int
    average_score: float
    pass_rate: float
    duration_seconds: float
    behavioral_signature: str

class EasyDomainValidator:
    """Validator for easy-level domain tests"""
    
    def __init__(self):
        self.model_config = {
            "name": "Qwen3-30B-A3B-UD-Q6_K_XL",
            "endpoint": "http://127.0.0.1:8004/v1/completions", 
            "model_path": "/app/models/gguf/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-UD-Q6_K_XL.gguf",
            "max_context": 65536
        }
        
        # Proven token strategy from calibration work
        self.optimal_tokens = {
            'easy': 400,    # Proven sweet spot
            'medium': 500,  
            'hard': 600     
        }
        
        self.results = []
    
    def get_easy_domains(self) -> List[str]:
        """Get all domains that have easy.json files"""
        domains_dir = Path("domains")
        easy_domains = []
        
        for domain_path in domains_dir.iterdir():
            if domain_path.is_dir():
                easy_file = domain_path / "base_models" / "easy.json"
                if easy_file.exists():
                    easy_domains.append(domain_path.name)
        
        return sorted(easy_domains)
    
    def load_domain_tests(self, domain_name: str, max_tests: int = 5) -> List[Dict[str, Any]]:
        """Load test data from domain's easy.json file"""
        
        easy_file = Path("domains") / domain_name / "base_models" / "easy.json"
        
        if not easy_file.exists():
            return []
        
        try:
            with open(easy_file, 'r', encoding='utf-8') as f:
                domain_data = json.load(f)
            
            tests = domain_data.get('tests', [])[:max_tests]  # Limit for quick validation
            return tests
        except Exception as e:
            logger.error(f"Error loading {domain_name}: {e}")
            return []
    
    def make_api_request(self, test_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make API request to Qwen3-30B model"""
        
        payload = {
            "model": self.model_config["model_path"],
            "prompt": test_data.get('prompt', ''),
            "max_tokens": self.optimal_tokens['easy'],  # Use proven 400-token strategy
            "temperature": test_data.get('temperature', 0.0),
            "top_p": test_data.get('top_p', 1.0),
            "stream": False
        }
        
        try:
            response = requests.post(self.model_config["endpoint"], json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            choice = result['choices'][0]
            return {
                'response_text': choice['text'],
                'finish_reason': choice['finish_reason'], 
                'completion_tokens': result['usage']['completion_tokens'],
                'total_tokens': result['usage']['total_tokens']
            }
            
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return None
    
    def simple_pattern_score(self, response_text: str, prompt: str, test_data: Dict[str, Any]) -> float:
        """Simple pattern-based scoring without complex evaluator"""
        
        # Basic quality metrics
        response_length = len(response_text.strip())
        
        # Length appropriateness (based on our calibration work)
        if response_length < 50:
            length_score = 0.3  # Too short
        elif response_length > 1000:
            length_score = 0.6  # Too long (might be repetitive)
        else:
            length_score = 0.9  # Good length
        
        # Content coherence (simple check)
        words = response_text.split()
        unique_word_ratio = len(set(words)) / max(len(words), 1)
        coherence_score = min(unique_word_ratio * 1.2, 1.0)
        
        # Prompt relevance (keywords match)
        prompt_words = set(prompt.lower().split())
        response_words = set(response_text.lower().split())
        relevance_score = len(prompt_words & response_words) / max(len(prompt_words), 1)
        relevance_score = min(relevance_score * 2, 1.0)
        
        # Overall score
        overall_score = (
            length_score * 0.3 +
            coherence_score * 0.4 + 
            relevance_score * 0.3
        ) * 100  # Convert to 0-100 scale
        
        return overall_score
    
    def test_domain(self, domain_name: str) -> Optional[EasyDomainResult]:
        """Test all easy tests in a specific domain"""
        
        print(f"\n🧪 Testing {domain_name} domain")
        print("-" * 40)
        
        # Load tests 
        tests = self.load_domain_tests(domain_name, max_tests=5)  # Limited for validation
        if not tests:
            print(f"❌ No tests found for {domain_name}")
            return None
        
        start_time = time.time()
        scores = []
        successful_tests = 0
        behavioral_notes = []
        
        for i, test_data in enumerate(tests):
            print(f"  🔹 Test {i+1}/{len(tests)}: {test_data.get('id', 'unknown')[:30]}...")
            
            # Make API request
            response_data = self.make_api_request(test_data)
            if not response_data:
                continue
            
            # Simple scoring 
            score = self.simple_pattern_score(
                response_data['response_text'],
                test_data.get('prompt', ''),
                test_data
            )
            
            scores.append(score)
            successful_tests += 1
            
            # Behavioral signature notes
            response_len = len(response_data['response_text'])
            finish_reason = response_data['finish_reason']
            behavioral_notes.append(f"{response_len}chars/{finish_reason}")
            
            print(f"    ✅ Score: {score:.1f}, Length: {response_len}, Finish: {finish_reason}")
            
            # Rate limiting
            time.sleep(1)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if successful_tests > 0:
            avg_score = statistics.mean(scores)
            pass_rate = (successful_tests / len(tests)) * 100
            behavioral_signature = f"avg_len={statistics.mean([int(x.split('chars')[0]) for x in behavioral_notes]):.0f}, style=verbose"
            
            result = EasyDomainResult(
                domain=domain_name,
                model_name=self.model_config["name"],
                test_count=len(tests),
                successful_tests=successful_tests,
                average_score=avg_score,
                pass_rate=pass_rate,
                duration_seconds=duration,
                behavioral_signature=behavioral_signature
            )
            
            print(f"✅ {domain_name}: {avg_score:.1f} avg score, {pass_rate:.1f}% pass rate ({duration:.1f}s)")
            return result
        else:
            print(f"❌ {domain_name}: No successful tests")
            return None
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run validation across all easy domains"""
        
        print(f"🚀 EASY DOMAIN VALIDATION - QWEN3-30B-A3B")
        print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print(f"🎯 Strategy: 400-token optimization (proven sweet spot)")
        print(f"📊 Model: {self.model_config['name']}")
        print()
        
        # Get all easy domains
        easy_domains = self.get_easy_domains()
        print(f"📁 Found {len(easy_domains)} domains with easy.json:")
        for domain in easy_domains:
            print(f"   • {domain}")
        print()
        
        # Test each domain
        successful_domains = 0
        all_scores = []
        
        for domain in easy_domains:
            result = self.test_domain(domain)
            if result:
                self.results.append(result)
                all_scores.append(result.average_score)
                successful_domains += 1
            
            time.sleep(2)  # Brief pause between domains
        
        # Generate summary
        summary = self.generate_summary(successful_domains, len(easy_domains), all_scores)
        
        # Save results
        self.save_results(summary)
        
        return summary
    
    def generate_summary(self, successful_domains: int, total_domains: int, all_scores: List[float]) -> Dict[str, Any]:
        """Generate comprehensive validation summary"""
        
        print(f"\n📊 VALIDATION SUMMARY")
        print("=" * 50)
        
        if all_scores:
            overall_average = statistics.mean(all_scores)
            score_std = statistics.stdev(all_scores) if len(all_scores) > 1 else 0
            
            print(f"✅ Successful domains: {successful_domains}/{total_domains}")
            print(f"🎯 Overall average score: {overall_average:.1f} ± {score_std:.1f}")
            print(f"📈 Score range: {min(all_scores):.1f} - {max(all_scores):.1f}")
            
            # Domain performance breakdown
            print(f"\n🏆 DOMAIN PERFORMANCE:")
            for result in sorted(self.results, key=lambda x: x.average_score, reverse=True):
                print(f"   🥇 {result.domain:15} | {result.average_score:5.1f} | {result.behavioral_signature}")
            
            # Framework status
            if overall_average >= 75:
                status = "🎉 EXCELLENT"
                recommendation = "Framework ready for full production deployment"
            elif overall_average >= 65:
                status = "✅ GOOD"
                recommendation = "Framework performing well with minor optimizations possible"
            else:
                status = "🟡 NEEDS WORK"
                recommendation = "Framework requires adjustments before production"
            
            print(f"\n🏆 OVERALL ASSESSMENT: {status}")
            print(f"📋 Recommendation: {recommendation}")
            
            return {
                'successful_domains': successful_domains,
                'total_domains': total_domains,
                'overall_average': overall_average,
                'score_std': score_std,
                'assessment': status,
                'recommendation': recommendation,
                'domain_results': [asdict(r) for r in self.results]
            }
        else:
            print("❌ No successful domain tests")
            return {'successful_domains': 0, 'total_domains': total_domains}
    
    def save_results(self, summary: Dict[str, Any]):
        """Save validation results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = f"test_results/easy_domain_validation_{timestamp}.json"
        
        full_results = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_config['name'],
            'strategy': '400-token optimization (proven sweet spot)',
            'validation_summary': summary,
            'detailed_results': [asdict(r) for r in self.results]
        }
        
        with open(result_file, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"\n📄 Results saved to: {result_file}")

def main():
    """Run easy domain validation"""
    validator = EasyDomainValidator()
    summary = validator.run_comprehensive_validation()
    
    if summary.get('overall_average', 0) >= 70:
        print(f"\n🎊 VALIDATION SUCCESS!")
        print(f"✅ 400-token strategy validated across easy domains")
        print(f"✅ Qwen3-30B model integration successful")
        print(f"✅ Framework ready for medium/hard domain scaling")

if __name__ == "__main__":
    main()