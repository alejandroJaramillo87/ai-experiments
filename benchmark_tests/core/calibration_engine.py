#!/usr/bin/env python3
"""
Systematic Base Model Calibration

Implements the systematic calibration path: base easy → medium → hard
Focus on the core domains first, then expand to specialized domains.

Progression Strategy:
1. Start with core domains (reasoning, creativity, language, social, knowledge, integration)
2. Calibrate easy level first, ensure quality thresholds met
3. Progress to medium only after easy calibration success
4. Progress to hard only after medium calibration success
5. Use statistical validation and pattern detection throughout

Based on .claude/CLAUDE.md requirements and production calibration framework.
"""

import json
import sys
import time
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from benchmark_runner import BenchmarkTestRunner, TestResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationCalibrationResult:
    """Results from evaluation-based calibration testing"""
    domain: str
    test_id: str 
    difficulty: str
    enhanced_score: float
    target_range: Tuple[float, float]
    validation_passed: bool
    calibration_status: str

@dataclass 
class SystematicCalibrationResult:
    """Results from systematic calibration progression"""
    domain: str
    difficulty_progression: Dict[str, Any]  # easy, medium, hard results
    overall_success: bool
    progression_halted_at: Optional[str]
    recommendation: str

class SystematicBaseCalibrator:
    """Systematic calibration for base models: easy → medium → hard"""
    
    def __init__(self):
        self.benchmark_runner = BenchmarkTestRunner()
        # Configure API endpoint
        self.benchmark_runner.configure_api(
            endpoint="http://localhost:8004/v1/completions",
            model="DeepSeek-R1-0528-Qwen3-8b",
            timeout=30
        )
        # Configure for enhanced evaluation
        self.benchmark_runner.enhanced_evaluation = True
        self.benchmark_runner.evaluation_mode = "full"
        
        # Target score ranges for different difficulty levels (based on evaluation quality)
        self.target_ranges = {
            'easy': (70, 85),      # Easy should score 70-85/100
            'medium': (60, 80),    # Medium should score 60-80/100  
            'hard': (50, 75)       # Hard should score 50-75/100
        }
        
        # Core domains to calibrate first (production-ready domains with good coverage)
        self.core_domains = [
            'reasoning',     # 300+ tests, critical for evaluation pipeline
            'creativity',    # 200+ tests, creative evaluation patterns
            'language',      # 230+ tests, linguistic diversity
            'social',        # 200+ tests, cultural understanding
            'knowledge',     # 200+ tests, factual reasoning 
            'integration'    # 200+ tests, cross-domain synthesis
        ]
        
        # Production readiness criteria (from CLAUDE.md)
        self.production_criteria = {
            'min_tests_per_difficulty': 3,     # Statistical significance
            'success_rate_threshold': 0.7,     # 70%+ tests achieve Good+ calibration
            'cross_domain_variance': 0.1       # <10% variance between domains
        }
        
        self.calibration_results = []
        
    def get_available_domain_files(self, domain: str) -> Dict[str, Path]:
        """Get available difficulty level files for a domain"""
        domain_dir = Path('domains') / domain / 'base_models'
        
        files = {}
        for difficulty in ['easy', 'medium', 'hard']:
            file_path = domain_dir / f"{difficulty}.json"
            if file_path.exists():
                files[difficulty] = file_path
                
        return files
        
    def calibrate_difficulty_level(self, domain: str, difficulty: str, 
                                 domain_file: Path, tests_per_level: int = 5) -> Tuple[bool, Dict[str, Any]]:
        """Calibrate a specific difficulty level for a domain using enhanced evaluation"""
        logger.info(f"🎯 Calibrating {domain} - {difficulty} level")
        
        # Load domain data to get test IDs
        try:
            with open(domain_file, 'r', encoding='utf-8') as f:
                domain_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {domain_file}: {e}")
            return False, {'error': str(e)}
            
        # Get test IDs (sample from beginning for consistency)
        test_ids = [test['id'] for test in domain_data.get('tests', [])][:tests_per_level]
        
        if len(test_ids) < tests_per_level:
            logger.warning(f"Only {len(test_ids)} tests available, wanted {tests_per_level}")
            
        # Run multiple samples per test for statistical validation
        calibration_results = []
        successful_calibrations = 0
        
        for test_id in test_ids:
            logger.info(f"  Testing {test_id}...")
            
            # Run multiple samples for statistical validation
            sample_scores = []
            for sample in range(3):  # 3 samples per test
                result = self.benchmark_runner.run_single_test(
                    domain_path=str(domain_file),
                    test_id=test_id,
                    enhanced_evaluation=True
                )
                
                if result and hasattr(result, 'enhanced_score'):
                    sample_scores.append(result.enhanced_score)
                    
                time.sleep(0.5)  # Rate limiting
                
            if sample_scores:
                # Calculate statistics for this test
                mean_score = statistics.mean(sample_scores)
                target_range = self.target_ranges[difficulty]
                
                # Determine if this test passed validation
                validation_passed = target_range[0] <= mean_score <= target_range[1]
                
                calibration_status = self._get_calibration_status(mean_score, target_range)
                
                eval_result = EvaluationCalibrationResult(
                    domain=domain,
                    test_id=test_id,
                    difficulty=difficulty,
                    enhanced_score=mean_score,
                    target_range=target_range,
                    validation_passed=validation_passed,
                    calibration_status=calibration_status
                )
                
                calibration_results.append(eval_result)
                if validation_passed:
                    successful_calibrations += 1
            else:
                logger.error(f"Failed to get enhanced scores for {test_id}")
                
        if not calibration_results:
            logger.error(f"All calibration tests failed for {domain} {difficulty}")
            return False, {'error': 'all_tests_failed'}
            
        # Analyze results
        success_rate = successful_calibrations / len(calibration_results)
        avg_score = statistics.mean([r.enhanced_score for r in calibration_results])
        score_std = statistics.stdev([r.enhanced_score for r in calibration_results]) if len(calibration_results) > 1 else 0
        
        # Determine calibration quality based on enhanced evaluation scores
        calibration_quality = self._assess_calibration_quality(avg_score, success_rate)
        should_continue = calibration_quality['should_continue']
        
        results_summary = {
            'domain': domain,
            'difficulty': difficulty,
            'tests_run': len(calibration_results),
            'successful_calibrations': successful_calibrations,
            'success_rate': success_rate,
            'avg_calibration_score': avg_score,  # Now this is the enhanced evaluation score
            'score_std': score_std,
            'calibration_quality': calibration_quality,
            'should_continue_progression': should_continue,
            'detailed_results': calibration_results
        }
        
        logger.info(f"  ✅ {domain} {difficulty}: {avg_score:.1f}±{score_std:.1f} score, "
                   f"{success_rate*100:.1f}% success rate ({calibration_quality['level']})")
        
        return should_continue, results_summary
    
    def _get_calibration_status(self, score: float, target_range: Tuple[float, float]) -> str:
        """Get calibration status based on score and target range"""
        min_target, max_target = target_range
        target_center = (min_target + max_target) / 2
        
        if min_target <= score <= max_target:
            if abs(score - target_center) <= 2:
                return "✅ EXCELLENT CALIBRATION"
            else:
                return "🟡 GOOD CALIBRATION"
        elif abs(score - target_center) <= 10:
            return "🟠 NEEDS CALIBRATION"
        else:
            return "❌ CALIBRATION BROKEN"
        
    def _assess_calibration_quality(self, avg_score: float, success_rate: float) -> Dict[str, Any]:
        """Assess calibration quality and determine if progression should continue"""
        
        # Determine quality level based on enhanced evaluation score thresholds
        if avg_score >= 80:  # Excellent evaluation scores
            level = "EXCELLENT"
            should_continue = True
            recommendation = "✅ Excellent calibration - continue to next difficulty"
        elif avg_score >= 70:  # Good evaluation scores
            level = "GOOD" 
            should_continue = True
            recommendation = "🟡 Good calibration - continue with monitoring"
        elif avg_score >= 60:  # Needs improvement
            level = "NEEDS_CALIBRATION"
            should_continue = False
            recommendation = "🟠 Calibration needs adjustment - halt progression and fix"
        else:  # Poor scores
            level = "BROKEN"
            should_continue = False
            recommendation = "❌ Calibration broken - system failure, halt immediately"
            
        # Additional check on success rate (70% of tests should pass target ranges)
        if success_rate < 0.7:
            should_continue = False
            recommendation += f" (Success rate {success_rate*100:.1f}% below 70.0% threshold)"
            
        return {
            'level': level,
            'should_continue': should_continue,
            'recommendation': recommendation,
            'score': avg_score,
            'success_rate': success_rate
        }
        
    def calibrate_domain_progression(self, domain: str) -> SystematicCalibrationResult:
        """Run systematic calibration progression for a single domain"""
        logger.info(f"🚀 Starting systematic calibration for {domain} domain")
        
        # Get available domain files
        domain_files = self.get_available_domain_files(domain)
        
        if not domain_files:
            logger.error(f"No domain files found for {domain}")
            return SystematicCalibrationResult(
                domain=domain,
                difficulty_progression={},
                overall_success=False,
                progression_halted_at=None,
                recommendation="❌ Domain files not found"
            )
            
        # Progression order: easy → medium → hard
        progression_order = ['easy', 'medium', 'hard']
        difficulty_results = {}
        progression_halted_at = None
        
        for difficulty in progression_order:
            if difficulty not in domain_files:
                logger.warning(f"Skipping {difficulty} - file not found for {domain}")
                continue
                
            # Run calibration for this difficulty level
            should_continue, results = self.calibrate_difficulty_level(
                domain, difficulty, domain_files[difficulty]
            )
            
            difficulty_results[difficulty] = results
            
            # Check if we should continue progression
            if not should_continue:
                progression_halted_at = difficulty
                logger.warning(f"⚠️ Progression halted at {difficulty} level for {domain}")
                logger.warning(f"   Reason: {results['calibration_quality']['recommendation']}")
                break
                
            logger.info(f"   ✅ {difficulty} level passed - continuing to next level")
            
        # Determine overall success
        overall_success = progression_halted_at is None and len(difficulty_results) > 0
        
        # Generate recommendation
        if overall_success:
            recommendation = "✅ Full progression successful - domain ready for production"
        elif progression_halted_at:
            recommendation = f"🟠 Progression halted at {progression_halted_at} - needs calibration adjustment"
        else:
            recommendation = "❌ Domain calibration failed - investigate domain files and evaluator setup"
            
        return SystematicCalibrationResult(
            domain=domain,
            difficulty_progression=difficulty_results,
            overall_success=overall_success,
            progression_halted_at=progression_halted_at,
            recommendation=recommendation
        )
        
    def run_systematic_base_calibration(self) -> Dict[str, Any]:
        """Run systematic calibration across all core domains"""
        logger.info("🎯 SYSTEMATIC BASE MODEL CALIBRATION")
        logger.info("=" * 60)
        logger.info("Progression: Base Easy → Medium → Hard")
        logger.info(f"Core domains: {', '.join(self.core_domains)}")
        logger.info("")
        
        start_time = time.time()
        domain_results = []
        successful_domains = 0
        
        for domain in self.core_domains:
            logger.info(f"📚 Processing domain: {domain}")
            logger.info("-" * 40)
            
            result = self.calibrate_domain_progression(domain)
            domain_results.append(result)
            
            if result.overall_success:
                successful_domains += 1
                
            logger.info(f"Result: {result.recommendation}")
            logger.info("")
            
        # Generate comprehensive report
        elapsed_time = time.time() - start_time
        self.generate_systematic_report(domain_results, successful_domains, elapsed_time)
        
        return {
            'domains_processed': len(self.core_domains),
            'successful_domains': successful_domains,
            'success_rate': successful_domains / len(self.core_domains),
            'domain_results': domain_results,
            'elapsed_time': elapsed_time
        }
        
    def generate_systematic_report(self, domain_results: List[SystematicCalibrationResult], 
                                 successful_domains: int, elapsed_time: float):
        """Generate comprehensive systematic calibration report"""
        
        print("\n" + "=" * 80)
        print("🎯 SYSTEMATIC BASE MODEL CALIBRATION REPORT")
        print("=" * 80)
        
        # Overall statistics
        total_domains = len(domain_results)
        success_rate = (successful_domains / total_domains * 100) if total_domains > 0 else 0
        
        print(f"📊 OVERALL RESULTS:")
        print(f"   Domains processed: {total_domains}")
        print(f"   Successful progressions: {successful_domains}")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Processing time: {elapsed_time/60:.1f} minutes")
        
        # Progression analysis
        progression_stats = {'easy': 0, 'medium': 0, 'hard': 0}
        halted_at_stats = {'easy': 0, 'medium': 0, 'hard': 0, 'none': 0}
        
        for result in domain_results:
            # Count successful difficulty levels
            for difficulty in result.difficulty_progression.keys():
                progression_stats[difficulty] = progression_stats.get(difficulty, 0) + 1
                
            # Count where progression was halted
            if result.progression_halted_at:
                halted_at_stats[result.progression_halted_at] += 1
            else:
                halted_at_stats['none'] += 1
                
        print(f"\n📈 PROGRESSION ANALYSIS:")
        print(f"   Easy level completed: {progression_stats.get('easy', 0)}/{total_domains}")
        print(f"   Medium level completed: {progression_stats.get('medium', 0)}/{total_domains}")
        print(f"   Hard level completed: {progression_stats.get('hard', 0)}/{total_domains}")
        print(f"   Full progressions: {halted_at_stats['none']}/{total_domains}")
        
        # Domain-by-domain results
        print(f"\n🔍 DOMAIN-BY-DOMAIN RESULTS:")
        for result in domain_results:
            status = "✅ SUCCESS" if result.overall_success else "❌ HALTED"
            halted_info = f" (at {result.progression_halted_at})" if result.progression_halted_at else ""
            print(f"   {result.domain:<15} {status}{halted_info}")
            
        # Production readiness assessment
        print(f"\n🎯 PRODUCTION READINESS ASSESSMENT:")
        if success_rate >= 70:
            print("   ✅ READY FOR PRODUCTION DEPLOYMENT")
            print("   ✅ Core domain calibration meets requirements")
            print("   ✅ Systematic progression validated")
            print("   📋 Next step: Proceed to instruct model calibration")
        elif success_rate >= 50:
            print("   🟡 PARTIAL READINESS - Review failed domains")
            print("   🟡 Some core domains need calibration adjustment")
            print("   🔧 Fix failing domains before proceeding")
        else:
            print("   ❌ NOT READY FOR PRODUCTION")
            print("   ❌ Major calibration issues across multiple domains")
            print("   🚨 Review evaluator framework and token optimization")
            
        # Detailed domain analysis
        print(f"\n📋 DETAILED DOMAIN ANALYSIS:")
        for result in domain_results:
            print(f"\n   {result.domain.upper()} DOMAIN:")
            print(f"     Status: {result.recommendation}")
            
            for difficulty, details in result.difficulty_progression.items():
                avg_score = details.get('avg_calibration_score', 0)
                success_rate = details.get('success_rate', 0) * 100
                quality = details.get('calibration_quality', {}).get('level', 'UNKNOWN')
                print(f"     {difficulty.capitalize():<8} {avg_score:6.1f} score | {success_rate:5.1f}% success | {quality}")
                
        # Save detailed results
        self.save_systematic_results(domain_results, successful_domains, elapsed_time)
        
    def save_systematic_results(self, domain_results: List[SystematicCalibrationResult], 
                              successful_domains: int, elapsed_time: float):
        """Save detailed calibration results to JSON"""
        results_dir = Path('test_results')
        results_dir.mkdir(exist_ok=True)
        
        results_file = results_dir / 'systematic_base_calibration_results.json'
        
        results_data = {
            'calibration_type': 'systematic_base_model',
            'framework_version': '1.0.0',
            'timestamp': time.time(),
            'summary': {
                'total_domains': len(domain_results),
                'successful_domains': successful_domains,
                'success_rate': successful_domains / len(domain_results) if domain_results else 0,
                'elapsed_time_minutes': elapsed_time / 60
            },
            'domain_results': [
                {
                    'domain': result.domain,
                    'overall_success': result.overall_success,
                    'progression_halted_at': result.progression_halted_at,
                    'recommendation': result.recommendation,
                    'difficulty_progression': result.difficulty_progression
                }
                for result in domain_results
            ]
        }
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            print(f"\n💾 Detailed results saved to: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save systematic results: {e}")

def main():
    """Main execution for systematic base model calibration"""
    calibrator = SystematicBaseCalibrator()
    
    # Run systematic calibration
    results = calibrator.run_systematic_base_calibration()
    
    # Return appropriate exit code
    if results['success_rate'] >= 0.7:
        print("\n🎉 Systematic calibration successful - ready for next phase!")
        return 0
    else:
        print("\n⚠️ Systematic calibration needs attention - review results above")
        return 1

if __name__ == "__main__":
    exit(main())