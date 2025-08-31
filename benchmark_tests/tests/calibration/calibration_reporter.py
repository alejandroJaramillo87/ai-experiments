#!/usr/bin/env python3
"""
Calibration Reporting System

Generates comprehensive reports for calibration validation results with
✅🟡🟠❌ status indicators and statistical analysis.

Author: Claude Code  
Version: 1.0.0 - Sequential Architecture
"""

import json
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from datetime import datetime

class CalibrationReporter:
    """
    Handles reporting and analysis of calibration validation results
    """
    
    def __init__(self):
        """Initialize calibration reporter"""
        self.report_timestamp = datetime.now().isoformat()
    
    def log_test_result(self, result) -> None:
        """
        Log individual test result to console
        
        Args:
            result: CalibrationResult object
        """
        print(f"\n📊 {result.test_name}")
        print(f"   Scores: {result.scores}")
        print(f"   Mean: {result.mean_score:.1f} ± {result.std_deviation:.1f}")
        print(f"   Target: {result.target_range[0]}-{result.target_range[1]} (mean: {result.target_mean:.1f})")
        print(f"   Status: {result.calibration_status}")
        if result.points_off_target > 0:
            print(f"   Points Off: {result.points_off_target:.1f}")
        print(f"   Samples: {result.sample_count}")
    
    def generate_report(self, results: List) -> Dict[str, Any]:
        """
        Generate comprehensive calibration report
        
        Args:
            results: List of CalibrationResult objects
            
        Returns:
            Comprehensive report dictionary
        """
        if not results:
            return {"error": "No calibration results provided"}
        
        # Basic statistics
        total_tests = len(results)
        successful_tests = [r for r in results if r.sample_count > 0]
        failed_tests = [r for r in results if r.sample_count == 0]
        
        # Calibration status counts
        status_counts = self._count_calibration_statuses(successful_tests)
        
        # Overall calibration assessment
        overall_status = self._determine_overall_status(status_counts, total_tests)
        
        # Statistical analysis
        stats_analysis = self._generate_statistical_analysis(successful_tests)
        
        # Individual test details
        test_details = []
        for result in results:
            test_details.append({
                "test_name": result.test_name,
                "mean_score": result.mean_score,
                "std_deviation": result.std_deviation,
                "target_range": result.target_range,
                "target_mean": result.target_mean,
                "calibration_status": result.calibration_status,
                "points_off_target": result.points_off_target,
                "sample_count": result.sample_count,
                "scores": result.scores
            })
        
        report = {
            "report_metadata": {
                "timestamp": self.report_timestamp,
                "framework_version": "1.0.0",
                "total_tests": total_tests,
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests)
            },
            "overall_assessment": {
                "overall_calibration_status": overall_status,
                "calibration_success_rate": len([r for r in successful_tests if r.calibration_status.startswith("✅")]) / max(len(successful_tests), 1),
                "production_ready_rate": len([r for r in successful_tests if r.calibration_status.startswith(("✅", "🟡"))]) / max(len(successful_tests), 1)
            },
            "calibration_status_breakdown": status_counts,
            "statistical_analysis": stats_analysis,
            "test_details": test_details,
            "recommendations": self._generate_recommendations(successful_tests, status_counts)
        }
        
        return report
    
    def _count_calibration_statuses(self, results: List) -> Dict[str, int]:
        """Count calibration status occurrences"""
        status_counts = {
            "perfect_calibration": 0,
            "good_calibration": 0, 
            "needs_calibration": 0,
            "calibration_broken": 0
        }
        
        for result in results:
            if result.calibration_status.startswith("✅"):
                status_counts["perfect_calibration"] += 1
            elif result.calibration_status.startswith("🟡"):
                status_counts["good_calibration"] += 1
            elif result.calibration_status.startswith("🟠"):
                status_counts["needs_calibration"] += 1
            elif result.calibration_status.startswith("❌"):
                status_counts["calibration_broken"] += 1
        
        return status_counts
    
    def _determine_overall_status(self, status_counts: Dict[str, int], total_tests: int) -> str:
        """Determine overall calibration status"""
        if total_tests == 0:
            return "❌ NO TESTS"
        
        # Calculate percentages
        perfect_rate = status_counts["perfect_calibration"] / total_tests
        good_rate = status_counts["good_calibration"] / total_tests
        production_ready_rate = perfect_rate + good_rate
        broken_rate = status_counts["calibration_broken"] / total_tests
        
        # Determine overall status
        if production_ready_rate >= 0.8:  # 80% perfect or good
            return "✅ CALIBRATION EXCELLENT"
        elif production_ready_rate >= 0.6:  # 60% perfect or good
            return "🟡 CALIBRATION GOOD"
        elif broken_rate <= 0.2:  # Less than 20% broken
            return "🟠 CALIBRATION NEEDS WORK"
        else:
            return "❌ CALIBRATION BROKEN"
    
    def _generate_statistical_analysis(self, results: List) -> Dict[str, Any]:
        """Generate statistical analysis of calibration results"""
        if not results:
            return {"error": "No results for statistical analysis"}
        
        # Collect all scores and deviations
        all_scores = []
        all_deviations = []
        all_points_off = []
        
        for result in results:
            all_scores.extend(result.scores)
            if result.std_deviation > 0:
                all_deviations.append(result.std_deviation)
            all_points_off.append(result.points_off_target)
        
        analysis = {}
        
        # Score analysis
        if all_scores:
            analysis["score_statistics"] = {
                "mean": statistics.mean(all_scores),
                "median": statistics.median(all_scores),
                "std_dev": statistics.stdev(all_scores) if len(all_scores) > 1 else 0,
                "min": min(all_scores),
                "max": max(all_scores),
                "range": max(all_scores) - min(all_scores)
            }
        
        # Deviation analysis
        if all_deviations:
            analysis["deviation_statistics"] = {
                "mean_deviation": statistics.mean(all_deviations),
                "max_deviation": max(all_deviations),
                "consistency_rating": "High" if statistics.mean(all_deviations) < 2.0 else "Medium" if statistics.mean(all_deviations) < 5.0 else "Low"
            }
        
        # Accuracy analysis
        if all_points_off:
            analysis["accuracy_statistics"] = {
                "mean_points_off": statistics.mean(all_points_off),
                "median_points_off": statistics.median(all_points_off),
                "max_points_off": max(all_points_off),
                "tests_within_5_points": len([p for p in all_points_off if p <= 5.0]),
                "tests_within_10_points": len([p for p in all_points_off if p <= 10.0])
            }
        
        return analysis
    
    def _generate_recommendations(self, results: List, status_counts: Dict[str, int]) -> List[str]:
        """Generate recommendations based on calibration results"""
        recommendations = []
        
        total_tests = len(results)
        if total_tests == 0:
            return ["No test results to analyze"]
        
        # Overall calibration recommendations
        broken_rate = status_counts["calibration_broken"] / total_tests
        needs_work_rate = status_counts["needs_calibration"] / total_tests
        
        if broken_rate > 0.2:  # More than 20% broken
            recommendations.append("🚨 CRITICAL: More than 20% of tests show broken calibration. Review evaluator implementation.")
        
        if needs_work_rate > 0.3:  # More than 30% need work
            recommendations.append("⚠️ HIGH PRIORITY: Multiple tests need calibration adjustment. Consider scoring formula tuning.")
        
        # Statistical consistency recommendations
        deviations = [r.std_deviation for r in results if r.std_deviation > 0]
        if deviations:
            mean_deviation = statistics.mean(deviations)
            if mean_deviation > 5.0:
                recommendations.append("📊 CONSISTENCY: High standard deviation in results. Consider increasing sample size or improving evaluation stability.")
            elif mean_deviation > 2.0:
                recommendations.append("📊 STABILITY: Moderate result variation detected. Monitor for evaluation consistency.")
        
        # Domain-specific recommendations
        cultural_tests = [r for r in results if "cultural" in r.test_name.lower()]
        if cultural_tests:
            cultural_broken = len([r for r in cultural_tests if r.calibration_status.startswith("❌")])
            if cultural_broken > 0:
                recommendations.append("🏛️ CULTURAL: Cultural reasoning tests showing calibration issues. Review cultural authenticity evaluation.")
        
        # Success recommendations
        perfect_rate = status_counts["perfect_calibration"] / total_tests
        if perfect_rate > 0.5:
            recommendations.append("✅ EXCELLENT: More than 50% perfect calibration achieved. System showing strong evaluation quality.")
        
        return recommendations
    
    def print_summary_report(self, report: Dict[str, Any]) -> None:
        """
        Print human-readable summary report to console
        
        Args:
            report: Report dictionary from generate_report()
        """
        print("\n" + "="*60)
        print("🎯 CALIBRATION VALIDATION SUMMARY REPORT")
        print("="*60)
        
        # Metadata
        metadata = report.get("report_metadata", {})
        print(f"📅 Timestamp: {metadata.get('timestamp', 'Unknown')}")
        print(f"🧪 Tests: {metadata.get('successful_tests', 0)}/{metadata.get('total_tests', 0)} successful")
        
        # Overall status
        overall = report.get("overall_assessment", {})
        print(f"\n🎯 Overall Status: {overall.get('overall_calibration_status', 'Unknown')}")
        print(f"📊 Success Rate: {overall.get('calibration_success_rate', 0):.1%}")
        print(f"🚀 Production Ready: {overall.get('production_ready_rate', 0):.1%}")
        
        # Status breakdown
        status = report.get("calibration_status_breakdown", {})
        print(f"\n📋 Calibration Status Breakdown:")
        print(f"   ✅ Perfect: {status.get('perfect_calibration', 0)}")
        print(f"   🟡 Good: {status.get('good_calibration', 0)}")
        print(f"   🟠 Needs Work: {status.get('needs_calibration', 0)}")
        print(f"   ❌ Broken: {status.get('calibration_broken', 0)}")
        
        # Statistical summary
        stats = report.get("statistical_analysis", {})
        if "score_statistics" in stats:
            score_stats = stats["score_statistics"]
            print(f"\n📈 Score Statistics:")
            print(f"   Mean: {score_stats.get('mean', 0):.1f}")
            print(f"   Range: {score_stats.get('min', 0):.1f} - {score_stats.get('max', 0):.1f}")
            print(f"   Std Dev: {score_stats.get('std_dev', 0):.1f}")
        
        # Top recommendations
        recommendations = report.get("recommendations", [])
        if recommendations:
            print(f"\n🔧 Key Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                print(f"   {i}. {rec}")
        
        # Individual test summary
        test_details = report.get("test_details", [])
        if test_details:
            print(f"\n📋 Individual Test Results:")
            for test in test_details:
                status_icon = test["calibration_status"].split()[0]
                print(f"   {status_icon} {test['test_name']}: {test['mean_score']:.1f} ± {test['std_deviation']:.1f}")
        
        print("\n" + "="*60)
    
    def save_detailed_report(self, report: Dict[str, Any], filename: str) -> bool:
        """
        Save detailed JSON report to file
        
        Args:
            report: Report dictionary
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            return True
        except Exception as e:
            print(f"❌ Failed to save report: {e}")
            return False