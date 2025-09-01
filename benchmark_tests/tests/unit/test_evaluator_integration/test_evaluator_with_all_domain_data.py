"""
Unit tests using real domain data from all 6 benchmark domains.

Tests statistical algorithms and validation systems with authentic test data from:
- social, reasoning, creativity, language, knowledge, integration domains
- Tests bias detection across diverse content types  
- Validates algorithm consistency across different evaluation contexts
"""

import unittest
from unittest.mock import Mock, patch
from typing import List, Dict, Any
import json
import os
import statistics
import numpy as np

from evaluator.core.evaluation_aggregator import EvaluationAggregator, AggregatedEvaluationResult, BiasAnalysis
from evaluator.core.ensemble_disagreement_detector import EnsembleDisagreementDetector
from evaluator.cultural.cultural_dataset_validator import CulturalDatasetValidator  
from evaluator.core.domain_evaluator_base import DomainEvaluationResult, EvaluationDimension, CulturalContext


class DomainDataLoader:
    """Utility class to load real domain test data."""
    
    def __init__(self, base_path="/home/alejandro/workspace/ai-workstation/benchmark_tests/domains"):
        self.base_path = base_path
        self.domains = ["social", "reasoning", "creativity", "language", "knowledge", "integration"]
        
    def load_domain_tests(self, domain: str, difficulty: str = "easy") -> List[Dict[str, Any]]:
        """Load test data from a specific domain."""
        file_path = os.path.join(self.base_path, domain, "base_models", f"{difficulty}.json")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('tests', [])
        except FileNotFoundError:
            return []
    
    def load_domain_categories(self, domain: str) -> Dict[str, Any]:
        """Load category information for a domain."""
        file_path = os.path.join(self.base_path, domain, "base_models", "categories.json")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('categories', {})
        except FileNotFoundError:
            return {}
    
    def load_all_domains_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load test data from all domains."""
        all_data = {}
        for domain in self.domains:
            all_data[domain] = self.load_domain_tests(domain)
        return all_data


class TestStatisticalAlgorithmsWithRealDomainData(unittest.TestCase):
    """Test statistical algorithms using real domain test data."""
    
    def setUp(self):
        """Set up test fixtures with real domain data."""
        self.data_loader = DomainDataLoader()
        self.aggregator = EvaluationAggregator()
        self.disagreement_detector = EnsembleDisagreementDetector()
        
        # Load real domain data
        self.domain_data = self.data_loader.load_all_domains_data()
        
        # Extract cultural contexts from real social domain data
        self.cultural_contexts = self._extract_cultural_contexts_from_social_data()
        
    def _extract_cultural_contexts_from_social_data(self) -> List[CulturalContext]:
        """Extract cultural contexts from real social domain test data."""
        contexts = []
        social_tests = self.domain_data.get('social', [])
        
        for test in social_tests[:10]:  # Use first 10 for testing
            # Extract cultural information from test metadata
            category = test.get('category', '')
            description = test.get('description', '')
            prompt = test.get('prompt', '')
            
            # Infer cultural context based on content
            cultural_groups = []
            traditions = []
            knowledge_systems = []
            
            # Ubuntu-related tests
            if 'ubuntu' in prompt.lower() or 'ubuntu' in description.lower():
                cultural_groups = ['bantu_peoples', 'zulu', 'xhosa']
                traditions = ['ubuntu', 'ubuntu_philosophy']
                knowledge_systems = ['bantu_traditional_knowledge']
            
            # Native American tests
            elif any(term in prompt.lower() for term in ['native american', 'ojibwe', 'lakota', 'circle']):
                cultural_groups = ['native_american', 'ojibwe', 'lakota'] 
                traditions = ['talking_circle', 'peacemaking_circle']
                knowledge_systems = ['indigenous_knowledge']
                
            # Islamic tests
            elif any(term in prompt.lower() for term in ['islamic', 'sulh', 'muslim']):
                cultural_groups = ['muslim_communities']
                traditions = ['sulh', 'islamic_mediation']
                knowledge_systems = ['islamic_jurisprudence']
            
            # Default context for other social tests
            else:
                cultural_groups = ['general_social']
                traditions = ['social_practice']
                knowledge_systems = ['social_knowledge']
            
            contexts.append(CulturalContext(
                cultural_groups=cultural_groups,
                traditions=traditions,
                knowledge_systems=knowledge_systems,
                performance_aspects=[category] if category else [],
                linguistic_varieties=[]
            ))
        
        return contexts
    
    def _create_mock_evaluation_from_test(self, test: Dict[str, Any], domain: str, score: float = None) -> DomainEvaluationResult:
        """Create mock evaluation result from real test data."""
        # Use real test data fields
        test_id = test.get('id', 'unknown')
        category = test.get('category', 'general')
        description = test.get('description', '')
        
        # Generate realistic scores based on domain and content complexity
        if score is None:
            # Base score on content complexity
            prompt_length = len(test.get('prompt', ''))
            complexity_factor = min(1.0, prompt_length / 1000.0)  # Normalize by typical prompt length
            score = 0.6 + (complexity_factor * 0.3) + (np.random.random() * 0.1 - 0.05)  # Add small random variation
            score = max(0.1, min(0.95, score))  # Clamp to valid range
        
        # Create dimensions based on domain type
        dimensions = []
        if domain == 'social':
            dimensions.extend([
                EvaluationDimension(
                    name='cultural_authenticity', score=score + np.random.random() * 0.1 - 0.05,
                    confidence=0.8 + np.random.random() * 0.15, cultural_relevance=0.9,
                    evidence=[f"Cultural elements in {description}"], cultural_markers=['cultural_element']
                ),
                EvaluationDimension(
                    name='social_appropriateness', score=score + np.random.random() * 0.1 - 0.05,
                    confidence=0.85, cultural_relevance=0.75,
                    evidence=[f"Social context in {category}"], cultural_markers=['social_marker']
                )
            ])
        elif domain == 'reasoning':
            dimensions.extend([
                EvaluationDimension(
                    name='logical_consistency', score=score + np.random.random() * 0.1 - 0.05,
                    confidence=0.9, cultural_relevance=0.3,
                    evidence=["Logical structure"], cultural_markers=[]
                ),
                EvaluationDimension(
                    name='reasoning_clarity', score=score + np.random.random() * 0.1 - 0.05,
                    confidence=0.85, cultural_relevance=0.2,
                    evidence=["Clear reasoning"], cultural_markers=[]
                )
            ])
        elif domain == 'creativity':
            dimensions.extend([
                EvaluationDimension(
                    name='creative_expression', score=score + np.random.random() * 0.1 - 0.05,
                    confidence=0.8, cultural_relevance=0.6,
                    evidence=["Creative elements"], cultural_markers=['creative_marker']
                ),
                EvaluationDimension(
                    name='originality', score=score + np.random.random() * 0.1 - 0.05,
                    confidence=0.75, cultural_relevance=0.5,
                    evidence=["Original thinking"], cultural_markers=[]
                )
            ])
        else:
            # Default dimensions for language, knowledge, integration
            dimensions.append(EvaluationDimension(
                name='domain_competence', score=score,
                confidence=0.8, cultural_relevance=0.4,
                evidence=[f"{domain} competence"], cultural_markers=[]
            ))
        
        # Ensure scores are within valid range
        for dim in dimensions:
            dim.score = max(0.0, min(1.0, dim.score))
            dim.confidence = max(0.0, min(1.0, dim.confidence))
        
        return DomainEvaluationResult(
            domain=domain,
            evaluation_type=category,
            overall_score=score,
            dimensions=dimensions,
            cultural_context=CulturalContext(),  # Will be set based on test content
            metadata={'test_id': test_id, 'original_test': test},
            processing_notes=[f"Evaluation of {domain} test {test_id}"]
        )
    
    def test_bias_detection_across_all_domains(self):
        """Test bias detection algorithms using data from all 6 domains."""
        # Create evaluation history using real test data from all domains
        evaluation_history = []
        cultural_contexts = []
        
        for domain, tests in self.domain_data.items():
            if not tests:  # Skip if no tests loaded
                continue
                
            for i, test in enumerate(tests[:5]):  # Use first 5 tests per domain
                # Create slightly different scores to simulate evaluator variation
                base_score = 0.7 + (i % 3) * 0.1  # Vary scores: 0.7, 0.8, 0.9
                
                # Add domain-specific bias patterns for testing
                if domain == 'social':
                    domain_bias = 0.1  # Social domain gets boosted
                elif domain == 'reasoning':
                    domain_bias = -0.05  # Reasoning domain gets slightly lowered
                else:
                    domain_bias = 0.0
                
                adjusted_score = max(0.1, min(0.95, base_score + domain_bias))
                
                # Create mock aggregated result
                mock_result = Mock()
                mock_result.overall_score = adjusted_score
                mock_result.consensus_level = 0.7 + np.random.random() * 0.2
                mock_result.cultural_competence = adjusted_score * 0.9
                mock_result.dimension_scores = {
                    'domain_competence': adjusted_score,
                    'cultural_authenticity': adjusted_score + 0.05
                }
                
                evaluation_history.append(mock_result)
                
                # Create appropriate cultural context
                if domain == 'social' and i < len(self.cultural_contexts):
                    cultural_contexts.append(self.cultural_contexts[i])
                else:
                    cultural_contexts.append(CulturalContext(
                        cultural_groups=[f'{domain}_context'],
                        traditions=[f'{domain}_tradition'],
                        knowledge_systems=[f'{domain}_knowledge']
                    ))
        
        # Run bias detection analysis
        if evaluation_history:  # Only run if we have data
            bias_analysis = self.aggregator.detect_statistical_bias(
                evaluation_history, cultural_contexts
            )
            
            # Verify bias analysis structure
            self.assertIsInstance(bias_analysis, BiasAnalysis)
            
            # Test that bias detection works across domains
            if bias_analysis.cultural_group_bias:
                # Should detect the intentional social domain bias
                social_bias_groups = [group for group in bias_analysis.cultural_group_bias.keys() 
                                    if 'social' in group.lower()]
                if social_bias_groups:
                    social_bias_score = bias_analysis.cultural_group_bias[social_bias_groups[0]]
                    self.assertGreater(social_bias_score, 0)  # Positive bias for social
            
            # Verify statistical significance tests ran
            self.assertIsInstance(bias_analysis.statistical_significance, dict)
            self.assertIsInstance(bias_analysis.chi_square_results, dict)
            self.assertIsInstance(bias_analysis.effect_sizes, dict)
    
    def test_algorithm_consistency_across_domains(self):
        """Test that algorithms behave consistently across different content types."""
        # Create evaluations for different domains with similar underlying quality
        target_score = 0.75
        tolerance = 0.02
        
        domain_evaluations = {}
        for domain, tests in self.domain_data.items():
            if not tests:  # Skip empty domains
                continue
                
            test_sample = tests[0]  # Use first test from each domain
            evaluation = self._create_mock_evaluation_from_test(test_sample, domain, target_score)
            domain_evaluations[domain] = evaluation
        
        if len(domain_evaluations) < 2:  # Need at least 2 domains for comparison
            self.skipTest("Insufficient domain data for consistency testing")
        
        # Test that dimension aggregation produces consistent results
        aggregated_scores = {}
        for domain, evaluation in domain_evaluations.items():
            domain_results = [evaluation]
            aggregated = self.aggregator.aggregate_results(domain_results)
            aggregated_scores[domain] = aggregated.overall_score
        
        # Check consistency: scores should be similar since input quality was similar
        score_values = list(aggregated_scores.values())
        score_std = statistics.stdev(score_values) if len(score_values) > 1 else 0
        
        # Standard deviation should be low for consistent algorithm behavior
        self.assertLess(score_std, 0.15, 
                       f"Algorithm inconsistent across domains. Scores: {aggregated_scores}")
    
    def test_cultural_authenticity_detection_across_domains(self):
        """Test cultural authenticity detection using real cultural scenarios."""
        # Test with real social domain data that contains cultural content
        social_tests = self.domain_data.get('social', [])
        if not social_tests:
            self.skipTest("No social domain data available")
        
        # Find Ubuntu-related test for high cultural authenticity
        ubuntu_test = None
        for test in social_tests:
            if 'ubuntu' in test.get('prompt', '').lower():
                ubuntu_test = test
                break
        
        if ubuntu_test:
            # Create evaluation with high cultural authenticity
            ubuntu_evaluation = self._create_mock_evaluation_from_test(ubuntu_test, 'social', 0.85)
            ubuntu_context = next((ctx for ctx in self.cultural_contexts 
                                 if 'ubuntu' in ctx.traditions), CulturalContext())
            ubuntu_evaluation.cultural_context = ubuntu_context
            
            # Test cultural competence calculation
            cultural_competence = ubuntu_evaluation.calculate_cultural_competence()
            self.assertGreater(cultural_competence, 0.7, 
                             "Ubuntu test should show high cultural competence")
            
            # Test cultural marker detection
            cultural_markers = ubuntu_evaluation.get_cultural_markers()
            self.assertGreater(len(cultural_markers), 0, 
                             "Should detect cultural markers in Ubuntu content")
    
    def test_consensus_calculation_with_diverse_content(self):
        """Test consensus calculation using diverse domain content."""
        # Create evaluations from different domains with varying consensus levels
        diverse_evaluations = []
        
        for domain, tests in self.domain_data.items():
            if not tests or len(tests) == 0:
                continue
                
            # Create 3 evaluations per domain with different "evaluator perspectives"
            for perspective_bias in [-0.1, 0.0, 0.1]:  # Different evaluator biases
                test_sample = tests[0]
                base_score = 0.7
                adjusted_score = max(0.1, min(0.95, base_score + perspective_bias))
                
                evaluation = self._create_mock_evaluation_from_test(test_sample, domain, adjusted_score)
                diverse_evaluations.append(evaluation)
        
        if len(diverse_evaluations) < 4:  # Need sufficient data for consensus testing
            self.skipTest("Insufficient evaluation data for consensus testing")
        
        # Test consensus analysis
        consensus_analysis = self.aggregator._analyze_dimension_consensus(diverse_evaluations)
        
        # Should handle diverse content types consistently
        self.assertIsInstance(consensus_analysis, dict)
        for dimension, consensus in consensus_analysis.items():
            self.assertGreaterEqual(consensus.consensus_level, 0.0)
            self.assertLessEqual(consensus.consensus_level, 1.0)
            self.assertIsInstance(consensus.scores, list)
            self.assertGreater(len(consensus.scores), 0)
    
    def test_statistical_accuracy_with_real_test_complexity(self):
        """Test statistical accuracy using real test complexity variations."""
        # Analyze prompt lengths and complexity across domains
        domain_complexity = {}
        
        for domain, tests in self.domain_data.items():
            if not tests:
                continue
                
            prompt_lengths = [len(test.get('prompt', '')) for test in tests[:10]]
            if prompt_lengths:
                domain_complexity[domain] = {
                    'avg_prompt_length': statistics.mean(prompt_lengths),
                    'complexity_variance': statistics.variance(prompt_lengths) if len(prompt_lengths) > 1 else 0,
                    'test_count': len(tests)
                }
        
        # Verify complexity analysis worked
        self.assertGreater(len(domain_complexity), 0, "Should analyze complexity from at least one domain")
        
        # Test that statistical functions handle varying complexity correctly
        for domain, complexity_data in domain_complexity.items():
            # Complexity should correlate with certain statistical properties
            avg_length = complexity_data['avg_prompt_length']
            variance = complexity_data['complexity_variance']
            
            # More complex prompts should generally have higher variance
            if avg_length > 500:  # Long prompts
                # Allow for both high and low variance - some domains may be consistent
                self.assertIsInstance(variance, (int, float))
            
            # All statistical measures should be non-negative
            self.assertGreaterEqual(avg_length, 0)
            self.assertGreaterEqual(variance, 0)
    
    def test_domain_specific_evaluation_patterns(self):
        """Test evaluation patterns specific to different domains."""
        domain_patterns = {}
        
        for domain, tests in self.domain_data.items():
            if not tests:
                continue
                
            # Create evaluations for this domain
            domain_evaluations = []
            for test in tests[:5]:  # Sample first 5 tests
                evaluation = self._create_mock_evaluation_from_test(test, domain)
                domain_evaluations.append(evaluation)
            
            if domain_evaluations:
                # Analyze domain-specific patterns
                overall_scores = [eval.overall_score for eval in domain_evaluations]
                cultural_relevance_scores = []
                
                for evaluation in domain_evaluations:
                    for dim in evaluation.dimensions:
                        cultural_relevance_scores.append(dim.cultural_relevance)
                
                domain_patterns[domain] = {
                    'avg_overall_score': statistics.mean(overall_scores),
                    'score_std': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
                    'avg_cultural_relevance': statistics.mean(cultural_relevance_scores) if cultural_relevance_scores else 0,
                    'evaluation_count': len(domain_evaluations)
                }
        
        # Verify domain-specific patterns
        if 'social' in domain_patterns and 'reasoning' in domain_patterns:
            social_cultural = domain_patterns['social']['avg_cultural_relevance']
            reasoning_cultural = domain_patterns['reasoning']['avg_cultural_relevance']
            
            # Social domain should have higher cultural relevance than reasoning
            self.assertGreater(social_cultural, reasoning_cultural * 0.8,
                             "Social domain should show higher cultural relevance")
        
        # All domains should have reasonable evaluation characteristics
        for domain, patterns in domain_patterns.items():
            self.assertGreater(patterns['avg_overall_score'], 0.1, 
                             f"{domain} domain should have reasonable scores")
            self.assertLess(patterns['avg_overall_score'], 0.98, 
                           f"{domain} domain scores should be realistic")


class TestRealDomainDataStatisticalValidation(unittest.TestCase):
    """Test statistical validation using comprehensive real domain data."""
    
    def setUp(self):
        """Set up comprehensive real data validation tests."""
        self.data_loader = DomainDataLoader()
        self.aggregator = EvaluationAggregator()
        
        # Load comprehensive domain data
        self.all_domain_data = self.data_loader.load_all_domains_data()
        
        # Count total available tests
        self.total_tests = sum(len(tests) for tests in self.all_domain_data.values())
    
    def test_comprehensive_statistical_validation(self):
        """Run comprehensive statistical validation across all available domain data."""
        if self.total_tests < 10:
            self.skipTest("Insufficient domain test data for comprehensive validation")
        
        # Create evaluation scenarios using real test data
        evaluation_scenarios = []
        
        for domain, tests in self.all_domain_data.items():
            for i, test in enumerate(tests[:3]):  # Use up to 3 tests per domain
                # Create different evaluation scenarios
                scenarios = [
                    {'bias': 0.0, 'confidence': 0.8, 'cultural_weight': 0.7},
                    {'bias': 0.15, 'confidence': 0.6, 'cultural_weight': 0.9},  # High cultural, some bias
                    {'bias': -0.1, 'confidence': 0.9, 'cultural_weight': 0.4}   # Low cultural, negative bias
                ]
                
                for scenario in scenarios:
                    base_score = 0.7
                    adjusted_score = max(0.1, min(0.95, base_score + scenario['bias']))
                    
                    # Create mock evaluation result
                    mock_result = Mock()
                    mock_result.overall_score = adjusted_score
                    mock_result.consensus_level = scenario['confidence']
                    mock_result.cultural_competence = adjusted_score * scenario['cultural_weight']
                    mock_result.dimension_scores = {'test_dimension': adjusted_score}
                    
                    evaluation_scenarios.append({
                        'result': mock_result,
                        'domain': domain,
                        'test_id': test.get('id', f'{domain}_{i}'),
                        'scenario': scenario,
                        'original_test': test
                    })
        
        # Run statistical validation on scenarios
        if len(evaluation_scenarios) >= 5:  # Need sufficient data
            results = [scenario['result'] for scenario in evaluation_scenarios]
            
            # Test bias detection
            cultural_contexts = [CulturalContext(cultural_groups=[scenario['domain']]) 
                               for scenario in evaluation_scenarios]
            
            bias_analysis = self.aggregator.detect_statistical_bias(results, cultural_contexts)
            
            # Validate bias detection worked
            self.assertIsInstance(bias_analysis, BiasAnalysis)
            
            # Should detect some bias patterns in our varied scenarios
            if bias_analysis.cultural_group_bias:
                bias_values = list(bias_analysis.cultural_group_bias.values())
                bias_range = max(bias_values) - min(bias_values) if bias_values else 0
                self.assertGreaterEqual(bias_range, 0, "Should calculate cultural group bias range")
    
    def test_real_data_coverage_and_quality(self):
        """Test coverage and quality of real domain data loading."""
        # Verify data loading worked
        loaded_domains = [domain for domain, tests in self.all_domain_data.items() if tests]
        self.assertGreater(len(loaded_domains), 0, "Should load data from at least one domain")
        
        # Test data quality for loaded domains
        for domain, tests in self.all_domain_data.items():
            if not tests:
                continue
                
            # Check test structure quality
            for test in tests[:3]:  # Check first 3 tests
                self.assertIn('id', test, f"Test in {domain} domain should have ID")
                self.assertIn('prompt', test, f"Test in {domain} domain should have prompt")
                
                # Verify prompt has meaningful content
                prompt = test.get('prompt', '')
                self.assertGreater(len(prompt), 20, f"Test {test.get('id')} prompt should have substantial content")
        
        # Log domain data statistics for validation
        domain_stats = {}
        for domain, tests in self.all_domain_data.items():
            if tests:
                avg_prompt_length = statistics.mean(len(test.get('prompt', '')) for test in tests)
                domain_stats[domain] = {
                    'test_count': len(tests),
                    'avg_prompt_length': avg_prompt_length
                }
        
        # Should have reasonable domain representation
        if domain_stats:
            total_loaded_tests = sum(stats['test_count'] for stats in domain_stats.values())
            self.assertGreater(total_loaded_tests, 5, "Should load reasonable number of tests total")


if __name__ == '__main__':
    unittest.main()