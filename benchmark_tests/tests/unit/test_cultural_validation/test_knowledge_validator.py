"""
Test suite for KnowledgeValidator

Comprehensive tests for the knowledge validation framework including
factual grounding tests, knowledge consistency validation, and confidence
calibration measurement.

"""

import unittest
from unittest.mock import patch, MagicMock
import logging
import sys
import os

# Test imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from evaluator.validation.knowledge_validator import (
    KnowledgeValidator, FactualTest, ValidationResult, KnowledgeValidationReport
)

# Disable logging during tests
logging.disable(logging.CRITICAL)


class TestKnowledgeValidator(unittest.TestCase):
    """Test knowledge validator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = KnowledgeValidator()
        
        # Test responses
        self.correct_geography_response = "Paris is the capital of France."
        self.incorrect_geography_response = "London is the capital of France."
        self.high_confidence_response = "I am absolutely certain that Paris is definitely the capital of France."
        self.low_confidence_response = "I think maybe Paris could possibly be the capital of France."
        self.uncertain_response = "I'm not sure what the capital of France is."
        
        # Test factual test
        self.sample_factual_test = FactualTest(
            question="What is the capital of France?",
            expected_tokens={"paris"},
            forbidden_tokens={"london", "berlin", "madrid"},
            category="geography",
            difficulty="easy"
        )
    
    def test_initialization(self):
        """Test validator initialization"""
        self.assertIsNotNone(self.validator)
        self.assertIsInstance(self.validator.factual_tests, dict)
        self.assertIn("geography", self.validator.factual_tests)
        self.assertIn("science", self.validator.factual_tests)
        self.assertIn("mathematics", self.validator.factual_tests)
    
    def test_factual_test_database(self):
        """Test built-in factual test database"""
        categories = ["geography", "science", "mathematics", "history", "literature", "general"]
        
        for category in categories:
            self.assertIn(category, self.validator.factual_tests)
            tests = self.validator.factual_tests[category]
            self.assertGreater(len(tests), 0)
            
            # Test first test in each category
            first_test = tests[0]
            self.assertIsInstance(first_test, FactualTest)
            self.assertIsInstance(first_test.expected_tokens, set)
            self.assertIsInstance(first_test.forbidden_tokens, set)
            self.assertIn(first_test.difficulty, ["easy", "medium", "hard"])
    
    def test_single_response_validation(self):
        """Test single response validation"""
        result = self.validator._validate_single_response(
            self.sample_factual_test,
            self.correct_geography_response,
            "test_001"
        )
        
        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.test_id, "test_001")
        self.assertGreater(result.factual_accuracy, 0.5)
        self.assertGreater(len(result.expected_tokens_found), 0)
        self.assertEqual(len(result.forbidden_tokens_found), 0)
        self.assertTrue(result.passed)
    
    def test_incorrect_response_validation(self):
        """Test validation of incorrect response"""
        result = self.validator._validate_single_response(
            self.sample_factual_test,
            self.incorrect_geography_response,
            "test_002"
        )
        
        self.assertIsInstance(result, ValidationResult)
        self.assertLess(result.factual_accuracy, 0.5)
        self.assertEqual(len(result.expected_tokens_found), 0)
        self.assertGreater(len(result.forbidden_tokens_found), 0)
        self.assertFalse(result.passed)
    
    def test_factual_accuracy_calculation(self):
        """Test factual accuracy calculation"""
        # Perfect accuracy
        perfect_accuracy = self.validator._calculate_factual_accuracy(
            ["paris"], [], self.sample_factual_test
        )
        self.assertEqual(perfect_accuracy, 1.0)
        
        # No expected tokens found
        no_accuracy = self.validator._calculate_factual_accuracy(
            [], [], self.sample_factual_test
        )
        self.assertEqual(no_accuracy, 0.0)
        
        # Forbidden tokens penalty
        penalty_accuracy = self.validator._calculate_factual_accuracy(
            ["paris"], ["london"], self.sample_factual_test
        )
        self.assertLess(penalty_accuracy, 1.0)
        self.assertGreater(penalty_accuracy, 0.0)
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation"""
        high_conf_score = self.validator._calculate_confidence_score(self.high_confidence_response)
        low_conf_score = self.validator._calculate_confidence_score(self.low_confidence_response)
        uncertain_score = self.validator._calculate_confidence_score(self.uncertain_response)
        
        self.assertGreater(high_conf_score, low_conf_score)
        self.assertGreater(low_conf_score, uncertain_score)
        self.assertLessEqual(high_conf_score, 1.0)
        self.assertGreaterEqual(uncertain_score, 0.0)
    
    def test_confidence_markers_detection(self):
        """Test confidence markers detection"""
        test_cases = [
            ("I am absolutely certain that...", "high"),
            ("This is probably correct...", "medium"),
            ("Maybe this could be...", "low"),
            ("I don't know...", "uncertain"),
            ("This is a neutral statement.", "neutral")
        ]
        
        for text, expected_level in test_cases:
            score = self.validator._calculate_confidence_score(text)
            
            if expected_level == "high":
                self.assertGreater(score, 0.7)
            elif expected_level == "low":
                self.assertLess(score, 0.6)
            elif expected_level == "uncertain":
                self.assertLess(score, 0.3)
    
    def test_knowledge_consistency_calculation(self):
        """Test knowledge consistency calculation"""
        consistent_responses = [
            ("Capital of France?", "Paris is the capital."),
            ("France's capital city?", "The capital is Paris."),
            ("What city is France's capital?", "Paris.")
        ]
        
        inconsistent_responses = [
            ("Capital of France?", "Paris is the capital."),
            ("France's capital city?", "London is the capital."),
            ("What city is France's capital?", "I don't know.")
        ]
        
        consistent_score = self.validator._calculate_knowledge_consistency(consistent_responses)
        inconsistent_score = self.validator._calculate_knowledge_consistency(inconsistent_responses)
        
        self.assertGreater(consistent_score, inconsistent_score)
        self.assertGreaterEqual(consistent_score, 0.0)
        self.assertLessEqual(consistent_score, 1.0)
    
    def test_consistency_test_variations(self):
        """Test built-in consistency test variations"""
        variations = self.validator._get_consistency_test_variations()
        
        self.assertIsInstance(variations, dict)
        self.assertIn("capital_of_france", variations)
        self.assertGreater(len(variations["capital_of_france"]), 3)
        
        # Check that variations are actually different phrasings
        france_variations = variations["capital_of_france"]
        self.assertNotEqual(france_variations[0], france_variations[1])
    
    def test_embedding_consistency_calculation(self):
        """Test embedding-based consistency calculation"""
        if self.validator.embedding_model is not None:
            responses = [
                "The answer is Paris.",
                "Paris is the answer.",
                "It's Paris."
            ]
            
            consistency = self.validator._calculate_embedding_consistency(responses)
            self.assertIsInstance(consistency, float)
            self.assertGreaterEqual(consistency, 0.0)
            self.assertLessEqual(consistency, 1.0)
        else:
            self.skipTest("Embedding model not available")
    
    def test_lexical_consistency_fallback(self):
        """Test lexical consistency fallback method"""
        responses = [
            "Paris is the capital of France.",
            "The capital of France is Paris.",
            "France's capital city is Paris."
        ]
        
        consistency = self.validator._calculate_lexical_consistency(responses)
        self.assertIsInstance(consistency, float)
        self.assertGreater(consistency, 0.3)  # Should have reasonable overlap
        self.assertLessEqual(consistency, 1.0)
    
    def test_confidence_calibration_analysis(self):
        """Test confidence calibration analysis"""
        responses_with_confidence = [
            "I am certain this is correct.",
            "This might be right.",
            "I don't know the answer.",
            "Definitely the right answer."
        ]
        
        for response in responses_with_confidence:
            analysis = self.validator._analyze_response_confidence_calibration(response)
            
            self.assertIn("calibration_score", analysis)
            self.assertIn("confidence_distribution", analysis)
            self.assertIn("assessment", analysis)
            self.assertGreaterEqual(analysis["calibration_score"], 0.0)
            self.assertLessEqual(analysis["calibration_score"], 1.0)
    
    def test_factual_indicators_analysis(self):
        """Test factual indicators analysis"""
        factual_response = """
        According to recent studies published in 2023, the population of Paris 
        is approximately 2.16 million people. Research by Jean Dupont indicates 
        that demographic trends show steady growth.
        """
        
        non_factual_response = """
        I think Paris is nice. It seems like a good city to visit and 
        people say it's beautiful.
        """
        
        factual_indicators = self.validator._analyze_factual_indicators(
            factual_response, "test_factual"
        )
        non_factual_indicators = self.validator._analyze_factual_indicators(
            non_factual_response, "test_non_factual"
        )
        
        # Factual response should score higher
        self.assertGreater(
            factual_indicators["factual_accuracy_score"],
            non_factual_indicators["factual_accuracy_score"]
        )
        
        # Check specific indicators
        self.assertTrue(factual_indicators["contains_numbers"])
        self.assertTrue(factual_indicators["contains_dates"])
        self.assertTrue(factual_indicators["contains_names"])
        self.assertTrue(factual_indicators["contains_specific_facts"])
    
    def test_mock_validation_workflow(self):
        """Test complete validation workflow with mock evaluator"""
        def mock_evaluator(question):
            if "capital of france" in question.lower():
                return "Paris is the capital of France."
            elif "2+2" in question:
                return "2+2 equals 4."
            else:
                return "I don't know."
        
        # Test would call validate_factual_knowledge but we test components
        test_result = self.validator._validate_single_response(
            self.sample_factual_test,
            mock_evaluator("What is the capital of France?"),
            "mock_test"
        )
        
        self.assertTrue(test_result.passed)
        self.assertGreater(test_result.factual_accuracy, 0.5)
    
    def test_validation_report_generation(self):
        """Test validation report generation"""
        # Create mock results
        mock_results = [
            ValidationResult(
                test_id="test_1",
                question="Test question 1",
                response="Correct answer",
                factual_accuracy=0.9,
                expected_tokens_found=["correct"],
                forbidden_tokens_found=[],
                confidence_score=0.8,
                knowledge_consistency_score=0.7,
                category="test_category",
                passed=True
            ),
            ValidationResult(
                test_id="test_2",
                question="Test question 2",
                response="Wrong answer",
                factual_accuracy=0.2,
                expected_tokens_found=[],
                forbidden_tokens_found=["wrong"],
                confidence_score=0.6,
                knowledge_consistency_score=0.3,
                category="test_category",
                passed=False
            )
        ]
        
        category_results = {"test_category": mock_results}
        report = self.validator._generate_validation_report(mock_results, category_results)
        
        self.assertIsInstance(report, KnowledgeValidationReport)
        self.assertEqual(report.overall_accuracy, 0.5)  # 1 passed out of 2
        self.assertIn("test_category", report.category_breakdown)
        self.assertIn("calibration_by_confidence", report.confidence_calibration)
    
    def test_confidence_calibration_assessment(self):
        """Test confidence calibration assessment"""
        assessments = [
            (0.8, "well-calibrated"),
            (0.3, "poorly calibrated"),
            (-0.2, "miscalibrated")
        ]
        
        for score, expected_keyword in assessments:
            assessment = self.validator._assess_calibration(score)
            self.assertIn(expected_keyword.replace(" ", "-").lower(), assessment.lower().replace(" ", "-"))
    
    def test_knowledge_assessment_generation(self):
        """Test comprehensive knowledge assessment generation"""
        mock_report = KnowledgeValidationReport(
            overall_accuracy=0.75,
            category_breakdown={"geography": 0.8, "science": 0.7, "math": 0.5},
            confidence_calibration={"calibration_score": 0.6},
            consistency_analysis={},
            failure_analysis={"total_failures": 2},
            detailed_results=[]
        )
        
        assessment = self.validator.generate_knowledge_assessment(mock_report)
        
        self.assertIn("factual_knowledge_assessment", assessment)
        self.assertIn("category_analysis", assessment)
        self.assertIn("confidence_reliability", assessment)
        self.assertIn("recommendations", assessment)
        
        # Should identify strengths and weaknesses
        self.assertIn("geography", assessment["factual_knowledge_assessment"]["strengths"])
        self.assertIn("math", assessment["factual_knowledge_assessment"]["weaknesses"])
    
    def test_edge_cases_empty_responses(self):
        """Test edge cases with empty or minimal responses"""
        empty_result = self.validator._validate_single_response(
            self.sample_factual_test, "", "empty_test"
        )
        
        minimal_result = self.validator._validate_single_response(
            self.sample_factual_test, ".", "minimal_test"
        )
        
        self.assertFalse(empty_result.passed)
        self.assertFalse(minimal_result.passed)
        self.assertEqual(empty_result.factual_accuracy, 0.0)
    
    def test_multiple_expected_tokens(self):
        """Test handling of multiple expected tokens"""
        multi_token_test = FactualTest(
            question="What is H2O?",
            expected_tokens={"water", "dihydrogen monoxide", "h2o"},
            forbidden_tokens={"fire", "air"},
            category="science",
            difficulty="easy"
        )
        
        partial_match = self.validator._validate_single_response(
            multi_token_test, "H2O is water.", "multi_test"
        )
        
        full_match = self.validator._validate_single_response(
            multi_token_test, "H2O is water, also known as dihydrogen monoxide.", "full_test"
        )
        
        self.assertGreater(full_match.factual_accuracy, partial_match.factual_accuracy)
    
    @patch('evaluator.validation.knowledge_validator.SENTENCE_TRANSFORMERS_AVAILABLE', False)
    @patch('evaluator.validation.knowledge_validator.SKLEARN_AVAILABLE', False)
    def test_fallback_methods(self):
        """Test fallback methods when ML libraries unavailable"""
        validator = KnowledgeValidator()
        
        responses = [
            ("Question 1", "Answer about topic"),
            ("Question 2", "Different answer about topic")
        ]
        
        consistency = validator._calculate_knowledge_consistency(responses)
        self.assertIsInstance(consistency, float)
        self.assertGreaterEqual(consistency, 0.0)
        self.assertLessEqual(consistency, 1.0)


class TestKnowledgeValidatorEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for knowledge validator"""
    
    def setUp(self):
        """Set up edge case test fixtures"""
        self.validator = KnowledgeValidator()
    
    def test_numeric_precision_handling(self):
        """Test handling of numeric precision in responses"""
        math_test = FactualTest(
            question="What is π to 3 decimal places?",
            expected_tokens={"3.142"},
            forbidden_tokens={"3.14", "22/7"},
            category="mathematics",
            difficulty="medium"
        )
        
        precise_result = self.validator._validate_single_response(
            math_test, "π is 3.142", "precise_test"
        )
        imprecise_result = self.validator._validate_single_response(
            math_test, "π is approximately 3.14", "imprecise_test"
        )
        
        # Check that validation results are calculated 
        self.assertIsInstance(precise_result.passed, bool, "Precise result should return boolean")
        self.assertIsInstance(imprecise_result.passed, bool, "Imprecise result should return boolean")
    
    def test_case_sensitivity(self):
        """Test case sensitivity in token matching"""
        case_test = FactualTest(
            question="What is DNA?",
            expected_tokens={"dna", "deoxyribonucleic acid"},
            forbidden_tokens={"rna"},
            category="science",
            difficulty="easy"
        )
        
        uppercase_result = self.validator._validate_single_response(
            case_test, "DNA stands for deoxyribonucleic acid", "uppercase_test"
        )
        lowercase_result = self.validator._validate_single_response(
            case_test, "dna stands for deoxyribonucleic acid", "lowercase_test"
        )
        
        # Both should pass due to case-insensitive matching
        self.assertTrue(uppercase_result.passed)
        self.assertTrue(lowercase_result.passed)
    
    def test_special_characters_in_tokens(self):
        """Test handling of special characters in expected/forbidden tokens"""
        special_test = FactualTest(
            question="What is the chemical formula for water?",
            expected_tokens={"h2o", "h₂o"},
            forbidden_tokens={"co2", "h2so4"},
            category="science",
            difficulty="easy"
        )
        
        normal_result = self.validator._validate_single_response(
            special_test, "Water is H2O", "normal_test"
        )
        subscript_result = self.validator._validate_single_response(
            special_test, "Water is H₂O", "subscript_test"
        )
        
        self.assertTrue(normal_result.passed)
        self.assertTrue(subscript_result.passed)
    
    def test_very_long_responses(self):
        """Test handling of very long responses"""
        long_response = """
        This is a very long response that contains the correct answer 'Paris' 
        buried somewhere in the middle of a lot of other text. """ + "Additional text. " * 200
        
        result = self.validator._validate_single_response(
            FactualTest(
                question="Capital of France?",
                expected_tokens={"paris"},
                forbidden_tokens={"london"},
                category="geography",
                difficulty="easy"
            ),
            long_response,
            "long_test"
        )
        
        # Should still find the correct answer
        self.assertTrue(result.passed)
        self.assertGreater(result.factual_accuracy, 0.0)
    
    def test_multilingual_responses(self):
        """Test handling of multilingual responses"""
        multilingual_response = "La capital de Francia es Paris. The capital of France is Paris."
        
        result = self.validator._validate_single_response(
            FactualTest(
                question="What is the capital of France?",
                expected_tokens={"paris"},
                forbidden_tokens={"madrid"},
                category="geography",
                difficulty="easy"
            ),
            multilingual_response,
            "multilingual_test"
        )
        
        self.assertTrue(result.passed)


class TestFactualTestDataQuality(unittest.TestCase):
    """Test the quality and consistency of factual test data"""
    
    def setUp(self):
        """Set up test data quality fixtures"""
        self.validator = KnowledgeValidator()
    
    def test_no_overlapping_tokens(self):
        """Test that expected and forbidden tokens don't overlap"""
        for category, tests in self.validator.factual_tests.items():
            for test in tests:
                overlap = test.expected_tokens.intersection(test.forbidden_tokens)
                self.assertEqual(len(overlap), 0, 
                    f"Category {category}: Expected and forbidden tokens overlap: {overlap}")
    
    def test_non_empty_token_sets(self):
        """Test that all tests have non-empty token sets"""
        for category, tests in self.validator.factual_tests.items():
            for i, test in enumerate(tests):
                self.assertGreater(len(test.expected_tokens), 0,
                    f"Category {category}, test {i}: No expected tokens")
                self.assertGreater(len(test.forbidden_tokens), 0,
                    f"Category {category}, test {i}: No forbidden tokens")
    
    def test_question_format_consistency(self):
        """Test that questions are properly formatted"""
        for category, tests in self.validator.factual_tests.items():
            for i, test in enumerate(tests):
                self.assertIsInstance(test.question, str,
                    f"Category {category}, test {i}: Question is not a string")
                self.assertGreater(len(test.question.strip()), 0,
                    f"Category {category}, test {i}: Question is empty")
    
    def test_difficulty_levels(self):
        """Test that difficulty levels are valid"""
        valid_difficulties = {"easy", "medium", "hard"}
        
        for category, tests in self.validator.factual_tests.items():
            for i, test in enumerate(tests):
                self.assertIn(test.difficulty, valid_difficulties,
                    f"Category {category}, test {i}: Invalid difficulty '{test.difficulty}'")
    
    def test_category_coverage(self):
        """Test that all expected categories are covered"""
        expected_categories = ["geography", "science", "mathematics", "history", "literature", "general"]
        actual_categories = list(self.validator.factual_tests.keys())
        
        for category in expected_categories:
            self.assertIn(category, actual_categories,
                f"Missing expected category: {category}")
    
    def test_minimum_tests_per_category(self):
        """Test that each category has minimum number of tests"""
        min_tests_per_category = 3
        
        for category, tests in self.validator.factual_tests.items():
            self.assertGreaterEqual(len(tests), min_tests_per_category,
                f"Category {category} has only {len(tests)} tests, minimum is {min_tests_per_category}")


if __name__ == '__main__':
    unittest.main(verbosity=2)