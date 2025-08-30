"""
Unit tests for evaluation configuration.

Tests the structure, values, and consistency of evaluation configuration settings,
including thresholds, weights, patterns, and specialized configurations.
"""

import unittest
from typing import Dict, Any

from evaluator.core.evaluation_config import (
    DEFAULT_CONFIG,
    FAST_CONFIG,
    DETAILED_CONFIG,
    REASONING_TYPE_PRESETS,
    ScoreThresholds,
    UniversalWeights
)


class TestScoreThresholds(unittest.TestCase):
    """Test cases for ScoreThresholds class."""
    
    def test_score_threshold_values(self):
        """Test that score thresholds have correct values and ordering."""
        self.assertEqual(ScoreThresholds.EXCELLENT, 85.0)
        self.assertEqual(ScoreThresholds.GOOD, 70.0)
        self.assertEqual(ScoreThresholds.SATISFACTORY, 55.0)
        self.assertEqual(ScoreThresholds.POOR, 40.0)
        self.assertEqual(ScoreThresholds.VERY_POOR, 25.0)
        
        # Test ordering (descending)
        self.assertGreater(ScoreThresholds.EXCELLENT, ScoreThresholds.GOOD)
        self.assertGreater(ScoreThresholds.GOOD, ScoreThresholds.SATISFACTORY)
        self.assertGreater(ScoreThresholds.SATISFACTORY, ScoreThresholds.POOR)
        self.assertGreater(ScoreThresholds.POOR, ScoreThresholds.VERY_POOR)


class TestUniversalWeights(unittest.TestCase):
    """Test cases for UniversalWeights class."""
    
    def test_universal_weight_values(self):
        """Test that universal weights are properly defined."""
        self.assertEqual(UniversalWeights.ORGANIZATION_QUALITY, 0.15)
        self.assertEqual(UniversalWeights.TECHNICAL_ACCURACY, 0.20)
        self.assertEqual(UniversalWeights.COMPLETENESS, 0.15)
        self.assertEqual(UniversalWeights.THOROUGHNESS, 0.15)
        self.assertEqual(UniversalWeights.RELIABILITY, 0.10)
        self.assertEqual(UniversalWeights.SCOPE_COVERAGE, 0.10)
        self.assertEqual(UniversalWeights.DOMAIN_APPROPRIATENESS, 0.15)
    
    def test_universal_weights_sum_to_one(self):
        """Test that universal weights sum to 1.0."""
        total_weight = (
            UniversalWeights.ORGANIZATION_QUALITY +
            UniversalWeights.TECHNICAL_ACCURACY +
            UniversalWeights.COMPLETENESS +
            UniversalWeights.THOROUGHNESS +
            UniversalWeights.RELIABILITY +
            UniversalWeights.SCOPE_COVERAGE +
            UniversalWeights.DOMAIN_APPROPRIATENESS
        )
        self.assertAlmostEqual(total_weight, 1.0, places=3)


class TestDefaultConfig(unittest.TestCase):
    """Test cases for DEFAULT_CONFIG."""
    
    def test_default_config_structure(self):
        """Test that DEFAULT_CONFIG has required top-level keys."""
        required_keys = [
            "weights", "thresholds", "text_analysis", "reasoning_type_configs",
            "test_type_configs", "domain_patterns", "linguistic_patterns",
            "quantitative_patterns", "advanced_analysis", "llm_evaluation",
            "coherence_detection", "domain_adaptation", "reporting",
            "edge_case_detection", "advanced_metrics_scoring",
            "consistency_validation_thresholds", "consistency_test_config",
            "knowledge_validation_config", "cultural_evaluation_thresholds",
            "cultural_evaluation_config", "domain_evaluation_config",
            "model_profiles"
        ]
        
        for key in required_keys:
            self.assertIn(key, DEFAULT_CONFIG, f"Missing key: {key}")
    
    def test_weights_structure(self):
        """Test the weights section structure."""
        weights = DEFAULT_CONFIG["weights"]
        
        required_weight_keys = [
            "organization_quality", "technical_accuracy", "completeness",
            "thoroughness", "reliability", "scope_coverage", "domain_appropriateness"
        ]
        
        for key in required_weight_keys:
            self.assertIn(key, weights)
            self.assertIsInstance(weights[key], (int, float))
            self.assertGreaterEqual(weights[key], 0.0)
            self.assertLessEqual(weights[key], 1.0)
        
        # Test weights sum to 1.0
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=3)
    
    def test_thresholds_structure(self):
        """Test the thresholds section structure."""
        thresholds = DEFAULT_CONFIG["thresholds"]
        
        required_threshold_keys = [
            "excellent_score", "good_score", "satisfactory_score", "poor_score",
            "minimum_word_count", "confidence_threshold", "coherence_failure_threshold",
            "coherence_penalty_threshold", "repetitive_loop_threshold",
            "meta_reasoning_threshold", "technical_conciseness_threshold",
            "creative_elaboration_threshold"
        ]
        
        for key in required_threshold_keys:
            self.assertIn(key, thresholds)
            self.assertIsInstance(thresholds[key], (int, float))
    
    def test_reasoning_type_configs_structure(self):
        """Test reasoning type configurations."""
        reasoning_configs = DEFAULT_CONFIG["reasoning_type_configs"]
        
        expected_types = ["chain_of_thought", "multi_hop", "verification", "mathematical", "backward", "scaffolded"]
        
        for reasoning_type in expected_types:
            self.assertIn(reasoning_type, reasoning_configs)
            config = reasoning_configs[reasoning_type]
            
            # Check required keys
            self.assertIn("weights", config)
            self.assertIn("required_patterns", config)
            self.assertIn("bonus_multiplier", config)
            
            # Check weights structure
            weights = config["weights"]
            self.assertIsInstance(weights, dict)
            total_weight = sum(weights.values())
            self.assertAlmostEqual(total_weight, 1.0, places=3)
            
            # Check patterns and multiplier
            self.assertIsInstance(config["required_patterns"], list)
            self.assertGreater(len(config["required_patterns"]), 0)
            self.assertIsInstance(config["bonus_multiplier"], (int, float))
            self.assertGreaterEqual(config["bonus_multiplier"], 1.0)
    
    def test_test_type_configs_structure(self):
        """Test test type configurations."""
        test_configs = DEFAULT_CONFIG["test_type_configs"]
        
        expected_types = ["linux", "creative", "reasoning"]
        
        for test_type in expected_types:
            self.assertIn(test_type, test_configs)
            config = test_configs[test_type]
            
            # Check required keys
            self.assertIn("weights", config)
            self.assertIn("keywords", config)
            self.assertIn("bonus_multiplier", config)
            
            # Check weights
            weights = config["weights"]
            self.assertIsInstance(weights, dict)
            total_weight = sum(weights.values())
            self.assertAlmostEqual(total_weight, 1.0, places=3)
    
    def test_domain_patterns_structure(self):
        """Test domain patterns configuration."""
        domain_patterns = DEFAULT_CONFIG["domain_patterns"]
        
        expected_domains = ["medical", "legal", "financial", "scientific", "engineering"]
        
        for domain in expected_domains:
            self.assertIn(domain, domain_patterns)
            config = domain_patterns[domain]
            
            # Check required keys
            required_keys = ["keywords", "technical_terms", "reasoning_patterns", "quality_indicators"]
            for key in required_keys:
                self.assertIn(key, config)
                self.assertIsInstance(config[key], list)
                self.assertGreater(len(config[key]), 0)
    
    def test_linguistic_patterns_structure(self):
        """Test linguistic patterns configuration."""
        linguistic_patterns = DEFAULT_CONFIG["linguistic_patterns"]
        
        expected_patterns = [
            "hedging_sophisticated", "hedging_basic", "certainty_high", "certainty_medium",
            "meta_cognitive", "self_correction"
        ]
        
        for pattern in expected_patterns:
            self.assertIn(pattern, linguistic_patterns)
            self.assertIsInstance(linguistic_patterns[pattern], list)
            self.assertGreater(len(linguistic_patterns[pattern]), 0)
    
    def test_advanced_analysis_structure(self):
        """Test advanced analysis configuration."""
        advanced_analysis = DEFAULT_CONFIG["advanced_analysis"]
        
        expected_sections = ["entropy_analysis", "semantic_coherence", "context_analysis", "quantization_analysis"]
        
        for section in expected_sections:
            self.assertIn(section, advanced_analysis)
            config = advanced_analysis[section]
            
            self.assertIn("enabled", config)
            self.assertIsInstance(config["enabled"], bool)
            
            # Each section should have weight parameters
            weight_keys = [key for key in config.keys() if "weight" in key]
            self.assertGreater(len(weight_keys), 0)
    
    def test_cultural_evaluation_structure(self):
        """Test cultural evaluation configuration."""
        cultural_config = DEFAULT_CONFIG["cultural_evaluation_config"]
        
        self.assertIn("enabled", cultural_config)
        self.assertIn("cultural_domains", cultural_config)
        self.assertIn("knowledge_domains", cultural_config)
        self.assertIn("authenticity_analysis", cultural_config)
        self.assertIn("tradition_validation", cultural_config)
        self.assertIn("coherence_checking", cultural_config)
        
        # Check that domains are lists
        self.assertIsInstance(cultural_config["cultural_domains"], list)
        self.assertIsInstance(cultural_config["knowledge_domains"], list)
        self.assertGreater(len(cultural_config["cultural_domains"]), 0)
        self.assertGreater(len(cultural_config["knowledge_domains"]), 0)
    
    def test_domain_evaluation_config_structure(self):
        """Test domain evaluation configuration."""
        domain_config = DEFAULT_CONFIG["domain_evaluation_config"]
        
        self.assertIn("enabled", domain_config)
        self.assertIn("router_config", domain_config)
        self.assertIn("aggregation_config", domain_config)
        self.assertIn("domain_settings", domain_config)
        self.assertIn("cultural_pattern_libraries", domain_config)
        
        # Check domain settings
        domain_settings = domain_config["domain_settings"]
        expected_domains = ["creativity", "language", "social", "reasoning", "knowledge", "integration"]
        
        for domain in expected_domains:
            self.assertIn(domain, domain_settings)
            settings = domain_settings[domain]
            
            self.assertIn("evaluator_class", settings)
            self.assertIn("dimensions", settings)
            self.assertIsInstance(settings["dimensions"], list)
            self.assertGreater(len(settings["dimensions"]), 0)
    
    def test_model_profiles_structure(self):
        """Test model profiles configuration."""
        model_profiles = DEFAULT_CONFIG["model_profiles"]
        
        expected_models = ["gpt_oss_20b", "claude_sonnet", "llama_70b", "qwen3_30b"]
        
        for model in expected_models:
            self.assertIn(model, model_profiles)
            profile = model_profiles[model]
            
            # Check expected sections
            self.assertIn("entropy_expectations", profile)
            self.assertIn("context_expectations", profile)
            self.assertIn("consistency_expectations", profile)
            self.assertIn("validation_expectations", profile)
            
            # Check that values are reasonable
            entropy_exp = profile["entropy_expectations"]
            self.assertIn("token_entropy_range", entropy_exp)
            self.assertIsInstance(entropy_exp["token_entropy_range"], list)
            self.assertEqual(len(entropy_exp["token_entropy_range"]), 2)


class TestSpecializedConfigs(unittest.TestCase):
    """Test cases for specialized configuration variants."""
    
    def test_fast_config_structure(self):
        """Test that FAST_CONFIG has proper structure."""
        # Should inherit from DEFAULT_CONFIG
        self.assertIn("weights", FAST_CONFIG)
        self.assertIn("thresholds", FAST_CONFIG)
        self.assertIn("text_analysis", FAST_CONFIG)
        
        # Check that weights still sum to 1.0
        weights = FAST_CONFIG["weights"]
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=3)
        
        # Should have modified text analysis settings for speed
        text_analysis = FAST_CONFIG["text_analysis"]
        self.assertIn("step_indicator_weight", text_analysis)
        self.assertIn("functional_completion_weight", text_analysis)
    
    def test_detailed_config_structure(self):
        """Test that DETAILED_CONFIG has proper structure."""
        # Should inherit from DEFAULT_CONFIG
        self.assertIn("weights", DETAILED_CONFIG)
        self.assertIn("thresholds", DETAILED_CONFIG)
        
        # Check weights
        weights = DETAILED_CONFIG["weights"]
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=3)
        
        # Should have modified coherence penalties
        coherence_config = DETAILED_CONFIG["coherence_detection"]
        self.assertIn("coherence_failure_penalties", coherence_config)
        penalties = coherence_config["coherence_failure_penalties"]
        
        # Penalties should be higher in detailed mode
        default_penalties = DEFAULT_CONFIG["coherence_detection"]["coherence_failure_penalties"]
        for key in penalties:
            if key in default_penalties:
                self.assertGreaterEqual(penalties[key], default_penalties[key])
    
    def test_reasoning_type_presets(self):
        """Test reasoning type preset configurations."""
        expected_presets = ["academic_research", "business_analysis", "educational_assessment"]
        
        for preset in expected_presets:
            self.assertIn(preset, REASONING_TYPE_PRESETS)
            config = REASONING_TYPE_PRESETS[preset]
            
            # Should have weights
            self.assertIn("weights", config)
            weights = config["weights"]
            total_weight = sum(weights.values())
            self.assertAlmostEqual(total_weight, 1.0, places=3)
            
            # Should have thresholds
            if "thresholds" in config:
                thresholds = config["thresholds"]
                self.assertIsInstance(thresholds, dict)


class TestConfigurationConsistency(unittest.TestCase):
    """Test cases for configuration consistency and validity."""
    
    def test_weight_consistency_across_configs(self):
        """Test that all weight configurations sum to 1.0."""
        configs_to_test = [DEFAULT_CONFIG, FAST_CONFIG, DETAILED_CONFIG]
        
        for config in configs_to_test:
            weights = config["weights"]
            total_weight = sum(weights.values())
            self.assertAlmostEqual(total_weight, 1.0, places=3, 
                                 msg=f"Weights don't sum to 1.0 in config: {weights}")
    
    def test_threshold_ordering(self):
        """Test that threshold values are properly ordered."""
        thresholds = DEFAULT_CONFIG["thresholds"]
        
        # Score thresholds should be in descending order
        self.assertGreater(thresholds["excellent_score"], thresholds["good_score"])
        self.assertGreater(thresholds["good_score"], thresholds["satisfactory_score"])
        self.assertGreater(thresholds["satisfactory_score"], thresholds["poor_score"])
    
    def test_bonus_multipliers_valid(self):
        """Test that bonus multipliers are >= 1.0."""
        reasoning_configs = DEFAULT_CONFIG["reasoning_type_configs"]
        
        for reasoning_type, config in reasoning_configs.items():
            bonus_multiplier = config["bonus_multiplier"]
            self.assertGreaterEqual(bonus_multiplier, 1.0, 
                                  f"Invalid bonus multiplier for {reasoning_type}: {bonus_multiplier}")
    
    def test_pattern_lists_non_empty(self):
        """Test that pattern lists are non-empty where required."""
        # Check linguistic patterns
        linguistic_patterns = DEFAULT_CONFIG["linguistic_patterns"]
        for pattern_type, patterns in linguistic_patterns.items():
            self.assertGreater(len(patterns), 0, f"Empty pattern list: {pattern_type}")
        
        # Check domain patterns
        domain_patterns = DEFAULT_CONFIG["domain_patterns"]
        for domain, config in domain_patterns.items():
            for pattern_type, patterns in config.items():
                self.assertGreater(len(patterns), 0, 
                                 f"Empty pattern list: {domain}.{pattern_type}")
    
    def test_enabled_flags_are_boolean(self):
        """Test that enabled flags are boolean values."""
        def check_enabled_flags(config_dict, prefix=""):
            for key, value in config_dict.items():
                if key == "enabled":
                    self.assertIsInstance(value, bool, 
                                        f"Non-boolean enabled flag at {prefix}.{key}")
                elif isinstance(value, dict):
                    check_enabled_flags(value, f"{prefix}.{key}")
        
        check_enabled_flags(DEFAULT_CONFIG)
    
    def test_score_ranges_valid(self):
        """Test that score ranges and thresholds are valid."""
        # Check advanced metrics scoring
        advanced_metrics = DEFAULT_CONFIG["advanced_metrics_scoring"]
        
        for category, thresholds in advanced_metrics.items():
            if isinstance(thresholds, dict):
                for threshold_name, value in thresholds.items():
                    if isinstance(value, (int, float)):
                        # Bonuses should be positive, penalties negative
                        if "bonus" in threshold_name:
                            self.assertGreater(value, 0, f"Negative bonus: {category}.{threshold_name}")
                        elif "penalty" in threshold_name:
                            self.assertLess(value, 0, f"Positive penalty: {category}.{threshold_name}")
    
    def test_cultural_thresholds_ordering(self):
        """Test that cultural evaluation thresholds are properly ordered."""
        cultural_thresholds = DEFAULT_CONFIG["cultural_evaluation_thresholds"]
        
        for category, thresholds in cultural_thresholds.items():
            if isinstance(thresholds, dict) and "excellent" in thresholds:
                # Test descending order for positive metrics
                if category in ["cultural_authenticity", "tradition_respect", "cross_cultural_coherence"]:
                    if "good" in thresholds:
                        self.assertGreater(thresholds["excellent"], thresholds["good"])
                    if "acceptable" in thresholds and "good" in thresholds:
                        self.assertGreater(thresholds["good"], thresholds["acceptable"])
    
    def test_model_profile_baselines_valid(self):
        """Test that model profile baseline values are valid."""
        model_profiles = DEFAULT_CONFIG["model_profiles"]
        
        for model_name, profile in model_profiles.items():
            # Check entropy expectations
            if "entropy_expectations" in profile:
                entropy_exp = profile["entropy_expectations"]
                if "token_entropy_range" in entropy_exp:
                    entropy_range = entropy_exp["token_entropy_range"]
                    self.assertEqual(len(entropy_range), 2)
                    self.assertLess(entropy_range[0], entropy_range[1])
                    self.assertGreater(entropy_range[0], 0)
                
                if "semantic_diversity_baseline" in entropy_exp:
                    baseline = entropy_exp["semantic_diversity_baseline"]
                    self.assertGreaterEqual(baseline, 0.0)
                    self.assertLessEqual(baseline, 1.0)
            
            # Check consistency expectations
            if "consistency_expectations" in profile:
                consistency_exp = profile["consistency_expectations"]
                for metric, value in consistency_exp.items():
                    if "baseline" in metric:
                        self.assertGreaterEqual(value, 0.0)
                        self.assertLessEqual(value, 1.0)


class TestConfigurationValues(unittest.TestCase):
    """Test cases for specific configuration values and ranges."""
    
    def test_coherence_failure_penalties(self):
        """Test coherence failure penalty values."""
        penalties = DEFAULT_CONFIG["coherence_detection"]["coherence_failure_penalties"]
        
        # All penalties should be positive (they're subtracted from score)
        for penalty_name, penalty_value in penalties.items():
            self.assertGreater(penalty_value, 0, f"Non-positive penalty: {penalty_name}")
            self.assertLess(penalty_value, 100, f"Excessive penalty: {penalty_name}")
    
    def test_context_limits_reasonable(self):
        """Test that context limits are reasonable."""
        model_profiles = DEFAULT_CONFIG["model_profiles"]
        
        for model_name, profile in model_profiles.items():
            if "context_expectations" in profile:
                context_exp = profile["context_expectations"]
                
                if "effective_context_limit" in context_exp:
                    limit = context_exp["effective_context_limit"]
                    self.assertGreater(limit, 1000)  # Should be at least 1K tokens
                    self.assertLess(limit, 100000)   # Should be less than 100K tokens
                
                if "saturation_onset" in context_exp:
                    onset = context_exp["saturation_onset"]
                    self.assertGreater(onset, 500)   # Should be at least 500 tokens
                    # Saturation should occur before the limit
                    if "effective_context_limit" in context_exp:
                        self.assertLess(onset, context_exp["effective_context_limit"])
    
    def test_domain_dimension_completeness(self):
        """Test that domain settings have reasonable numbers of dimensions."""
        domain_settings = DEFAULT_CONFIG["domain_evaluation_config"]["domain_settings"]
        
        for domain_name, settings in domain_settings.items():
            dimensions = settings["dimensions"]
            
            # Each domain should have at least 3 dimensions
            self.assertGreaterEqual(len(dimensions), 3, 
                                  f"Too few dimensions for {domain_name}: {len(dimensions)}")
            
            # But not too many (for practicality)
            self.assertLessEqual(len(dimensions), 10, 
                               f"Too many dimensions for {domain_name}: {len(dimensions)}")
            
            # All dimensions should be non-empty strings
            for dimension in dimensions:
                self.assertIsInstance(dimension, str)
                self.assertGreater(len(dimension), 0)
    
    def test_temperature_ranges_valid(self):
        """Test that temperature ranges in domain settings are valid."""
        domain_settings = DEFAULT_CONFIG["domain_evaluation_config"]["domain_settings"]
        
        for domain_name, settings in domain_settings.items():
            if "temperature_range" in settings:
                temp_range = settings["temperature_range"]
                self.assertEqual(len(temp_range), 2)
                self.assertGreaterEqual(temp_range[0], 0.0)
                self.assertLessEqual(temp_range[1], 2.0)
                self.assertLess(temp_range[0], temp_range[1])


if __name__ == '__main__':
    unittest.main()