"""
Evaluation Configuration

Configuration settings for the UniversalEvaluator system.
Defines scoring weights, thresholds, and specialized patterns for different reasoning types.

"""

from enum import Enum
from typing import Dict, List


class ScoreThresholds:
    """Score threshold definitions for evaluation categories"""
    EXCELLENT = 75.0    # Reduced from 85.0 to address conservative scoring (-15 gap)
    GOOD = 60.0         # Reduced from 70.0 to better align with mid-range responses  
    SATISFACTORY = 45.0 # Reduced from 55.0 to properly score adequate responses
    POOR = 30.0         # Reduced from 40.0 to maintain proportional spacing
    VERY_POOR = 15.0    # Reduced from 25.0 to maintain proportional spacing


class UniversalWeights:
    """Default weights for universal evaluation metrics"""
    ORGANIZATION_QUALITY = 0.15
    TECHNICAL_ACCURACY = 0.20
    COMPLETENESS = 0.15
    THOROUGHNESS = 0.15
    RELIABILITY = 0.10
    SCOPE_COVERAGE = 0.10
    DOMAIN_APPROPRIATENESS = 0.15


# Main configuration dictionary
DEFAULT_CONFIG = {
    # Scoring weights for universal metrics
    "weights": {
        "organization_quality": UniversalWeights.ORGANIZATION_QUALITY,
        "technical_accuracy": UniversalWeights.TECHNICAL_ACCURACY,
        "completeness": UniversalWeights.COMPLETENESS,
        "thoroughness": UniversalWeights.THOROUGHNESS,
        "reliability": UniversalWeights.RELIABILITY,
        "scope_coverage": UniversalWeights.SCOPE_COVERAGE,
        "domain_appropriateness": UniversalWeights.DOMAIN_APPROPRIATENESS
    },
    
    # Quality thresholds
    "thresholds": {
        "excellent_score": ScoreThresholds.EXCELLENT,
        "good_score": ScoreThresholds.GOOD,
        "satisfactory_score": ScoreThresholds.SATISFACTORY,
        "poor_score": ScoreThresholds.POOR,
        "minimum_word_count": 50,
        "confidence_threshold": 70.0,
        "coherence_failure_threshold": 30.0,  # Below this = major coherence failure
        "coherence_penalty_threshold": 70.0,  # Below this = apply coherence penalty
        "repetitive_loop_threshold": 5,       # More than 5 repetitions = loop detected
        "meta_reasoning_threshold": 10,       # More than 10 meta-statements = excessive
        "technical_conciseness_threshold": 200, # Word count below this for technical bonus
        "creative_elaboration_threshold": 800   # Word count above this for creative bonus
    },
    
    # Text analysis parameters
    "text_analysis": {
        "min_sentence_length": 5,
        "max_sentence_length": 100,
        "vocabulary_diversity_threshold": 0.4,
        "step_indicator_weight": 10,
        "logic_connector_weight": 8,
        "evidence_indicator_weight": 12,
        "verification_indicator_weight": 15,
        "functional_completion_weight": 0.7,  # Weight for task completion vs formatting
        "content_length_weight": 0.2,        # Weight for content length
        "formatting_coverage_weight": 0.1     # Weight for formatting indicators
    },
    
    # Reasoning type specific configurations
    "reasoning_type_configs": {
        "chain_of_thought": {
            "weights": {
                "organization_quality": 0.25,  # Higher weight for step clarity
                "technical_accuracy": 0.25,  # Higher weight for logical flow
                "completeness": 0.10,
                "thoroughness": 0.15,
                "reliability": 0.05,
                "scope_coverage": 0.10,
                "domain_appropriateness": 0.10
            },
            "required_patterns": ["step", "first", "second", "then", "therefore"],
            "bonus_multiplier": 1.2
        },
        
        "multi_hop": {
            "weights": {
                "organization_quality": 0.10,
                "technical_accuracy": 0.15,
                "completeness": 0.30,  # Higher weight for evidence integration
                "thoroughness": 0.20,  # Higher weight for synthesis
                "reliability": 0.05,
                "scope_coverage": 0.10,
                "domain_appropriateness": 0.10
            },
            "required_patterns": ["document", "according to", "based on", "evidence shows"],
            "bonus_multiplier": 1.3
        },
        
        "verification": {
            "weights": {
                "organization_quality": 0.10,
                "technical_accuracy": 0.15,
                "completeness": 0.10,
                "thoroughness": 0.15,
                "reliability": 0.35,  # Much higher weight for self-checking
                "scope_coverage": 0.05,
                "domain_appropriateness": 0.10
            },
            "required_patterns": ["verify", "check", "confirm", "validate", "review"],
            "bonus_multiplier": 1.5
        },
        
        "mathematical": {
            "weights": {
                "organization_quality": 0.20,
                "technical_accuracy": 0.30,  # Higher weight for logical precision
                "completeness": 0.05,
                "thoroughness": 0.20,
                "reliability": 0.15,
                "scope_coverage": 0.05,
                "domain_appropriateness": 0.05
            },
            "required_patterns": ["calculate", "equation", "probability", "therefore"],
            "bonus_multiplier": 1.4
        },
        
        "backward": {
            "weights": {
                "organization_quality": 0.15,
                "technical_accuracy": 0.20,
                "completeness": 0.20,
                "thoroughness": 0.25,  # Higher weight for reconstruction analysis
                "reliability": 0.10,
                "scope_coverage": 0.05,
                "domain_appropriateness": 0.05
            },
            "required_patterns": ["work backward", "reverse", "trace", "reconstruct"],
            "bonus_multiplier": 1.3
        },
        
        "scaffolded": {
            "weights": {
                "organization_quality": 0.30,  # Highest weight for structured approach
                "technical_accuracy": 0.20,
                "completeness": 0.15,
                "thoroughness": 0.15,
                "reliability": 0.10,
                "scope_coverage": 0.05,
                "domain_appropriateness": 0.05
            },
            "required_patterns": ["analysis", "evidence", "reasoning", "conclusion"],
            "bonus_multiplier": 1.2
        }
    },
    
    # Test type specific configurations for universal evaluation
    "test_type_configs": {
        "linux": {
            "weights": {
                "organization_quality": 0.20,  # Command structure clarity
                "technical_accuracy": 0.35,   # Highest weight for correct syntax/security
                "completeness": 0.15,         # Solution completeness
                "thoroughness": 0.10,         # Documentation/explanation
                "reliability": 0.15,          # Best practices/security
                "scope_coverage": 0.03,       # Edge cases
                "domain_appropriateness": 0.02  # Linux-specific terminology
            },
            "keywords": ["command", "script", "bash", "sudo", "systemctl", "grep", "awk"],
            "best_practices": ["error handling", "logging", "security", "validation"],
            "dangerous_patterns": ["rm -rf /", "chmod 777", "* * * * *"],
            "bonus_multiplier": 1.2
        },
        
        "creative": {
            "weights": {
                "organization_quality": 0.15,  # Structure and flow
                "technical_accuracy": 0.10,    # Logical coherence
                "completeness": 0.20,          # Addressing all constraints
                "thoroughness": 0.25,          # Depth of creative exploration
                "reliability": 0.15,           # Consistency with requirements
                "scope_coverage": 0.10,        # Breadth of ideas
                "domain_appropriateness": 0.05  # Creative language
            },
            "keywords": ["creative", "innovative", "original", "unique", "alternative"],
            "quality_indicators": ["perspective", "approach", "consideration", "exploration"],
            "constraint_adherence": ["requirement", "specification", "criteria"],
            "bonus_multiplier": 1.3
        },
        
        "reasoning": {
            "weights": {
                "organization_quality": 0.15,  # Traditional step clarity
                "technical_accuracy": 0.20,    # Logical consistency
                "completeness": 0.15,          # Evidence integration
                "thoroughness": 0.15,          # Analysis depth
                "reliability": 0.10,           # Verification effort
                "scope_coverage": 0.10,        # Comprehensive coverage
                "domain_appropriateness": 0.15  # Reasoning patterns
            },
            "keywords": ["analysis", "reasoning", "logic", "evidence", "conclusion"],
            "logical_connectors": ["because", "therefore", "thus", "hence", "given that"],
            "verification_patterns": ["verify", "check", "confirm", "validate"],
            "bonus_multiplier": 1.0  # Default baseline
        }
    },
    
    # Domain-specific patterns and vocabulary
    "domain_patterns": {
        "medical": {
            "keywords": ["diagnosis", "symptoms", "treatment", "patient", "clinical", "medical"],
            "technical_terms": ["differential", "pathophysiology", "etiology", "prognosis"],
            "reasoning_patterns": ["history", "examination", "assessment", "plan"],
            "quality_indicators": ["systematic", "comprehensive", "evidence-based"]
        },
        
        "legal": {
            "keywords": ["precedent", "case", "court", "ruling", "law", "statute"],
            "technical_terms": ["jurisprudence", "appellant", "defendant", "jurisdiction"],
            "reasoning_patterns": ["facts", "issue", "holding", "reasoning"],
            "quality_indicators": ["cite", "distinguish", "overrule", "affirm"]
        },
        
        "financial": {
            "keywords": ["market", "investment", "risk", "return", "portfolio", "analysis"],
            "technical_terms": ["volatility", "correlation", "arbitrage", "diversification"],
            "reasoning_patterns": ["valuation", "scenario", "sensitivity", "recommendation"],
            "quality_indicators": ["quantitative", "model", "assumption", "stress-test"]
        },
        
        "scientific": {
            "keywords": ["hypothesis", "experiment", "data", "results", "conclusion"],
            "technical_terms": ["methodology", "variable", "control", "statistical"],
            "reasoning_patterns": ["observation", "hypothesis", "test", "analysis"],
            "quality_indicators": ["peer-reviewed", "replicate", "validate", "significance"]
        },
        
        "engineering": {
            "keywords": ["system", "design", "analysis", "specification", "performance"],
            "technical_terms": ["optimization", "constraint", "parameter", "simulation"],
            "reasoning_patterns": ["requirements", "design", "implementation", "testing"],
            "quality_indicators": ["systematic", "methodical", "validated", "robust"]
        }
    },
    
    # Advanced linguistic analysis patterns
    "linguistic_patterns": {
        "hedging_sophisticated": [
            "arguably", "presumably", "seemingly", "apparently", "potentially",
            "conceivably", "plausibly", "presumably", "ostensibly"
        ],
        
        "hedging_basic": [
            "maybe", "perhaps", "possibly", "might", "could", "may"
        ],
        
        "certainty_high": [
            "definitely", "certainly", "undoubtedly", "clearly", "obviously",
            "unquestionably", "indisputably", "conclusively"
        ],
        
        "certainty_medium": [
            "likely", "probably", "generally", "typically", "usually",
            "commonly", "frequently", "normally"
        ],
        
        "meta_cognitive": [
            "I need to consider", "let me think", "on reflection",
            "reconsidering", "upon further analysis", "stepping back",
            "taking a different approach", "re-examining"
        ],
        
        "self_correction": [
            "actually", "rather", "in fact", "more precisely", "to clarify",
            "correction", "amendment", "revised thinking", "better stated"
        ]
    },
    
    # Quantitative reasoning patterns
    "quantitative_patterns": {
        "numerical_precision": [
            r"\d+\.\d+%",  # Percentages with decimals
            r"\$[\d,]+\.\d+",  # Currency with decimals
            r"\d+\.\d+\s*(million|billion|trillion)",  # Large numbers
            r"±\s*\d+",  # Plus/minus indicators
        ],
        
        "statistical_terms": [
            "correlation", "regression", "standard deviation", "confidence interval",
            "p-value", "significant", "sample size", "population", "variance"
        ],
        
        "mathematical_operators": [
            "equals", "approximately", "greater than", "less than",
            "multiplied by", "divided by", "squared", "cubed"
        ],
        
        "probability_language": [
            "probability", "likelihood", "chance", "odds", "risk",
            "expected value", "distribution", "random", "stochastic"
        ]
    },
    
    # ENHANCEMENT: Advanced analysis configurations
    "advanced_analysis": {
        "entropy_analysis": {
            "enabled": True,
            "token_entropy_weight": 0.15,
            "semantic_entropy_weight": 0.20,
            "entropy_quality_ratio_weight": 0.10,
            "semantic_diversity_threshold": 0.6,  # Above this = bonus
            "low_entropy_penalty_threshold": 0.3,  # Below this = penalty
            "repetitive_pattern_penalty": 5.0
        },
        
        "semantic_coherence": {
            "enabled": True,
            "coherence_score_weight": 0.25,
            "prompt_completion_coherence_weight": 0.15,
            "semantic_drift_penalty_threshold": 0.3,  # High drift penalty
            "topic_consistency_bonus_threshold": 0.8,  # High consistency bonus
            "coherence_excellence_threshold": 0.85
        },
        
        "context_analysis": {
            "enabled": True,
            "context_health_weight": 0.20,
            "saturation_detection_penalty": 8.0,  # Severe penalty for saturation
            "degradation_penalty_per_point": 2.0,
            "quality_retention_bonus_threshold": 0.9,
            "context_efficiency_bonus_threshold": 0.8
        },
        
        "quantization_analysis": {
            "enabled": True,
            "numerical_stability_weight": 0.30,
            "factual_consistency_weight": 0.25,
            "quantization_impact_penalty_weight": 0.20,
            "high_impact_penalty_threshold": 0.7,
            "moderate_impact_penalty_threshold": 0.4,
            "minimal_impact_bonus_threshold": 0.2
        }
    },
    
    # LLM evaluation integration settings
    "llm_evaluation": {
        "enabled": False,  # Default to disabled
        "endpoint_url": None,  # To be configured if used
        "model_name": "claude-3-sonnet",
        "temperature": 0.1,
        "max_tokens": 1000,
        "evaluation_prompt_template": """
        Evaluate the reasoning quality of the following response on a scale of 0-100.
        
        Response to evaluate:
        {response_text}
        
        Reasoning type: {reasoning_type}
        
        Please provide:
        1. Overall score (0-100)
        2. Strengths of the reasoning
        3. Areas for improvement
        4. Specific recommendations
        
        Focus on logical coherence, evidence usage, and reasoning clarity.
        """,
        "timeout_seconds": 30,
        "retry_attempts": 3
    },
    
    # Coherence detection configuration
    "coherence_detection": {
        "enabled": True,
        "repetitive_phrase_patterns": [
            "The user might want", "The user wants", "I need to", "Let me",
            "We need to", "report" or "analysis", "summary" or "interpretation"
        ],
        "meta_reasoning_patterns": [
            "I think", "I should", "maybe I", "perhaps I", "let me think",
            "I'm not sure", "I wonder if", "I guess", "I believe", "I suppose"
        ],
        "broken_completion_patterns": [
            "(stop)", "...", "continues", "and so on", "etc.",
            "more of the same", "similar pattern", "keeps going"
        ],
        "system_error_patterns": [
            "assistant", "I am an AI", "I cannot", "I don't know",
            "error:", "failed:", "exception:", "unable to"
        ],
        "coherence_failure_penalties": {
            "repetitive_loop": 50,        # Severe penalty for loops
            "meta_reasoning_spiral": 30,  # Moderate penalty for meta-spirals
            "incomplete_response": 40,    # High penalty for incomplete responses
            "system_error": 60,           # Highest penalty for system errors
            "general_incoherence": 35     # Default penalty for other issues
        }
    },
    
    # Domain adaptation configuration
    "domain_adaptation": {
        "enabled": True,
        "length_normalization_ranges": {
            "linux": {
                "optimal_min": 100,
                "optimal_max": 300,
                "penalty_threshold": 500,
                "bonus_multiplier": 1.1
            },
            "creative": {
                "penalty_threshold": 200,
                "bonus_threshold": 800,
                "bonus_multiplier": 1.05
            },
            "reasoning": {
                "optimal_min": 300,
                "optimal_max": 600,
                "penalty_threshold": 150,
                "bonus_multiplier": 1.02
            }
        },
        "technical_domain_adjustments": {
            "conciseness_bonus": 1.2,      # Bonus for concise technical responses
            "accuracy_threshold": 70,      # Technical accuracy threshold for bonus
            "structure_bonus": 1.15,       # Bonus for technical structure
            "completeness_boost": 1.2,     # Boost completeness for technical content
            "thoroughness_boost": 1.15     # Boost thoroughness for technical content
        }
    },
    
    # Export and reporting settings
    "reporting": {
        "include_detailed_analysis": True,
        "include_recommendations": True,
        "include_confidence_scores": True,
        "export_formats": ["json", "csv", "html"],
        "decimal_precision": 1,
        "timestamp_format": "ISO8601",
        "include_coherence_analysis": True,
        "include_domain_adjustments": True,
        "include_functional_completion_breakdown": True
    },
    
    # Edge case detection patterns
    "edge_case_detection": {
        "enabled": True,
        "test_patterns": {
            "empty_response": r"^\s*$",
            "single_word": r"^\w+\s*$",
            "only_punctuation": r"^[^\w\s]*$",
            "excessive_repetition": r"(\b\w+\b)(?:\s+\1){4,}",  # Same word repeated 5+ times
            "meta_only": r"^(?:I|Let me|I need to|I should).*$",
            "code_only": r"^```[\s\S]*```$",
            "list_only": r"^(?:\d+\.\s*.*\n?)+$"
        },
        "handling_strategies": {
            "empty_response": "assign_minimum_score",
            "single_word": "severe_penalty",
            "only_punctuation": "assign_minimum_score",
            "excessive_repetition": "coherence_failure",
            "meta_only": "meta_reasoning_penalty",
            "code_only": "technical_evaluation",
            "list_only": "structure_only_penalty"
        }
    },
    
    # ENHANCEMENT: Advanced metrics thresholds and scoring adjustments
    "advanced_metrics_scoring": {
        "entropy_bonuses": {
            "high_semantic_diversity": 3.0,     # >0.6 semantic diversity
            "excellent_entropy_ratio": 2.0,     # >0.8 entropy/quality ratio
            "optimal_token_entropy": 1.5       # 4.0-6.0 token entropy range
        },
        "entropy_penalties": {
            "low_semantic_diversity": -4.0,     # <0.3 semantic diversity
            "repetitive_patterns": -5.0,        # Detected repetitive loops
            "entropy_collapse": -6.0            # <2.0 token entropy (collapse)
        },
        "coherence_bonuses": {
            "excellent_coherence": 4.0,         # >0.85 overall coherence
            "strong_topic_consistency": 2.5,    # >0.8 topic consistency
            "smooth_semantic_flow": 2.0         # >0.75 semantic flow
        },
        "coherence_penalties": {
            "poor_coherence": -6.0,             # <0.4 overall coherence
            "high_semantic_drift": -4.0,        # >0.7 drift score
            "topic_inconsistency": -3.0         # <0.3 topic consistency
        },
        "context_bonuses": {
            "excellent_context_health": 3.0,    # >0.9 context health
            "efficient_context_usage": 2.0,     # >0.8 efficiency
            "stable_quality_retention": 1.5     # >0.9 quality retention
        },
        "context_penalties": {
            "context_saturation": -8.0,         # Saturation detected
            "severe_degradation": -6.0,         # >3 severe degradation points
            "poor_context_health": -4.0         # <0.3 context health
        },
        "quantization_bonuses": {
            "excellent_numerical_stability": 2.0, # >0.9 numerical stability
            "high_factual_consistency": 2.5,     # >0.9 factual consistency
            "minimal_quantization_impact": 1.0    # <0.2 impact score
        },
        "quantization_penalties": {
            "high_quantization_impact": -8.0,    # >0.7 impact score
            "poor_numerical_stability": -5.0,    # <0.4 numerical stability
            "low_factual_consistency": -6.0,     # <0.4 factual consistency
            "calculation_avoidance": -4.0        # Detected avoidance patterns
        },
        "consistency_bonuses": {
            "excellent_consistency": 4.0,          # >0.9 cross-phrasing consistency
            "high_internal_consistency": 2.5,      # >0.8 internal consistency
            "good_confidence_consistency": 2.0     # >0.7 confidence consistency
        },
        "consistency_penalties": {
            "poor_consistency": -5.0,              # <0.4 cross-phrasing consistency
            "internal_contradictions": -6.0,       # Detected contradictions
            "confidence_inconsistency": -3.0       # <0.3 confidence consistency
        },
        "validation_bonuses": {
            "high_factual_accuracy": 5.0,          # >0.8 factual accuracy
            "excellent_knowledge_consistency": 3.0, # >0.8 knowledge consistency
            "well_calibrated_confidence": 3.0      # >0.7 confidence calibration
        },
        "validation_penalties": {
            "low_factual_accuracy": -6.0,          # <0.4 factual accuracy
            "poor_knowledge_consistency": -4.0,    # <0.3 knowledge consistency
            "miscalibrated_confidence": -3.0,      # <0.3 confidence calibration
            "failed_validation": -8.0              # Overall validation failure
        },
        "cultural_bonuses": {
            "excellent_cultural_authenticity": 4.0,  # >0.85 cultural authenticity
            "high_tradition_respect": 3.0,           # >0.8 tradition respect
            "excellent_coherence": 3.0,              # >0.85 cross-cultural coherence
            "comprehensive_contextualization": 2.0,  # Multiple context categories
            "proper_attribution": 2.0,               # Good community attribution
            "living_tradition_recognition": 1.5      # Recognizes evolving traditions
        },
        "cultural_penalties": {
            "poor_cultural_authenticity": -6.0,      # <0.3 cultural authenticity
            "tradition_violations": -8.0,            # Critical tradition violations
            "framework_imposition": -5.0,            # Inappropriate framework imposition
            "stereotype_indicators": -4.0,           # Stereotyping detected
            "appropriation_markers": -7.0,           # Cultural appropriation detected
            "sacred_knowledge_violations": -10.0,    # Sacred knowledge mishandling
            "poor_coherence": -4.0                   # <0.3 cross-cultural coherence
        }
    },
    
    # Consistency and validation thresholds
    "consistency_validation_thresholds": {
        "consistency": {
            "excellent": 0.9,
            "good": 0.7,
            "moderate": 0.5,
            "poor": 0.3
        },
        "factual_accuracy": {
            "excellent": 0.9,
            "good": 0.75,
            "moderate": 0.6,
            "poor": 0.4
        },
        "confidence_calibration": {
            "well_calibrated": 0.7,
            "reasonably_calibrated": 0.5,
            "poorly_calibrated": 0.3,
            "miscalibrated": 0.0
        },
        "knowledge_consistency": {
            "highly_consistent": 0.8,
            "moderately_consistent": 0.6,
            "low_consistency": 0.4,
            "inconsistent": 0.0
        }
    },
    
    # Built-in consistency test configurations
    "consistency_test_config": {
        "enabled": True,
        "test_categories": [
            "basic_math", "factual_knowledge", "logical_reasoning",
            "definitional", "comparative", "causal"
        ],
        "tests_per_category": 3,
        "similarity_threshold": 0.6,
        "confidence_analysis": True,
        "failure_detection": True
    },
    
    # Knowledge validation configurations
    "knowledge_validation_config": {
        "enabled": True,
        "validation_categories": [
            "geography", "science", "mathematics", 
            "history", "literature", "general"
        ],
        "tests_per_category": 5,
        "factual_accuracy_threshold": 0.6,
        "confidence_calibration_analysis": True,
        "forbidden_token_penalty": 0.5,
        "expected_token_bonus": 0.3
    },
    
    # Cultural evaluation thresholds
    "cultural_evaluation_thresholds": {
        "cultural_authenticity": {
            "excellent": 0.85,
            "good": 0.7,
            "acceptable": 0.5,
            "concerning": 0.3,
            "problematic": 0.1
        },
        "tradition_respect": {
            "excellent": 0.85,
            "good": 0.7,
            "acceptable": 0.5,
            "concerning": 0.3,
            "critical_issues": 0.1
        },
        "cross_cultural_coherence": {
            "excellent": 0.85,
            "good": 0.7,
            "acceptable": 0.5,
            "concerning": 0.3,
            "problematic": 0.1
        },
        "stereotype_severity": {
            "acceptable": 0.1,
            "concerning": 0.3,
            "problematic": 0.5,
            "severe": 0.8
        },
        "appropriation_risk": {
            "low": 0.1,
            "medium": 0.3,
            "high": 0.5,
            "critical": 0.8
        },
        "framework_imposition": {
            "minimal": 0.9,
            "low": 0.7,
            "moderate": 0.5,
            "high": 0.3,
            "severe": 0.1
        }
    },
    
    # Cultural evaluation configurations
    "cultural_evaluation_config": {
        "enabled": True,
        "cultural_domains": [
            "traditional_scientific", "historical_systems", "geographic_cultural",
            "mathematical_traditions", "social_systems", "material_cultural"
        ],
        "knowledge_domains": [
            "healing", "spiritual", "ecological", "social", "technical", "educational"
        ],
        "authenticity_analysis": True,
        "tradition_validation": True,
        "coherence_checking": True,
        "sacred_knowledge_protection": True,
        "community_attribution_required": True
    },
    
    # Domain-specific evaluation configurations
    "domain_evaluation_config": {
        "enabled": True,
        "router_config": {
            "lazy_loading": True,
            "cache_evaluators": True,
            "parallel_evaluation": False,  # Sequential by default for consistency
            "fallback_to_general": True,
            "max_concurrent_evaluators": 3
        },
        "aggregation_config": {
            "consensus_threshold": 0.7,
            "outlier_threshold": 2.0,  # Standard deviations
            "cultural_weighting": True,
            "dimension_aggregation_method": "weighted_average",
            "overall_score_method": "cultural_competence_weighted"
        },
        "domain_settings": {
            "creativity": {
                "evaluator_class": "CreativityEvaluator",
                "temperature_range": [0.6, 0.9],
                "pattern_libraries": ["griot", "dreamtime", "kamishibai", "oral_performance"],
                "dimensions": [
                    "cultural_creative_patterns",
                    "rhythmic_quality", 
                    "narrative_coherence",
                    "originality_within_bounds",
                    "performance_quality",
                    "collaborative_creation"
                ]
            },
            "language": {
                "evaluator_class": "LanguageEvaluator", 
                "pattern_libraries": ["code_switching", "register_variation", "dialectal_patterns", "historical_linguistics", "narrative_structure"],
                "dimensions": [
                    "register_appropriateness",
                    "code_switching_quality",
                    "pragmatic_competence",
                    "multilingual_patterns",
                    "dialectal_competence",
                    "sociolinguistic_awareness",
                    "historical_linguistics",
                    "narrative_structure",
                    "semantic_sophistication"
                ]
            },
            "social": {
                "evaluator_class": "SocialEvaluator",
                "pattern_libraries": ["hierarchy_patterns", "relationship_maintenance", "etiquette_systems", "conflict_resolution", "intercultural_competence"],
                "dimensions": [
                    "social_appropriateness",
                    "hierarchy_navigation", 
                    "relationship_maintenance",
                    "community_dynamics",
                    "cultural_etiquette",
                    "conflict_resolution",
                    "intercultural_competence"
                ]
            },
            "reasoning": {
                "evaluator_class": "ReasoningEvaluator",
                "pattern_libraries": ["logic_frameworks", "cultural_reasoning", "holistic_thinking"],
                "dimensions": [
                    "reasoning_pattern_recognition",
                    "cultural_logic_frameworks",
                    "holistic_vs_analytical",
                    "relational_reasoning",
                    "multi_logic_integration"
                ]
            },
            "knowledge": {
                "evaluator_class": "KnowledgeEvaluator",
                "pattern_libraries": ["traditional_knowledge", "sacred_boundaries", "community_attribution"],
                "dimensions": [
                    "traditional_knowledge_accuracy",
                    "cultural_contextualization", 
                    "sacred_knowledge_respect",
                    "community_attribution",
                    "knowledge_system_integration"
                ]
            },
            "integration": {
                "evaluator_class": "IntegrationEvaluator",
                "pattern_libraries": ["cross_domain", "synthesis_patterns", "integration_markers"],
                "dimensions": [
                    "cross_domain_coherence",
                    "cultural_authenticity_integration",
                    "logical_consistency_across_domains",
                    "creative_appropriateness", 
                    "social_awareness_integration",
                    "synthesis_quality"
                ],
                "integration_types": [
                    "knowledge_reasoning_synthesis",
                    "social_creative_solutions",
                    "multilingual_knowledge_expression",
                    "culturally_sensitive_reasoning",
                    "comprehensive_integration"
                ]
            }
        },
        "cultural_pattern_libraries": {
            "griot": {
                "storytelling_structures": ["call_response", "moral_embedding", "historical_weaving"],
                "performance_markers": ["rhythmic_speech", "audience_engagement", "improvisation"],
                "cultural_values": ["community_wisdom", "oral_preservation", "moral_instruction"]
            },
            "dreamtime": {
                "narrative_structures": ["landscape_embodiment", "ancestor_presence", "cyclical_time"],
                "knowledge_integration": ["ecological_wisdom", "spiritual_geography", "social_law"],
                "storytelling_ethics": ["sacred_boundaries", "initiation_levels", "place_connection"]
            },
            "kamishibai": {
                "visual_narrative": ["image_text_harmony", "dramatic_pacing", "audience_participation"],
                "educational_purpose": ["moral_lessons", "cultural_values", "community_bonding"],
                "performance_style": ["theatrical_delivery", "voice_modulation", "timing_mastery"]
            },
            "oral_performance": {
                "rhythmic_patterns": ["meter_consistency", "stress_patterns", "breath_phrasing"],
                "memory_aids": ["alliteration", "repetition", "formulaic_phrases"],
                "audience_interaction": ["call_response", "participation_cues", "collective_memory"]
            },
            "cross_domain": {
                "integration_markers": ["synthesize", "integrate", "combine", "merge", "unify"],
                "transition_indicators": ["furthermore", "additionally", "however", "therefore", "similarly"],
                "coherence_patterns": ["holistic", "comprehensive", "multifaceted", "interdisciplinary"]
            },
            "synthesis_patterns": {
                "knowledge_reasoning": ["traditional knowledge shows", "cultural understanding combined with", "evidence from traditional systems"],
                "social_creativity": ["creative solution to social", "community healing through art", "artistic practices address"],
                "language_knowledge": ["multilingual expression of", "code-switching reflects", "cultural translation conveys"],
                "reasoning_social": ["cultural sensitivity in reasoning", "ethical reasoning considers", "multiple perspectives in analysis"]
            }
        }
    },
    
    # ENHANCEMENT: Model-specific configuration profiles
    "model_profiles": {
        "gpt_oss_20b": {
            "base_model_优化": True,  # Optimized for base completion tasks
            "instruct_model_penalty": 0.9,  # 10% penalty for instruct format mismatch
            "entropy_expectations": {
                "token_entropy_range": [3.5, 5.5],   # Expected range for this model
                "semantic_diversity_baseline": 0.45   # Baseline expectation
            },
            "context_expectations": {
                "effective_context_limit": 4096,      # Tokens before degradation
                "saturation_onset": 3000              # Early saturation detection
            },
            "quantization_profiles": {
                "fp16": {"impact_threshold": 0.1},
                "int8": {"impact_threshold": 0.3},
                "int4": {"impact_threshold": 0.6}
            },
            "consistency_expectations": {
                "internal_consistency_baseline": 0.7,  # Expected internal consistency
                "confidence_consistency_baseline": 0.6, # Expected confidence consistency
                "cross_phrasing_baseline": 0.5        # Expected cross-phrasing consistency
            },
            "validation_expectations": {
                "factual_accuracy_baseline": 0.6,     # Expected factual accuracy
                "knowledge_consistency_baseline": 0.5, # Expected knowledge consistency
                "confidence_calibration_baseline": 0.4 # Expected confidence calibration
            },
            "cultural_expectations": {
                "cultural_authenticity_baseline": 0.5,  # Expected cultural authenticity
                "tradition_respect_baseline": 0.5,      # Expected tradition respect
                "cross_cultural_coherence_baseline": 0.6, # Expected coherence
                "stereotype_tolerance": 0.2,             # Maximum acceptable stereotype level
                "appropriation_tolerance": 0.1           # Maximum acceptable appropriation risk
            }
        },
        
        "claude_sonnet": {
            "instruct_model_优化": True,
            "base_model_penalty": 0.95,
            "entropy_expectations": {
                "token_entropy_range": [4.0, 6.0],
                "semantic_diversity_baseline": 0.55
            },
            "context_expectations": {
                "effective_context_limit": 8192,
                "saturation_onset": 6000
            },
            "consistency_expectations": {
                "internal_consistency_baseline": 0.8,  # Higher expectations for instruct model
                "confidence_consistency_baseline": 0.7,
                "cross_phrasing_baseline": 0.7
            },
            "validation_expectations": {
                "factual_accuracy_baseline": 0.75,
                "knowledge_consistency_baseline": 0.7,
                "confidence_calibration_baseline": 0.6
            },
            "cultural_expectations": {
                "cultural_authenticity_baseline": 0.7,  # Higher expectations for instruct model
                "tradition_respect_baseline": 0.7,      # Higher expectations for tradition respect
                "cross_cultural_coherence_baseline": 0.8, # Higher coherence expectations
                "stereotype_tolerance": 0.1,             # Lower tolerance for stereotypes
                "appropriation_tolerance": 0.05          # Very low appropriation tolerance
            }
        },
        
        "llama_70b": {
            "balanced_优化": True,
            "entropy_expectations": {
                "token_entropy_range": [4.2, 6.2],
                "semantic_diversity_baseline": 0.50
            },
            "context_expectations": {
                "effective_context_limit": 2048,
                "saturation_onset": 1500
            },
            "consistency_expectations": {
                "internal_consistency_baseline": 0.75,
                "confidence_consistency_baseline": 0.65,
                "cross_phrasing_baseline": 0.6
            },
            "validation_expectations": {
                "factual_accuracy_baseline": 0.7,
                "knowledge_consistency_baseline": 0.6,
                "confidence_calibration_baseline": 0.5
            }
        },
        
        "qwen3_30b": {
            "balanced_优化": True,
            "entropy_expectations": {
                "token_entropy_range": [3.8, 5.8],
                "semantic_diversity_baseline": 0.48
            },
            "context_expectations": {
                "effective_context_limit": 4096,
                "saturation_onset": 3000
            },
            "consistency_expectations": {
                "internal_consistency_baseline": 0.72,
                "confidence_consistency_baseline": 0.62,
                "cross_phrasing_baseline": 0.55
            },
            "validation_expectations": {
                "factual_accuracy_baseline": 0.65,
                "knowledge_consistency_baseline": 0.55,
                "confidence_calibration_baseline": 0.45
            }
        }
    }
}


# Specialized configurations for different use cases
FAST_CONFIG = {
    **DEFAULT_CONFIG,
    "weights": {
        "organization_quality": 0.20,
        "technical_accuracy": 0.30,
        "completeness": 0.15,
        "thoroughness": 0.15,
        "reliability": 0.10,
        "scope_coverage": 0.05,
        "domain_appropriateness": 0.05
    },
    "text_analysis": {
        **DEFAULT_CONFIG["text_analysis"],
        "step_indicator_weight": 15,  # Faster computation
        "logic_connector_weight": 12,
        "functional_completion_weight": 0.6,  # Reduced for speed
        "content_length_weight": 0.3,
        "formatting_coverage_weight": 0.1
    },
    "coherence_detection": {
        **DEFAULT_CONFIG["coherence_detection"],
        "enabled": True  # Keep coherence detection even in fast mode
    },
    "domain_adaptation": {
        **DEFAULT_CONFIG["domain_adaptation"],
        "enabled": False  # Disable for speed
    }
}

DETAILED_CONFIG = {
    **DEFAULT_CONFIG,
    "weights": {
        "organization_quality": 0.12,
        "technical_accuracy": 0.18,
        "completeness": 0.18,
        "thoroughness": 0.18,
        "reliability": 0.12,
        "scope_coverage": 0.12,
        "domain_appropriateness": 0.10
    },
    "text_analysis": {
        **DEFAULT_CONFIG["text_analysis"],
        "vocabulary_diversity_threshold": 0.3,  # More detailed analysis
        "min_sentence_length": 3,
        "functional_completion_weight": 0.8,  # Higher emphasis on functional completion
        "content_length_weight": 0.15,
        "formatting_coverage_weight": 0.05
    },
    "coherence_detection": {
        **DEFAULT_CONFIG["coherence_detection"],
        "coherence_failure_penalties": {
            "repetitive_loop": 60,        # Even more severe in detailed mode
            "meta_reasoning_spiral": 40,
            "incomplete_response": 50,
            "system_error": 70,
            "general_incoherence": 45
        }
    }
}

# Configuration presets for different reasoning types
REASONING_TYPE_PRESETS = {
    "academic_research": {
        **DETAILED_CONFIG,
        "thresholds": {
            **DEFAULT_CONFIG["thresholds"],
            "excellent_score": 90.0,
            "good_score": 80.0,
            "coherence_failure_threshold": 25.0  # Stricter for academic
        }
    },
    
    "business_analysis": {
        **DEFAULT_CONFIG,
        "weights": {
            "organization_quality": 0.20,
            "technical_accuracy": 0.25,
            "completeness": 0.20,
            "thoroughness": 0.15,
            "reliability": 0.10,
            "scope_coverage": 0.05,
            "domain_appropriateness": 0.05
        }
    },
    
    "educational_assessment": {
        **FAST_CONFIG,
        "thresholds": {
            **DEFAULT_CONFIG["thresholds"],
            "excellent_score": 80.0,
            "good_score": 65.0,
            "satisfactory_score": 50.0
        }
    }
}