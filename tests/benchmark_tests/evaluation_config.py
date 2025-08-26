"""
Evaluation Configuration

Configuration settings for the UniversalEvaluator system.
Defines scoring weights, thresholds, and specialized patterns for different reasoning types.

Author: Claude Code
Version: 1.0.0
"""

from enum import Enum
from typing import Dict, List


class ScoreThresholds:
    """Score threshold definitions for evaluation categories"""
    EXCELLENT = 85.0
    GOOD = 70.0
    SATISFACTORY = 55.0
    POOR = 40.0
    VERY_POOR = 25.0


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
        "confidence_threshold": 70.0
    },
    
    # Text analysis parameters
    "text_analysis": {
        "min_sentence_length": 5,
        "max_sentence_length": 100,
        "vocabulary_diversity_threshold": 0.4,
        "step_indicator_weight": 10,
        "logic_connector_weight": 8,
        "evidence_indicator_weight": 12,
        "verification_indicator_weight": 15
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
    
    # Export and reporting settings
    "reporting": {
        "include_detailed_analysis": True,
        "include_recommendations": True,
        "include_confidence_scores": True,
        "export_formats": ["json", "csv", "html"],
        "decimal_precision": 1,
        "timestamp_format": "ISO8601"
    }
}


# Specialized configurations for different use cases
FAST_CONFIG = {
    **DEFAULT_CONFIG,
    "weights": {
        "step_clarity": 0.20,
        "logical_consistency": 0.30,
        "evidence_integration": 0.15,
        "analysis_depth": 0.15,
        "verification_effort": 0.10,
        "comprehensive_coverage": 0.05,
        "reasoning_pattern": 0.05
    },
    "text_analysis": {
        **DEFAULT_CONFIG["text_analysis"],
        "step_indicator_weight": 15,  # Faster computation
        "logic_connector_weight": 12
    }
}

DETAILED_CONFIG = {
    **DEFAULT_CONFIG,
    "weights": {
        "step_clarity": 0.12,
        "logical_consistency": 0.18,
        "evidence_integration": 0.18,
        "analysis_depth": 0.18,
        "verification_effort": 0.12,
        "comprehensive_coverage": 0.12,
        "reasoning_pattern": 0.10
    },
    "text_analysis": {
        **DEFAULT_CONFIG["text_analysis"],
        "vocabulary_diversity_threshold": 0.3,  # More detailed analysis
        "min_sentence_length": 3
    }
}

# Configuration presets for different reasoning types
REASONING_TYPE_PRESETS = {
    "academic_research": {
        **DETAILED_CONFIG,
        "thresholds": {
            **DEFAULT_CONFIG["thresholds"],
            "excellent_score": 90.0,
            "good_score": 80.0
        }
    },
    
    "business_analysis": {
        **DEFAULT_CONFIG,
        "weights": {
            "step_clarity": 0.20,
            "logical_consistency": 0.25,
            "evidence_integration": 0.20,
            "analysis_depth": 0.15,
            "verification_effort": 0.10,
            "comprehensive_coverage": 0.05,
            "reasoning_pattern": 0.05
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