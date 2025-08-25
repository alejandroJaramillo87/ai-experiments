"""
Evaluation Configuration

Configuration settings for the ReasoningEvaluator system.
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


class ReasoningWeights:
    """Default weights for different reasoning metrics"""
    STEP_CLARITY = 0.15
    LOGICAL_CONSISTENCY = 0.20
    EVIDENCE_INTEGRATION = 0.15
    ANALYSIS_DEPTH = 0.15
    VERIFICATION_EFFORT = 0.10
    COMPREHENSIVE_COVERAGE = 0.10
    REASONING_PATTERN = 0.15


# Main configuration dictionary
DEFAULT_CONFIG = {
    # Scoring weights for different metrics
    "weights": {
        "step_clarity": ReasoningWeights.STEP_CLARITY,
        "logical_consistency": ReasoningWeights.LOGICAL_CONSISTENCY,
        "evidence_integration": ReasoningWeights.EVIDENCE_INTEGRATION,
        "analysis_depth": ReasoningWeights.ANALYSIS_DEPTH,
        "verification_effort": ReasoningWeights.VERIFICATION_EFFORT,
        "comprehensive_coverage": ReasoningWeights.COMPREHENSIVE_COVERAGE,
        "reasoning_pattern": ReasoningWeights.REASONING_PATTERN
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
                "step_clarity": 0.25,  # Higher weight for step clarity
                "logical_consistency": 0.25,  # Higher weight for logical flow
                "evidence_integration": 0.10,
                "analysis_depth": 0.15,
                "verification_effort": 0.05,
                "comprehensive_coverage": 0.10,
                "reasoning_pattern": 0.10
            },
            "required_patterns": ["step", "first", "second", "then", "therefore"],
            "bonus_multiplier": 1.2
        },
        
        "multi_hop": {
            "weights": {
                "step_clarity": 0.10,
                "logical_consistency": 0.15,
                "evidence_integration": 0.30,  # Higher weight for evidence integration
                "analysis_depth": 0.20,  # Higher weight for synthesis
                "verification_effort": 0.05,
                "comprehensive_coverage": 0.10,
                "reasoning_pattern": 0.10
            },
            "required_patterns": ["document", "according to", "based on", "evidence shows"],
            "bonus_multiplier": 1.3
        },
        
        "verification": {
            "weights": {
                "step_clarity": 0.10,
                "logical_consistency": 0.15,
                "evidence_integration": 0.10,
                "analysis_depth": 0.15,
                "verification_effort": 0.35,  # Much higher weight for self-checking
                "comprehensive_coverage": 0.05,
                "reasoning_pattern": 0.10
            },
            "required_patterns": ["verify", "check", "confirm", "validate", "review"],
            "bonus_multiplier": 1.5
        },
        
        "mathematical": {
            "weights": {
                "step_clarity": 0.20,
                "logical_consistency": 0.30,  # Higher weight for logical precision
                "evidence_integration": 0.05,
                "analysis_depth": 0.20,
                "verification_effort": 0.15,
                "comprehensive_coverage": 0.05,
                "reasoning_pattern": 0.05
            },
            "required_patterns": ["calculate", "equation", "probability", "therefore"],
            "bonus_multiplier": 1.4
        },
        
        "backward": {
            "weights": {
                "step_clarity": 0.15,
                "logical_consistency": 0.20,
                "evidence_integration": 0.20,
                "analysis_depth": 0.25,  # Higher weight for reconstruction analysis
                "verification_effort": 0.10,
                "comprehensive_coverage": 0.05,
                "reasoning_pattern": 0.05
            },
            "required_patterns": ["work backward", "reverse", "trace", "reconstruct"],
            "bonus_multiplier": 1.3
        },
        
        "scaffolded": {
            "weights": {
                "step_clarity": 0.30,  # Highest weight for structured approach
                "logical_consistency": 0.20,
                "evidence_integration": 0.15,
                "analysis_depth": 0.15,
                "verification_effort": 0.10,
                "comprehensive_coverage": 0.05,
                "reasoning_pattern": 0.05
            },
            "required_patterns": ["analysis", "evidence", "reasoning", "conclusion"],
            "bonus_multiplier": 1.2
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