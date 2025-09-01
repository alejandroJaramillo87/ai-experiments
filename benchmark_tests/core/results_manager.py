#!/usr/bin/env python3
"""
Test Results Manager - Cognitive Pattern Detection Framework

Manages comprehensive test result storage with cognitive ability mapping and pattern detection.
Each test run gets a unique folder with complete response data and cognitive analysis.

Core Functions:
- Unique run identification and storage
- Cognitive pattern analysis across reasoning/memory/creativity/social/integration
- Statistical validation of detected patterns
- Result inspection and validation tools
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import statistics
import logging
from scipy import stats
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class RunMetadata:
    """Metadata for a test run"""
    run_id: str
    model_name: str
    model_path: str
    timestamp: str
    test_configuration: Dict[str, Any]
    cognitive_domains_tested: List[str]
    total_tests: int
    execution_duration_seconds: float

@dataclass 
class CognitivePattern:
    """Detected cognitive pattern in model responses"""
    cognitive_domain: str  # reasoning, memory, creativity, social, integration
    pattern_type: str      # strength, weakness, bias, inconsistency
    confidence_score: float # 0-1 statistical confidence
    evidence_tests: List[str]  # Test IDs supporting this pattern
    statistical_measures: Dict[str, float]  # p-values, effect sizes, etc.
    description: str       # Human-readable pattern description
    severity: str         # low, medium, high, critical

@dataclass
class CognitiveProfile:
    """Overall cognitive ability profile for a model"""
    model_name: str
    run_id: str
    
    # Cognitive ability scores (0-100)
    reasoning_score: float
    memory_score: float  
    creativity_score: float
    social_score: float
    integration_score: float
    
    # Pattern analysis
    detected_patterns: List[CognitivePattern]
    strengths: List[str]        # Cognitive areas of excellence
    weaknesses: List[str]       # Cognitive areas needing improvement
    blind_spots: List[str]      # Systematic gaps in capability
    
    # Statistical validation
    pattern_confidence: float   # Overall confidence in pattern detection
    sample_size: int           # Number of tests analyzed

class TestResultsManager:
    """Manages test result storage and cognitive pattern detection"""
    
    def __init__(self, base_results_dir: str = "test_results"):
        self.base_results_dir = Path(base_results_dir)
        self.base_results_dir.mkdir(exist_ok=True)
        
        # Cognitive domain mappings
        self.cognitive_domains = {
            'reasoning': ['reasoning', 'abstract_reasoning', 'logical', 'causal', 'inference'],
            'memory': ['knowledge', 'factual', 'historical', 'recall', 'contextual'],
            'creativity': ['creativity', 'narrative', 'artistic', 'innovation', 'synthesis'],
            'social': ['social', 'cultural', 'empathy', 'conflict', 'interpersonal'],
            'integration': ['integration', 'cross_domain', 'complex', 'holistic', 'synthesis']
        }
        
        # Statistical thresholds
        self.pattern_confidence_threshold = 0.7  # Minimum confidence for pattern detection
        self.effect_size_threshold = 0.3         # Minimum effect size for practical significance
        self.min_sample_size = 5                 # Minimum tests needed for pattern validation
    
    def create_run_directory(self, 
                           model_name: str, 
                           model_path: str,
                           test_configuration: Dict[str, Any]) -> str:
        """Create unique directory for test run"""
        
        # Generate unique run identifier
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_id = f"{model_name.replace('/', '_')}_{timestamp}_{uuid.uuid4().hex[:8]}"
        
        # Create directory structure
        run_dir = self.base_results_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (run_dir / "raw_responses").mkdir(exist_ok=True)
        (run_dir / "domain_analysis").mkdir(exist_ok=True)
        (run_dir / "pattern_detection").mkdir(exist_ok=True)
        
        # Save run metadata
        metadata = RunMetadata(
            run_id=run_id,
            model_name=model_name,
            model_path=model_path,
            timestamp=datetime.now().isoformat(),
            test_configuration=test_configuration,
            cognitive_domains_tested=[],  # Will be updated as tests run
            total_tests=0,
            execution_duration_seconds=0.0
        )
        
        self.save_run_metadata(run_dir, metadata)
        logger.info(f"Created test run directory: {run_dir}")
        
        return str(run_dir)
    
    def save_run_metadata(self, run_dir: Path, metadata: RunMetadata):
        """Save run metadata"""
        metadata_file = run_dir / "run_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
    
    def save_test_response(self,
                          run_dir: str,
                          test_id: str, 
                          prompt: str,
                          response_text: str,
                          evaluation_results: Dict[str, Any],
                          test_metadata: Dict[str, Any]):
        """Save individual test response with evaluation results"""
        
        run_path = Path(run_dir)
        response_file = run_path / "raw_responses" / f"{test_id}.json"
        
        # Convert complex objects to serializable format
        serializable_evaluation = self._make_serializable(evaluation_results)
        
        response_data = {
            'test_id': test_id,
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'response_text': response_text,
            'response_length': len(response_text),
            'evaluation_results': serializable_evaluation,
            'test_metadata': test_metadata,
            'cognitive_domain': self._classify_cognitive_domain(test_metadata),
        }
        
        try:
            with open(response_file, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save response {test_id}: {e}")
            # Save minimal version if full save fails
            minimal_data = {
                'test_id': test_id,
                'timestamp': datetime.now().isoformat(),
                'response_text': response_text,
                'error': f"Evaluation serialization failed: {str(e)}"
            }
            with open(response_file, 'w', encoding='utf-8') as f:
                json.dump(minimal_data, f, indent=2, ensure_ascii=False)
    
    def _classify_cognitive_domain(self, test_metadata: Dict[str, Any]) -> str:
        """Classify test into cognitive domain based on metadata"""
        
        # Check domain from test metadata
        if 'domain' in test_metadata:
            domain = test_metadata['domain'].lower()
            for cognitive_domain, keywords in self.cognitive_domains.items():
                if any(keyword in domain for keyword in keywords):
                    return cognitive_domain
        
        # Check from test ID or prompt keywords
        test_id = test_metadata.get('id', '').lower()
        for cognitive_domain, keywords in self.cognitive_domains.items():
            if any(keyword in test_id for keyword in keywords):
                return cognitive_domain
                
        return 'integration'  # Default for unclear cases
    
    def analyze_cognitive_patterns(self, run_dir: str) -> CognitiveProfile:
        """Analyze cognitive patterns from completed test run"""
        
        run_path = Path(run_dir)
        
        # Load all test responses
        responses = self._load_all_responses(run_path)
        if not responses:
            logger.warning(f"No responses found in {run_dir}")
            return self._create_empty_profile(run_dir)
        
        # Group by cognitive domain
        domain_responses = self._group_by_cognitive_domain(responses)
        
        # Calculate cognitive scores
        cognitive_scores = self._calculate_cognitive_scores(domain_responses)
        
        # Detect patterns
        detected_patterns = self._detect_cognitive_patterns(domain_responses)
        
        # Generate strengths/weaknesses
        strengths, weaknesses, blind_spots = self._identify_strengths_weaknesses(
            cognitive_scores, detected_patterns
        )
        
        # Calculate overall confidence
        pattern_confidence = self._calculate_pattern_confidence(detected_patterns)
        
        # Create cognitive profile
        profile = CognitiveProfile(
            model_name=self._get_model_name_from_run(run_path),
            run_id=run_path.name,
            reasoning_score=cognitive_scores.get('reasoning', 0),
            memory_score=cognitive_scores.get('memory', 0),
            creativity_score=cognitive_scores.get('creativity', 0),
            social_score=cognitive_scores.get('social', 0),
            integration_score=cognitive_scores.get('integration', 0),
            detected_patterns=detected_patterns,
            strengths=strengths,
            weaknesses=weaknesses,
            blind_spots=blind_spots,
            pattern_confidence=pattern_confidence,
            sample_size=len(responses)
        )
        
        # Save profile
        self._save_cognitive_profile(run_path, profile)
        
        return profile
    
    def _load_all_responses(self, run_path: Path) -> List[Dict[str, Any]]:
        """Load all test responses from run directory"""
        responses = []
        responses_dir = run_path / "raw_responses"
        
        if not responses_dir.exists():
            return responses
            
        for response_file in responses_dir.glob("*.json"):
            try:
                with open(response_file, 'r', encoding='utf-8') as f:
                    response_data = json.load(f)
                    responses.append(response_data)
            except Exception as e:
                logger.error(f"Error loading {response_file}: {e}")
                
        return responses
    
    def _group_by_cognitive_domain(self, responses: List[Dict]) -> Dict[str, List[Dict]]:
        """Group responses by cognitive domain"""
        domain_groups = {}
        
        for response in responses:
            domain = response.get('cognitive_domain', 'integration')
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(response)
            
        return domain_groups
    
    def _calculate_cognitive_scores(self, domain_responses: Dict[str, List[Dict]]) -> Dict[str, float]:
        """Calculate cognitive scores by domain"""
        cognitive_scores = {}
        
        for domain, responses in domain_responses.items():
            scores = []
            
            for response in responses:
                # Extract score from evaluation results
                eval_results = response.get('evaluation_results', {})
                
                # Try various score field names
                score = None
                for score_field in ['calibration_score', 'overall_score', 'quality_score', 'score']:
                    if score_field in eval_results and eval_results[score_field] is not None:
                        score = float(eval_results[score_field])
                        break
                
                if score is not None:
                    scores.append(score)
            
            # Calculate domain average
            if scores:
                cognitive_scores[domain] = statistics.mean(scores)
            else:
                cognitive_scores[domain] = 0.0
                
        return cognitive_scores
    
    def _detect_cognitive_patterns(self, domain_responses: Dict[str, List[Dict]]) -> List[CognitivePattern]:
        """Detect cognitive patterns using statistical analysis"""
        patterns = []
        
        for domain, responses in domain_responses.items():
            if len(responses) < self.min_sample_size:
                continue
                
            # Extract scores for analysis
            scores = []
            for response in responses:
                eval_results = response.get('evaluation_results', {})
                for score_field in ['calibration_score', 'overall_score', 'quality_score', 'score']:
                    if score_field in eval_results and eval_results[score_field] is not None:
                        scores.append(float(eval_results[score_field]))
                        break
            
            if not scores:
                continue
                
            # Statistical analysis
            mean_score = statistics.mean(scores)
            std_score = statistics.stdev(scores) if len(scores) > 1 else 0
            
            # Detect patterns based on performance
            if mean_score >= 80:  # Strong performance
                pattern = CognitivePattern(
                    cognitive_domain=domain,
                    pattern_type="strength",
                    confidence_score=min((mean_score - 70) / 30, 1.0),
                    evidence_tests=[r['test_id'] for r in responses],
                    statistical_measures={
                        'mean': mean_score,
                        'std': std_score,
                        'sample_size': len(scores)
                    },
                    description=f"Strong performance in {domain} (μ={mean_score:.1f})",
                    severity="high" if mean_score >= 90 else "medium"
                )
                patterns.append(pattern)
                
            elif mean_score <= 50:  # Weak performance
                pattern = CognitivePattern(
                    cognitive_domain=domain,
                    pattern_type="weakness", 
                    confidence_score=min((60 - mean_score) / 30, 1.0),
                    evidence_tests=[r['test_id'] for r in responses],
                    statistical_measures={
                        'mean': mean_score,
                        'std': std_score,
                        'sample_size': len(scores)
                    },
                    description=f"Weak performance in {domain} (μ={mean_score:.1f})",
                    severity="critical" if mean_score <= 30 else "high"
                )
                patterns.append(pattern)
                
            # Detect inconsistency patterns
            if std_score > 25:  # High variability
                pattern = CognitivePattern(
                    cognitive_domain=domain,
                    pattern_type="inconsistency",
                    confidence_score=min(std_score / 30, 1.0),
                    evidence_tests=[r['test_id'] for r in responses],
                    statistical_measures={
                        'mean': mean_score,
                        'std': std_score,
                        'sample_size': len(scores)
                    },
                    description=f"Inconsistent performance in {domain} (σ={std_score:.1f})",
                    severity="medium"
                )
                patterns.append(pattern)
        
        return patterns
    
    def _identify_strengths_weaknesses(self, 
                                     cognitive_scores: Dict[str, float], 
                                     patterns: List[CognitivePattern]) -> Tuple[List[str], List[str], List[str]]:
        """Identify cognitive strengths, weaknesses, and blind spots"""
        
        # Sort domains by score
        sorted_domains = sorted(cognitive_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Identify strengths (top domains with good scores)
        strengths = []
        for domain, score in sorted_domains[:2]:  # Top 2 domains
            if score >= 70:
                strengths.append(f"{domain.title()}: {score:.1f} - {self._get_strength_description(domain, score)}")
        
        # Identify weaknesses (bottom domains with poor scores)
        weaknesses = []
        for domain, score in sorted_domains[-2:]:  # Bottom 2 domains
            if score <= 60:
                weaknesses.append(f"{domain.title()}: {score:.1f} - {self._get_weakness_description(domain, score)}")
        
        # Identify blind spots from critical patterns
        blind_spots = []
        for pattern in patterns:
            if pattern.severity == "critical" and pattern.pattern_type in ["weakness", "bias"]:
                blind_spots.append(f"{pattern.cognitive_domain.title()}: {pattern.description}")
        
        return strengths, weaknesses, blind_spots
    
    def _get_strength_description(self, domain: str, score: float) -> str:
        """Get description for cognitive strength"""
        descriptions = {
            'reasoning': "Strong logical analysis and inference capabilities",
            'memory': "Excellent factual recall and contextual understanding", 
            'creativity': "High originality and synthetic thinking",
            'social': "Strong empathy and cultural competency",
            'integration': "Excellent cross-domain synthesis abilities"
        }
        return descriptions.get(domain, "Strong performance in this cognitive area")
    
    def _get_weakness_description(self, domain: str, score: float) -> str:
        """Get description for cognitive weakness"""
        descriptions = {
            'reasoning': "Struggles with logical analysis and complex inference",
            'memory': "Difficulty with factual recall and contextual understanding",
            'creativity': "Limited originality and synthetic thinking capabilities", 
            'social': "Challenges with empathy and cultural understanding",
            'integration': "Difficulty synthesizing across multiple domains"
        }
        return descriptions.get(domain, "Challenges in this cognitive area")
    
    def _calculate_pattern_confidence(self, patterns: List[CognitivePattern]) -> float:
        """Calculate overall confidence in pattern detection"""
        if not patterns:
            return 0.0
            
        confidences = [p.confidence_score for p in patterns]
        return statistics.mean(confidences)
    
    def _get_model_name_from_run(self, run_path: Path) -> str:
        """Extract model name from run metadata"""
        try:
            metadata_file = run_path / "run_metadata.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                return metadata.get('model_name', run_path.name)
        except Exception:
            return run_path.name.split('_')[0]
    
    def _save_cognitive_profile(self, run_path: Path, profile: CognitiveProfile):
        """Save cognitive profile to run directory"""
        profile_file = run_path / "cognitive_profile.json"
        
        # Convert to dict for JSON serialization
        profile_dict = asdict(profile)
        
        with open(profile_file, 'w') as f:
            json.dump(profile_dict, f, indent=2)
        
        logger.info(f"Saved cognitive profile to {profile_file}")
    
    def _create_empty_profile(self, run_dir: str) -> CognitiveProfile:
        """Create empty cognitive profile for failed analysis"""
        return CognitiveProfile(
            model_name="unknown",
            run_id=Path(run_dir).name,
            reasoning_score=0.0,
            memory_score=0.0,
            creativity_score=0.0,
            social_score=0.0,
            integration_score=0.0,
            detected_patterns=[],
            strengths=[],
            weaknesses=[],
            blind_spots=[],
            pattern_confidence=0.0,
            sample_size=0
        )
    
    def get_cognitive_summary_report(self, run_dir: str) -> str:
        """Generate human-readable cognitive summary report"""
        
        profile = self.analyze_cognitive_patterns(run_dir)
        
        report = f"""
🧠 COGNITIVE PROFILE ANALYSIS
=====================================
Model: {profile.model_name}
Run ID: {profile.run_id}
Sample Size: {profile.sample_size} tests
Pattern Confidence: {profile.pattern_confidence:.2f}

📊 COGNITIVE SCORES:
    Reasoning:    {profile.reasoning_score:.1f}/100
    Memory:       {profile.memory_score:.1f}/100  
    Creativity:   {profile.creativity_score:.1f}/100
    Social:       {profile.social_score:.1f}/100
    Integration:  {profile.integration_score:.1f}/100

🎯 STRENGTHS:
"""
        
        for strength in profile.strengths:
            report += f"    ✅ {strength}\n"
            
        report += "\n⚠️  WEAKNESSES:\n"
        for weakness in profile.weaknesses:
            report += f"    ❌ {weakness}\n"
            
        if profile.blind_spots:
            report += "\n🚨 CRITICAL BLIND SPOTS:\n"
            for blind_spot in profile.blind_spots:
                report += f"    🔍 {blind_spot}\n"
        
        report += f"\n🔍 DETECTED PATTERNS ({len(profile.detected_patterns)}):\n"
        for pattern in profile.detected_patterns:
            report += f"    {pattern.pattern_type.upper()}: {pattern.description} (confidence: {pattern.confidence_score:.2f})\n"
            
        return report
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert complex objects to JSON-serializable format"""
        
        if hasattr(obj, '__dict__'):
            # Handle dataclass or custom objects
            return {key: self._make_serializable(value) for key, value in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            # Fallback: convert to string
            return str(obj)