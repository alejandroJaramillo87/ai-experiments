from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
import time

from ..core.domain_evaluator_base import DomainEvaluationResult, CulturalContext, MultiDimensionalEvaluator
from ..core.evaluation_aggregator import ValidationFlag, BiasAnalysis, EvaluationAggregator
from .validation_runner import ValidationRunner, ValidationRequest, MultiModelValidationResult
from ..cultural.cultural_dataset_validator import CulturalDatasetValidator, DatasetValidationResult
from ..core.ensemble_disagreement_detector import EnsembleDisagreementDetector, EnsembleDisagreementResult
from ..data.open_cultural_apis import OpenCulturalAPIsIntegration, CulturalValidationResult as APIValidationResult
from .community_flagging_system import CommunityFlaggingSystem, FlagCategory, FlagSeverity
from ..data.domain_metadata_extractor import DomainMetadataExtractor, CulturalValidationResult as MetadataValidationResult


@dataclass
class IntegratedValidationConfig:
    """Configuration for integrated validation system."""
    enable_multi_model_validation: bool = True
    enable_wikipedia_validation: bool = True
    enable_cultural_datasets: bool = True
    enable_ensemble_disagreement: bool = True
    enable_open_apis: bool = True
    enable_community_flagging: bool = True
    enable_bias_detection: bool = True
    
    # Validation thresholds
    confidence_threshold: float = 0.5
    disagreement_threshold: float = 0.3
    bias_threshold: float = 0.3
    cultural_authenticity_threshold: float = 0.4
    
    # API configurations
    api_configs: Dict[str, Any] = None
    
    # Component-specific configs
    validation_runner_config: Dict[str, Any] = None
    dataset_validator_config: Dict[str, Any] = None
    community_system_config: Dict[str, Any] = None


@dataclass
class ComprehensiveValidationResult:
    """Complete validation result from all validation systems."""
    primary_evaluation: DomainEvaluationResult
    
    # Validation results from different components
    multi_model_validation: Optional[MultiModelValidationResult] = None
    wikipedia_validation: Optional[MetadataValidationResult] = None
    dataset_validation: Optional[DatasetValidationResult] = None
    disagreement_analysis: Optional[EnsembleDisagreementResult] = None  # Renamed for test compatibility
    open_apis_validation: Optional[APIValidationResult] = None  # Renamed for test compatibility
    bias_analysis: Optional[BiasAnalysis] = None
    
    # Aggregated metrics  
    overall_confidence: float = 0.0  # Renamed for test compatibility
    validation_consensus_score: float = 0.0
    cultural_authenticity_score: float = 0.0
    
    # All validation flags from all systems
    all_validation_flags: List[ValidationFlag] = None
    
    # Community flagging
    community_flags: List[str] = None  # Renamed for test compatibility
    
    # Processing metadata
    validation_time: float = 0.0
    components_used: List[str] = None
    validation_summary: str = ""
    recommendations: List[str] = None  # Added for test compatibility
    processing_metadata: Dict[str, Any] = None  # Added for test compatibility
    
    def __post_init__(self):
        """Initialize additional fields for compatibility."""
        if self.recommendations is None:
            self.recommendations = []
        if self.processing_metadata is None:
            self.processing_metadata = {
                'validation_time': self.validation_time,
                'components_used': self.components_used if self.components_used else []
            }
        if self.all_validation_flags is None:
            self.all_validation_flags = []
        if self.community_flags is None:
            self.community_flags = []
        if self.components_used is None:
            self.components_used = []
        
        # Maintain backward compatibility with old field names
        self.overall_validation_confidence = self.overall_confidence
        self.ensemble_disagreement = self.disagreement_analysis
        self.api_validation = self.open_apis_validation
        self.auto_generated_flags = self.community_flags


class IntegratedValidationSystem:
    """Integrated system that coordinates all validation components."""
    
    def __init__(self, config: IntegratedValidationConfig = None):
        self.config = config if config is not None else IntegratedValidationConfig()
        self.components = self._initialize_components()
        
        # Core components
        self.evaluation_aggregator = EvaluationAggregator()
        self.metadata_extractor = DomainMetadataExtractor()
        
    def _initialize_components(self) -> Dict[str, Any]:
        """Initialize validation components based on configuration."""
        components = {}
        
        if self.config.enable_multi_model_validation:
            components['validation_runner'] = ValidationRunner(
                self.config.validation_runner_config or {}
            )
        
        if self.config.enable_cultural_datasets:
            components['dataset_validator'] = CulturalDatasetValidator(
                self.config.dataset_validator_config or {}
            )
        
        if self.config.enable_ensemble_disagreement:
            components['ensemble_detector'] = EnsembleDisagreementDetector()
        
        if self.config.enable_open_apis:
            components['api_integration'] = OpenCulturalAPIsIntegration(
                self.config.api_configs or {}
            )
        
        if self.config.enable_community_flagging:
            components['community_system'] = CommunityFlaggingSystem(
                self.config.community_system_config or {}
            )
        
        return components
    
    async def comprehensive_validation(self, 
                                     content: str,
                                     cultural_context: CulturalContext,
                                     evaluation_result: DomainEvaluationResult,
                                     evaluator: MultiDimensionalEvaluator,
                                     test_metadata: Dict[str, Any] = None) -> ComprehensiveValidationResult:
        """
        Perform comprehensive validation using all enabled components.
        
        Args:
            content: Content that was evaluated
            cultural_context: Cultural context
            evaluation_result: Primary evaluation result
            evaluator: Evaluator used for primary evaluation
            test_metadata: Test metadata
            
        Returns:
            ComprehensiveValidationResult with all validation findings
        """
        start_time = time.time()
        components_used = []
        all_validation_flags = []
        
        # Initialize results containers
        multi_model_validation = None
        wikipedia_validation = None
        dataset_validation = None
        ensemble_disagreement = None
        api_validation = None
        bias_analysis = None
        auto_generated_flags = []
        
        # 1. Multi-model validation
        if self.config.enable_multi_model_validation and 'validation_runner' in self.components:
            try:
                components_used.append('multi_model_validation')
                
                validation_request = ValidationRequest(
                    content=content,
                    cultural_context=cultural_context,
                    evaluation_claims=self._extract_evaluation_claims(evaluation_result),
                    evaluation_dimension="overall",
                    original_score=evaluation_result.overall_score
                )
                
                multi_model_validation = await self.components['validation_runner'].validate_evaluation(
                    validation_request
                )
                
                all_validation_flags.extend(multi_model_validation.validation_flags)
                
            except Exception as e:
                all_validation_flags.append(ValidationFlag(
                    flag_type='validation_error',
                    severity='medium',
                    description=f"Multi-model validation failed: {str(e)}",
                    affected_dimensions=[],
                    cultural_groups=cultural_context.cultural_groups,
                    recommendation="Check multi-model validation configuration"
                ))
        
        # 2. Wikipedia validation
        if self.config.enable_wikipedia_validation:
            try:
                components_used.append('wikipedia_validation')
                
                wikipedia_validation = self.metadata_extractor.validate_cultural_context(
                    cultural_context, use_wikipedia=True
                )
                
            except Exception as e:
                all_validation_flags.append(ValidationFlag(
                    flag_type='validation_error',
                    severity='medium',
                    description=f"Wikipedia validation failed: {str(e)}",
                    affected_dimensions=[],
                    cultural_groups=cultural_context.cultural_groups,
                    recommendation="Check Wikipedia API availability"
                ))
        
        # 3. Cultural dataset validation
        if self.config.enable_cultural_datasets and 'dataset_validator' in self.components:
            try:
                components_used.append('cultural_datasets')
                
                dataset_validation = self.components['dataset_validator'].validate_cultural_evaluation(
                    cultural_context, evaluation_result
                )
                
                all_validation_flags.extend(dataset_validation.validation_flags)
                
            except Exception as e:
                all_validation_flags.append(ValidationFlag(
                    flag_type='validation_error',
                    severity='medium',
                    description=f"Dataset validation failed: {str(e)}",
                    affected_dimensions=[],
                    cultural_groups=cultural_context.cultural_groups,
                    recommendation="Check cultural datasets availability"
                ))
        
        # 4. Ensemble disagreement detection
        if self.config.enable_ensemble_disagreement and 'ensemble_detector' in self.components:
            try:
                components_used.append('ensemble_disagreement')
                
                ensemble_disagreement = self.components['ensemble_detector'].detect_evaluation_disagreement(
                    content, cultural_context, evaluation_result.domain, 
                    test_metadata or {}, evaluator
                )
                
                all_validation_flags.extend(ensemble_disagreement.validation_flags)
                
            except Exception as e:
                all_validation_flags.append(ValidationFlag(
                    flag_type='validation_error',
                    severity='medium',
                    description=f"Ensemble disagreement detection failed: {str(e)}",
                    affected_dimensions=[],
                    cultural_groups=cultural_context.cultural_groups,
                    recommendation="Check ensemble disagreement detector configuration"
                ))
        
        # 5. Open Cultural APIs validation
        if self.config.enable_open_apis and 'api_integration' in self.components:
            try:
                components_used.append('open_cultural_apis')
                
                api_validation = await self.components['api_integration'].validate_cultural_context(
                    cultural_context
                )
                
                all_validation_flags.extend(api_validation.validation_flags)
                
            except Exception as e:
                all_validation_flags.append(ValidationFlag(
                    flag_type='validation_error',
                    severity='medium',
                    description=f"Cultural APIs validation failed: {str(e)}",
                    affected_dimensions=[],
                    cultural_groups=cultural_context.cultural_groups,
                    recommendation="Check cultural APIs availability and configuration"
                ))
        
        # 6. Bias analysis
        if self.config.enable_bias_detection:
            try:
                components_used.append('bias_detection')
                
                # For bias analysis, we'd typically need historical data
                # For now, we'll do a simplified analysis based on current result
                from .evaluation_aggregator import AggregatedEvaluationResult
                
                # Create a mock aggregated result for bias analysis
                mock_aggregated = AggregatedEvaluationResult(
                    overall_score=evaluation_result.overall_score,
                    domain_scores={evaluation_result.domain: evaluation_result.overall_score},
                    dimension_scores={dim.name: dim.score for dim in evaluation_result.dimensions},
                    cultural_competence=evaluation_result.calculate_cultural_competence(),
                    cultural_markers=evaluation_result.get_cultural_markers(),
                    consensus_level=1.0,  # Single evaluation
                    evaluation_coverage=1.0,
                    metadata=evaluation_result.metadata,
                    processing_notes=evaluation_result.processing_notes,
                    domain_results=[evaluation_result]
                )
                
                bias_analysis = self.evaluation_aggregator.detect_statistical_bias(
                    [mock_aggregated], [cultural_context]
                )
                
            except Exception as e:
                all_validation_flags.append(ValidationFlag(
                    flag_type='validation_error',
                    severity='medium',
                    description=f"Bias analysis failed: {str(e)}",
                    affected_dimensions=[],
                    cultural_groups=cultural_context.cultural_groups,
                    recommendation="Check bias detection system configuration"
                ))
        
        # 7. Community flagging (auto-flag generation)
        if self.config.enable_community_flagging and 'community_system' in self.components:
            try:
                components_used.append('community_flagging')
                
                auto_generated_flags = self.components['community_system'].auto_flag_evaluation(
                    evaluation_result, all_validation_flags
                )
                
            except Exception as e:
                all_validation_flags.append(ValidationFlag(
                    flag_type='validation_error',
                    severity='medium',
                    description=f"Community flagging failed: {str(e)}",
                    affected_dimensions=[],
                    cultural_groups=cultural_context.cultural_groups,
                    recommendation="Check community flagging system configuration"
                ))
        
        # Calculate aggregated metrics
        overall_validation_confidence = self._calculate_overall_validation_confidence([
            multi_model_validation, wikipedia_validation, dataset_validation,
            ensemble_disagreement, api_validation, bias_analysis
        ])
        
        validation_consensus_score = self._calculate_validation_consensus([
            multi_model_validation, ensemble_disagreement, api_validation
        ])
        
        cultural_authenticity_score = self._calculate_cultural_authenticity_score([
            wikipedia_validation, dataset_validation, api_validation, evaluation_result
        ])
        
        # Generate validation summary
        validation_summary = self._generate_validation_summary(
            components_used, all_validation_flags, overall_validation_confidence
        )
        
        validation_time = time.time() - start_time
        
        return ComprehensiveValidationResult(
            primary_evaluation=evaluation_result,
            multi_model_validation=multi_model_validation,
            wikipedia_validation=wikipedia_validation,
            dataset_validation=dataset_validation,
            disagreement_analysis=ensemble_disagreement,  # Updated field name
            open_apis_validation=api_validation,  # Updated field name
            bias_analysis=bias_analysis,
            overall_confidence=overall_validation_confidence,  # Updated field name
            validation_consensus_score=validation_consensus_score,
            cultural_authenticity_score=cultural_authenticity_score,
            all_validation_flags=all_validation_flags,
            community_flags=auto_generated_flags,  # Updated field name
            validation_time=validation_time,
            components_used=components_used,
            validation_summary=validation_summary,
            recommendations=[],  # Initialize empty recommendations
            processing_metadata={  # Initialize processing metadata
                'validation_time': validation_time,
                'components_used': components_used
            }
        )
    
    def _extract_evaluation_claims(self, evaluation_result: DomainEvaluationResult) -> List[str]:
        """Extract evaluation claims for validation."""
        claims = []
        
        # Extract claims from dimension evidence
        for dimension in evaluation_result.dimensions:
            for evidence in dimension.evidence:
                if len(evidence) > 10:  # Meaningful evidence
                    claims.append(evidence)
        
        # Add cultural markers as claims
        cultural_markers = evaluation_result.get_cultural_markers()
        for marker in cultural_markers:
            claims.append(f"Cultural marker detected: {marker}")
        
        # Add overall assessment claim
        claims.append(f"Overall evaluation score: {evaluation_result.overall_score:.2f}")
        
        return claims[:10]  # Limit to top 10 claims
    
    def _calculate_overall_validation_confidence(self, validation_results: List[Any]) -> float:
        """Calculate overall validation confidence from all components."""
        confidences = []
        
        for result in validation_results:
            if result is None:
                continue
            
            if hasattr(result, 'consensus_score'):
                confidences.append(result.consensus_score)
            elif hasattr(result, 'validation_confidence'):
                confidences.append(result.validation_confidence)
            elif hasattr(result, 'overall_confidence'):
                confidences.append(result.overall_confidence)
            elif hasattr(result, 'evaluation_reliability_score'):
                confidences.append(result.evaluation_reliability_score)
        
        if not confidences:
            return 0.5  # Default confidence
        
        return sum(confidences) / len(confidences)
    
    def _calculate_validation_consensus(self, validation_results: List[Any]) -> float:
        """Calculate consensus between validation components."""
        consensus_scores = []
        
        for result in validation_results:
            if result is None:
                continue
            
            if hasattr(result, 'consensus_score'):
                consensus_scores.append(result.consensus_score)
            elif hasattr(result, 'disagreement_level'):
                # Convert disagreement to consensus
                consensus_scores.append(1.0 - result.disagreement_level)
        
        if not consensus_scores:
            return 1.0  # Perfect consensus if no disagreement data
        
        return sum(consensus_scores) / len(consensus_scores)
    
    def _calculate_cultural_authenticity_score(self, validation_results: List[Any]) -> float:
        """Calculate cultural authenticity score from validation components."""
        authenticity_scores = []
        
        for result in validation_results:
            if result is None:
                continue
            
            if hasattr(result, 'validation_confidence'):
                authenticity_scores.append(result.validation_confidence)
            elif hasattr(result, 'coverage_score'):
                authenticity_scores.append(result.coverage_score)
            elif hasattr(result, 'cultural_authenticity_consensus'):
                authenticity_scores.append(result.cultural_authenticity_consensus)
            elif hasattr(result, 'calculate_cultural_competence'):
                authenticity_scores.append(result.calculate_cultural_competence())
        
        if not authenticity_scores:
            return 0.5  # Default score
        
        return sum(authenticity_scores) / len(authenticity_scores)
    
    def _generate_validation_summary(self, 
                                   components_used: List[str],
                                   validation_flags: List[ValidationFlag],
                                   confidence: float) -> str:
        """Generate human-readable validation summary."""
        summary_parts = []
        
        # Components used
        summary_parts.append(f"Validation performed using {len(components_used)} components: {', '.join(components_used)}")
        
        # Overall confidence
        confidence_desc = "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
        summary_parts.append(f"Overall validation confidence: {confidence_desc} ({confidence:.2f})")
        
        # Flag summary
        if validation_flags:
            high_flags = len([f for f in validation_flags if f.severity == 'high'])
            medium_flags = len([f for f in validation_flags if f.severity == 'medium'])
            low_flags = len([f for f in validation_flags if f.severity == 'low'])
            
            flag_summary = f"Generated {len(validation_flags)} validation flags"
            if high_flags:
                flag_summary += f" ({high_flags} high priority"
                if medium_flags or low_flags:
                    flag_summary += f", {medium_flags + low_flags} others"
                flag_summary += ")"
            
            summary_parts.append(flag_summary)
        else:
            summary_parts.append("No validation issues detected")
        
        return ". ".join(summary_parts) + "."
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of integrated validation system."""
        status = {
            'configuration': {
                'multi_model_validation': self.config.enable_multi_model_validation,
                'wikipedia_validation': self.config.enable_wikipedia_validation,
                'cultural_datasets': self.config.enable_cultural_datasets,
                'ensemble_disagreement': self.config.enable_ensemble_disagreement,
                'open_apis': self.config.enable_open_apis,
                'community_flagging': self.config.enable_community_flagging,
                'bias_detection': self.config.enable_bias_detection
            },
            'components': {
                'total_components': len(self.components),
                'available_components': list(self.components.keys())
            },
            'thresholds': {
                'confidence_threshold': self.config.confidence_threshold,
                'disagreement_threshold': self.config.disagreement_threshold,
                'bias_threshold': self.config.bias_threshold,
                'cultural_authenticity_threshold': self.config.cultural_authenticity_threshold
            }
        }
        
        # Get component-specific status
        for component_name, component in self.components.items():
            if hasattr(component, 'get_system_status'):
                status[f'{component_name}_status'] = component.get_system_status()
            elif hasattr(component, 'get_usage_stats'):
                status[f'{component_name}_stats'] = component.get_usage_stats()
            elif hasattr(component, 'get_api_status'):
                status[f'{component_name}_api_status'] = component.get_api_status()
        
        return status
    
    def create_validation_report(self, validation_result: ComprehensiveValidationResult) -> Dict[str, Any]:
        """Create comprehensive validation report."""
        report = {
            'validation_metadata': {
                'timestamp': time.time(),
                'validation_time_seconds': validation_result.validation_time,
                'components_used': validation_result.components_used,
                'summary': validation_result.validation_summary
            },
            'primary_evaluation': {
                'domain': validation_result.primary_evaluation.domain,
                'evaluation_type': validation_result.primary_evaluation.evaluation_type,
                'overall_score': validation_result.primary_evaluation.overall_score,
                'cultural_competence': validation_result.primary_evaluation.calculate_cultural_competence(),
                'dimensions_count': len(validation_result.primary_evaluation.dimensions)
            },
            'validation_metrics': {
                'overall_confidence': validation_result.overall_validation_confidence,
                'consensus_score': validation_result.validation_consensus_score,
                'cultural_authenticity': validation_result.cultural_authenticity_score
            },
            'validation_flags': {
                'total_flags': len(validation_result.all_validation_flags),
                'flags_by_severity': self._count_flags_by_severity(validation_result.all_validation_flags),
                'flags_by_type': self._count_flags_by_type(validation_result.all_validation_flags),
                'auto_generated_flags': len(validation_result.auto_generated_flags)
            },
            'component_results': {
                'multi_model_validation': {
                    'available': validation_result.multi_model_validation is not None,
                    'consensus_score': getattr(validation_result.multi_model_validation, 'consensus_score', None),
                    'disagreement_level': getattr(validation_result.multi_model_validation, 'disagreement_level', None)
                } if validation_result.multi_model_validation else None,
                'dataset_validation': {
                    'available': validation_result.dataset_validation is not None,
                    'validation_confidence': getattr(validation_result.dataset_validation, 'validation_confidence', None),
                    'coverage_score': getattr(validation_result.dataset_validation, 'coverage_score', None)
                } if validation_result.dataset_validation else None,
                'api_validation': {
                    'available': validation_result.api_validation is not None,
                    'overall_confidence': getattr(validation_result.api_validation, 'overall_confidence', None),
                    'coverage_score': getattr(validation_result.api_validation, 'coverage_score', None)
                } if validation_result.api_validation else None
            }
        }
        
        return report
    
    # Component access properties for compatibility with tests
    @property
    def validation_runner(self):
        """Get validation runner component."""
        return self.components.get('validation_runner')
    
    @property
    def dataset_validator(self):
        """Get dataset validator component."""
        return self.components.get('dataset_validator')
    
    @property
    def community_flagging_system(self):
        """Get community flagging system component."""
        return self.components.get('community_system')
    
    @property
    def disagreement_detector(self):
        """Get ensemble disagreement detector component."""
        return self.components.get('ensemble_detector')
    
    @property
    def api_integration(self):
        """Get API integration component."""
        return self.components.get('api_integration')
    
    # Add convenient method aliases for tests
    def comprehensive_validate(self, *args, **kwargs):
        """Alias for comprehensive_validation method."""
        return self.comprehensive_validation(*args, **kwargs)
    
    def _count_flags_by_severity(self, flags: List[ValidationFlag]) -> Dict[str, int]:
        """Count validation flags by severity."""
        severity_counts = {}
        for flag in flags:
            severity_counts[flag.severity] = severity_counts.get(flag.severity, 0) + 1
        return severity_counts
    
    def _count_flags_by_type(self, flags: List[ValidationFlag]) -> Dict[str, int]:
        """Count validation flags by type."""
        type_counts = {}
        for flag in flags:
            type_counts[flag.flag_type] = type_counts.get(flag.flag_type, 0) + 1
        return type_counts