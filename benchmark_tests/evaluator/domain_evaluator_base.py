from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class CulturalContext:
    """Represents cultural context information extracted from test metadata."""
    
    def __init__(self, 
                 traditions: List[str] = None,
                 knowledge_systems: List[str] = None, 
                 performance_aspects: List[str] = None,
                 cultural_groups: List[str] = None,
                 linguistic_varieties: List[str] = None):
        self.traditions = traditions or []
        self.knowledge_systems = knowledge_systems or []
        self.performance_aspects = performance_aspects or []
        self.cultural_groups = cultural_groups or []
        self.linguistic_varieties = linguistic_varieties or []


@dataclass
class EvaluationDimension:
    """Represents a single evaluation dimension with score and cultural context."""
    name: str
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    cultural_relevance: float  # 0.0 to 1.0
    evidence: List[str]  # Supporting evidence/examples
    cultural_markers: List[str]  # Detected cultural patterns
    
    def __post_init__(self):
        # Ensure scores are within valid ranges
        self.score = max(0.0, min(1.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.cultural_relevance = max(0.0, min(1.0, self.cultural_relevance))


@dataclass
class DomainEvaluationResult:
    """Result from a domain-specific evaluator."""
    domain: str
    evaluation_type: str
    overall_score: float  # 0.0 to 1.0
    dimensions: List[EvaluationDimension]
    cultural_context: CulturalContext
    metadata: Dict[str, Any]
    processing_notes: List[str]
    
    def get_dimension_score(self, dimension_name: str) -> Optional[float]:
        """Get score for a specific dimension."""
        for dim in self.dimensions:
            if dim.name == dimension_name:
                return dim.score
        return None
    
    def get_cultural_markers(self) -> List[str]:
        """Get all cultural markers detected across dimensions."""
        markers = []
        for dim in self.dimensions:
            markers.extend(dim.cultural_markers)
        return list(set(markers))
    
    def calculate_cultural_competence(self) -> float:
        """Calculate overall cultural competence score."""
        if not self.dimensions:
            return 0.0
        
        total_cultural_score = 0.0
        total_weight = 0.0
        
        for dim in self.dimensions:
            weight = dim.confidence * dim.cultural_relevance
            total_cultural_score += dim.score * weight
            total_weight += weight
        
        return total_cultural_score / total_weight if total_weight > 0 else 0.0


class BaseDomainEvaluator(ABC):
    """Abstract base class for domain-specific evaluators."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._initialize_evaluator()
    
    @abstractmethod
    def _initialize_evaluator(self):
        """Initialize domain-specific components and resources."""
        pass
    
    @abstractmethod
    def get_supported_evaluation_types(self) -> List[str]:
        """Return list of evaluation types this evaluator supports."""
        pass
    
    @abstractmethod
    def evaluate(self, 
                response_text: str, 
                test_metadata: Dict[str, Any], 
                cultural_context: CulturalContext) -> DomainEvaluationResult:
        """
        Evaluate a response using domain-specific metrics.
        
        Args:
            response_text: The text response to evaluate
            test_metadata: Metadata about the test case
            cultural_context: Cultural context information
            
        Returns:
            DomainEvaluationResult with scores and analysis
        """
        pass
    
    def preprocess_response(self, response_text: str) -> str:
        """Preprocess response text before evaluation. Override if needed."""
        return response_text.strip()
    
    def extract_cultural_markers(self, text: str, cultural_context: CulturalContext) -> List[str]:
        """Extract cultural markers from text. Override for domain-specific detection."""
        markers = []
        
        # Basic cultural marker detection
        text_lower = text.lower()
        
        for tradition in cultural_context.traditions:
            if tradition.lower() in text_lower:
                markers.append(f"tradition:{tradition}")
        
        for group in cultural_context.cultural_groups:
            if group.lower() in text_lower:
                markers.append(f"cultural_group:{group}")
        
        for variety in cultural_context.linguistic_varieties:
            if variety.lower() in text_lower:
                markers.append(f"linguistic:{variety}")
        
        return markers
    
    def calculate_dimension_score(self, 
                                metrics: Dict[str, float], 
                                weights: Dict[str, float] = None) -> float:
        """Calculate weighted score from multiple metrics."""
        if not metrics:
            return 0.0
        
        if weights is None:
            weights = {k: 1.0 for k in metrics.keys()}
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, score in metrics.items():
            weight = weights.get(metric, 1.0)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def validate_inputs(self, 
                       response_text: str, 
                       test_metadata: Dict[str, Any], 
                       cultural_context: CulturalContext) -> bool:
        """Validate inputs before evaluation."""
        if not response_text or not response_text.strip():
            return False
        
        if not isinstance(test_metadata, dict):
            return False
        
        if not isinstance(cultural_context, CulturalContext):
            return False
        
        return True
    
    def get_evaluation_metadata(self) -> Dict[str, Any]:
        """Get metadata about this evaluator."""
        return {
            'evaluator_class': self.__class__.__name__,
            'supported_types': self.get_supported_evaluation_types(),
            'config': self.config,
            'version': getattr(self, 'VERSION', '1.0.0')
        }


class MultiDimensionalEvaluator(BaseDomainEvaluator):
    """Base class for evaluators that assess multiple dimensions."""
    
    @abstractmethod
    def get_evaluation_dimensions(self) -> List[str]:
        """Return list of dimensions this evaluator assesses."""
        pass
    
    @abstractmethod
    def evaluate_dimension(self, 
                          dimension: str,
                          response_text: str, 
                          test_metadata: Dict[str, Any], 
                          cultural_context: CulturalContext) -> EvaluationDimension:
        """Evaluate a specific dimension."""
        pass
    
    def evaluate(self, 
                response_text: str, 
                test_metadata: Dict[str, Any], 
                cultural_context: CulturalContext) -> DomainEvaluationResult:
        """Evaluate all dimensions and aggregate results."""
        if not self.validate_inputs(response_text, test_metadata, cultural_context):
            return self._create_empty_result(cultural_context)
        
        processed_text = self.preprocess_response(response_text)
        dimensions = []
        
        for dim_name in self.get_evaluation_dimensions():
            try:
                dim_result = self.evaluate_dimension(
                    dim_name, processed_text, test_metadata, cultural_context
                )
                dimensions.append(dim_result)
            except Exception as e:
                # Create failed dimension result
                dimensions.append(EvaluationDimension(
                    name=dim_name,
                    score=0.0,
                    confidence=0.0,
                    cultural_relevance=0.0,
                    evidence=[f"Evaluation failed: {str(e)}"],
                    cultural_markers=[]
                ))
        
        overall_score = self._calculate_overall_score(dimensions)
        
        return DomainEvaluationResult(
            domain=self.get_domain_name(),
            evaluation_type=self._get_primary_evaluation_type(test_metadata),
            overall_score=overall_score,
            dimensions=dimensions,
            cultural_context=cultural_context,
            metadata=self.get_evaluation_metadata(),
            processing_notes=self._get_processing_notes(dimensions)
        )
    
    def _calculate_overall_score(self, dimensions: List[EvaluationDimension]) -> float:
        """Calculate overall score from dimension scores."""
        if not dimensions:
            return 0.0
        
        total_score = 0.0
        total_weight = 0.0
        
        for dim in dimensions:
            weight = dim.confidence * dim.cultural_relevance
            total_score += dim.score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _create_empty_result(self, cultural_context: CulturalContext) -> DomainEvaluationResult:
        """Create empty result for invalid inputs."""
        return DomainEvaluationResult(
            domain=self.get_domain_name(),
            evaluation_type="unknown",
            overall_score=0.0,
            dimensions=[],
            cultural_context=cultural_context,
            metadata=self.get_evaluation_metadata(),
            processing_notes=["Invalid inputs provided"]
        )
    
    def _get_processing_notes(self, dimensions: List[EvaluationDimension]) -> List[str]:
        """Generate processing notes from dimension results."""
        notes = []
        
        successful_dims = [d for d in dimensions if d.score > 0]
        failed_dims = [d for d in dimensions if d.score == 0 and d.confidence == 0]
        
        if successful_dims:
            notes.append(f"Successfully evaluated {len(successful_dims)} dimensions")
        
        if failed_dims:
            notes.append(f"Failed to evaluate {len(failed_dims)} dimensions: {[d.name for d in failed_dims]}")
        
        return notes
    
    @abstractmethod
    def get_domain_name(self) -> str:
        """Return the domain name this evaluator handles."""
        pass
    
    def _get_primary_evaluation_type(self, test_metadata: Dict[str, Any]) -> str:
        """Extract primary evaluation type from metadata."""
        return test_metadata.get('evaluation_type', 'general')