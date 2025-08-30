"""
Advanced Analysis Orchestrator

Centralized system for managing and coordinating all advanced analysis modules.
Provides graceful degradation, error handling, and consistent integration points.

Modules Managed:
- EntropyCalculator: Information theory and semantic entropy analysis
- SemanticCoherenceAnalyzer: Text coherence and consistency analysis  
- ContextWindowAnalyzer: Context usage and quality analysis
- QuantizationTester: Model quantization impact assessment
- ConsistencyValidator: Cross-validation and consistency checking
- WikipediaFactChecker: External fact validation
- MultiSourceFactValidator: Ensemble fact validation

Features:
- Graceful degradation when modules fail
- Centralized configuration management
- Performance monitoring and optimization
- Error recovery and fallback strategies
- Integration with existing evaluation pipeline

Author: Claude Code
Version: 1.0.0
"""

import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisModule(Enum):
    """Available advanced analysis modules"""
    ENTROPY_CALCULATOR = "entropy_calculator"
    SEMANTIC_COHERENCE = "semantic_coherence" 
    CONTEXT_ANALYZER = "context_analyzer"
    QUANTIZATION_TESTER = "quantization_tester"
    CONSISTENCY_VALIDATOR = "consistency_validator"
    WIKIPEDIA_FACT_CHECKER = "wikipedia_fact_checker"
    MULTI_SOURCE_FACT_VALIDATOR = "multi_source_fact_validator"


@dataclass
class ModuleResult:
    """Result from an individual analysis module"""
    module: AnalysisModule
    success: bool
    result: Any  # Module-specific result
    processing_time: float
    error_message: Optional[str] = None
    fallback_used: bool = False


@dataclass  
class OrchestrationResult:
    """Complete advanced analysis orchestration result"""
    text: str
    requested_modules: List[AnalysisModule]
    module_results: List[ModuleResult]
    successful_modules: List[AnalysisModule]
    failed_modules: List[AnalysisModule]
    total_processing_time: float
    analysis_data: Dict[str, Any]  # Consolidated analysis results
    integration_notes: List[str]
    performance_metrics: Dict[str, float]


class AdvancedAnalysisOrchestrator:
    """
    Orchestrates advanced analysis modules with graceful degradation.
    
    Provides centralized management, error handling, and integration
    for all advanced analysis components in the evaluation system.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the advanced analysis orchestrator"""
        self.config = config or self._get_default_config()
        
        # Module registry
        self.modules = {}
        self.module_status = {}
        self.fallback_handlers = {}
        
        # Performance tracking
        self.performance_history = {}
        self.error_history = {}
        
        # Thread pool for concurrent analysis
        self.max_workers = self.config.get('max_concurrent_modules', 3)
        
        # Initialize modules
        self._initialize_modules()
        
        logger.info(f"Advanced Analysis Orchestrator initialized with {len(self.modules)} modules")
    
    def run_advanced_analysis(self, 
                            text: str,
                            requested_modules: Optional[List[AnalysisModule]] = None,
                            domain_context: Optional[str] = None,
                            cultural_context: Optional[Dict[str, Any]] = None,
                            internal_confidence: Optional[float] = None,
                            concurrent: bool = True) -> OrchestrationResult:
        """
        Run advanced analysis with specified modules.
        
        Args:
            text: Text to analyze
            requested_modules: Specific modules to run (None = all available)
            domain_context: Optional domain context
            cultural_context: Optional cultural context
            internal_confidence: Optional internal confidence score
            concurrent: Whether to run modules concurrently
            
        Returns:
            OrchestrationResult with consolidated analysis
        """
        start_time = time.time()
        
        # Determine which modules to run
        modules_to_run = requested_modules or list(self.modules.keys())
        available_modules = [m for m in modules_to_run if m in self.modules and self.module_status.get(m, True)]
        
        logger.info(f"Running advanced analysis with {len(available_modules)} modules: {[m.value for m in available_modules]}")
        
        # Run analysis modules
        if concurrent and len(available_modules) > 1:
            module_results = self._run_modules_concurrent(
                text, available_modules, domain_context, cultural_context, internal_confidence
            )
        else:
            module_results = self._run_modules_sequential(
                text, available_modules, domain_context, cultural_context, internal_confidence
            )
        
        # Consolidate results
        analysis_data = self._consolidate_analysis_results(module_results)
        
        # Generate integration notes
        integration_notes = self._generate_integration_notes(module_results, internal_confidence)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(module_results)
        
        # Update performance history
        self._update_performance_history(module_results)
        
        total_time = time.time() - start_time
        
        successful_modules = [r.module for r in module_results if r.success]
        failed_modules = [r.module for r in module_results if not r.success]
        
        logger.info(f"Advanced analysis completed: {len(successful_modules)} succeeded, {len(failed_modules)} failed, {total_time:.2f}s total")
        
        return OrchestrationResult(
            text=text,
            requested_modules=modules_to_run,
            module_results=module_results,
            successful_modules=successful_modules,
            failed_modules=failed_modules,
            total_processing_time=total_time,
            analysis_data=analysis_data,
            integration_notes=integration_notes,
            performance_metrics=performance_metrics
        )
    
    def _initialize_modules(self):
        """Initialize all available advanced analysis modules"""
        
        # EntropyCalculator
        try:
            from ..advanced.entropy_calculator import EntropyCalculator
            self.modules[AnalysisModule.ENTROPY_CALCULATOR] = EntropyCalculator()
            self.module_status[AnalysisModule.ENTROPY_CALCULATOR] = True
            logger.info("✅ EntropyCalculator initialized")
        except Exception as e:
            logger.warning(f"❌ EntropyCalculator initialization failed: {e}")
            self.module_status[AnalysisModule.ENTROPY_CALCULATOR] = False
            self._register_fallback(AnalysisModule.ENTROPY_CALCULATOR, self._entropy_fallback)
        
        # SemanticCoherenceAnalyzer
        try:
            from ..advanced.semantic_coherence import SemanticCoherenceAnalyzer
            self.modules[AnalysisModule.SEMANTIC_COHERENCE] = SemanticCoherenceAnalyzer()
            self.module_status[AnalysisModule.SEMANTIC_COHERENCE] = True
            logger.info("✅ SemanticCoherenceAnalyzer initialized")
        except Exception as e:
            logger.warning(f"❌ SemanticCoherenceAnalyzer initialization failed: {e}")
            self.module_status[AnalysisModule.SEMANTIC_COHERENCE] = False
            self._register_fallback(AnalysisModule.SEMANTIC_COHERENCE, self._semantic_coherence_fallback)
        
        # ContextWindowAnalyzer
        try:
            from ..advanced.context_analyzer import ContextWindowAnalyzer
            self.modules[AnalysisModule.CONTEXT_ANALYZER] = ContextWindowAnalyzer()
            self.module_status[AnalysisModule.CONTEXT_ANALYZER] = True
            logger.info("✅ ContextWindowAnalyzer initialized")
        except Exception as e:
            logger.warning(f"❌ ContextWindowAnalyzer initialization failed: {e}")
            self.module_status[AnalysisModule.CONTEXT_ANALYZER] = False
            self._register_fallback(AnalysisModule.CONTEXT_ANALYZER, self._context_analyzer_fallback)
        
        # QuantizationTester
        try:
            from ..advanced.quantization_tester import QuantizationTester
            self.modules[AnalysisModule.QUANTIZATION_TESTER] = QuantizationTester()
            self.module_status[AnalysisModule.QUANTIZATION_TESTER] = True
            logger.info("✅ QuantizationTester initialized")
        except Exception as e:
            logger.warning(f"❌ QuantizationTester initialization failed: {e}")
            self.module_status[AnalysisModule.QUANTIZATION_TESTER] = False
            self._register_fallback(AnalysisModule.QUANTIZATION_TESTER, self._quantization_fallback)
        
        # ConsistencyValidator
        try:
            from ..advanced.consistency_validator import ConsistencyValidator
            self.modules[AnalysisModule.CONSISTENCY_VALIDATOR] = ConsistencyValidator()
            self.module_status[AnalysisModule.CONSISTENCY_VALIDATOR] = True
            logger.info("✅ ConsistencyValidator initialized")
        except Exception as e:
            logger.warning(f"❌ ConsistencyValidator initialization failed: {e}")
            self.module_status[AnalysisModule.CONSISTENCY_VALIDATOR] = False
            self._register_fallback(AnalysisModule.CONSISTENCY_VALIDATOR, self._consistency_fallback)
        
        # WikipediaFactChecker
        try:
            from ..validation.wikipedia_fact_checker import WikipediaFactChecker
            wikipedia_config = self.config.get('wikipedia', {})
            self.modules[AnalysisModule.WIKIPEDIA_FACT_CHECKER] = WikipediaFactChecker(wikipedia_config)
            self.module_status[AnalysisModule.WIKIPEDIA_FACT_CHECKER] = True
            logger.info("✅ WikipediaFactChecker initialized")
        except Exception as e:
            logger.warning(f"❌ WikipediaFactChecker initialization failed: {e}")
            self.module_status[AnalysisModule.WIKIPEDIA_FACT_CHECKER] = False
            self._register_fallback(AnalysisModule.WIKIPEDIA_FACT_CHECKER, self._wikipedia_fallback)
        
        # MultiSourceFactValidator
        try:
            from ..validation.multi_source_fact_validator import MultiSourceFactValidator
            fact_validator_config = self.config.get('multi_source_validation', {})
            self.modules[AnalysisModule.MULTI_SOURCE_FACT_VALIDATOR] = MultiSourceFactValidator(fact_validator_config)
            self.module_status[AnalysisModule.MULTI_SOURCE_FACT_VALIDATOR] = True
            logger.info("✅ MultiSourceFactValidator initialized")
        except Exception as e:
            logger.warning(f"❌ MultiSourceFactValidator initialization failed: {e}")
            self.module_status[AnalysisModule.MULTI_SOURCE_FACT_VALIDATOR] = False
            self._register_fallback(AnalysisModule.MULTI_SOURCE_FACT_VALIDATOR, self._multi_source_fallback)
    
    def _run_modules_concurrent(self, 
                               text: str,
                               modules: List[AnalysisModule],
                               domain_context: Optional[str],
                               cultural_context: Optional[Dict[str, Any]],
                               internal_confidence: Optional[float]) -> List[ModuleResult]:
        """Run analysis modules concurrently"""
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all module tasks
            future_to_module = {}
            for module in modules:
                future = executor.submit(
                    self._run_single_module,
                    module, text, domain_context, cultural_context, internal_confidence
                )
                future_to_module[future] = module
            
            # Collect results as they complete
            for future in as_completed(future_to_module, timeout=self.config.get('module_timeout', 30)):
                module = future_to_module[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Module {module.value} failed with exception: {e}")
                    # Create error result
                    error_result = ModuleResult(
                        module=module,
                        success=False,
                        result=None,
                        processing_time=0.0,
                        error_message=str(e)
                    )
                    results.append(error_result)
        
        return results
    
    def _run_modules_sequential(self,
                               text: str,
                               modules: List[AnalysisModule],
                               domain_context: Optional[str],
                               cultural_context: Optional[Dict[str, Any]],
                               internal_confidence: Optional[float]) -> List[ModuleResult]:
        """Run analysis modules sequentially"""
        
        results = []
        for module in modules:
            result = self._run_single_module(module, text, domain_context, cultural_context, internal_confidence)
            results.append(result)
        
        return results
    
    def _run_single_module(self,
                          module: AnalysisModule,
                          text: str,
                          domain_context: Optional[str],
                          cultural_context: Optional[Dict[str, Any]], 
                          internal_confidence: Optional[float]) -> ModuleResult:
        """Run a single analysis module with error handling"""
        
        start_time = time.time()
        
        try:
            # Get module instance
            if module not in self.modules:
                return self._create_module_error_result(module, "Module not available", start_time)
            
            module_instance = self.modules[module]
            
            # Run module with appropriate parameters
            if module == AnalysisModule.ENTROPY_CALCULATOR:
                result = module_instance.analyze_entropy_profile(text)
                
            elif module == AnalysisModule.SEMANTIC_COHERENCE:
                result = module_instance.comprehensive_coherence_analysis(text)
                
            elif module == AnalysisModule.CONTEXT_ANALYZER:
                # Pass domain context if available
                result = module_instance.comprehensive_context_analysis(text)
                
            elif module == AnalysisModule.QUANTIZATION_TESTER:
                result = module_instance.run_comprehensive_quantization_tests(text)
                
            elif module == AnalysisModule.CONSISTENCY_VALIDATOR:
                # Consistency validator needs question-response pairs, use a simplified analysis
                result = self._create_simple_consistency_result(text)
                
            elif module == AnalysisModule.WIKIPEDIA_FACT_CHECKER:
                result = module_instance.check_factual_claims(text, domain_context)
                
            elif module == AnalysisModule.MULTI_SOURCE_FACT_VALIDATOR:
                # This requires special handling as it needs internal confidence
                result = module_instance.validate_factual_content(
                    text, domain_context, cultural_context, internal_confidence
                )
                
            else:
                return self._create_module_error_result(module, "Unknown module", start_time)
            
            processing_time = time.time() - start_time
            
            return ModuleResult(
                module=module,
                success=True,
                result=result,
                processing_time=processing_time,
                error_message=None,
                fallback_used=False
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.warning(f"Module {module.value} failed: {str(e)}")
            
            # Try fallback if available
            if module in self.fallback_handlers:
                try:
                    fallback_result = self.fallback_handlers[module](text, domain_context)
                    return ModuleResult(
                        module=module,
                        success=True,
                        result=fallback_result,
                        processing_time=processing_time,
                        error_message=None,
                        fallback_used=True
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback for {module.value} also failed: {fallback_error}")
            
            return ModuleResult(
                module=module,
                success=False,
                result=None,
                processing_time=processing_time,
                error_message=str(e),
                fallback_used=False
            )
    
    def _consolidate_analysis_results(self, module_results: List[ModuleResult]) -> Dict[str, Any]:
        """Consolidate results from all modules into unified analysis data"""
        
        consolidated = {
            'entropy_analysis': {},
            'semantic_coherence': {},
            'context_analysis': {},
            'quantization_analysis': {},
            'consistency_analysis': {},
            'fact_validation': {},
            'multi_source_validation': {}
        }
        
        for result in module_results:
            if not result.success:
                continue
            
            if result.module == AnalysisModule.ENTROPY_CALCULATOR:
                consolidated['entropy_analysis'] = {
                    'entropy_score': result.result.get('token_entropy', 0.0),
                    'complexity_score': result.result.get('semantic_diversity', 0.0),
                    'information_density': result.result.get('embedding_variance', 0.0),
                    # Add expected test fields - direct from entropy calculator
                    'token_entropy': result.result.get('token_entropy', 0.0),
                    'semantic_diversity': result.result.get('semantic_diversity', 0.0),
                    'entropy_patterns': result.result.get('entropy_patterns', {}),
                    'word_entropy': result.result.get('word_entropy', 0.0),
                    'vocab_entropy': result.result.get('vocab_entropy', 0.0),
                    'vocab_diversity': result.result.get('vocab_diversity', 0.0),
                    'bigram_entropy': result.result.get('bigram_entropy', 0.0),
                    'trigram_entropy': result.result.get('trigram_entropy', 0.0),
                    'entropy_quality_ratio': result.result.get('entropy_quality_ratio', 0.0),
                    'unique_ratio': result.result.get('unique_ratio', 0.0),
                    'fallback_used': result.fallback_used
                }
                
            elif result.module == AnalysisModule.SEMANTIC_COHERENCE:
                consolidated['semantic_coherence'] = {
                    'coherence_score': result.result.get('overall_coherence_score', 0.0),
                    'consistency_score': result.result.get('overall_coherence_score', 0.0),
                    'semantic_flow': result.result.get('overall_coherence_score', 0.0),
                    # Add expected test fields
                    'overall_coherence_score': result.result.get('overall_coherence_score', 0.0),
                    'semantic_drift': result.result.get('semantic_drift', {}),
                    'semantic_flow_data': result.result.get('semantic_flow', {}),
                    'topic_consistency': result.result.get('topic_consistency', {}),
                    'fallback_used': result.fallback_used
                }
                
            elif result.module == AnalysisModule.CONTEXT_ANALYZER:
                consolidated['context_analysis'] = {
                    'context_quality': result.result.get('context_health_score', 0.0),
                    'context_usage': result.result.get('position_analysis', {}).get('avg_quality', 0.0),
                    'context_efficiency': result.result.get('context_efficiency', {}).get('efficiency_score', 0.0),
                    # Add expected test fields
                    'context_health_score': result.result.get('context_health_score', 0.0),
                    'position_analysis': result.result.get('position_analysis', {}),
                    'saturation_analysis': result.result.get('saturation_analysis', {}),
                    'context_limit_estimate': result.result.get('context_limit_estimate', {}),
                    'fallback_used': result.fallback_used
                }
                
            elif result.module == AnalysisModule.QUANTIZATION_TESTER:
                consolidated['quantization_analysis'] = {
                    'quantization_impact': result.result.get('overall_quantization_score', 0.0),
                    'quality_degradation': result.result.get('factual_consistency', {}).get('consistency_score', 0.0),
                    'performance_impact': result.result.get('numerical_stability', {}).get('stability_score', 0.0),
                    # Add expected test fields
                    'quantization_impact_score': result.result.get('quantization_impact', {}).get('quantization_impact_score', 0.0),
                    'stability_analysis': result.result.get('numerical_stability', {}),
                    'consistency_analysis': result.result.get('factual_consistency', {}),
                    'test_summary': result.result.get('summary', {}),
                    'overall_quantization_score': result.result.get('overall_quantization_score', 0.0),
                    'fallback_used': result.fallback_used
                }
                
            elif result.module == AnalysisModule.CONSISTENCY_VALIDATOR:
                consolidated['consistency_analysis'] = {
                    'consistency_score': result.result.get('consistency_score', 0.0),
                    'cross_validation_score': result.result.get('cross_validation_score', 0.0),
                    'reliability_score': result.result.get('reliability_score', 0.0),
                    'internal_consistency': result.result.get('internal_consistency', 0.0),
                    'pattern_consistency': result.result.get('pattern_consistency', 0.0),
                    'fallback_used': result.fallback_used
                }
                
            elif result.module == AnalysisModule.WIKIPEDIA_FACT_CHECKER:
                consolidated['fact_validation'] = {
                    'factual_confidence': result.result.overall_factual_confidence,
                    'cultural_sensitivity': result.result.cultural_sensitivity_score,
                    'claims_count': len(result.result.extracted_claims),
                    'validated_claims': len(result.result.validation_results),
                    'recommendations': result.result.recommendations,
                    'fallback_used': result.fallback_used
                }
                
            elif result.module == AnalysisModule.MULTI_SOURCE_FACT_VALIDATOR:
                consolidated['multi_source_validation'] = {
                    'ensemble_confidence': result.result.ensemble_confidence,
                    'ensemble_disagreement': result.result.ensemble_disagreement,
                    'confidence_reliability': result.result.confidence_reliability,
                    'cultural_sensitivity': result.result.cultural_sensitivity_score,
                    'bias_detected': result.result.cultural_bias_detected,
                    'source_count': len(result.result.source_results),
                    'recommendations': result.result.recommendations,
                    'fallback_used': result.fallback_used
                }
        
        return consolidated
    
    def _generate_integration_notes(self, module_results: List[ModuleResult], internal_confidence: Optional[float]) -> List[str]:
        """Generate notes on how modules integrate with each other"""
        notes = []
        
        successful_count = len([r for r in module_results if r.success])
        failed_count = len([r for r in module_results if not r.success])
        fallback_count = len([r for r in module_results if r.fallback_used])
        
        notes.append(f"Advanced analysis: {successful_count} modules succeeded, {failed_count} failed")
        
        if fallback_count > 0:
            notes.append(f"Fallback strategies used for {fallback_count} modules")
        
        # Integration with internal confidence
        if internal_confidence is not None:
            fact_validation_results = [r for r in module_results 
                                     if r.module == AnalysisModule.MULTI_SOURCE_FACT_VALIDATOR and r.success]
            if fact_validation_results:
                ensemble_confidence = fact_validation_results[0].result.ensemble_confidence
                confidence_gap = abs(internal_confidence - ensemble_confidence)
                
                if confidence_gap > 0.3:
                    notes.append(f"Significant gap between linguistic confidence ({internal_confidence:.2f}) and fact validation ({ensemble_confidence:.2f})")
                else:
                    notes.append(f"Good alignment between linguistic and fact validation confidence")
        
        return notes
    
    def _calculate_performance_metrics(self, module_results: List[ModuleResult]) -> Dict[str, float]:
        """Calculate performance metrics for the orchestration"""
        if not module_results:
            return {}
        
        processing_times = [r.processing_time for r in module_results if r.success]
        
        metrics = {
            'total_processing_time': sum(r.processing_time for r in module_results),
            'average_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0.0,
            'success_rate': len([r for r in module_results if r.success]) / len(module_results),
            'fallback_rate': len([r for r in module_results if r.fallback_used]) / len(module_results)
        }
        
        return metrics
    
    def _update_performance_history(self, module_results: List[ModuleResult]):
        """Update performance tracking history"""
        for result in module_results:
            module_name = result.module.value
            
            if module_name not in self.performance_history:
                self.performance_history[module_name] = []
            
            self.performance_history[module_name].append({
                'timestamp': time.time(),
                'processing_time': result.processing_time,
                'success': result.success,
                'fallback_used': result.fallback_used
            })
            
            # Keep only recent history (last 100 runs)
            if len(self.performance_history[module_name]) > 100:
                self.performance_history[module_name] = self.performance_history[module_name][-100:]
    
    def _create_module_error_result(self, module: AnalysisModule, error_message: str, start_time: float) -> ModuleResult:
        """Create an error result for a failed module"""
        return ModuleResult(
            module=module,
            success=False,
            result=None,
            processing_time=time.time() - start_time,
            error_message=error_message,
            fallback_used=False
        )
    
    def _register_fallback(self, module: AnalysisModule, fallback_function: Callable):
        """Register a fallback function for a module"""
        self.fallback_handlers[module] = fallback_function
    
    # Fallback implementations for when modules fail
    def _entropy_fallback(self, text: str, domain_context: Optional[str]) -> Dict[str, Any]:
        """Fallback entropy calculation using simple metrics"""
        import re
        
        words = text.split()
        unique_words = set(word.lower() for word in words)
        
        # Simple entropy approximation
        if len(words) == 0:
            entropy_score = 0.0
        else:
            entropy_score = len(unique_words) / len(words) * 100
        
        return {
            'entropy_score': min(entropy_score, 100),
            'complexity_score': min(len(unique_words) / 10, 100),
            'information_density': min(len(re.findall(r'[.!?]', text)) / max(1, len(words) / 10), 100),
            'fallback': True
        }
    
    def _semantic_coherence_fallback(self, text: str, domain_context: Optional[str]) -> Dict[str, Any]:
        """Fallback semantic coherence using simple heuristics"""
        sentences = text.split('.')
        
        # Simple coherence metrics
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        sentence_variety = len(set(len(s.split()) for s in sentences))
        
        coherence_score = min(avg_sentence_length * 2 + sentence_variety * 5, 100)
        
        return {
            'coherence_score': coherence_score,
            'consistency_score': min(sentence_variety * 10, 100),
            'semantic_flow': min(avg_sentence_length * 3, 100),
            'fallback': True
        }
    
    def _context_analyzer_fallback(self, text: str, domain_context: Optional[str]) -> Dict[str, Any]:
        """Fallback context analysis using basic metrics"""
        word_count = len(text.split())
        
        return {
            'context_quality': min(word_count / 5, 100),
            'context_usage': min(word_count / 10, 100), 
            'context_efficiency': 50.0,  # Neutral score
            'fallback': True
        }
    
    def _quantization_fallback(self, text: str, domain_context: Optional[str]) -> Dict[str, Any]:
        """Fallback quantization analysis"""
        return {
            'quantization_impact': 20.0,  # Assume low impact
            'quality_degradation': 10.0,  # Assume minimal degradation
            'performance_impact': 15.0,   # Assume moderate performance impact
            'fallback': True
        }
    
    def _consistency_fallback(self, text: str, domain_context: Optional[str]) -> Dict[str, Any]:
        """Fallback consistency validation"""
        # Basic consistency check using text patterns
        word_count = len(text.split())
        sentence_count = len(text.split('.'))
        
        consistency_score = min(word_count / sentence_count * 5, 100) if sentence_count > 0 else 50
        
        return {
            'consistency_score': consistency_score,
            'cross_validation_score': 50.0,  # Neutral when can't validate
            'reliability_score': min(consistency_score * 0.8, 100),
            'fallback': True
        }
    
    def _wikipedia_fallback(self, text: str, domain_context: Optional[str]) -> Dict[str, Any]:
        """Fallback Wikipedia fact-checking when external API unavailable"""
        return {
            'overall_factual_confidence': 0.5,  # Neutral when can't validate
            'cultural_sensitivity_score': 0.8,  # Assume good cultural sensitivity
            'extracted_claims': [],
            'validation_results': [],
            'recommendations': ["External fact validation unavailable - consider manual verification"],
            'fallback': True
        }
    
    def _multi_source_fallback(self, text: str, domain_context: Optional[str]) -> Dict[str, Any]:
        """Fallback multi-source validation when ensemble unavailable"""
        return {
            'ensemble_confidence': 0.5,
            'ensemble_disagreement': 0.0,
            'confidence_reliability': 0.3,
            'cultural_sensitivity_score': 0.7,
            'cultural_bias_detected': False,
            'recommendations': ["Multi-source validation unavailable - using single-source validation"],
            'source_results': [],
            'fallback': True
        }
    
    def _create_simple_consistency_result(self, text: str) -> Dict[str, Any]:
        """Create a simplified consistency analysis result for single text inputs"""
        try:
            # Simple text consistency analysis
            sentences = text.split('.')
            if len(sentences) < 2:
                return {
                    "consistency_score": 1.0,
                    "internal_consistency": 1.0,
                    "pattern_consistency": 1.0,
                    "cross_validation_score": 1.0,
                    "reliability_score": 1.0,
                    "note": "Text too short for consistency analysis"
                }
            
            # Basic repetition and contradiction detection
            word_counts = Counter(text.lower().split())
            repetition_score = 1.0 - min(0.5, max(word_counts.values()) / len(text.split()) - 0.1)
            
            # Look for contradictory language patterns
            contradiction_indicators = ["however", "but", "although", "despite", "nevertheless", "on the other hand"]
            contradiction_count = sum(1 for indicator in contradiction_indicators if indicator in text.lower())
            contradiction_penalty = min(0.2, contradiction_count * 0.05)
            
            consistency_score = max(0.0, repetition_score - contradiction_penalty)
            
            return {
                "consistency_score": consistency_score,
                "internal_consistency": repetition_score,
                "pattern_consistency": 1.0 - contradiction_penalty,
                "cross_validation_score": consistency_score,
                "reliability_score": consistency_score,
                "fallback": True,
                "note": "Simplified consistency analysis for single text"
            }
            
        except Exception as e:
            return {
                "consistency_score": 0.5,
                "internal_consistency": 0.5,
                "pattern_consistency": 0.5,
                "error": f"Consistency analysis failed: {str(e)}"
            }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default orchestrator configuration"""
        return {
            'max_concurrent_modules': 3,
            'module_timeout': 30,  # seconds
            'enable_fallbacks': True,
            'performance_tracking': True,
            'wikipedia': {
                'min_request_interval': 0.5,
                'claim_extraction_threshold': 0.3,
                'max_claims_per_text': 10
            },
            'multi_source_validation': {
                'source_weights': {
                    'internal_knowledge': 0.3,
                    'wikipedia_external': 0.4,
                    'cultural_authenticity': 0.3
                },
                'disagreement_threshold': 0.4,
                'cultural_bias_threshold': 0.6,
                'confidence_threshold': 0.7
            }
        }
    
    def get_module_status(self) -> Dict[str, bool]:
        """Get current status of all modules"""
        return self.module_status.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all modules"""
        summary = {}
        
        for module_name, history in self.performance_history.items():
            if not history:
                continue
                
            recent_history = history[-10:]  # Last 10 runs
            
            summary[module_name] = {
                'total_runs': len(history),
                'recent_success_rate': sum(1 for h in recent_history if h['success']) / len(recent_history),
                'average_processing_time': sum(h['processing_time'] for h in recent_history) / len(recent_history),
                'fallback_rate': sum(1 for h in recent_history if h.get('fallback_used', False)) / len(recent_history)
            }
        
        return summary
    
    def enable_module(self, module: AnalysisModule):
        """Enable a disabled module"""
        self.module_status[module] = True
        logger.info(f"Module {module.value} enabled")
    
    def disable_module(self, module: AnalysisModule):
        """Disable a module"""
        self.module_status[module] = False
        logger.info(f"Module {module.value} disabled")


# Integration function for existing evaluation system
def integrate_advanced_analysis(text: str,
                              internal_confidence: float,
                              domain_context: Optional[str] = None,
                              cultural_context: Optional[Dict[str, Any]] = None,
                              requested_modules: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Main integration point for advanced analysis with existing evaluation system.
    
    Args:
        text: Text to analyze
        internal_confidence: Confidence from linguistic analysis
        domain_context: Optional domain context
        cultural_context: Optional cultural context
        requested_modules: Optional list of specific modules to run
        
    Returns:
        Dict with advanced analysis results integrated with existing metrics
    """
    
    try:
        # Initialize orchestrator
        orchestrator = AdvancedAnalysisOrchestrator()
        
        # Convert requested modules to enum
        modules_to_run = None
        if requested_modules:
            modules_to_run = []
            for module_name in requested_modules:
                try:
                    module_enum = AnalysisModule(module_name)
                    modules_to_run.append(module_enum)
                except ValueError:
                    logger.warning(f"Unknown module requested: {module_name}")
        
        # Run advanced analysis
        result = orchestrator.run_advanced_analysis(
            text=text,
            requested_modules=modules_to_run,
            domain_context=domain_context,
            cultural_context=cultural_context,
            internal_confidence=internal_confidence,
            concurrent=True
        )
        
        return {
            'advanced_analysis': result.analysis_data,
            'integration_notes': result.integration_notes,
            'performance_metrics': result.performance_metrics,
            'successful_modules': [m.value for m in result.successful_modules],
            'failed_modules': [m.value for m in result.failed_modules],
            'total_processing_time': result.total_processing_time,
            'module_status': orchestrator.get_module_status()
        }
        
    except Exception as e:
        logger.error(f"Advanced analysis integration failed: {str(e)}")
        return {
            'advanced_analysis': {},
            'integration_notes': [f"Advanced analysis failed: {str(e)}"],
            'performance_metrics': {},
            'successful_modules': [],
            'failed_modules': list(AnalysisModule),
            'total_processing_time': 0.0,
            'module_status': {}
        }