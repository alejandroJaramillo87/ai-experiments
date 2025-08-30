"""
AI Workstation Consciousness System

Advanced autonomous optimization and consciousness system for RTX 5090 + AMD 9950X 
AI engineering workstations. Provides comprehensive intelligence across:

Phase 1 - Container Consciousness:
- Docker-based AI service orchestration monitoring
- Service lifecycle semantic event extraction  
- Cross-container resource correlation analysis

Phase 2 - Hardware Specialization:
- RTX 5090 Blackwall architecture optimization
- AMD Zen 5 workload performance tuning
- 15-fan thermal intelligence management

Phase 3 - Multi-Model Oracle:
- Intelligent resource optimization engine
- ML-based workload performance prediction
- Autonomous optimization strategy execution

Author: AI Workstation Intelligence System
"""

# Main controller - unified orchestration interface
from .ai_workstation_controller import AIWorkstationController, AIWorkstationMode, SystemHealthStatus

# Phase 1: Container Consciousness
from .container_consciousness.ai_container_detector import AIContainerOrchestratorDetector
from .container_consciousness.service_lifecycle_extractor import AIServiceLifecycleExtractor
from .container_consciousness.container_correlator import ContainerResourceCorrelator

# Phase 2: Hardware Specialization
from .hardware_specialization.rtx5090_blackwell_detector import RTX5090BlackwallDetector
from .hardware_specialization.amd_zen5_detector import AMDZen5WorkloadDetector
from .hardware_specialization.thermal_intelligence_detector import ThermalIntelligenceDetector

# Phase 3: Multi-Model Oracle
from .multi_model_oracle.resource_oracle import MultiModelResourceOracle
from .multi_model_oracle.workload_predictor import AIWorkloadPredictor, WorkloadFeatures, PerformancePrediction
from .multi_model_oracle.strategy_engine import PerformanceStrategyEngine, OptimizationStrategy, OptimizationAction

# Phase 4: Natural Language Intelligence
from .natural_language_intelligence.natural_language_orchestrator import NaturalLanguageOrchestrator
from .natural_language_intelligence.intent_understanding_engine import IntentUnderstandingEngine, QueryIntent, SystemComponent

__all__ = [
    # Main controller
    'AIWorkstationController',
    'AIWorkstationMode', 
    'SystemHealthStatus',
    
    # Phase 1: Container Consciousness
    'AIContainerOrchestratorDetector',
    'AIServiceLifecycleExtractor', 
    'ContainerResourceCorrelator',
    
    # Phase 2: Hardware Specialization
    'RTX5090BlackwallDetector',
    'AMDZen5WorkloadDetector',
    'ThermalIntelligenceDetector',
    
    # Phase 3: Multi-Model Oracle
    'MultiModelResourceOracle',
    'AIWorkloadPredictor',
    'WorkloadFeatures',
    'PerformancePrediction',
    'PerformanceStrategyEngine',
    'OptimizationStrategy',
    'OptimizationAction',
    
    # Phase 4: Natural Language Intelligence
    'NaturalLanguageOrchestrator',
    'IntentUnderstandingEngine',
    'QueryIntent',
    'SystemComponent'
]

__version__ = '2.0.0'  # Major version bump for comprehensive consciousness system