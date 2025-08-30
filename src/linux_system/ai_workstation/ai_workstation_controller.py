"""
AI Workstation Controller

Unified orchestrator for the AI workstation consciousness system. Integrates
container consciousness, hardware specialization, and multi-model oracle into
a cohesive autonomous optimization platform for RTX 5090 + AMD 9950X systems.

This controller provides:
- Unified AI workstation consciousness interface
- Autonomous optimization coordination  
- Cross-component intelligence integration
- Real-time system adaptation
- Performance optimization orchestration
"""

import logging
import asyncio
import time
import threading
import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque

# Temporal intelligence system imports
from ..temporal_intelligence.temporal_system_collector import TemporalSystemCollector
from ..temporal_intelligence.core_structures import SystemEvent, EventType, EventSeverity

# Container consciousness imports
from .container_consciousness.ai_container_detector import AIContainerOrchestratorDetector
from .container_consciousness.service_lifecycle_extractor import AIServiceLifecycleExtractor
from .container_consciousness.container_correlator import ContainerResourceCorrelator

# Hardware specialization imports
from .hardware_specialization.rtx5090_blackwell_detector import RTX5090BlackwallDetector
from .hardware_specialization.amd_zen5_detector import AMDZen5WorkloadDetector
from .hardware_specialization.thermal_intelligence_detector import ThermalIntelligenceDetector

# Multi-model oracle imports
from .multi_model_oracle.resource_oracle import MultiModelResourceOracle
from .multi_model_oracle.workload_predictor import AIWorkloadPredictor, WorkloadFeatures
from .multi_model_oracle.strategy_engine import PerformanceStrategyEngine, OptimizationPlan

# Phase 4: Natural Language Intelligence imports
from .natural_language_intelligence.natural_language_orchestrator import NaturalLanguageOrchestrator


class AIWorkstationMode(Enum):
    """Operating modes for the AI workstation"""
    MONITORING = "monitoring"           # Passive monitoring only
    LEARNING = "learning"              # Active learning with limited optimization
    OPTIMIZING = "optimizing"          # Full autonomous optimization
    PERFORMANCE = "performance"        # Maximum performance mode
    EFFICIENCY = "efficiency"          # Energy/thermal efficiency focused
    MAINTENANCE = "maintenance"        # System maintenance and diagnostics


class SystemHealthStatus(Enum):
    """Overall system health assessment"""
    EXCELLENT = "excellent"    # All systems operating optimally
    GOOD = "good"             # Normal operation with minor issues
    FAIR = "fair"             # Some performance degradation
    POOR = "poor"             # Significant issues affecting performance
    CRITICAL = "critical"     # System stability at risk


@dataclass
class AIWorkstationStatus:
    """Current status of the AI workstation system"""
    timestamp: datetime
    mode: AIWorkstationMode
    health_status: SystemHealthStatus
    
    # Component status
    container_consciousness_active: bool
    hardware_specialization_active: bool
    multi_model_oracle_active: bool
    temporal_intelligence_active: bool
    
    # Performance metrics
    overall_performance_score: float  # 0.0 to 1.0
    optimization_effectiveness: float # 0.0 to 1.0
    thermal_efficiency: float        # 0.0 to 1.0
    resource_utilization: Dict[str, float]
    
    # System insights
    active_workloads: List[str]
    optimization_opportunities: List[str]
    critical_alerts: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API/logging"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'mode': self.mode.value,
            'health_status': self.health_status.value,
            'container_consciousness_active': self.container_consciousness_active,
            'hardware_specialization_active': self.hardware_specialization_active,
            'multi_model_oracle_active': self.multi_model_oracle_active,
            'temporal_intelligence_active': self.temporal_intelligence_active,
            'overall_performance_score': self.overall_performance_score,
            'optimization_effectiveness': self.optimization_effectiveness,
            'thermal_efficiency': self.thermal_efficiency,
            'resource_utilization': self.resource_utilization,
            'active_workloads': self.active_workloads,
            'optimization_opportunities': self.optimization_opportunities,
            'critical_alerts': self.critical_alerts,
            'recommendations': self.recommendations
        }


class AIWorkstationController:
    """
    Unified AI Workstation Consciousness Controller
    
    Orchestrates all AI workstation intelligence components to provide autonomous
    optimization, predictive performance management, and adaptive resource allocation
    for RTX 5090 + AMD 9950X systems running multi-model AI workloads.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(f"{__name__}.AIWorkstationController")
        
        # Configuration
        self.config_path = config_path or "/etc/ai-workstation/config.yaml"
        self.config = self._load_configuration()
        
        # Operating state
        self.mode = AIWorkstationMode.MONITORING
        self.running = False
        self.start_time = None
        
        # Core components - will be initialized during startup
        self.temporal_collector: Optional[TemporalSystemCollector] = None
        
        # Phase 1: Container Consciousness
        self.container_detector: Optional[AIContainerOrchestratorDetector] = None
        self.service_extractor: Optional[AIServiceLifecycleExtractor] = None
        self.container_correlator: Optional[ContainerResourceCorrelator] = None
        
        # Phase 2: Hardware Specialization
        self.rtx5090_detector: Optional[RTX5090BlackwallDetector] = None
        self.amd_zen5_detector: Optional[AMDZen5WorkloadDetector] = None
        self.thermal_detector: Optional[ThermalIntelligenceDetector] = None
        
        # Phase 3: Multi-Model Oracle
        self.resource_oracle: Optional[MultiModelResourceOracle] = None
        self.workload_predictor: Optional[AIWorkloadPredictor] = None
        self.strategy_engine: Optional[PerformanceStrategyEngine] = None
        
        # Phase 4: Natural Language Intelligence
        self.natural_language_orchestrator: Optional[NaturalLanguageOrchestrator] = None
        
        # Integration state
        self.status_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(list)
        self.optimization_log = deque(maxlen=500)
        
        # Event handling
        self.event_handlers = defaultdict(list)
        self.alert_handlers = []
        
        # Control tasks
        self.control_tasks = []
        
        self.logger.info("AIWorkstationController initialized")
    
    async def start_ai_workstation(self, mode: AIWorkstationMode = AIWorkstationMode.MONITORING) -> Dict[str, Any]:
        """Start the AI workstation consciousness system"""
        if self.running:
            return {"success": False, "error": "AI workstation already running"}
        
        self.logger.info(f"Starting AI workstation in {mode.value} mode")
        self.mode = mode
        self.running = True
        self.start_time = datetime.now()
        
        try:
            # Initialize core temporal intelligence
            await self._initialize_temporal_intelligence()
            
            # Initialize Phase 1: Container Consciousness
            await self._initialize_container_consciousness()
            
            # Initialize Phase 2: Hardware Specialization
            await self._initialize_hardware_specialization()
            
            # Initialize Phase 3: Multi-Model Oracle
            await self._initialize_multi_model_oracle()
            
            # Initialize Phase 4: Natural Language Intelligence
            await self._initialize_natural_language_intelligence()
            
            # Start integration and coordination loops
            await self._start_coordination_loops()
            
            # Register event handlers
            self._register_event_handlers()
            
            self.logger.info(f"AI workstation started successfully in {mode.value} mode")
            
            return {
                "success": True,
                "mode": mode.value,
                "start_time": self.start_time.isoformat(),
                "components_initialized": self._get_component_status()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to start AI workstation: {e}")
            await self.stop_ai_workstation()
            return {"success": False, "error": str(e)}
    
    async def stop_ai_workstation(self) -> Dict[str, Any]:
        """Stop the AI workstation consciousness system gracefully"""
        self.logger.info("Stopping AI workstation...")
        self.running = False
        
        try:
            # Stop coordination loops
            for task in self.control_tasks:
                if not task.done():
                    task.cancel()
            
            # Stop Multi-Model Oracle components
            if self.strategy_engine:
                await self.strategy_engine.stop_optimization_engine()
            
            # Stop hardware specialization components
            if self.rtx5090_detector:
                self.rtx5090_detector.stop()
            if self.amd_zen5_detector:
                self.amd_zen5_detector.stop()
            if self.thermal_detector:
                self.thermal_detector.stop()
            
            # Stop container consciousness components
            if self.container_detector:
                self.container_detector.stop()
            
            # Stop temporal intelligence
            if self.temporal_collector:
                await self.temporal_collector.stop_collection()
            
            # Wait for tasks to complete
            if self.control_tasks:
                await asyncio.gather(*self.control_tasks, return_exceptions=True)
            
            self.logger.info("AI workstation stopped successfully")
            
            return {
                "success": True,
                "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                "final_status": await self.get_workstation_status()
            }
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            return {"success": False, "error": str(e)}
    
    async def set_workstation_mode(self, mode: AIWorkstationMode) -> Dict[str, Any]:
        """Change the operating mode of the AI workstation"""
        if not self.running:
            return {"success": False, "error": "AI workstation not running"}
        
        previous_mode = self.mode
        self.mode = mode
        
        self.logger.info(f"Switching from {previous_mode.value} to {mode.value} mode")
        
        try:
            # Adjust component behavior based on mode
            await self._adjust_components_for_mode(mode)
            
            self.logger.info(f"Successfully switched to {mode.value} mode")
            
            return {
                "success": True,
                "previous_mode": previous_mode.value,
                "new_mode": mode.value,
                "adjustments_applied": True
            }
            
        except Exception as e:
            # Revert to previous mode on failure
            self.mode = previous_mode
            self.logger.error(f"Failed to switch mode: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_workstation_status(self) -> AIWorkstationStatus:
        """Get comprehensive status of the AI workstation"""
        try:
            # Collect component status
            component_status = self._get_component_status()
            
            # Collect performance metrics
            performance_score = await self._calculate_overall_performance()
            optimization_effectiveness = await self._calculate_optimization_effectiveness()
            thermal_efficiency = await self._calculate_thermal_efficiency()
            resource_utilization = await self._collect_resource_utilization()
            
            # Collect system insights
            active_workloads = await self._identify_active_workloads()
            optimization_opportunities = await self._identify_optimization_opportunities()
            critical_alerts = await self._collect_critical_alerts()
            recommendations = await self._generate_recommendations()
            
            # Determine overall health
            health_status = self._assess_system_health(
                performance_score, thermal_efficiency, critical_alerts
            )
            
            status = AIWorkstationStatus(
                timestamp=datetime.now(),
                mode=self.mode,
                health_status=health_status,
                container_consciousness_active=component_status.get('container_consciousness', False),
                hardware_specialization_active=component_status.get('hardware_specialization', False),
                multi_model_oracle_active=component_status.get('multi_model_oracle', False),
                temporal_intelligence_active=component_status.get('temporal_intelligence', False),
                overall_performance_score=performance_score,
                optimization_effectiveness=optimization_effectiveness,
                thermal_efficiency=thermal_efficiency,
                resource_utilization=resource_utilization,
                active_workloads=active_workloads,
                optimization_opportunities=optimization_opportunities,
                critical_alerts=critical_alerts,
                recommendations=recommendations
            )
            
            # Store in history
            self.status_history.append(status)
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to collect status: {e}")
            # Return minimal status on error
            return AIWorkstationStatus(
                timestamp=datetime.now(),
                mode=self.mode,
                health_status=SystemHealthStatus.POOR,
                container_consciousness_active=False,
                hardware_specialization_active=False,
                multi_model_oracle_active=False,
                temporal_intelligence_active=False,
                overall_performance_score=0.0,
                optimization_effectiveness=0.0,
                thermal_efficiency=0.0,
                resource_utilization={},
                active_workloads=[],
                optimization_opportunities=[],
                critical_alerts=[f"Status collection failed: {str(e)}"],
                recommendations=["Check system logs for detailed error information"]
            )
    
    async def execute_optimization(self, optimization_targets: Dict[str, float]) -> Dict[str, Any]:
        """Execute targeted optimization with specific goals"""
        if not self.running or self.mode == AIWorkstationMode.MONITORING:
            return {"success": False, "error": "Optimization not available in current mode"}
        
        if not self.strategy_engine:
            return {"success": False, "error": "Strategy engine not available"}
        
        try:
            self.logger.info(f"Executing optimization with targets: {optimization_targets}")
            
            # Get current performance baseline
            current_status = await self.get_workstation_status()
            current_performance = {
                'overall_score': current_status.overall_performance_score,
                'thermal_efficiency': current_status.thermal_efficiency,
                'resource_utilization': current_status.resource_utilization
            }
            
            # Generate optimization plan
            optimization_plan = await self.strategy_engine.generate_optimization_strategy(
                current_performance, optimization_targets
            )
            
            # Execute the plan
            execution_result = await self.strategy_engine.execute_optimization_plan(optimization_plan)
            
            # Log the optimization
            self.optimization_log.append({
                'timestamp': datetime.now().isoformat(),
                'targets': optimization_targets,
                'plan_id': optimization_plan.plan_id,
                'result': execution_result
            })
            
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Optimization execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_performance_insights(self) -> Dict[str, Any]:
        """Get comprehensive performance insights and predictions"""
        if not self.running:
            return {"error": "AI workstation not running"}
        
        try:
            insights = {
                "timestamp": datetime.now().isoformat(),
                "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0,
                "current_mode": self.mode.value
            }
            
            # Container consciousness insights
            if self.container_correlator:
                container_insights = await self.container_correlator.get_correlation_insights()
                insights["container_intelligence"] = container_insights
            
            # Hardware specialization insights
            hardware_insights = {}
            if self.rtx5090_detector:
                rtx_insights = self.rtx5090_detector.get_blackwall_insights()
                hardware_insights["rtx5090_blackwall"] = rtx_insights
            
            if self.amd_zen5_detector:
                zen5_insights = self.amd_zen5_detector.get_zen5_insights()
                hardware_insights["amd_zen5"] = zen5_insights
            
            if self.thermal_detector:
                thermal_insights = self.thermal_detector.get_thermal_insights()
                hardware_insights["thermal_intelligence"] = thermal_insights
            
            insights["hardware_specialization"] = hardware_insights
            
            # Multi-model oracle insights
            oracle_insights = {}
            if self.workload_predictor:
                prediction_insights = self.workload_predictor.get_workload_insights()
                oracle_insights["workload_predictions"] = prediction_insights
            
            if self.strategy_engine:
                optimization_status = self.strategy_engine.get_optimization_status()
                oracle_insights["optimization_engine"] = optimization_status
            
            insights["multi_model_oracle"] = oracle_insights
            
            # Performance trends
            if len(self.status_history) > 5:
                trends = self._calculate_performance_trends()
                insights["performance_trends"] = trends
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to collect performance insights: {e}")
            return {"error": str(e)}
    
    async def process_natural_language_query(self, query: str, 
                                           session_id: Optional[str] = None,
                                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process natural language queries about the AI workstation consciousness system.
        
        This method provides the digital symbiosis interface between human language
        and machine consciousness, enabling intuitive interaction with the sophisticated
        consciousness systems.
        
        Args:
            query: Natural language question from user
            session_id: Optional session ID for conversational continuity
            context: Optional additional context for query processing
            
        Returns:
            Complete response with natural language answer, confidence, visualizations,
            follow-up suggestions, and technical insights
        """
        if not self.running:
            return {
                'answer': 'AI workstation is not running. Please start the system first.',
                'confidence': 0.0,
                'structured_data': {'error': 'System not running'},
                'visualizations': [],
                'follow_up_suggestions': ['Start the AI workstation system'],
                'timestamp': datetime.now(),
                'processing_time_ms': 0
            }
        
        if not self.natural_language_orchestrator:
            return {
                'answer': 'Natural language processing is not available. The consciousness system may still be initializing.',
                'confidence': 0.0,
                'structured_data': {'error': 'Natural language orchestrator not available'},
                'visualizations': [],
                'follow_up_suggestions': ['Wait for system initialization to complete'],
                'timestamp': datetime.now(),
                'processing_time_ms': 0
            }
        
        try:
            # Process query through the natural language orchestrator
            return await self.natural_language_orchestrator.process_natural_language_query(
                query, session_id, context
            )
            
        except Exception as e:
            self.logger.error(f"Natural language query processing failed: {e}")
            return {
                'answer': f'I encountered an error processing your question: {str(e)}. Please try rephrasing your question or check system status.',
                'confidence': 0.0,
                'structured_data': {'error': str(e)},
                'visualizations': [],
                'follow_up_suggestions': [
                    'Check AI workstation system status',
                    'Try asking about specific components like GPU or CPU',
                    'Ask simpler questions about current system state'
                ],
                'timestamp': datetime.now(),
                'processing_time_ms': 0
            }
    
    async def get_real_time_intelligence_stream(self, stream_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Get real-time intelligence data stream for WebSocket streaming.
        
        Provides structured data from all consciousness systems for real-time
        monitoring and visualization in the frontend.
        
        Args:
            stream_type: Type of stream ('comprehensive', 'hardware', 'containers', 'thermal', 'predictions')
            
        Returns:
            Structured real-time data from consciousness systems
        """
        if not self.running:
            return {"error": "AI workstation not running"}
        
        try:
            base_data = {
                "timestamp": datetime.now().isoformat(),
                "stream_type": stream_type,
                "system_mode": self.mode.value,
                "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0
            }
            
            if stream_type in ["comprehensive", "hardware"]:
                # Hardware specialization data
                hardware_data = {}
                if self.rtx5090_detector:
                    rtx_data = self.rtx5090_detector.get_blackwall_insights()
                    hardware_data["rtx5090_blackwall"] = rtx_data
                
                if self.amd_zen5_detector:
                    zen5_data = self.amd_zen5_detector.get_zen5_insights()
                    hardware_data["amd_zen5"] = zen5_data
                    
                if self.thermal_detector:
                    thermal_data = self.thermal_detector.get_thermal_insights()
                    hardware_data["thermal_intelligence"] = thermal_data
                
                base_data["hardware_specialization"] = hardware_data
            
            if stream_type in ["comprehensive", "containers"]:
                # Container consciousness data
                if self.container_correlator:
                    container_insights = await self.container_correlator.get_correlation_insights()
                    base_data["container_intelligence"] = container_insights
            
            if stream_type in ["comprehensive", "predictions"]:
                # Multi-model oracle predictions
                oracle_data = {}
                if self.workload_predictor:
                    prediction_insights = self.workload_predictor.get_workload_insights()
                    oracle_data["predictions"] = prediction_insights
                
                if self.strategy_engine:
                    optimization_status = self.strategy_engine.get_optimization_status()
                    oracle_data["optimization_status"] = optimization_status
                
                base_data["multi_model_oracle"] = oracle_data
            
            if stream_type in ["comprehensive", "temporal"]:
                # Temporal intelligence data (if available from performance insights)
                performance_trends = {}
                if len(self.status_history) > 5:
                    trends = self._calculate_performance_trends()
                    performance_trends = trends
                
                base_data["temporal_intelligence"] = performance_trends
            
            # Add system status summary for all stream types
            current_status = await self.get_workstation_status()
            base_data["system_summary"] = {
                "health_status": current_status.health_status.value,
                "overall_performance_score": current_status.overall_performance_score,
                "thermal_efficiency": current_status.thermal_efficiency,
                "active_workloads": current_status.active_workloads,
                "critical_alerts": current_status.critical_alerts[:3]  # Limit alerts
            }
            
            return base_data
            
        except Exception as e:
            self.logger.error(f"Failed to get real-time intelligence stream: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "stream_type": stream_type
            }
    
    def get_natural_language_orchestrator_status(self) -> Dict[str, Any]:
        """Get status of the natural language orchestrator"""
        if not self.natural_language_orchestrator:
            return {"status": "not_initialized"}
        
        return self.natural_language_orchestrator.get_orchestrator_status()
    
    # Internal initialization methods
    async def _initialize_temporal_intelligence(self) -> None:
        """Initialize the temporal intelligence system"""
        self.logger.info("Initializing temporal intelligence system...")
        
        # Create temporal collector with AI workstation configuration
        self.temporal_collector = TemporalSystemCollector(
            storage_path=self.config.get('temporal_storage_path', '/var/lib/ai-workstation/temporal'),
            collection_interval=self.config.get('collection_interval', 30)
        )
        
        # Start temporal collection
        await self.temporal_collector.start_collection()
        
        self.logger.info("Temporal intelligence system initialized")
    
    async def _initialize_container_consciousness(self) -> None:
        """Initialize Phase 1: Container Consciousness components"""
        self.logger.info("Initializing container consciousness...")
        
        # Initialize container detector
        self.container_detector = AIContainerOrchestratorDetector()
        
        # Initialize service lifecycle extractor
        self.service_extractor = AIServiceLifecycleExtractor()
        
        # Initialize container correlator with sophisticated analysis
        self.container_correlator = ContainerResourceCorrelator()
        
        # Start components if not in monitoring mode
        if self.mode != AIWorkstationMode.MONITORING:
            self.container_detector.start()
        
        self.logger.info("Container consciousness initialized")
    
    async def _initialize_hardware_specialization(self) -> None:
        """Initialize Phase 2: Hardware Specialization components"""
        self.logger.info("Initializing hardware specialization...")
        
        # Initialize RTX 5090 Blackwall detector
        try:
            self.rtx5090_detector = RTX5090BlackwallDetector()
            if self.mode != AIWorkstationMode.MONITORING:
                self.rtx5090_detector.start()
        except Exception as e:
            self.logger.warning(f"RTX 5090 detector initialization failed: {e}")
        
        # Initialize AMD Zen 5 detector
        try:
            self.amd_zen5_detector = AMDZen5WorkloadDetector()
            if self.mode != AIWorkstationMode.MONITORING:
                self.amd_zen5_detector.start()
        except Exception as e:
            self.logger.warning(f"AMD Zen 5 detector initialization failed: {e}")
        
        # Initialize thermal intelligence detector
        try:
            self.thermal_detector = ThermalIntelligenceDetector()
            if self.mode != AIWorkstationMode.MONITORING:
                self.thermal_detector.start()
        except Exception as e:
            self.logger.warning(f"Thermal detector initialization failed: {e}")
        
        self.logger.info("Hardware specialization initialized")
    
    async def _initialize_multi_model_oracle(self) -> None:
        """Initialize Phase 3: Multi-Model Oracle components"""
        self.logger.info("Initializing multi-model oracle...")
        
        # Initialize workload predictor
        self.workload_predictor = AIWorkloadPredictor()
        
        # Initialize resource oracle with all detectors
        self.resource_oracle = MultiModelResourceOracle(
            container_detector=self.container_detector,
            rtx5090_detector=self.rtx5090_detector,
            amd_zen5_detector=self.amd_zen5_detector,
            thermal_detector=self.thermal_detector
        )
        
        # Initialize strategy engine with oracle and predictor
        self.strategy_engine = PerformanceStrategyEngine(
            resource_oracle=self.resource_oracle,
            workload_predictor=self.workload_predictor
        )
        
        # Start oracle components if in optimization modes
        if self.mode in [AIWorkstationMode.OPTIMIZING, AIWorkstationMode.PERFORMANCE, AIWorkstationMode.EFFICIENCY]:
            await self.strategy_engine.start_optimization_engine()
        
        self.logger.info("Multi-model oracle initialized")
    
    async def _initialize_natural_language_intelligence(self) -> None:
        """Initialize Phase 4: Natural Language Intelligence"""
        self.logger.info("Initializing natural language intelligence...")
        
        # Initialize natural language orchestrator with reference to this controller
        self.natural_language_orchestrator = NaturalLanguageOrchestrator(consciousness_controller=self)
        
        self.logger.info("Natural language intelligence initialized")
    
    async def _start_coordination_loops(self) -> None:
        """Start coordination and monitoring loops"""
        self.logger.info("Starting coordination loops...")
        
        # Status monitoring loop
        self.control_tasks.append(asyncio.create_task(self._status_monitoring_loop()))
        
        # Cross-component coordination loop
        self.control_tasks.append(asyncio.create_task(self._coordination_loop()))
        
        # Performance optimization loop (if in optimization modes)
        if self.mode in [AIWorkstationMode.OPTIMIZING, AIWorkstationMode.PERFORMANCE, AIWorkstationMode.EFFICIENCY]:
            self.control_tasks.append(asyncio.create_task(self._optimization_loop()))
        
        self.logger.info("Coordination loops started")
    
    async def _status_monitoring_loop(self) -> None:
        """Monitor system status and health continuously"""
        while self.running:
            try:
                # Collect and log current status
                status = await self.get_workstation_status()
                
                # Check for critical conditions
                if status.health_status == SystemHealthStatus.CRITICAL:
                    await self._handle_critical_condition(status)
                
                # Log performance metrics for trend analysis
                self.performance_metrics['overall_score'].append(status.overall_performance_score)
                self.performance_metrics['thermal_efficiency'].append(status.thermal_efficiency)
                
                # Maintain metrics history size
                for metric_list in self.performance_metrics.values():
                    if len(metric_list) > 1000:
                        metric_list.pop(0)
                
                await asyncio.sleep(30)  # Status check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Status monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _coordination_loop(self) -> None:
        """Coordinate between different system components"""
        while self.running:
            try:
                # Coordinate container consciousness with hardware specialization
                if self.container_correlator and self.thermal_detector:
                    await self._coordinate_thermal_container_optimization()
                
                # Coordinate resource oracle with workload predictor
                if self.resource_oracle and self.workload_predictor:
                    await self._coordinate_prediction_optimization()
                
                await asyncio.sleep(60)  # Coordination every minute
                
            except Exception as e:
                self.logger.error(f"Coordination error: {e}")
                await asyncio.sleep(120)
    
    async def _optimization_loop(self) -> None:
        """Autonomous optimization execution loop"""
        while self.running and self.mode in [AIWorkstationMode.OPTIMIZING, AIWorkstationMode.PERFORMANCE]:
            try:
                # Check if optimization is needed based on performance trends
                should_optimize = await self._should_trigger_optimization()
                
                if should_optimize:
                    # Determine optimization targets based on mode
                    targets = await self._determine_optimization_targets()
                    
                    # Execute optimization
                    result = await self.execute_optimization(targets)
                    
                    if result.get('success'):
                        self.logger.info("Autonomous optimization executed successfully")
                    else:
                        self.logger.warning(f"Autonomous optimization failed: {result.get('error')}")
                
                await asyncio.sleep(300)  # Optimization check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(600)
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load AI workstation configuration"""
        config = {
            "temporal_storage_path": "/var/lib/ai-workstation/temporal",
            "collection_interval": 30,
            "optimization_thresholds": {
                "performance_degradation": 0.15,  # 15% degradation triggers optimization
                "thermal_warning": 75.0,          # Temperature threshold
                "resource_utilization": 0.90      # Resource utilization threshold
            },
            "safety_limits": {
                "max_gpu_power": 575,    # Watts
                "max_gpu_temp": 83,      # Celsius
                "max_cpu_temp": 90       # Celsius
            }
        }
        
        try:
            config_path = Path(self.config_path)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    config.update(loaded_config)
        except Exception as e:
            self.logger.warning(f"Failed to load config from {self.config_path}: {e}")
        
        return config
    
    def _get_component_status(self) -> Dict[str, bool]:
        """Get status of all components"""
        return {
            'temporal_intelligence': self.temporal_collector is not None,
            'container_consciousness': all([
                self.container_detector is not None,
                self.service_extractor is not None,
                self.container_correlator is not None
            ]),
            'hardware_specialization': any([
                self.rtx5090_detector is not None,
                self.amd_zen5_detector is not None,
                self.thermal_detector is not None
            ]),
            'multi_model_oracle': all([
                self.resource_oracle is not None,
                self.workload_predictor is not None,
                self.strategy_engine is not None
            ])
        }
    
    async def _calculate_overall_performance(self) -> float:
        """Calculate overall system performance score"""
        # This would integrate with all monitoring components
        # For now, return a calculated score based on available metrics
        base_score = 0.75  # Baseline performance
        
        # Adjust based on component health
        component_status = self._get_component_status()
        active_components = sum(component_status.values())
        component_factor = active_components / len(component_status)
        
        return min(1.0, base_score * component_factor)
    
    async def _calculate_optimization_effectiveness(self) -> float:
        """Calculate effectiveness of recent optimizations"""
        if not self.optimization_log:
            return 0.5  # Neutral score when no optimizations
        
        recent_optimizations = [opt for opt in self.optimization_log 
                               if datetime.fromisoformat(opt['timestamp']) > datetime.now() - timedelta(hours=24)]
        
        if not recent_optimizations:
            return 0.5
        
        success_rate = sum(1 for opt in recent_optimizations if opt['result'].get('success', False)) / len(recent_optimizations)
        return success_rate
    
    async def _calculate_thermal_efficiency(self) -> float:
        """Calculate thermal management efficiency"""
        if self.thermal_detector:
            thermal_insights = self.thermal_detector.get_thermal_insights()
            # This would be calculated from actual thermal data
            return thermal_insights.get('efficiency_score', 0.7)
        
        return 0.5  # Neutral score when thermal detector unavailable
    
    async def _collect_resource_utilization(self) -> Dict[str, float]:
        """Collect current resource utilization across all components"""
        utilization = {}
        
        # GPU utilization from RTX 5090 detector
        if self.rtx5090_detector:
            gpu_metrics = self.rtx5090_detector.get_blackwall_insights()
            utilization['gpu'] = gpu_metrics.get('utilization', 0.0)
        
        # CPU utilization from AMD Zen 5 detector
        if self.amd_zen5_detector:
            cpu_metrics = self.amd_zen5_detector.get_zen5_insights()
            utilization['cpu'] = cpu_metrics.get('utilization', 0.0)
        
        # Memory utilization (placeholder - would integrate with system monitoring)
        utilization['memory'] = 0.6
        utilization['vram'] = 0.7
        
        return utilization
    
    def _assess_system_health(self, performance_score: float, thermal_efficiency: float, 
                            critical_alerts: List[str]) -> SystemHealthStatus:
        """Assess overall system health"""
        if critical_alerts:
            return SystemHealthStatus.CRITICAL
        
        if performance_score >= 0.9 and thermal_efficiency >= 0.8:
            return SystemHealthStatus.EXCELLENT
        elif performance_score >= 0.7 and thermal_efficiency >= 0.6:
            return SystemHealthStatus.GOOD
        elif performance_score >= 0.5:
            return SystemHealthStatus.FAIR
        else:
            return SystemHealthStatus.POOR
    
    async def _identify_active_workloads(self) -> List[str]:
        """Identify currently active AI workloads"""
        workloads = []
        
        if self.container_detector:
            # Get active container workloads
            container_state = self.container_detector.get_container_state()
            for container, state in container_state.items():
                if state.get('running', False):
                    workloads.append(f"container:{container}")
        
        return workloads
    
    async def _identify_optimization_opportunities(self) -> List[str]:
        """Identify current optimization opportunities"""
        opportunities = []
        
        # Check resource utilization inefficiencies
        utilization = await self._collect_resource_utilization()
        for resource, usage in utilization.items():
            if usage > 0.9:
                opportunities.append(f"{resource}_overutilization")
            elif usage < 0.3:
                opportunities.append(f"{resource}_underutilization")
        
        # Check thermal optimization opportunities
        if self.thermal_detector:
            thermal_insights = self.thermal_detector.get_thermal_insights()
            if thermal_insights.get('optimization_potential', 0) > 0.3:
                opportunities.append("thermal_optimization")
        
        return opportunities
    
    async def _collect_critical_alerts(self) -> List[str]:
        """Collect critical system alerts"""
        alerts = []
        
        # Check thermal alerts
        if self.thermal_detector:
            thermal_insights = self.thermal_detector.get_thermal_insights()
            max_temp = thermal_insights.get('max_temperature', 0)
            if max_temp > self.config['safety_limits']['max_gpu_temp']:
                alerts.append(f"High GPU temperature: {max_temp}°C")
        
        return alerts
    
    async def _generate_recommendations(self) -> List[str]:
        """Generate system recommendations"""
        recommendations = []
        
        # Performance recommendations
        performance_score = await self._calculate_overall_performance()
        if performance_score < 0.7:
            recommendations.append("Consider enabling optimization mode for better performance")
        
        # Thermal recommendations
        thermal_efficiency = await self._calculate_thermal_efficiency()
        if thermal_efficiency < 0.6:
            recommendations.append("Review thermal management settings")
        
        return recommendations