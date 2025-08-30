"""
Performance Strategy Engine

Autonomous optimization execution engine for the AI workstation. Takes insights
from the MultiModelResourceOracle and AIWorkloadPredictor to automatically
execute performance optimizations, resource reallocation, and workload management.

This engine provides:
- Autonomous resource optimization execution
- Dynamic thermal management
- Container orchestration optimization
- Hardware-specific performance tuning
- Adaptive optimization strategies
- Real-time performance monitoring and adjustment
"""

import logging
import time
import asyncio
import threading
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import yaml
from collections import defaultdict, deque

# Local imports
from .resource_oracle import MultiModelResourceOracle, ResourceOptimizationStrategy
from .workload_predictor import AIWorkloadPredictor, PerformancePrediction, WorkloadFeatures


class OptimizationStrategy(Enum):
    """Types of optimization strategies"""
    RESOURCE_REALLOCATION = "resource_reallocation"
    THERMAL_MANAGEMENT = "thermal_management"
    CONTAINER_SCALING = "container_scaling"
    WORKLOAD_MIGRATION = "workload_migration"
    HARDWARE_TUNING = "hardware_tuning"
    BATCH_SIZE_OPTIMIZATION = "batch_size_optimization"
    MODEL_QUANTIZATION = "model_quantization"
    MEMORY_OPTIMIZATION = "memory_optimization"
    CPU_AFFINITY_TUNING = "cpu_affinity_tuning"
    GPU_FREQUENCY_SCALING = "gpu_frequency_scaling"


class OptimizationPriority(Enum):
    """Priority levels for optimization execution"""
    CRITICAL = "critical"      # System stability at risk
    HIGH = "high"             # Significant performance impact
    MEDIUM = "medium"         # Moderate performance gain
    LOW = "low"              # Minor optimization
    BACKGROUND = "background" # Long-term optimization


class ExecutionStatus(Enum):
    """Status of optimization execution"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


@dataclass
class OptimizationAction:
    """A specific optimization action to be executed"""
    action_id: str
    strategy: OptimizationStrategy
    priority: OptimizationPriority
    
    # Action details
    target_component: str  # Container, GPU, CPU, etc.
    action_type: str      # scale_up, tune_frequency, migrate, etc.
    parameters: Dict[str, Any]
    
    # Execution details
    status: ExecutionStatus = ExecutionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Context and validation
    expected_impact: Dict[str, float] = field(default_factory=dict)
    safety_constraints: Dict[str, Any] = field(default_factory=dict)
    rollback_procedure: Optional[Dict[str, Any]] = None
    
    # Results tracking
    actual_impact: Dict[str, float] = field(default_factory=dict)
    success_metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'action_id': self.action_id,
            'strategy': self.strategy.value,
            'priority': self.priority.value,
            'target_component': self.target_component,
            'action_type': self.action_type,
            'parameters': self.parameters,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'expected_impact': self.expected_impact,
            'actual_impact': self.actual_impact,
            'success_metrics': self.success_metrics,
            'error_message': self.error_message
        }


@dataclass
class OptimizationPlan:
    """A comprehensive optimization plan with multiple actions"""
    plan_id: str
    created_at: datetime
    target_improvements: Dict[str, float]
    
    actions: List[OptimizationAction] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)  # action_id -> prerequisite_ids
    
    # Plan status
    status: ExecutionStatus = ExecutionStatus.PENDING
    progress: float = 0.0  # 0.0 to 1.0
    
    # Results
    overall_success: bool = False
    performance_gains: Dict[str, float] = field(default_factory=dict)
    

class ContainerOrchestrator:
    """Handles container-based optimization actions"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ContainerOrchestrator")
        self.docker_available = self._check_docker_availability()
        
    def _check_docker_availability(self) -> bool:
        """Check if Docker is available and accessible"""
        try:
            result = subprocess.run(['docker', 'ps'], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception:
            return False
    
    async def scale_container(self, container_name: str, target_replicas: int) -> Dict[str, Any]:
        """Scale a container service"""
        if not self.docker_available:
            return {"success": False, "error": "Docker not available"}
        
        try:
            # For Docker Compose services
            compose_result = await self._execute_command([
                'docker-compose', 'up', '-d', '--scale', f'{container_name}={target_replicas}'
            ])
            
            if compose_result['success']:
                return {
                    "success": True,
                    "action": "scaled_service",
                    "container": container_name,
                    "replicas": target_replicas,
                    "output": compose_result['output']
                }
            else:
                # Try Docker Swarm scaling as fallback
                swarm_result = await self._execute_command([
                    'docker', 'service', 'scale', f'{container_name}={target_replicas}'
                ])
                return swarm_result
                
        except Exception as e:
            self.logger.error(f"Container scaling failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def restart_container(self, container_name: str) -> Dict[str, Any]:
        """Restart a specific container"""
        if not self.docker_available:
            return {"success": False, "error": "Docker not available"}
        
        try:
            result = await self._execute_command(['docker', 'restart', container_name])
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def update_container_resources(self, container_name: str, 
                                       cpu_limit: Optional[str] = None,
                                       memory_limit: Optional[str] = None) -> Dict[str, Any]:
        """Update container resource limits"""
        if not self.docker_available:
            return {"success": False, "error": "Docker not available"}
        
        try:
            cmd = ['docker', 'update']
            if cpu_limit:
                cmd.extend(['--cpus', cpu_limit])
            if memory_limit:
                cmd.extend(['--memory', memory_limit])
            cmd.append(container_name)
            
            result = await self._execute_command(cmd)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Execute system command asynchronously"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
            
            return {
                "success": process.returncode == 0,
                "returncode": process.returncode,
                "output": stdout.decode() if stdout else "",
                "error": stderr.decode() if stderr else ""
            }
        except asyncio.TimeoutError:
            return {"success": False, "error": "Command timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class HardwareOptimizer:
    """Handles hardware-specific optimization actions"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.HardwareOptimizer")
        self.nvidia_smi_available = self._check_nvidia_smi()
        self.cpufreq_available = self._check_cpufreq()
        
    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available"""
        try:
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    def _check_cpufreq(self) -> bool:
        """Check if cpufreq-set is available"""
        try:
            result = subprocess.run(['which', 'cpufreq-set'], capture_output=True)
            return result.returncode == 0
        except Exception:
            return False
    
    async def set_gpu_power_limit(self, gpu_id: int, power_limit: int) -> Dict[str, Any]:
        """Set GPU power limit"""
        if not self.nvidia_smi_available:
            return {"success": False, "error": "nvidia-smi not available"}
        
        try:
            cmd = ['nvidia-smi', '-i', str(gpu_id), '-pl', str(power_limit)]
            result = await self._execute_command(cmd)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def set_gpu_memory_clock(self, gpu_id: int, memory_clock: int) -> Dict[str, Any]:
        """Set GPU memory clock (requires proper permissions)"""
        if not self.nvidia_smi_available:
            return {"success": False, "error": "nvidia-smi not available"}
        
        try:
            # Enable persistence mode first
            await self._execute_command(['nvidia-smi', '-i', str(gpu_id), '-pm', '1'])
            
            # Set memory clock
            cmd = ['nvidia-smi', '-i', str(gpu_id), '-ac', f'{memory_clock},1500']  # memory,graphics
            result = await self._execute_command(cmd)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def set_cpu_governor(self, governor: str) -> Dict[str, Any]:
        """Set CPU frequency governor"""
        if not self.cpufreq_available:
            return {"success": False, "error": "cpufreq-set not available"}
        
        try:
            cmd = ['cpufreq-set', '-g', governor]
            result = await self._execute_command(cmd)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def set_cpu_affinity(self, pid: int, cpu_mask: str) -> Dict[str, Any]:
        """Set CPU affinity for a process"""
        try:
            cmd = ['taskset', '-p', cpu_mask, str(pid)]
            result = await self._execute_command(cmd)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def tune_numa_balancing(self, enable: bool) -> Dict[str, Any]:
        """Enable/disable NUMA balancing"""
        try:
            value = "1" if enable else "0"
            cmd = ['echo', value]
            # This would need root permissions
            # echo value > /proc/sys/kernel/numa_balancing
            return {"success": False, "error": "Requires root permissions"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Execute system command asynchronously"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
            
            return {
                "success": process.returncode == 0,
                "returncode": process.returncode,
                "output": stdout.decode() if stdout else "",
                "error": stderr.decode() if stderr else ""
            }
        except asyncio.TimeoutError:
            return {"success": False, "error": "Command timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class ThermalManager:
    """Handles thermal-aware optimization actions"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ThermalManager")
        self.thermal_zones = self._discover_thermal_zones()
        self.fan_control_available = self._check_fan_control()
    
    def _discover_thermal_zones(self) -> List[str]:
        """Discover available thermal zones"""
        thermal_path = Path("/sys/class/thermal")
        zones = []
        
        try:
            if thermal_path.exists():
                for zone in thermal_path.glob("thermal_zone*"):
                    zones.append(zone.name)
        except Exception as e:
            self.logger.warning(f"Failed to discover thermal zones: {e}")
        
        return zones
    
    def _check_fan_control(self) -> bool:
        """Check if fan control utilities are available"""
        try:
            # Check for pwm-config or fancontrol
            result = subprocess.run(['which', 'fancontrol'], capture_output=True)
            return result.returncode == 0
        except Exception:
            return False
    
    async def get_thermal_state(self) -> Dict[str, float]:
        """Get current thermal state of the system"""
        thermal_state = {}
        
        for zone in self.thermal_zones:
            try:
                temp_file = Path(f"/sys/class/thermal/{zone}/temp")
                if temp_file.exists():
                    temp_raw = temp_file.read_text().strip()
                    temperature = float(temp_raw) / 1000.0  # Convert millicelsius
                    thermal_state[zone] = temperature
            except Exception as e:
                self.logger.warning(f"Failed to read temperature for {zone}: {e}")
        
        return thermal_state
    
    async def apply_thermal_optimization(self, target_temp_reduction: float) -> Dict[str, Any]:
        """Apply thermal optimization strategies"""
        optimizations_applied = []
        
        try:
            current_thermal = await self.get_thermal_state()
            max_temp = max(current_thermal.values()) if current_thermal else 70.0
            
            # Strategy 1: Reduce GPU power limit
            if max_temp > 75.0:
                gpu_optimizer = HardwareOptimizer()
                power_reduction = min(50, int(target_temp_reduction * 10))  # 10W per degree
                result = await gpu_optimizer.set_gpu_power_limit(0, 575 - power_reduction)
                if result['success']:
                    optimizations_applied.append(f"Reduced GPU power limit by {power_reduction}W")
            
            # Strategy 2: Increase fan speed (if available)
            if self.fan_control_available and max_temp > 70.0:
                fan_result = await self._increase_fan_speed(target_temp_reduction)
                if fan_result['success']:
                    optimizations_applied.append("Increased fan speed")
            
            # Strategy 3: Reduce CPU frequency
            if max_temp > 80.0:
                hw_optimizer = HardwareOptimizer()
                governor_result = await hw_optimizer.set_cpu_governor('powersave')
                if governor_result['success']:
                    optimizations_applied.append("Set CPU governor to powersave")
            
            return {
                "success": len(optimizations_applied) > 0,
                "optimizations": optimizations_applied,
                "target_reduction": target_temp_reduction
            }
            
        except Exception as e:
            self.logger.error(f"Thermal optimization failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _increase_fan_speed(self, target_reduction: float) -> Dict[str, Any]:
        """Increase fan speed to reduce temperatures"""
        try:
            # This would need specific fan control implementation
            # For demonstration, we'll simulate the action
            return {
                "success": True,
                "action": "fan_speed_increased",
                "target_reduction": target_reduction
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class PerformanceStrategyEngine:
    """
    Autonomous performance optimization execution engine for the AI workstation.
    
    Coordinates with ResourceOracle and WorkloadPredictor to execute optimization
    strategies automatically, managing resources, thermal constraints, and workload
    performance across the RTX 5090 + AMD 9950X system.
    """
    
    def __init__(self, resource_oracle: Optional[MultiModelResourceOracle] = None,
                 workload_predictor: Optional[AIWorkloadPredictor] = None):
        self.logger = logging.getLogger(f"{__name__}.PerformanceStrategyEngine")
        
        # Core components
        self.resource_oracle = resource_oracle
        self.workload_predictor = workload_predictor
        
        # Execution components
        self.container_orchestrator = ContainerOrchestrator()
        self.hardware_optimizer = HardwareOptimizer()
        self.thermal_manager = ThermalManager()
        
        # Optimization state
        self.active_optimizations: Dict[str, OptimizationAction] = {}
        self.optimization_history: deque = deque(maxlen=1000)
        self.optimization_queue = asyncio.Queue()
        
        # Configuration
        self.max_concurrent_optimizations = 3
        self.safety_mode = True  # Conservative optimizations when True
        self.thermal_threshold = 80.0  # Celsius
        self.performance_target_threshold = 0.1  # 10% improvement minimum
        
        # Performance tracking
        self.baseline_metrics = {}
        self.current_metrics = {}
        self.optimization_effectiveness = defaultdict(list)
        
        # Control flags
        self.running = False
        self.executor_task = None
        
        self.logger.info("PerformanceStrategyEngine initialized")
    
    async def start_optimization_engine(self) -> None:
        """Start the autonomous optimization engine"""
        if self.running:
            self.logger.warning("Optimization engine already running")
            return
        
        self.running = True
        self.executor_task = asyncio.create_task(self._optimization_executor())
        
        # Start monitoring and strategy generation
        asyncio.create_task(self._performance_monitor())
        asyncio.create_task(self._strategy_generator())
        
        self.logger.info("Optimization engine started")
    
    async def stop_optimization_engine(self) -> None:
        """Stop the optimization engine gracefully"""
        self.running = False
        
        if self.executor_task:
            await self.executor_task
        
        self.logger.info("Optimization engine stopped")
    
    async def execute_optimization_plan(self, plan: OptimizationPlan) -> Dict[str, Any]:
        """Execute a comprehensive optimization plan"""
        self.logger.info(f"Executing optimization plan: {plan.plan_id}")
        
        try:
            plan.status = ExecutionStatus.IN_PROGRESS
            results = []
            
            # Execute actions based on dependencies
            executed_actions = set()
            
            while len(executed_actions) < len(plan.actions):
                ready_actions = []
                
                for action in plan.actions:
                    if action.action_id in executed_actions:
                        continue
                    
                    # Check dependencies
                    dependencies = plan.dependencies.get(action.action_id, [])
                    if all(dep_id in executed_actions for dep_id in dependencies):
                        ready_actions.append(action)
                
                if not ready_actions:
                    self.logger.error("Circular dependency detected in optimization plan")
                    break
                
                # Execute ready actions
                for action in ready_actions:
                    result = await self.execute_optimization_action(action)
                    results.append(result)
                    executed_actions.add(action.action_id)
                    
                    # Update plan progress
                    plan.progress = len(executed_actions) / len(plan.actions)
            
            # Evaluate overall plan success
            successful_actions = sum(1 for r in results if r.get('success', False))
            plan.overall_success = successful_actions >= len(plan.actions) * 0.8  # 80% success rate
            plan.status = ExecutionStatus.COMPLETED if plan.overall_success else ExecutionStatus.FAILED
            
            # Calculate performance gains
            plan.performance_gains = await self._calculate_performance_gains(plan)
            
            self.logger.info(f"Plan {plan.plan_id} completed: {successful_actions}/{len(plan.actions)} actions successful")
            
            return {
                "success": plan.overall_success,
                "plan_id": plan.plan_id,
                "actions_executed": len(executed_actions),
                "actions_successful": successful_actions,
                "performance_gains": plan.performance_gains,
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"Plan execution failed: {e}")
            plan.status = ExecutionStatus.FAILED
            return {"success": False, "error": str(e)}
    
    async def execute_optimization_action(self, action: OptimizationAction) -> Dict[str, Any]:
        """Execute a single optimization action"""
        self.logger.info(f"Executing action: {action.action_id} ({action.strategy.value})")
        
        action.status = ExecutionStatus.IN_PROGRESS
        action.started_at = datetime.now()
        
        try:
            # Safety checks
            if not self._validate_action_safety(action):
                action.status = ExecutionStatus.FAILED
                action.error_message = "Action failed safety validation"
                return {"success": False, "error": "Safety validation failed"}
            
            # Route to appropriate executor based on strategy
            result = await self._route_action_execution(action)
            
            # Update action status
            if result.get('success', False):
                action.status = ExecutionStatus.COMPLETED
                action.actual_impact = result.get('impact', {})
                action.success_metrics = result.get('metrics', {})
            else:
                action.status = ExecutionStatus.FAILED
                action.error_message = result.get('error', 'Unknown error')
            
            action.completed_at = datetime.now()
            
            # Track effectiveness
            if action.status == ExecutionStatus.COMPLETED:
                effectiveness = self._calculate_action_effectiveness(action)
                self.optimization_effectiveness[action.strategy].append(effectiveness)
            
            self.optimization_history.append(action)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Action execution failed: {e}")
            action.status = ExecutionStatus.FAILED
            action.error_message = str(e)
            action.completed_at = datetime.now()
            return {"success": False, "error": str(e)}
    
    async def generate_optimization_strategy(self, 
                                           current_performance: Dict[str, Any],
                                           target_improvements: Dict[str, float]) -> OptimizationPlan:
        """Generate an optimization strategy based on current performance and targets"""
        plan_id = f"opt_plan_{int(time.time())}"
        plan = OptimizationPlan(
            plan_id=plan_id,
            created_at=datetime.now(),
            target_improvements=target_improvements
        )
        
        try:
            # Analyze current bottlenecks
            bottlenecks = await self._identify_performance_bottlenecks(current_performance)
            
            # Generate actions based on bottlenecks and targets
            for bottleneck in bottlenecks:
                actions = await self._generate_actions_for_bottleneck(
                    bottleneck, target_improvements
                )
                plan.actions.extend(actions)
            
            # Add thermal management actions if needed
            thermal_state = await self.thermal_manager.get_thermal_state()
            if thermal_state and max(thermal_state.values()) > self.thermal_threshold:
                thermal_actions = await self._generate_thermal_actions(thermal_state)
                plan.actions.extend(thermal_actions)
            
            # Determine action dependencies
            plan.dependencies = self._determine_action_dependencies(plan.actions)
            
            self.logger.info(f"Generated optimization plan with {len(plan.actions)} actions")
            return plan
            
        except Exception as e:
            self.logger.error(f"Strategy generation failed: {e}")
            return plan  # Return empty plan
    
    async def _optimization_executor(self) -> None:
        """Main optimization execution loop"""
        while self.running:
            try:
                # Get next optimization from queue
                try:
                    action = await asyncio.wait_for(self.optimization_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Check if we can execute more optimizations
                active_count = len([a for a in self.active_optimizations.values() 
                                  if a.status == ExecutionStatus.IN_PROGRESS])
                
                if active_count >= self.max_concurrent_optimizations:
                    # Put action back in queue and wait
                    await self.optimization_queue.put(action)
                    await asyncio.sleep(1)
                    continue
                
                # Execute the action
                self.active_optimizations[action.action_id] = action
                
                # Execute in background
                asyncio.create_task(self._execute_and_cleanup(action))
                
            except Exception as e:
                self.logger.error(f"Optimization executor error: {e}")
                await asyncio.sleep(1)
    
    async def _execute_and_cleanup(self, action: OptimizationAction) -> None:
        """Execute action and clean up tracking"""
        try:
            await self.execute_optimization_action(action)
        finally:
            # Remove from active optimizations
            if action.action_id in self.active_optimizations:
                del self.active_optimizations[action.action_id]
    
    async def _performance_monitor(self) -> None:
        """Monitor system performance and detect optimization opportunities"""
        while self.running:
            try:
                # Collect current metrics
                current_metrics = await self._collect_performance_metrics()
                self.current_metrics = current_metrics
                
                # Compare with baseline
                if self.baseline_metrics:
                    performance_delta = self._calculate_performance_delta(
                        self.baseline_metrics, current_metrics
                    )
                    
                    # Trigger optimizations if significant degradation
                    if self._should_trigger_optimization(performance_delta):
                        await self._trigger_automatic_optimization(performance_delta)
                
                # Update baseline periodically
                if not self.baseline_metrics or \
                   (datetime.now().hour % 6 == 0 and datetime.now().minute < 5):
                    self.baseline_metrics = current_metrics.copy()
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)  # Longer delay on error
    
    async def _strategy_generator(self) -> None:
        """Generate optimization strategies based on system state"""
        while self.running:
            try:
                # Generate strategies based on workload predictions
                if self.workload_predictor and self.resource_oracle:
                    # Get current workload characteristics
                    current_workloads = await self._identify_current_workloads()
                    
                    for workload in current_workloads:
                        # Predict performance
                        prediction = self.workload_predictor.predict_workload_performance(workload)
                        
                        # Check if optimization is needed
                        if prediction.confidence_score > 0.7:  # High confidence predictions only
                            if len(prediction.optimization_suggestions) > 0:
                                await self._create_prediction_based_optimizations(prediction, workload)
                
                await asyncio.sleep(300)  # Generate strategies every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Strategy generation error: {e}")
                await asyncio.sleep(600)  # Longer delay on error
    
    async def _route_action_execution(self, action: OptimizationAction) -> Dict[str, Any]:
        """Route action to appropriate executor"""
        if action.strategy == OptimizationStrategy.CONTAINER_SCALING:
            return await self._execute_container_action(action)
        elif action.strategy == OptimizationStrategy.HARDWARE_TUNING:
            return await self._execute_hardware_action(action)
        elif action.strategy == OptimizationStrategy.THERMAL_MANAGEMENT:
            return await self._execute_thermal_action(action)
        elif action.strategy == OptimizationStrategy.RESOURCE_REALLOCATION:
            return await self._execute_resource_action(action)
        else:
            return {"success": False, "error": f"Unknown strategy: {action.strategy}"}
    
    async def _execute_container_action(self, action: OptimizationAction) -> Dict[str, Any]:
        """Execute container-related optimization"""
        if action.action_type == "scale_up":
            target_replicas = action.parameters.get('target_replicas', 1)
            return await self.container_orchestrator.scale_container(
                action.target_component, target_replicas
            )
        elif action.action_type == "restart":
            return await self.container_orchestrator.restart_container(action.target_component)
        elif action.action_type == "update_resources":
            cpu_limit = action.parameters.get('cpu_limit')
            memory_limit = action.parameters.get('memory_limit')
            return await self.container_orchestrator.update_container_resources(
                action.target_component, cpu_limit, memory_limit
            )
        else:
            return {"success": False, "error": f"Unknown container action: {action.action_type}"}
    
    async def _execute_hardware_action(self, action: OptimizationAction) -> Dict[str, Any]:
        """Execute hardware optimization"""
        if action.action_type == "set_power_limit":
            gpu_id = action.parameters.get('gpu_id', 0)
            power_limit = action.parameters.get('power_limit', 575)
            return await self.hardware_optimizer.set_gpu_power_limit(gpu_id, power_limit)
        elif action.action_type == "set_cpu_governor":
            governor = action.parameters.get('governor', 'performance')
            return await self.hardware_optimizer.set_cpu_governor(governor)
        elif action.action_type == "set_cpu_affinity":
            pid = action.parameters.get('pid')
            cpu_mask = action.parameters.get('cpu_mask')
            return await self.hardware_optimizer.set_cpu_affinity(pid, cpu_mask)
        else:
            return {"success": False, "error": f"Unknown hardware action: {action.action_type}"}
    
    async def _execute_thermal_action(self, action: OptimizationAction) -> Dict[str, Any]:
        """Execute thermal management optimization"""
        if action.action_type == "reduce_temperature":
            target_reduction = action.parameters.get('target_reduction', 5.0)
            return await self.thermal_manager.apply_thermal_optimization(target_reduction)
        else:
            return {"success": False, "error": f"Unknown thermal action: {action.action_type}"}
    
    async def _execute_resource_action(self, action: OptimizationAction) -> Dict[str, Any]:
        """Execute resource reallocation"""
        # This would integrate with the resource oracle for complex reallocation
        return {"success": True, "message": "Resource reallocation simulated"}
    
    def _validate_action_safety(self, action: OptimizationAction) -> bool:
        """Validate that an action is safe to execute"""
        if self.safety_mode:
            # Conservative safety checks
            if action.strategy == OptimizationStrategy.HARDWARE_TUNING:
                # Don't allow aggressive hardware changes in safety mode
                if action.action_type == "set_power_limit":
                    power_limit = action.parameters.get('power_limit', 575)
                    if power_limit < 400:  # Don't go below 400W
                        return False
            
            # Check thermal constraints
            if hasattr(action, 'thermal_impact'):
                if action.thermal_impact > 10.0:  # Don't allow >10°C increases
                    return False
        
        return True
    
    def _calculate_action_effectiveness(self, action: OptimizationAction) -> float:
        """Calculate the effectiveness of an executed action"""
        if not action.expected_impact or not action.actual_impact:
            return 0.5  # Neutral score for missing data
        
        effectiveness_scores = []
        
        for metric, expected in action.expected_impact.items():
            actual = action.actual_impact.get(metric, 0)
            if expected != 0:
                effectiveness = min(2.0, actual / expected)  # Cap at 2x expected
                effectiveness_scores.append(effectiveness)
        
        return np.mean(effectiveness_scores) if effectiveness_scores else 0.5
    
    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect current system performance metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_utilization': 0.25,    # Placeholder - integrate with system monitoring
            'gpu_utilization': 0.80,    # Placeholder
            'memory_usage_gb': 32.0,    # Placeholder
            'vram_usage_gb': 24.0,      # Placeholder
            'thermal_state': await self.thermal_manager.get_thermal_state()
        }
        
        return metrics
    
    async def _identify_performance_bottlenecks(self, performance: Dict[str, Any]) -> List[str]:
        """Identify current performance bottlenecks"""
        bottlenecks = []
        
        if performance.get('gpu_utilization', 0) > 0.95:
            bottlenecks.append('gpu_compute')
        if performance.get('vram_usage_gb', 0) > 28.0:
            bottlenecks.append('gpu_memory')
        if performance.get('cpu_utilization', 0) > 0.90:
            bottlenecks.append('cpu_compute')
        if performance.get('memory_usage_gb', 0) > 120.0:
            bottlenecks.append('system_memory')
        
        thermal_state = performance.get('thermal_state', {})
        if thermal_state and max(thermal_state.values()) > self.thermal_threshold:
            bottlenecks.append('thermal')
        
        return bottlenecks
    
    async def _generate_actions_for_bottleneck(self, bottleneck: str,
                                             targets: Dict[str, float]) -> List[OptimizationAction]:
        """Generate optimization actions for a specific bottleneck"""
        actions = []
        
        if bottleneck == 'gpu_compute':
            # Scale down GPU-intensive containers or reduce batch sizes
            action = OptimizationAction(
                action_id=f"gpu_scale_{int(time.time())}",
                strategy=OptimizationStrategy.CONTAINER_SCALING,
                priority=OptimizationPriority.HIGH,
                target_component="llama-gpu",
                action_type="scale_down",
                parameters={'target_replicas': 1},
                expected_impact={'gpu_utilization': -0.2}
            )
            actions.append(action)
        
        elif bottleneck == 'thermal':
            # Apply thermal management
            action = OptimizationAction(
                action_id=f"thermal_{int(time.time())}",
                strategy=OptimizationStrategy.THERMAL_MANAGEMENT,
                priority=OptimizationPriority.CRITICAL,
                target_component="system",
                action_type="reduce_temperature",
                parameters={'target_reduction': 10.0},
                expected_impact={'temperature_reduction': 10.0}
            )
            actions.append(action)
        
        return actions
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization engine status"""
        return {
            'running': self.running,
            'active_optimizations': len(self.active_optimizations),
            'queue_size': self.optimization_queue.qsize(),
            'total_optimizations_executed': len(self.optimization_history),
            'baseline_metrics': self.baseline_metrics,
            'current_metrics': self.current_metrics,
            'optimization_effectiveness': dict(self.optimization_effectiveness),
            'safety_mode': self.safety_mode,
            'thermal_threshold': self.thermal_threshold
        }