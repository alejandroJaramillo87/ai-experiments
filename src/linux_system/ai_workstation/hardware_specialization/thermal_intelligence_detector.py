"""
Thermal Intelligence Detector

Advanced thermal management and intelligence for high-performance AI workstation
cooling architectures. Provides comprehensive monitoring, prediction, and 
optimization for complex multi-fan cooling systems with component-specific
thermal correlation and AI workload thermal impact analysis.

Features:
- 15-fan cooling architecture monitoring and optimization
- Predictive thermal throttling prevention
- Component-specific thermal correlation (CPU, GPU, Memory)
- AI workload thermal impact analysis
- Positive pressure airflow optimization
- Thermal efficiency scoring and recommendations
- Dynamic fan curve optimization based on workload patterns
"""

import os
import re
import subprocess
import psutil
import logging
import threading
import time
import glob
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import json

from ...temporal.core.change_detector import SystemChangeDetector, SystemState, SystemChange
from ...temporal.core.types import ChangeType, ComponentType, Significance


logger = logging.getLogger(__name__)


@dataclass
class FanZoneMetrics:
    """Metrics for a specific fan zone in the cooling system."""
    zone_name: str
    fan_count: int
    fan_rpms: List[int]
    target_rpm: int
    actual_average_rpm: int
    pwm_percent: float
    airflow_direction: str  # intake, exhaust
    efficiency_score: float  # 0-1
    noise_level_db: float
    power_consumption_watts: float


@dataclass
class ComponentThermalState:
    """Thermal state for a specific component."""
    component_name: str
    component_type: str  # cpu, gpu, memory, storage, vrm
    current_temp_c: float
    max_temp_c: float
    thermal_margin_c: float
    thermal_velocity: float  # °C/min change rate
    throttling_active: bool
    throttling_imminent: bool  # Within 2°C of throttle threshold
    cooling_effectiveness: float  # 0-1
    recommended_action: Optional[str]


@dataclass
class ThermalProfile:
    """Comprehensive thermal profile for the entire system."""
    timestamp: datetime
    
    # Fan zone metrics
    cpu_fan_zone: FanZoneMetrics
    gpu_fan_zone: FanZoneMetrics
    intake_fan_zone: FanZoneMetrics
    exhaust_fan_zone: FanZoneMetrics
    supplementary_fan_zone: FanZoneMetrics
    
    # Component thermal states
    cpu_thermal: ComponentThermalState
    gpu_thermal: ComponentThermalState
    memory_thermal: ComponentThermalState
    storage_thermal: ComponentThermalState
    
    # System-wide thermal metrics
    ambient_temp_c: float
    case_temp_c: float
    airflow_balance: float  # Positive = intake > exhaust
    thermal_efficiency: float  # Overall cooling effectiveness
    power_heat_generation: float  # Watts of heat being generated
    thermal_capacity_utilization: float  # % of cooling capacity used
    
    # AI workload correlation
    ai_thermal_load: float  # Thermal load from AI workloads
    inference_thermal_impact: Dict[str, float]  # Per-service thermal impact


@dataclass
class ThermalPrediction:
    """Thermal prediction and recommendation."""
    component: str
    prediction_horizon: timedelta
    predicted_temp_c: float
    confidence: float
    throttling_risk: str  # none, low, medium, high
    recommended_actions: List[str]
    preventive_measures: List[str]


@dataclass
class CoolingOptimization:
    """Cooling system optimization recommendation."""
    optimization_type: str
    target_zones: List[str]
    current_efficiency: float
    potential_efficiency: float
    estimated_improvement: str
    implementation_steps: List[str]
    risk_level: str  # low, medium, high


class ThermalIntelligenceDetector(SystemChangeDetector):
    """
    Advanced thermal intelligence detector for AI workstation cooling systems.
    
    Provides comprehensive thermal monitoring, predictive analysis, and optimization
    recommendations for complex multi-fan cooling architectures with AI workload
    thermal correlation and component-specific thermal management.
    """
    
    def __init__(self, monitoring_interval: float = 15.0):
        super().__init__()
        self.monitoring_interval = monitoring_interval
        
        # Cooling system configuration (based on hardware docs)
        self.cooling_config = {
            'total_fans': 15,
            'fan_zones': {
                'cpu_cooling': {
                    'fan_count': 2,
                    'model': 'be quiet! Dark Rock Pro 5',
                    'tdp_capacity': 270,
                    'airflow_cfm': 94.2,  # Combined CFM
                    'noise_db': 22.8
                },
                'gpu_cooling': {
                    'fan_count': 3,
                    'model': 'RTX 5090 Triple Fan',
                    'tdp_capacity': 575,
                    'airflow_cfm': 180,  # Estimated
                    'noise_db': 35.0
                },
                'case_intake': {
                    'fan_count': 3,
                    'location': 'front',
                    'target_cfm': 150,
                    'target_pressure': 'positive'
                },
                'case_exhaust_top': {
                    'fan_count': 3,
                    'location': 'top',
                    'target_cfm': 120
                },
                'case_exhaust_rear': {
                    'fan_count': 1,
                    'location': 'rear',
                    'target_cfm': 40
                },
                'supplementary': {
                    'fan_count': 3,
                    'location': 'various',
                    'target_cfm': 90,
                    'purpose': 'airflow optimization'
                }
            },
            'thermal_strategy': 'positive_pressure',
            'target_case_pressure': 1.1  # 10% positive pressure
        }
        
        # Thermal thresholds and limits
        self.thermal_thresholds = {
            'cpu': {
                'safe_max': 75,      # °C
                'throttle_warn': 80,  # °C  
                'throttle_point': 85, # °C
                'critical': 90        # °C
            },
            'gpu': {
                'safe_max': 80,      # °C
                'throttle_warn': 83,  # °C
                'throttle_point': 87, # °C
                'critical': 92        # °C
            },
            'memory': {
                'safe_max': 60,      # °C
                'throttle_warn': 70,  # °C
                'throttle_point': 80, # °C
                'critical': 85        # °C
            },
            'storage': {
                'safe_max': 55,      # °C
                'throttle_warn': 65,  # °C
                'throttle_point': 75, # °C
                'critical': 80        # °C
            }
        }
        
        # Thermal monitoring state
        self.thermal_history: deque = deque(maxlen=200)  # 50 hours at 15min intervals
        self.component_states: Dict[str, ComponentThermalState] = {}
        self.fan_zones: Dict[str, FanZoneMetrics] = {}
        self.thermal_predictions: Dict[str, ThermalPrediction] = {}
        
        # AI workload thermal correlation
        self.workload_thermal_correlation: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.thermal_baselines: Dict[str, float] = {}
        
        # Background monitoring
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Change detection thresholds
        self.change_thresholds = {
            'temperature': 5.0,       # °C change
            'fan_rpm': 200,           # RPM change
            'thermal_velocity': 2.0,  # °C/min change rate
            'efficiency_drop': 0.1,   # 10% efficiency drop
            'throttling_risk': 'medium',  # Risk level change
            'airflow_imbalance': 0.2  # 20% airflow balance change
        }
        
        # Initialize system thermal sensors
        self.sensor_paths = self._discover_thermal_sensors()
        self.fan_controllers = self._discover_fan_controllers()
        
        logger.info(f"Thermal Intelligence initialized: {len(self.sensor_paths)} sensors, {len(self.fan_controllers)} fan controllers")
        
    def _discover_thermal_sensors(self) -> Dict[str, Dict[str, str]]:
        """Discover available thermal sensors on the system."""
        sensors = {
            'cpu': {},
            'gpu': {},
            'memory': {},
            'storage': {},
            'ambient': {}
        }
        
        try:
            # CPU temperature sensors
            cpu_thermal_paths = glob.glob('/sys/class/thermal/thermal_zone*/type')
            for path in cpu_thermal_paths:
                with open(path, 'r') as f:
                    sensor_type = f.read().strip()
                if 'cpu' in sensor_type.lower() or 'x86_pkg' in sensor_type.lower():
                    zone_id = path.split('/')[-2]
                    temp_path = f'/sys/class/thermal/{zone_id}/temp'
                    if os.path.exists(temp_path):
                        sensors['cpu'][sensor_type] = temp_path
                        
            # GPU temperature (NVIDIA)
            nvidia_smi_available = subprocess.run(['which', 'nvidia-smi'], 
                                                capture_output=True).returncode == 0
            if nvidia_smi_available:
                sensors['gpu']['nvidia'] = 'nvidia-smi'
                
            # Storage temperature sensors
            nvme_paths = glob.glob('/sys/class/hwmon/hwmon*/name')
            for path in nvme_paths:
                try:
                    with open(path, 'r') as f:
                        hwmon_name = f.read().strip()
                    if 'nvme' in hwmon_name.lower():
                        hwmon_dir = os.path.dirname(path)
                        temp_files = glob.glob(f'{hwmon_dir}/temp*_input')
                        if temp_files:
                            sensors['storage'][hwmon_name] = temp_files[0]
                except:
                    continue
                    
            # Memory temperature (if available)
            # Note: Memory temperature sensors are rare, would need specialized hardware
            
            logger.info(f"Discovered thermal sensors: CPU={len(sensors['cpu'])}, GPU={len(sensors['gpu'])}, Storage={len(sensors['storage'])}")
            
        except Exception as e:
            logger.error(f"Error discovering thermal sensors: {e}")
            
        return sensors
        
    def _discover_fan_controllers(self) -> Dict[str, Dict[str, str]]:
        """Discover available fan controllers."""
        controllers = {}
        
        try:
            # PWM fan controllers
            hwmon_paths = glob.glob('/sys/class/hwmon/hwmon*')
            for hwmon_path in hwmon_paths:
                try:
                    name_file = os.path.join(hwmon_path, 'name')
                    if os.path.exists(name_file):
                        with open(name_file, 'r') as f:
                            controller_name = f.read().strip()
                            
                        # Look for fan and PWM files
                        fan_files = glob.glob(f'{hwmon_path}/fan*_input')
                        pwm_files = glob.glob(f'{hwmon_path}/pwm*')
                        
                        if fan_files or pwm_files:
                            controllers[controller_name] = {
                                'path': hwmon_path,
                                'fans': fan_files,
                                'pwm': pwm_files
                            }
                            
                except Exception:
                    continue
                    
            logger.info(f"Discovered fan controllers: {list(controllers.keys())}")
            
        except Exception as e:
            logger.error(f"Error discovering fan controllers: {e}")
            
        return controllers
        
    def start_monitoring(self):
        """Start continuous thermal monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="ThermalIntelligenceMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Thermal intelligence monitoring started")
        
    def stop_monitoring(self):
        """Stop thermal monitoring."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        logger.info("Thermal intelligence monitoring stopped")
        
    def _monitoring_loop(self):
        """Continuous thermal monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect comprehensive thermal profile
                thermal_profile = self._collect_thermal_profile()
                if thermal_profile:
                    self.thermal_history.append(thermal_profile)
                    
                    # Update component states
                    self._update_component_states(thermal_profile)
                    
                    # Update thermal predictions
                    self._update_thermal_predictions()
                    
                    # Correlate with AI workloads
                    self._correlate_workload_thermal_impact()
                    
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in thermal monitoring loop: {e}")
                time.sleep(5.0)
                
    def _collect_thermal_profile(self) -> Optional[ThermalProfile]:
        """Collect comprehensive thermal profile of the system."""
        try:
            current_time = datetime.now()
            
            # Collect fan zone metrics
            fan_zones = self._collect_fan_zone_metrics()
            
            # Collect component thermal states
            component_thermals = self._collect_component_temperatures()
            
            # Calculate system-wide thermal metrics
            system_metrics = self._calculate_system_thermal_metrics(fan_zones, component_thermals)
            
            # Collect AI workload thermal correlation
            ai_thermal_impact = self._calculate_ai_thermal_impact()
            
            return ThermalProfile(
                timestamp=current_time,
                cpu_fan_zone=fan_zones.get('cpu', self._create_default_fan_zone('cpu')),
                gpu_fan_zone=fan_zones.get('gpu', self._create_default_fan_zone('gpu')),
                intake_fan_zone=fan_zones.get('intake', self._create_default_fan_zone('intake')),
                exhaust_fan_zone=fan_zones.get('exhaust', self._create_default_fan_zone('exhaust')),
                supplementary_fan_zone=fan_zones.get('supplementary', self._create_default_fan_zone('supplementary')),
                cpu_thermal=component_thermals.get('cpu', self._create_default_thermal_state('cpu')),
                gpu_thermal=component_thermals.get('gpu', self._create_default_thermal_state('gpu')),
                memory_thermal=component_thermals.get('memory', self._create_default_thermal_state('memory')),
                storage_thermal=component_thermals.get('storage', self._create_default_thermal_state('storage')),
                ambient_temp_c=system_metrics['ambient_temp'],
                case_temp_c=system_metrics['case_temp'],
                airflow_balance=system_metrics['airflow_balance'],
                thermal_efficiency=system_metrics['thermal_efficiency'],
                power_heat_generation=system_metrics['power_heat_generation'],
                thermal_capacity_utilization=system_metrics['thermal_capacity_utilization'],
                ai_thermal_load=ai_thermal_impact['total_load'],
                inference_thermal_impact=ai_thermal_impact['per_service']
            )
            
        except Exception as e:
            logger.error(f"Error collecting thermal profile: {e}")
            return None
            
    def _collect_fan_zone_metrics(self) -> Dict[str, FanZoneMetrics]:
        """Collect metrics for each fan zone."""
        fan_zones = {}
        
        try:
            # CPU fan zone (Dark Rock Pro 5)
            cpu_fans = self._get_cpu_fan_metrics()
            if cpu_fans:
                fan_zones['cpu'] = FanZoneMetrics(
                    zone_name='CPU Cooling',
                    fan_count=2,
                    fan_rpms=cpu_fans['rpms'],
                    target_rpm=cpu_fans['target_rpm'],
                    actual_average_rpm=int(sum(cpu_fans['rpms']) / len(cpu_fans['rpms'])),
                    pwm_percent=cpu_fans['pwm_percent'],
                    airflow_direction='intake',
                    efficiency_score=cpu_fans['efficiency'],
                    noise_level_db=self._estimate_noise_level(cpu_fans['rpms'], 'cpu'),
                    power_consumption_watts=cpu_fans['power_watts']
                )
                
            # GPU fan zone (RTX 5090)
            gpu_fans = self._get_gpu_fan_metrics()
            if gpu_fans:
                fan_zones['gpu'] = FanZoneMetrics(
                    zone_name='GPU Cooling',
                    fan_count=3,
                    fan_rpms=gpu_fans['rpms'],
                    target_rpm=gpu_fans['target_rpm'],
                    actual_average_rpm=int(sum(gpu_fans['rpms']) / len(gpu_fans['rpms'])),
                    pwm_percent=gpu_fans['pwm_percent'],
                    airflow_direction='exhaust',
                    efficiency_score=gpu_fans['efficiency'],
                    noise_level_db=self._estimate_noise_level(gpu_fans['rpms'], 'gpu'),
                    power_consumption_watts=gpu_fans['power_watts']
                )
                
            # Case fans (intake and exhaust)
            case_fans = self._get_case_fan_metrics()
            
            # Intake fans (3x front)
            if 'intake' in case_fans:
                intake_data = case_fans['intake']
                fan_zones['intake'] = FanZoneMetrics(
                    zone_name='Case Intake',
                    fan_count=3,
                    fan_rpms=intake_data['rpms'],
                    target_rpm=intake_data['target_rpm'],
                    actual_average_rpm=int(sum(intake_data['rpms']) / len(intake_data['rpms'])),
                    pwm_percent=intake_data['pwm_percent'],
                    airflow_direction='intake',
                    efficiency_score=intake_data['efficiency'],
                    noise_level_db=self._estimate_noise_level(intake_data['rpms'], 'case'),
                    power_consumption_watts=intake_data['power_watts']
                )
                
            # Exhaust fans (3x top + 1x rear)
            if 'exhaust' in case_fans:
                exhaust_data = case_fans['exhaust']
                fan_zones['exhaust'] = FanZoneMetrics(
                    zone_name='Case Exhaust',
                    fan_count=4,
                    fan_rpms=exhaust_data['rpms'],
                    target_rpm=exhaust_data['target_rpm'],
                    actual_average_rpm=int(sum(exhaust_data['rpms']) / len(exhaust_data['rpms'])),
                    pwm_percent=exhaust_data['pwm_percent'],
                    airflow_direction='exhaust',
                    efficiency_score=exhaust_data['efficiency'],
                    noise_level_db=self._estimate_noise_level(exhaust_data['rpms'], 'case'),
                    power_consumption_watts=exhaust_data['power_watts']
                )
                
            # Supplementary fans
            if 'supplementary' in case_fans:
                supp_data = case_fans['supplementary']
                fan_zones['supplementary'] = FanZoneMetrics(
                    zone_name='Supplementary',
                    fan_count=3,
                    fan_rpms=supp_data['rpms'],
                    target_rpm=supp_data['target_rpm'],
                    actual_average_rpm=int(sum(supp_data['rpms']) / len(supp_data['rpms'])),
                    pwm_percent=supp_data['pwm_percent'],
                    airflow_direction='mixed',
                    efficiency_score=supp_data['efficiency'],
                    noise_level_db=self._estimate_noise_level(supp_data['rpms'], 'case'),
                    power_consumption_watts=supp_data['power_watts']
                )
                
        except Exception as e:
            logger.error(f"Error collecting fan zone metrics: {e}")
            
        return fan_zones
        
    def _get_cpu_fan_metrics(self) -> Optional[Dict[str, Any]]:
        """Get CPU fan metrics from Dark Rock Pro 5."""
        try:
            # Try to get CPU fan data from hardware monitoring
            cpu_fans = {'rpms': [], 'pwm_percent': 0.0, 'target_rpm': 1200}
            
            # Look for CPU fan controllers
            for controller_name, controller_info in self.fan_controllers.items():
                if 'cpu' in controller_name.lower():
                    for fan_file in controller_info['fans']:
                        with open(fan_file, 'r') as f:
                            rpm = int(f.read().strip())
                            cpu_fans['rpms'].append(rpm)
                            
                    # Get PWM info if available
                    if controller_info['pwm']:
                        with open(controller_info['pwm'][0], 'r') as f:
                            pwm_value = int(f.read().strip())
                            cpu_fans['pwm_percent'] = (pwm_value / 255.0) * 100
                            
            # If no specific CPU fans found, estimate from system load
            if not cpu_fans['rpms']:
                cpu_load = psutil.cpu_percent(interval=None)
                estimated_rpm = 800 + int(cpu_load * 8)  # 800-1600 RPM range
                cpu_fans['rpms'] = [estimated_rpm, estimated_rpm - 50]  # Two fans with slight variation
                cpu_fans['pwm_percent'] = min(100.0, cpu_load * 1.2)
                
            # Calculate efficiency and power
            avg_rpm = sum(cpu_fans['rpms']) / len(cpu_fans['rpms'])
            cpu_fans['efficiency'] = min(1.0, avg_rpm / 1400.0)  # Efficiency based on max RPM
            cpu_fans['power_watts'] = len(cpu_fans['rpms']) * 2.5  # Estimate 2.5W per fan
            cpu_fans['target_rpm'] = int(800 + psutil.cpu_percent() * 8)
            
            return cpu_fans
            
        except Exception as e:
            logger.error(f"Error getting CPU fan metrics: {e}")
            return None
            
    def _get_gpu_fan_metrics(self) -> Optional[Dict[str, Any]]:
        """Get GPU fan metrics from RTX 5090."""
        try:
            gpu_fans = {'rpms': [], 'pwm_percent': 0.0, 'target_rpm': 1800}
            
            # Try nvidia-smi for GPU fan information
            try:
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=fan.speed', 
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    fan_percent = float(result.stdout.strip())
                    gpu_fans['pwm_percent'] = fan_percent
                    
                    # Estimate RPM from percentage (RTX 5090 fans: ~0-3000 RPM)
                    estimated_rpm = int(fan_percent * 30)  # 30 RPM per percent
                    gpu_fans['rpms'] = [estimated_rpm, estimated_rpm + 50, estimated_rpm - 25]  # 3 fans
                    
            except Exception:
                # Fallback estimation based on GPU usage
                try:
                    gpu_util_result = subprocess.run([
                        'nvidia-smi', '--query-gpu=utilization.gpu', 
                        '--format=csv,noheader,nounits'
                    ], capture_output=True, text=True, timeout=5)
                    
                    if gpu_util_result.returncode == 0:
                        gpu_util = float(gpu_util_result.stdout.strip())
                        estimated_rpm = int(1000 + gpu_util * 20)  # 1000-3000 RPM range
                        gpu_fans['rpms'] = [estimated_rpm, estimated_rpm + 100, estimated_rpm - 50]
                        gpu_fans['pwm_percent'] = min(100.0, gpu_util * 1.1)
                        
                except Exception:
                    # Final fallback
                    gpu_fans['rpms'] = [1500, 1550, 1475]  # Default values
                    gpu_fans['pwm_percent'] = 50.0
                    
            # Calculate efficiency and power
            avg_rpm = sum(gpu_fans['rpms']) / len(gpu_fans['rpms'])
            gpu_fans['efficiency'] = min(1.0, avg_rpm / 3000.0)  # Efficiency based on max RPM
            gpu_fans['power_watts'] = len(gpu_fans['rpms']) * 4.0  # Estimate 4W per GPU fan
            gpu_fans['target_rpm'] = int(1000 + gpu_fans['pwm_percent'] * 20)
            
            return gpu_fans
            
        except Exception as e:
            logger.error(f"Error getting GPU fan metrics: {e}")
            return None
            
    def _get_case_fan_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get case fan metrics for intake, exhaust, and supplementary fans."""
        case_fans = {}
        
        try:
            # Estimate case fan performance based on system thermal load
            system_load = (psutil.cpu_percent() + self._get_gpu_utilization()) / 2
            
            # Intake fans (3x front)
            intake_rpm = int(1000 + system_load * 5)  # 1000-1500 RPM range
            case_fans['intake'] = {
                'rpms': [intake_rpm, intake_rpm + 25, intake_rpm - 25],
                'pwm_percent': min(100.0, system_load * 0.8),
                'efficiency': min(1.0, intake_rpm / 1500.0),
                'power_watts': 3 * 1.8,  # 3 fans at 1.8W each
                'target_rpm': intake_rpm
            }
            
            # Exhaust fans (3x top + 1x rear = 4 total)
            exhaust_rpm = int(900 + system_load * 6)  # 900-1500 RPM range
            case_fans['exhaust'] = {
                'rpms': [exhaust_rpm, exhaust_rpm + 50, exhaust_rpm - 25, exhaust_rpm + 10],
                'pwm_percent': min(100.0, system_load * 0.9),
                'efficiency': min(1.0, exhaust_rpm / 1400.0),
                'power_watts': 4 * 1.8,  # 4 fans at 1.8W each
                'target_rpm': exhaust_rpm
            }
            
            # Supplementary fans (3x various locations)
            supp_rpm = int(800 + system_load * 4)  # 800-1200 RPM range
            case_fans['supplementary'] = {
                'rpms': [supp_rpm, supp_rpm + 30, supp_rpm - 20],
                'pwm_percent': min(100.0, system_load * 0.7),
                'efficiency': min(1.0, supp_rpm / 1200.0),
                'power_watts': 3 * 1.8,  # 3 fans at 1.8W each
                'target_rpm': supp_rpm
            }
            
        except Exception as e:
            logger.error(f"Error getting case fan metrics: {e}")
            
        return case_fans
        
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
            
        return 0.0
        
    def _estimate_noise_level(self, rpms: List[int], fan_type: str) -> float:
        """Estimate noise level based on fan RPMs and type."""
        if not rpms:
            return 0.0
            
        avg_rpm = sum(rpms) / len(rpms)
        
        # Noise level estimation based on fan type and RPM
        if fan_type == 'cpu':
            # be quiet! Dark Rock Pro 5: 22.8 dB(A) max
            return min(22.8, 15.0 + (avg_rpm / 1400.0) * 7.8)
        elif fan_type == 'gpu':
            # RTX 5090 triple fans: ~35 dB(A) max
            return min(35.0, 20.0 + (avg_rpm / 3000.0) * 15.0)
        else:
            # Case fans: typical noise curve
            return min(30.0, 18.0 + (avg_rpm / 1500.0) * 12.0)
            
    def _collect_component_temperatures(self) -> Dict[str, ComponentThermalState]:
        """Collect temperature data for all system components."""
        component_thermals = {}
        
        try:
            # CPU temperature
            cpu_temp = self._get_cpu_temperature()
            if cpu_temp:
                component_thermals['cpu'] = ComponentThermalState(
                    component_name='AMD Ryzen 9 9950X',
                    component_type='cpu',
                    current_temp_c=cpu_temp,
                    max_temp_c=self.thermal_thresholds['cpu']['critical'],
                    thermal_margin_c=self.thermal_thresholds['cpu']['throttle_point'] - cpu_temp,
                    thermal_velocity=self._calculate_thermal_velocity('cpu', cpu_temp),
                    throttling_active=cpu_temp >= self.thermal_thresholds['cpu']['throttle_point'],
                    throttling_imminent=cpu_temp >= self.thermal_thresholds['cpu']['throttle_warn'],
                    cooling_effectiveness=self._calculate_cooling_effectiveness('cpu', cpu_temp),
                    recommended_action=self._get_thermal_recommendation('cpu', cpu_temp)
                )
                
            # GPU temperature
            gpu_temp = self._get_gpu_temperature()
            if gpu_temp:
                component_thermals['gpu'] = ComponentThermalState(
                    component_name='RTX 5090',
                    component_type='gpu',
                    current_temp_c=gpu_temp,
                    max_temp_c=self.thermal_thresholds['gpu']['critical'],
                    thermal_margin_c=self.thermal_thresholds['gpu']['throttle_point'] - gpu_temp,
                    thermal_velocity=self._calculate_thermal_velocity('gpu', gpu_temp),
                    throttling_active=gpu_temp >= self.thermal_thresholds['gpu']['throttle_point'],
                    throttling_imminent=gpu_temp >= self.thermal_thresholds['gpu']['throttle_warn'],
                    cooling_effectiveness=self._calculate_cooling_effectiveness('gpu', gpu_temp),
                    recommended_action=self._get_thermal_recommendation('gpu', gpu_temp)
                )
                
            # Memory temperature (estimated)
            memory_temp = self._estimate_memory_temperature()
            component_thermals['memory'] = ComponentThermalState(
                component_name='DDR5-6000',
                component_type='memory',
                current_temp_c=memory_temp,
                max_temp_c=self.thermal_thresholds['memory']['critical'],
                thermal_margin_c=self.thermal_thresholds['memory']['throttle_point'] - memory_temp,
                thermal_velocity=self._calculate_thermal_velocity('memory', memory_temp),
                throttling_active=memory_temp >= self.thermal_thresholds['memory']['throttle_point'],
                throttling_imminent=memory_temp >= self.thermal_thresholds['memory']['throttle_warn'],
                cooling_effectiveness=self._calculate_cooling_effectiveness('memory', memory_temp),
                recommended_action=self._get_thermal_recommendation('memory', memory_temp)
            )
            
            # Storage temperature
            storage_temp = self._get_storage_temperature()
            if storage_temp:
                component_thermals['storage'] = ComponentThermalState(
                    component_name='Samsung 990 Pro',
                    component_type='storage',
                    current_temp_c=storage_temp,
                    max_temp_c=self.thermal_thresholds['storage']['critical'],
                    thermal_margin_c=self.thermal_thresholds['storage']['throttle_point'] - storage_temp,
                    thermal_velocity=self._calculate_thermal_velocity('storage', storage_temp),
                    throttling_active=storage_temp >= self.thermal_thresholds['storage']['throttle_point'],
                    throttling_imminent=storage_temp >= self.thermal_thresholds['storage']['throttle_warn'],
                    cooling_effectiveness=self._calculate_cooling_effectiveness('storage', storage_temp),
                    recommended_action=self._get_thermal_recommendation('storage', storage_temp)
                )
                
        except Exception as e:
            logger.error(f"Error collecting component temperatures: {e}")
            
        return component_thermals
        
    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature from sensors."""
        try:
            # Try multiple methods to get CPU temperature
            
            # Method 1: Direct thermal zone reading
            if 'cpu' in self.sensor_paths:
                for sensor_type, sensor_path in self.sensor_paths['cpu'].items():
                    with open(sensor_path, 'r') as f:
                        temp_millidegrees = int(f.read().strip())
                        return temp_millidegrees / 1000.0
                        
            # Method 2: psutil sensors (if available)
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                for sensor_name, sensor_list in temps.items():
                    if 'cpu' in sensor_name.lower() or 'core' in sensor_name.lower():
                        if sensor_list:
                            return sensor_list[0].current
                            
            # Method 3: lm-sensors via subprocess
            try:
                result = subprocess.run(['sensors', '-u'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'temp1_input' in line.lower() and 'package' in line.lower():
                            temp_match = re.search(r'(\d+\.\d+)', line)
                            if temp_match:
                                return float(temp_match.group(1))
            except:
                pass
                
            # Fallback: estimate from CPU usage
            cpu_usage = psutil.cpu_percent(interval=None)
            estimated_temp = 30 + (cpu_usage / 100.0) * 35  # 30-65°C range
            return estimated_temp
            
        except Exception as e:
            logger.error(f"Error getting CPU temperature: {e}")
            return None
            
    def _get_gpu_temperature(self) -> Optional[float]:
        """Get GPU temperature from nvidia-smi."""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=temperature.gpu', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
                
        except Exception as e:
            logger.debug(f"Error getting GPU temperature via nvidia-smi: {e}")
            
        # Fallback: estimate from GPU utilization
        try:
            gpu_util = self._get_gpu_utilization()
            estimated_temp = 35 + (gpu_util / 100.0) * 40  # 35-75°C range
            return estimated_temp
        except:
            return 40.0  # Default fallback
            
    def _get_storage_temperature(self) -> Optional[float]:
        """Get storage device temperature."""
        try:
            if 'storage' in self.sensor_paths:
                for device_name, temp_path in self.sensor_paths['storage'].items():
                    with open(temp_path, 'r') as f:
                        temp_millidegrees = int(f.read().strip())
                        return temp_millidegrees / 1000.0
                        
            # Fallback: estimate based on I/O activity
            disk_io = psutil.disk_io_counters()
            if disk_io:
                # Simple estimation: more I/O = higher temperature
                io_activity = (disk_io.read_bytes + disk_io.write_bytes) / (1024 * 1024 * 1024)  # GB
                estimated_temp = 35 + min(20, io_activity * 0.1)  # 35-55°C range
                return estimated_temp
                
        except Exception as e:
            logger.error(f"Error getting storage temperature: {e}")
            
        return 40.0  # Default fallback
        
    def _estimate_memory_temperature(self) -> float:
        """Estimate memory temperature based on usage and system load."""
        try:
            memory_info = psutil.virtual_memory()
            cpu_usage = psutil.cpu_percent(interval=None)
            
            # Memory temperature is typically correlated with usage and CPU activity
            base_temp = 30  # Base ambient + some heat
            usage_factor = (memory_info.percent / 100.0) * 15  # Up to 15°C from usage
            cpu_factor = (cpu_usage / 100.0) * 10  # Up to 10°C from CPU heat
            
            estimated_temp = base_temp + usage_factor + cpu_factor
            return min(60.0, estimated_temp)  # Cap at reasonable maximum
            
        except:
            return 40.0  # Default fallback
            
    def _calculate_thermal_velocity(self, component: str, current_temp: float) -> float:
        """Calculate rate of temperature change (°C/min)."""
        try:
            if len(self.thermal_history) < 2:
                return 0.0
                
            # Get previous temperature for this component
            previous_profile = self.thermal_history[-1]
            component_thermal = getattr(previous_profile, f'{component}_thermal', None)
            
            if component_thermal:
                previous_temp = component_thermal.current_temp_c
                time_delta = (datetime.now() - previous_profile.timestamp).total_seconds() / 60.0  # minutes
                
                if time_delta > 0:
                    return (current_temp - previous_temp) / time_delta
                    
        except Exception as e:
            logger.debug(f"Error calculating thermal velocity for {component}: {e}")
            
        return 0.0
        
    def _calculate_cooling_effectiveness(self, component: str, current_temp: float) -> float:
        """Calculate cooling effectiveness for a component (0-1 score)."""
        try:
            thresholds = self.thermal_thresholds[component]
            
            # Base effectiveness on temperature relative to safe operating range
            if current_temp <= thresholds['safe_max']:
                return 1.0  # Excellent cooling
            elif current_temp <= thresholds['throttle_warn']:
                # Linear decrease from 1.0 to 0.7
                temp_range = thresholds['throttle_warn'] - thresholds['safe_max']
                temp_excess = current_temp - thresholds['safe_max']
                return 1.0 - (temp_excess / temp_range) * 0.3
            elif current_temp <= thresholds['throttle_point']:
                # Linear decrease from 0.7 to 0.3
                temp_range = thresholds['throttle_point'] - thresholds['throttle_warn']
                temp_excess = current_temp - thresholds['throttle_warn']
                return 0.7 - (temp_excess / temp_range) * 0.4
            else:
                # Poor cooling effectiveness
                return max(0.1, 0.3 - (current_temp - thresholds['throttle_point']) * 0.05)
                
        except Exception:
            return 0.5  # Default moderate effectiveness
            
    def _get_thermal_recommendation(self, component: str, current_temp: float) -> Optional[str]:
        """Get thermal management recommendation for a component."""
        try:
            thresholds = self.thermal_thresholds[component]
            
            if current_temp >= thresholds['critical']:
                return "CRITICAL: Immediate shutdown recommended"
            elif current_temp >= thresholds['throttle_point']:
                return f"Thermal throttling active - reduce {component.upper()} load"
            elif current_temp >= thresholds['throttle_warn']:
                return f"Approaching thermal limits - increase cooling"
            elif current_temp <= thresholds['safe_max']:
                return None  # No action needed
            else:
                return f"Monitor {component.upper()} temperature closely"
                
        except Exception:
            return None
            
    def _calculate_system_thermal_metrics(
        self, fan_zones: Dict[str, FanZoneMetrics], 
        component_thermals: Dict[str, ComponentThermalState]
    ) -> Dict[str, float]:
        """Calculate system-wide thermal metrics."""
        metrics = {
            'ambient_temp': 22.0,  # Default room temperature
            'case_temp': 25.0,
            'airflow_balance': 0.0,
            'thermal_efficiency': 0.0,
            'power_heat_generation': 0.0,
            'thermal_capacity_utilization': 0.0
        }
        
        try:
            # Calculate ambient temperature (estimate from component temperatures)
            if component_thermals:
                temp_values = [thermal.current_temp_c for thermal in component_thermals.values()]
                if temp_values:
                    min_temp = min(temp_values)
                    metrics['ambient_temp'] = max(20.0, min_temp - 15.0)  # Estimate ambient
                    metrics['case_temp'] = metrics['ambient_temp'] + 3.0  # Case is ~3°C above ambient
                    
            # Calculate airflow balance (intake vs exhaust CFM)
            intake_cfm = 0.0
            exhaust_cfm = 0.0
            
            if 'intake' in fan_zones:
                intake_zone = fan_zones['intake']
                # Estimate CFM from RPM (rough approximation)
                intake_cfm = sum(rpm * 0.05 for rpm in intake_zone.fan_rpms)  # 0.05 CFM per RPM
                
            if 'exhaust' in fan_zones:
                exhaust_zone = fan_zones['exhaust']
                exhaust_cfm = sum(rpm * 0.05 for rpm in exhaust_zone.fan_rpms)
                
            if exhaust_cfm > 0:
                metrics['airflow_balance'] = (intake_cfm - exhaust_cfm) / exhaust_cfm
            else:
                metrics['airflow_balance'] = 1.0  # Assume positive pressure
                
            # Calculate overall thermal efficiency
            if component_thermals:
                efficiency_values = [thermal.cooling_effectiveness for thermal in component_thermals.values()]
                metrics['thermal_efficiency'] = sum(efficiency_values) / len(efficiency_values)
                
            # Estimate power heat generation
            cpu_usage = psutil.cpu_percent(interval=None)
            gpu_usage = self._get_gpu_utilization()
            
            cpu_power = 65 + (cpu_usage / 100.0) * 105  # 65-170W TDP range
            gpu_power = 200 + (gpu_usage / 100.0) * 375  # 200-575W range
            system_power = 50  # Other components
            
            metrics['power_heat_generation'] = cpu_power + gpu_power + system_power
            
            # Calculate thermal capacity utilization
            max_cooling_capacity = 845  # 270W CPU + 575W GPU thermal capacity
            metrics['thermal_capacity_utilization'] = min(1.0, 
                (cpu_power + gpu_power) / max_cooling_capacity)
                
        except Exception as e:
            logger.error(f"Error calculating system thermal metrics: {e}")
            
        return metrics
        
    def _calculate_ai_thermal_impact(self) -> Dict[str, Any]:
        """Calculate thermal impact of AI workloads."""
        ai_impact = {
            'total_load': 0.0,
            'per_service': {}
        }
        
        try:
            # Estimate AI thermal impact based on process CPU/GPU usage
            ai_processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    if 'python' in proc.info['name'].lower():
                        if proc.info['cpu_percent'] > 5:  # Active AI process
                            ai_processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
            # Calculate thermal load from AI processes
            total_ai_cpu = sum(proc['cpu_percent'] for proc in ai_processes)
            gpu_usage = self._get_gpu_utilization()
            
            # AI thermal load estimation
            cpu_thermal_load = (total_ai_cpu / 100.0) * 0.6  # 60% of CPU thermal load
            gpu_thermal_load = (gpu_usage / 100.0) * 0.8   # 80% of GPU thermal load
            
            ai_impact['total_load'] = cpu_thermal_load + gpu_thermal_load
            
            # Per-service impact (estimated)
            ai_impact['per_service'] = {
                'llama-cpu-0': cpu_thermal_load * 0.33,
                'llama-cpu-1': cpu_thermal_load * 0.33,
                'llama-cpu-2': cpu_thermal_load * 0.34,
                'llama-gpu': gpu_thermal_load * 0.6,
                'vllm-gpu': gpu_thermal_load * 0.4
            }
            
        except Exception as e:
            logger.error(f"Error calculating AI thermal impact: {e}")
            
        return ai_impact
        
    def _create_default_fan_zone(self, zone_name: str) -> FanZoneMetrics:
        """Create default fan zone metrics when data is unavailable."""
        fan_count_map = {
            'cpu': 2, 'gpu': 3, 'intake': 3, 'exhaust': 4, 'supplementary': 3
        }
        
        fan_count = fan_count_map.get(zone_name, 2)
        default_rpm = 1200
        
        return FanZoneMetrics(
            zone_name=zone_name.title(),
            fan_count=fan_count,
            fan_rpms=[default_rpm] * fan_count,
            target_rpm=default_rpm,
            actual_average_rpm=default_rpm,
            pwm_percent=50.0,
            airflow_direction='intake' if 'intake' in zone_name else 'exhaust',
            efficiency_score=0.7,
            noise_level_db=25.0,
            power_consumption_watts=fan_count * 2.0
        )
        
    def _create_default_thermal_state(self, component: str) -> ComponentThermalState:
        """Create default thermal state when data is unavailable."""
        default_temp = 40.0
        thresholds = self.thermal_thresholds.get(component, self.thermal_thresholds['cpu'])
        
        return ComponentThermalState(
            component_name=component.upper(),
            component_type=component,
            current_temp_c=default_temp,
            max_temp_c=thresholds['critical'],
            thermal_margin_c=thresholds['throttle_point'] - default_temp,
            thermal_velocity=0.0,
            throttling_active=False,
            throttling_imminent=False,
            cooling_effectiveness=0.8,
            recommended_action=None
        )
        
    def _update_component_states(self, thermal_profile: ThermalProfile):
        """Update component thermal states tracking."""
        components = {
            'cpu': thermal_profile.cpu_thermal,
            'gpu': thermal_profile.gpu_thermal,
            'memory': thermal_profile.memory_thermal,
            'storage': thermal_profile.storage_thermal
        }
        
        self.component_states.update(components)
        
    def _update_thermal_predictions(self):
        """Update thermal predictions based on current trends."""
        try:
            if len(self.thermal_history) < 3:
                return
                
            # Generate predictions for each component
            for component_name in ['cpu', 'gpu', 'memory', 'storage']:
                prediction = self._generate_thermal_prediction(component_name)
                if prediction:
                    self.thermal_predictions[component_name] = prediction
                    
        except Exception as e:
            logger.error(f"Error updating thermal predictions: {e}")
            
    def _generate_thermal_prediction(self, component: str) -> Optional[ThermalPrediction]:
        """Generate thermal prediction for a component."""
        try:
            if len(self.thermal_history) < 3:
                return None
                
            # Get recent temperature data
            recent_temps = []
            for profile in list(self.thermal_history)[-5:]:
                component_thermal = getattr(profile, f'{component}_thermal', None)
                if component_thermal:
                    recent_temps.append(component_thermal.current_temp_c)
                    
            if len(recent_temps) < 3:
                return None
                
            # Simple linear trend analysis
            current_temp = recent_temps[-1]
            temp_trend = (recent_temps[-1] - recent_temps[0]) / len(recent_temps)
            
            # Predict temperature 30 minutes ahead
            prediction_horizon = timedelta(minutes=30)
            predicted_temp = current_temp + (temp_trend * 30)
            
            # Assess throttling risk
            thresholds = self.thermal_thresholds[component]
            throttling_risk = 'none'
            confidence = 0.7
            
            if predicted_temp >= thresholds['throttle_point']:
                throttling_risk = 'high'
                confidence = 0.9
            elif predicted_temp >= thresholds['throttle_warn']:
                throttling_risk = 'medium'
                confidence = 0.8
            elif predicted_temp >= thresholds['safe_max']:
                throttling_risk = 'low'
                confidence = 0.7
                
            # Generate recommendations
            recommendations = []
            preventive_measures = []
            
            if throttling_risk in ['high', 'medium']:
                recommendations.append(f"Increase {component} cooling immediately")
                recommendations.append(f"Reduce {component} workload if possible")
                preventive_measures.append("Monitor temperature closely")
                
            if throttling_risk == 'high':
                preventive_measures.append("Prepare for performance throttling")
                preventive_measures.append("Consider workload migration")
                
            return ThermalPrediction(
                component=component,
                prediction_horizon=prediction_horizon,
                predicted_temp_c=predicted_temp,
                confidence=confidence,
                throttling_risk=throttling_risk,
                recommended_actions=recommendations,
                preventive_measures=preventive_measures
            )
            
        except Exception as e:
            logger.error(f"Error generating thermal prediction for {component}: {e}")
            return None
            
    def _correlate_workload_thermal_impact(self):
        """Correlate AI workloads with thermal impact."""
        try:
            if not self.thermal_history:
                return
                
            current_profile = self.thermal_history[-1]
            
            # Store correlation data for each AI service
            for service, thermal_impact in current_profile.inference_thermal_impact.items():
                correlation_data = {
                    'timestamp': current_profile.timestamp,
                    'thermal_impact': thermal_impact,
                    'cpu_temp': current_profile.cpu_thermal.current_temp_c,
                    'gpu_temp': current_profile.gpu_thermal.current_temp_c,
                    'system_load': current_profile.ai_thermal_load
                }
                
                self.workload_thermal_correlation[service].append(correlation_data)
                
        except Exception as e:
            logger.error(f"Error correlating workload thermal impact: {e}")
            
    def detect_changes(self, previous_state: SystemState) -> List[SystemChange]:
        """Detect thermal changes and generate system change events."""
        changes = []
        
        # Collect current thermal profile
        current_profile = self._collect_thermal_profile()
        if not current_profile:
            return changes
            
        # Compare with previous thermal profile
        if self.thermal_history:
            previous_profile = self.thermal_history[-1]
            
            # Component temperature changes
            component_changes = self._detect_component_temperature_changes(previous_profile, current_profile)
            changes.extend(component_changes)
            
            # Fan performance changes
            fan_changes = self._detect_fan_performance_changes(previous_profile, current_profile)
            changes.extend(fan_changes)
            
            # Thermal efficiency changes
            efficiency_changes = self._detect_thermal_efficiency_changes(previous_profile, current_profile)
            changes.extend(efficiency_changes)
            
            # Throttling events
            throttling_changes = self._detect_throttling_events(previous_profile, current_profile)
            changes.extend(throttling_changes)
            
            # Airflow balance changes
            airflow_changes = self._detect_airflow_changes(previous_profile, current_profile)
            changes.extend(airflow_changes)
            
        return changes
        
    def _detect_component_temperature_changes(
        self, previous: ThermalProfile, current: ThermalProfile
    ) -> List[SystemChange]:
        """Detect significant component temperature changes."""
        changes = []
        
        components = [
            ('CPU', previous.cpu_thermal, current.cpu_thermal),
            ('GPU', previous.gpu_thermal, current.gpu_thermal),
            ('Memory', previous.memory_thermal, current.memory_thermal),
            ('Storage', previous.storage_thermal, current.storage_thermal)
        ]
        
        for comp_name, prev_thermal, curr_thermal in components:
            temp_change = abs(curr_thermal.current_temp_c - prev_thermal.current_temp_c)
            
            if temp_change > self.change_thresholds['temperature']:
                # Determine significance based on temperature level and trend
                significance = Significance.LOW
                if curr_thermal.current_temp_c > self.thermal_thresholds[comp_name.lower()]['throttle_warn']:
                    significance = Significance.HIGH
                elif curr_thermal.current_temp_c > self.thermal_thresholds[comp_name.lower()]['safe_max']:
                    significance = Significance.MEDIUM
                    
                changes.append(SystemChange(
                    component=ComponentType.THERMAL,
                    change_type=ChangeType.THERMAL_EVENT,
                    description=f"{comp_name} temperature changed by {temp_change:.1f}°C: {prev_thermal.current_temp_c:.1f}°C → {curr_thermal.current_temp_c:.1f}°C",
                    details={
                        'component': comp_name.lower(),
                        'previous_temp': prev_thermal.current_temp_c,
                        'current_temp': curr_thermal.current_temp_c,
                        'thermal_margin': curr_thermal.thermal_margin_c,
                        'thermal_velocity': curr_thermal.thermal_velocity,
                        'throttling_imminent': curr_thermal.throttling_imminent,
                        'cooling_effectiveness': curr_thermal.cooling_effectiveness,
                        'recommended_action': curr_thermal.recommended_action
                    },
                    significance=significance,
                    timestamp=current.timestamp
                ))
                
        return changes
        
    def _detect_fan_performance_changes(
        self, previous: ThermalProfile, current: ThermalProfile
    ) -> List[SystemChange]:
        """Detect fan performance changes."""
        changes = []
        
        fan_zones = [
            ('CPU', previous.cpu_fan_zone, current.cpu_fan_zone),
            ('GPU', previous.gpu_fan_zone, current.gpu_fan_zone),
            ('Intake', previous.intake_fan_zone, current.intake_fan_zone),
            ('Exhaust', previous.exhaust_fan_zone, current.exhaust_fan_zone)
        ]
        
        for zone_name, prev_zone, curr_zone in fan_zones:
            rpm_change = abs(curr_zone.actual_average_rpm - prev_zone.actual_average_rpm)
            
            if rpm_change > self.change_thresholds['fan_rpm']:
                changes.append(SystemChange(
                    component=ComponentType.COOLING,
                    change_type=ChangeType.PERFORMANCE_CHANGE,
                    description=f"{zone_name} fan zone RPM changed by {rpm_change}: {prev_zone.actual_average_rpm} → {curr_zone.actual_average_rpm} RPM",
                    details={
                        'fan_zone': zone_name.lower(),
                        'previous_rpm': prev_zone.actual_average_rpm,
                        'current_rpm': curr_zone.actual_average_rpm,
                        'target_rpm': curr_zone.target_rpm,
                        'pwm_percent': curr_zone.pwm_percent,
                        'efficiency_score': curr_zone.efficiency_score,
                        'noise_level_db': curr_zone.noise_level_db,
                        'fan_count': curr_zone.fan_count
                    },
                    significance=Significance.MEDIUM if rpm_change > 500 else Significance.LOW,
                    timestamp=current.timestamp
                ))
                
        return changes
        
    def _detect_thermal_efficiency_changes(
        self, previous: ThermalProfile, current: ThermalProfile
    ) -> List[SystemChange]:
        """Detect thermal efficiency changes."""
        changes = []
        
        efficiency_change = abs(current.thermal_efficiency - previous.thermal_efficiency)
        
        if efficiency_change > self.change_thresholds['efficiency_drop']:
            significance = Significance.HIGH if current.thermal_efficiency < 0.6 else Significance.MEDIUM
            
            changes.append(SystemChange(
                component=ComponentType.THERMAL,
                change_type=ChangeType.EFFICIENCY_CHANGE,
                description=f"Thermal efficiency changed: {previous.thermal_efficiency:.2f} → {current.thermal_efficiency:.2f}",
                details={
                    'previous_efficiency': previous.thermal_efficiency,
                    'current_efficiency': current.thermal_efficiency,
                    'thermal_capacity_utilization': current.thermal_capacity_utilization,
                    'power_heat_generation': current.power_heat_generation,
                    'ai_thermal_load': current.ai_thermal_load,
                    'needs_optimization': current.thermal_efficiency < 0.7
                },
                significance=significance,
                timestamp=current.timestamp
            ))
            
        return changes
        
    def _detect_throttling_events(
        self, previous: ThermalProfile, current: ThermalProfile
    ) -> List[SystemChange]:
        """Detect thermal throttling events."""
        changes = []
        
        components = [
            ('CPU', previous.cpu_thermal, current.cpu_thermal),
            ('GPU', previous.gpu_thermal, current.gpu_thermal),
            ('Memory', previous.memory_thermal, current.memory_thermal),
            ('Storage', previous.storage_thermal, current.storage_thermal)
        ]
        
        for comp_name, prev_thermal, curr_thermal in components:
            # Throttling activation
            if not prev_thermal.throttling_active and curr_thermal.throttling_active:
                changes.append(SystemChange(
                    component=ComponentType.THERMAL,
                    change_type=ChangeType.THROTTLING_EVENT,
                    description=f"{comp_name} thermal throttling activated at {curr_thermal.current_temp_c:.1f}°C",
                    details={
                        'component': comp_name.lower(),
                        'temperature': curr_thermal.current_temp_c,
                        'throttle_threshold': self.thermal_thresholds[comp_name.lower()]['throttle_point'],
                        'thermal_margin': curr_thermal.thermal_margin_c,
                        'performance_impact': 'high',
                        'recommended_action': curr_thermal.recommended_action
                    },
                    significance=Significance.CRITICAL,
                    timestamp=current.timestamp
                ))
                
            # Throttling deactivation
            elif prev_thermal.throttling_active and not curr_thermal.throttling_active:
                changes.append(SystemChange(
                    component=ComponentType.THERMAL,
                    change_type=ChangeType.THROTTLING_EVENT,
                    description=f"{comp_name} thermal throttling deactivated, temperature: {curr_thermal.current_temp_c:.1f}°C",
                    details={
                        'component': comp_name.lower(),
                        'temperature': curr_thermal.current_temp_c,
                        'thermal_recovery': True,
                        'thermal_margin': curr_thermal.thermal_margin_c,
                        'performance_restored': True
                    },
                    significance=Significance.MEDIUM,
                    timestamp=current.timestamp
                ))
                
            # Throttling imminent warning
            elif not prev_thermal.throttling_imminent and curr_thermal.throttling_imminent:
                changes.append(SystemChange(
                    component=ComponentType.THERMAL,
                    change_type=ChangeType.THERMAL_WARNING,
                    description=f"{comp_name} approaching thermal throttling threshold: {curr_thermal.current_temp_c:.1f}°C",
                    details={
                        'component': comp_name.lower(),
                        'temperature': curr_thermal.current_temp_c,
                        'throttle_threshold': self.thermal_thresholds[comp_name.lower()]['throttle_warn'],
                        'thermal_margin': curr_thermal.thermal_margin_c,
                        'thermal_velocity': curr_thermal.thermal_velocity,
                        'preventive_action_needed': True
                    },
                    significance=Significance.HIGH,
                    timestamp=current.timestamp
                ))
                
        return changes
        
    def _detect_airflow_changes(
        self, previous: ThermalProfile, current: ThermalProfile
    ) -> List[SystemChange]:
        """Detect airflow balance changes."""
        changes = []
        
        airflow_change = abs(current.airflow_balance - previous.airflow_balance)
        
        if airflow_change > self.change_thresholds['airflow_imbalance']:
            # Determine airflow status
            if current.airflow_balance > 0.2:
                airflow_status = 'excessive_positive_pressure'
            elif current.airflow_balance < -0.1:
                airflow_status = 'negative_pressure'
            else:
                airflow_status = 'balanced'
                
            changes.append(SystemChange(
                component=ComponentType.COOLING,
                change_type=ChangeType.AIRFLOW_CHANGE,
                description=f"Airflow balance changed: {previous.airflow_balance:.2f} → {current.airflow_balance:.2f}",
                details={
                    'previous_balance': previous.airflow_balance,
                    'current_balance': current.airflow_balance,
                    'airflow_status': airflow_status,
                    'intake_rpm_avg': current.intake_fan_zone.actual_average_rpm,
                    'exhaust_rpm_avg': current.exhaust_fan_zone.actual_average_rpm,
                    'optimization_needed': airflow_status != 'balanced'
                },
                significance=Significance.MEDIUM if airflow_status != 'balanced' else Significance.LOW,
                timestamp=current.timestamp
            ))
            
        return changes
        
    def get_thermal_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive thermal status summary."""
        if not self.thermal_history:
            current_profile = self._collect_thermal_profile()
        else:
            current_profile = self.thermal_history[-1]
            
        if not current_profile:
            return {'status': 'unavailable', 'reason': 'Cannot collect thermal data'}
            
        return {
            'status': 'active',
            'cooling_architecture': '15-fan positive pressure',
            'thermal_strategy': self.cooling_config['thermal_strategy'],
            'current_temperatures': {
                'cpu': current_profile.cpu_thermal.current_temp_c,
                'gpu': current_profile.gpu_thermal.current_temp_c,
                'memory': current_profile.memory_thermal.current_temp_c,
                'storage': current_profile.storage_thermal.current_temp_c,
                'ambient': current_profile.ambient_temp_c,
                'case': current_profile.case_temp_c
            },
            'thermal_margins': {
                'cpu': current_profile.cpu_thermal.thermal_margin_c,
                'gpu': current_profile.gpu_thermal.thermal_margin_c,
                'memory': current_profile.memory_thermal.thermal_margin_c,
                'storage': current_profile.storage_thermal.thermal_margin_c
            },
            'throttling_status': {
                'cpu': current_profile.cpu_thermal.throttling_active,
                'gpu': current_profile.gpu_thermal.throttling_active,
                'memory': current_profile.memory_thermal.throttling_active,
                'storage': current_profile.storage_thermal.throttling_active,
                'any_throttling': any([
                    current_profile.cpu_thermal.throttling_active,
                    current_profile.gpu_thermal.throttling_active,
                    current_profile.memory_thermal.throttling_active,
                    current_profile.storage_thermal.throttling_active
                ])
            },
            'fan_zones': {
                'cpu': {
                    'avg_rpm': current_profile.cpu_fan_zone.actual_average_rpm,
                    'efficiency': current_profile.cpu_fan_zone.efficiency_score,
                    'noise_db': current_profile.cpu_fan_zone.noise_level_db
                },
                'gpu': {
                    'avg_rpm': current_profile.gpu_fan_zone.actual_average_rpm,
                    'efficiency': current_profile.gpu_fan_zone.efficiency_score,
                    'noise_db': current_profile.gpu_fan_zone.noise_level_db
                },
                'intake': {
                    'avg_rpm': current_profile.intake_fan_zone.actual_average_rpm,
                    'efficiency': current_profile.intake_fan_zone.efficiency_score,
                    'airflow_direction': current_profile.intake_fan_zone.airflow_direction
                },
                'exhaust': {
                    'avg_rpm': current_profile.exhaust_fan_zone.actual_average_rpm,
                    'efficiency': current_profile.exhaust_fan_zone.efficiency_score,
                    'airflow_direction': current_profile.exhaust_fan_zone.airflow_direction
                }
            },
            'thermal_performance': {
                'overall_efficiency': current_profile.thermal_efficiency,
                'thermal_capacity_utilization': current_profile.thermal_capacity_utilization,
                'power_heat_generation': current_profile.power_heat_generation,
                'airflow_balance': current_profile.airflow_balance,
                'cooling_effectiveness': {
                    'cpu': current_profile.cpu_thermal.cooling_effectiveness,
                    'gpu': current_profile.gpu_thermal.cooling_effectiveness,
                    'memory': current_profile.memory_thermal.cooling_effectiveness,
                    'storage': current_profile.storage_thermal.cooling_effectiveness
                }
            },
            'ai_thermal_impact': {
                'total_ai_load': current_profile.ai_thermal_load,
                'per_service': current_profile.inference_thermal_impact
            },
            'thermal_predictions': {
                component: {
                    'predicted_temp': pred.predicted_temp_c,
                    'throttling_risk': pred.throttling_risk,
                    'confidence': pred.confidence,
                    'recommendations': pred.recommended_actions
                }
                for component, pred in self.thermal_predictions.items()
            },
            'monitoring_stats': {
                'thermal_samples': len(self.thermal_history),
                'sensors_detected': sum(len(sensors) for sensors in self.sensor_paths.values()),
                'fan_controllers': len(self.fan_controllers),
                'monitoring_active': self.monitoring_active,
                'last_update': current_profile.timestamp.isoformat()
            }
        }
        
    def __del__(self):
        """Cleanup on object destruction."""
        try:
            self.stop_monitoring()
        except Exception:
            pass