"""
Thermal Intelligence Management Detector - Advanced Thermal Monitoring
======================================================================

Specialized detector for comprehensive thermal intelligence across the entire
AI workstation ecosystem. Integrates thermal monitoring of CPU (AMD 9950X),
GPU (RTX 5090), and system-wide thermal management with predictive analytics,
cooling optimization, and workload correlation analysis.

Key Capabilities:
- Multi-zone thermal monitoring (CPU, GPU, Memory, Storage, Ambient)
- Predictive thermal throttling analysis with machine learning insights
- Cooling system effectiveness analysis (15-fan architecture optimization)
- Workload-thermal correlation with AI inference impact assessment
- Thermal stress testing and sustained workload suitability analysis
- Power-thermal efficiency optimization recommendations
- Environmental factor correlation (ambient temperature, case airflow)
- Thermal protection and emergency response coordination
"""

import asyncio
import json
import logging
import subprocess
import psutil
import os
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque, defaultdict
import statistics

from ..base_collector import BaseCollector
from ...temporal.types import SystemChange, ChangeType

logger = logging.getLogger(__name__)


@dataclass
class ThermalZone:
    """Individual thermal zone monitoring data."""
    zone_id: str
    zone_name: str
    sensor_type: str  # 'cpu', 'gpu', 'memory', 'storage', 'ambient', 'case'
    current_temp_c: float
    min_temp_c: float
    max_temp_c: float
    critical_temp_c: float
    hysteresis_temp_c: float
    thermal_state: str  # 'normal', 'warning', 'critical', 'throttling'
    cooling_device: Optional[str]
    power_correlation: float


@dataclass
class CoolingSystem:
    """Cooling system analysis data."""
    fan_id: str
    fan_name: str
    current_rpm: int
    max_rpm: int
    duty_cycle_percent: float
    cooling_zones: List[str]
    effectiveness_score: float
    noise_level_estimate: str  # 'quiet', 'moderate', 'loud', 'very_loud'
    power_draw_w: Optional[float]


@dataclass
class ThermalPrediction:
    """Thermal prediction and forecasting data."""
    prediction_type: str  # 'throttle_risk', 'temp_trend', 'cooling_requirement'
    confidence_level: float
    time_horizon_minutes: int
    predicted_temp_c: Optional[float]
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    recommended_actions: List[str]
    workload_impact: Dict[str, Any]


@dataclass
class WorkloadThermalCorrelation:
    """Correlation between workload and thermal behavior."""
    workload_type: str  # 'ai_inference', 'training', 'idle', 'mixed'
    cpu_thermal_impact: float
    gpu_thermal_impact: float
    memory_thermal_impact: float
    overall_thermal_load: float
    sustainable_duration_minutes: Optional[int]
    optimization_opportunities: List[str]


class ThermalIntelligenceDetector:
    """
    Specialized detector for comprehensive thermal intelligence management.
    
    Monitors thermal behavior across the entire AI workstation with predictive
    analytics, cooling optimization, and workload correlation for optimal
    sustained AI inference performance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize thermal intelligence detector."""
        self.config = config or {}
        
        # Thermal specifications for the AI workstation
        self.thermal_specs = {
            'cpu_specs': {
                'model': 'AMD 9950X',
                'max_temp_c': 95,
                'throttle_temp_c': 90,
                'warning_temp_c': 85,
                'idle_temp_c': 35,
                'tdp_w': 170
            },
            'gpu_specs': {
                'model': 'RTX 5090',
                'max_temp_c': 93,
                'throttle_temp_c': 88,
                'warning_temp_c': 83,
                'idle_temp_c': 30,
                'tdp_w': 575
            },
            'memory_specs': {
                'type': 'DDR5-6000',
                'max_temp_c': 85,
                'warning_temp_c': 70,
                'capacity_gb': 128
            },
            'cooling_specs': {
                'fan_count': 15,
                'case_fans': 6,
                'cpu_fans': 3,
                'gpu_fans': 3,
                'exhaust_fans': 3,
                'max_cfm': 200,  # Estimated total airflow
                'noise_threshold_db': 50
            }
        }
        
        # Thermal thresholds and alerts
        self.thresholds = {
            'cpu_warning': 85,
            'cpu_critical': 90,
            'cpu_throttle': 93,
            'gpu_warning': 83,
            'gpu_critical': 88,
            'gpu_throttle': 90,
            'memory_warning': 70,
            'memory_critical': 80,
            'ambient_high': 30,
            'ambient_critical': 35,
            'temp_rise_rate_warning': 2.0,  # °C per minute
            'temp_rise_rate_critical': 5.0,
            'sustained_load_temp': 80,  # Max temp for sustained workload
            'cooling_inefficiency_threshold': 60  # Effectiveness percentage
        }
        
        # Historical thermal data for trend analysis
        self.thermal_history = deque(maxlen=120)  # 2 hours at 1-minute intervals
        self.workload_thermal_history = deque(maxlen=60)  # 1 hour
        self.cooling_performance_history = deque(maxlen=30)  # 30 minutes
        
        # Thermal zones mapping
        self.thermal_zones = {}
        self.cooling_devices = {}
        
        # Prediction models (simplified)
        self.prediction_models = {
            'temp_trend': self._predict_temperature_trend,
            'throttle_risk': self._predict_throttle_risk,
            'cooling_requirement': self._predict_cooling_requirement
        }
        
        logger.info("ThermalIntelligenceDetector initialized for AI workstation thermal management")
    
    async def collect_thermal_intelligence(self) -> Dict[str, Any]:
        """Collect comprehensive thermal intelligence data."""
        try:
            # Discover and monitor thermal zones
            thermal_zones = await self._discover_thermal_zones()
            
            # Monitor cooling systems
            cooling_analysis = await self._analyze_cooling_systems()
            
            # Thermal trend analysis
            thermal_trends = await self._analyze_thermal_trends()
            
            # Workload-thermal correlation
            workload_correlation = await self._analyze_workload_thermal_correlation()
            
            # Predictive thermal analysis
            thermal_predictions = await self._generate_thermal_predictions()
            
            # System thermal health assessment
            thermal_health = await self._assess_thermal_health()
            
            # Environmental factor analysis
            environmental_analysis = await self._analyze_environmental_factors()
            
            # Optimization recommendations
            optimization_recommendations = await self._generate_thermal_optimization_recommendations(
                thermal_zones, cooling_analysis, workload_correlation
            )
            
            # Emergency thermal response planning
            emergency_response = await self._analyze_emergency_thermal_response()
            
            return {
                'thermal_zones': thermal_zones,
                'cooling_system_analysis': cooling_analysis,
                'thermal_trends': thermal_trends,
                'workload_thermal_correlation': workload_correlation,
                'thermal_predictions': thermal_predictions,
                'thermal_health_assessment': thermal_health,
                'environmental_analysis': environmental_analysis,
                'optimization_recommendations': optimization_recommendations,
                'emergency_thermal_response': emergency_response,
                'thermal_specifications': self.thermal_specs,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error collecting thermal intelligence: {e}")
            return {'error': str(e)}
    
    async def _discover_thermal_zones(self) -> Dict[str, Any]:
        """Discover and monitor all thermal zones in the system."""
        try:
            thermal_zones_data = {
                'total_zones': 0,
                'zones_by_type': {},
                'critical_zones': [],
                'thermal_overview': {},
                'zone_details': {}
            }
            
            zones_by_type = defaultdict(list)
            all_zones = []
            
            # Get thermal data from psutil (if available)
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                
                for sensor_name, sensor_list in temps.items():
                    for sensor in sensor_list:
                        zone_type = self._classify_thermal_zone(sensor_name, sensor.label)
                        
                        zone = ThermalZone(
                            zone_id=f"{sensor_name}_{sensor.label}".replace(' ', '_'),
                            zone_name=f"{sensor_name} - {sensor.label}",
                            sensor_type=zone_type,
                            current_temp_c=sensor.current,
                            min_temp_c=getattr(sensor, 'min', 0) or 0,
                            max_temp_c=getattr(sensor, 'max', 100) or 100,
                            critical_temp_c=getattr(sensor, 'critical', 95) or 95,
                            hysteresis_temp_c=getattr(sensor, 'hysteresis', 5) or 5,
                            thermal_state=self._assess_thermal_state(sensor.current, zone_type),
                            cooling_device=None,
                            power_correlation=0.0
                        )
                        
                        zones_by_type[zone_type].append(zone)
                        all_zones.append(zone)
                        
                        # Check for critical zones
                        if zone.thermal_state in ['critical', 'throttling']:
                            thermal_zones_data['critical_zones'].append({
                                'zone_id': zone.zone_id,
                                'zone_name': zone.zone_name,
                                'current_temp': zone.current_temp_c,
                                'thermal_state': zone.thermal_state
                            })
            
            # Alternative thermal detection methods
            if not all_zones:
                all_zones = await self._detect_thermal_zones_alternative()
            
            # Process collected zones
            thermal_zones_data['total_zones'] = len(all_zones)
            thermal_zones_data['zones_by_type'] = {
                zone_type: len(zones) for zone_type, zones in zones_by_type.items()
            }
            
            # Calculate thermal overview
            if all_zones:
                current_temps = [zone.current_temp_c for zone in all_zones]
                thermal_zones_data['thermal_overview'] = {
                    'average_temperature_c': round(statistics.mean(current_temps), 2),
                    'max_temperature_c': round(max(current_temps), 2),
                    'min_temperature_c': round(min(current_temps), 2),
                    'temperature_range_c': round(max(current_temps) - min(current_temps), 2),
                    'zones_in_warning': len([z for z in all_zones if z.thermal_state == 'warning']),
                    'zones_in_critical': len([z for z in all_zones if z.thermal_state == 'critical']),
                    'zones_throttling': len([z for z in all_zones if z.thermal_state == 'throttling'])
                }
                
                # Detailed zone information
                thermal_zones_data['zone_details'] = {
                    zone.zone_id: {
                        'zone_name': zone.zone_name,
                        'sensor_type': zone.sensor_type,
                        'current_temp_c': round(zone.current_temp_c, 2),
                        'thermal_state': zone.thermal_state,
                        'critical_temp_c': zone.critical_temp_c,
                        'temperature_margin_c': round(zone.critical_temp_c - zone.current_temp_c, 2)
                    }
                    for zone in all_zones
                }
            
            # Store zones for later use
            self.thermal_zones = {zone.zone_id: zone for zone in all_zones}
            
            return thermal_zones_data
            
        except Exception as e:
            logger.error(f"Error discovering thermal zones: {e}")
            return {'error': str(e)}
    
    def _classify_thermal_zone(self, sensor_name: str, sensor_label: str) -> str:
        """Classify thermal zone based on sensor name and label."""
        sensor_lower = f"{sensor_name} {sensor_label}".lower()
        
        if any(keyword in sensor_lower for keyword in ['cpu', 'core', 'package', 'processor']):
            return 'cpu'
        elif any(keyword in sensor_lower for keyword in ['gpu', 'graphics', 'nvidia', 'rtx']):
            return 'gpu'
        elif any(keyword in sensor_lower for keyword in ['memory', 'ram', 'dimm']):
            return 'memory'
        elif any(keyword in sensor_lower for keyword in ['storage', 'ssd', 'nvme', 'disk']):
            return 'storage'
        elif any(keyword in sensor_lower for keyword in ['motherboard', 'system', 'chipset']):
            return 'motherboard'
        elif any(keyword in sensor_lower for keyword in ['ambient', 'case', 'intake']):
            return 'ambient'
        else:
            return 'unknown'
    
    def _assess_thermal_state(self, current_temp: float, zone_type: str) -> str:
        """Assess thermal state based on temperature and zone type."""
        # Get type-specific thresholds
        if zone_type == 'cpu':
            warning_temp = self.thresholds['cpu_warning']
            critical_temp = self.thresholds['cpu_critical']
            throttle_temp = self.thresholds['cpu_throttle']
        elif zone_type == 'gpu':
            warning_temp = self.thresholds['gpu_warning']
            critical_temp = self.thresholds['gpu_critical']
            throttle_temp = self.thresholds['gpu_throttle']
        elif zone_type == 'memory':
            warning_temp = self.thresholds['memory_warning']
            critical_temp = self.thresholds['memory_critical']
            throttle_temp = 85  # Memory doesn't throttle, but this is danger zone
        else:
            # Generic thresholds
            warning_temp = 70
            critical_temp = 80
            throttle_temp = 85
        
        if current_temp >= throttle_temp:
            return 'throttling'
        elif current_temp >= critical_temp:
            return 'critical'
        elif current_temp >= warning_temp:
            return 'warning'
        else:
            return 'normal'
    
    async def _detect_thermal_zones_alternative(self) -> List[ThermalZone]:
        """Alternative thermal zone detection methods."""
        zones = []
        
        try:
            # Try hwmon interface
            hwmon_path = '/sys/class/hwmon'
            if os.path.exists(hwmon_path):
                for hwmon_dir in os.listdir(hwmon_path):
                    hwmon_full_path = os.path.join(hwmon_path, hwmon_dir)
                    
                    # Read hwmon name
                    name_file = os.path.join(hwmon_full_path, 'name')
                    if os.path.exists(name_file):
                        try:
                            with open(name_file, 'r') as f:
                                hwmon_name = f.read().strip()
                            
                            # Look for temperature inputs
                            for file in os.listdir(hwmon_full_path):
                                if file.startswith('temp') and file.endswith('_input'):
                                    temp_file = os.path.join(hwmon_full_path, file)
                                    try:
                                        with open(temp_file, 'r') as f:
                                            temp_millicelsius = int(f.read().strip())
                                            temp_celsius = temp_millicelsius / 1000.0
                                        
                                        # Create zone
                                        zone_type = self._classify_thermal_zone(hwmon_name, file)
                                        zone = ThermalZone(
                                            zone_id=f"{hwmon_name}_{file}",
                                            zone_name=f"{hwmon_name} {file}",
                                            sensor_type=zone_type,
                                            current_temp_c=temp_celsius,
                                            min_temp_c=0,
                                            max_temp_c=100,
                                            critical_temp_c=90,
                                            hysteresis_temp_c=5,
                                            thermal_state=self._assess_thermal_state(temp_celsius, zone_type),
                                            cooling_device=None,
                                            power_correlation=0.0
                                        )
                                        zones.append(zone)
                                        
                                    except (ValueError, IOError):
                                        continue
                        except IOError:
                            continue
            
        except Exception as e:
            logger.warning(f"Alternative thermal detection failed: {e}")
        
        return zones
    
    async def _analyze_cooling_systems(self) -> Dict[str, Any]:
        """Analyze cooling system effectiveness."""
        try:
            cooling_analysis = {
                'cooling_devices': {},
                'fan_performance': {},
                'cooling_effectiveness': 0.0,
                'noise_analysis': {},
                'power_efficiency': {},
                'optimization_opportunities': []
            }
            
            # Get fan information
            fans_data = await self._get_fan_data()
            
            if fans_data:
                cooling_analysis['cooling_devices'] = fans_data
                
                # Analyze fan performance
                total_fans = len(fans_data)
                active_fans = len([f for f in fans_data.values() if f.get('current_rpm', 0) > 0])
                avg_duty_cycle = statistics.mean([f.get('duty_cycle_percent', 0) for f in fans_data.values()]) if fans_data else 0
                
                cooling_analysis['fan_performance'] = {
                    'total_fans_detected': total_fans,
                    'active_fans': active_fans,
                    'fan_utilization_percent': (active_fans / max(total_fans, 1)) * 100,
                    'average_duty_cycle_percent': round(avg_duty_cycle, 2),
                    'max_rpm_detected': max([f.get('max_rpm', 0) for f in fans_data.values()]) if fans_data else 0
                }
                
                # Calculate cooling effectiveness
                cooling_effectiveness = self._calculate_cooling_effectiveness(fans_data)
                cooling_analysis['cooling_effectiveness'] = cooling_effectiveness
                
                # Noise analysis
                cooling_analysis['noise_analysis'] = self._analyze_cooling_noise(fans_data)
                
                # Power efficiency
                cooling_analysis['power_efficiency'] = self._analyze_cooling_power_efficiency(fans_data)
            
            # Cooling optimization opportunities
            if cooling_analysis['cooling_effectiveness'] < self.thresholds['cooling_inefficiency_threshold']:
                cooling_analysis['optimization_opportunities'].append({
                    'type': 'cooling_inefficiency',
                    'description': f'Cooling effectiveness at {cooling_analysis["cooling_effectiveness"]:.1f}%',
                    'recommendation': 'Review fan curves and airflow optimization'
                })
            
            return cooling_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing cooling systems: {e}")
            return {'error': str(e)}
    
    async def _get_fan_data(self) -> Dict[str, Any]:
        """Get fan data from system sensors."""
        fans_data = {}
        
        try:
            # Try psutil for fan data
            if hasattr(psutil, 'sensors_fans'):
                fans = psutil.sensors_fans()
                
                for fan_name, fan_list in fans.items():
                    for i, fan in enumerate(fan_list):
                        fan_id = f"{fan_name}_fan_{i}"
                        fans_data[fan_id] = {
                            'fan_name': f"{fan_name} Fan {i}",
                            'current_rpm': fan.current,
                            'max_rpm': getattr(fan, 'max', 3000) or 3000,
                            'duty_cycle_percent': (fan.current / max(getattr(fan, 'max', 3000), 1)) * 100,
                            'status': 'active' if fan.current > 0 else 'inactive'
                        }
            
            # Alternative fan detection
            if not fans_data:
                fans_data = await self._detect_fans_alternative()
            
        except Exception as e:
            logger.warning(f"Error getting fan data: {e}")
        
        return fans_data
    
    async def _detect_fans_alternative(self) -> Dict[str, Any]:
        """Alternative fan detection methods."""
        fans_data = {}
        
        try:
            # Try pwm interface
            hwmon_path = '/sys/class/hwmon'
            if os.path.exists(hwmon_path):
                for hwmon_dir in os.listdir(hwmon_path):
                    hwmon_full_path = os.path.join(hwmon_path, hwmon_dir)
                    
                    # Look for fan inputs
                    for file in os.listdir(hwmon_full_path):
                        if file.startswith('fan') and file.endswith('_input'):
                            fan_file = os.path.join(hwmon_full_path, file)
                            try:
                                with open(fan_file, 'r') as f:
                                    fan_rpm = int(f.read().strip())
                                
                                fan_id = f"hwmon_{hwmon_dir}_{file}"
                                fans_data[fan_id] = {
                                    'fan_name': f"System Fan {file}",
                                    'current_rpm': fan_rpm,
                                    'max_rpm': 3000,  # Estimated
                                    'duty_cycle_percent': (fan_rpm / 3000) * 100,
                                    'status': 'active' if fan_rpm > 0 else 'inactive'
                                }
                                
                            except (ValueError, IOError):
                                continue
            
        except Exception as e:
            logger.warning(f"Alternative fan detection failed: {e}")
        
        return fans_data
    
    def _calculate_cooling_effectiveness(self, fans_data: Dict[str, Any]) -> float:
        """Calculate overall cooling system effectiveness."""
        if not fans_data or not self.thermal_zones:
            return 0.0
        
        # Simple effectiveness calculation based on thermal margins and fan utilization
        thermal_margins = []
        for zone in self.thermal_zones.values():
            if zone.sensor_type in ['cpu', 'gpu']:
                margin = zone.critical_temp_c - zone.current_temp_c
                thermal_margins.append(max(0, margin))
        
        if not thermal_margins:
            return 50.0  # Default moderate effectiveness
        
        # Average thermal margin as percentage of critical threshold
        avg_margin = statistics.mean(thermal_margins)
        effectiveness = min(100, (avg_margin / 20) * 100)  # 20°C margin = 100% effective
        
        # Factor in fan utilization
        active_fans = len([f for f in fans_data.values() if f.get('current_rpm', 0) > 0])
        total_fans = len(fans_data)
        fan_factor = (active_fans / max(total_fans, 1)) * 100
        
        # Combined effectiveness score
        combined_effectiveness = (effectiveness * 0.7) + (fan_factor * 0.3)
        
        return round(combined_effectiveness, 2)
    
    def _analyze_cooling_noise(self, fans_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cooling system noise characteristics."""
        noise_analysis = {
            'estimated_noise_level': 'unknown',
            'noise_sources': [],
            'quiet_operation_possible': True
        }
        
        if not fans_data:
            return noise_analysis
        
        # Estimate noise based on fan RPMs
        high_rpm_fans = len([f for f in fans_data.values() if f.get('current_rpm', 0) > 2000])
        total_fans = len(fans_data)
        avg_rpm = statistics.mean([f.get('current_rpm', 0) for f in fans_data.values()])
        
        if avg_rpm > 2500:
            noise_analysis['estimated_noise_level'] = 'loud'
            noise_analysis['quiet_operation_possible'] = False
        elif avg_rpm > 1800:
            noise_analysis['estimated_noise_level'] = 'moderate'
        elif avg_rpm > 1000:
            noise_analysis['estimated_noise_level'] = 'quiet'
        else:
            noise_analysis['estimated_noise_level'] = 'very_quiet'
        
        # Identify noise sources
        for fan_id, fan_data in fans_data.items():
            if fan_data.get('current_rpm', 0) > 2200:
                noise_analysis['noise_sources'].append({
                    'fan_id': fan_id,
                    'fan_name': fan_data.get('fan_name', 'Unknown'),
                    'current_rpm': fan_data.get('current_rpm', 0),
                    'noise_contribution': 'high'
                })
        
        return noise_analysis
    
    def _analyze_cooling_power_efficiency(self, fans_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cooling system power efficiency."""
        power_efficiency = {
            'estimated_cooling_power_w': 0.0,
            'power_per_cfm': 0.0,
            'efficiency_rating': 'unknown'
        }
        
        if not fans_data:
            return power_efficiency
        
        # Estimate power consumption (rough calculation)
        total_power = 0
        for fan_data in fans_data.values():
            rpm = fan_data.get('current_rpm', 0)
            # Rough estimate: 1-3W per fan depending on RPM
            if rpm > 0:
                fan_power = 1 + (rpm / 3000) * 2  # 1-3W range
                total_power += fan_power
        
        power_efficiency['estimated_cooling_power_w'] = round(total_power, 2)
        
        # Efficiency rating
        if total_power < 20:
            power_efficiency['efficiency_rating'] = 'excellent'
        elif total_power < 40:
            power_efficiency['efficiency_rating'] = 'good'
        elif total_power < 60:
            power_efficiency['efficiency_rating'] = 'moderate'
        else:
            power_efficiency['efficiency_rating'] = 'poor'
        
        return power_efficiency
    
    async def _analyze_thermal_trends(self) -> Dict[str, Any]:
        """Analyze thermal trends and patterns."""
        try:
            thermal_trends = {
                'trend_analysis': {},
                'temperature_patterns': {},
                'thermal_stability': 'unknown',
                'trend_predictions': []
            }
            
            # Store current thermal snapshot
            current_snapshot = {
                'timestamp': datetime.now(),
                'thermal_zones': {zone_id: zone.current_temp_c for zone_id, zone in self.thermal_zones.items()},
                'overall_thermal_load': self._calculate_overall_thermal_load()
            }
            self.thermal_history.append(current_snapshot)
            
            # Analyze trends if we have sufficient history
            if len(self.thermal_history) > 5:
                thermal_trends['trend_analysis'] = self._calculate_thermal_trends()
                thermal_trends['temperature_patterns'] = self._identify_temperature_patterns()
                thermal_trends['thermal_stability'] = self._assess_thermal_stability()
            
            return thermal_trends
            
        except Exception as e:
            logger.error(f"Error analyzing thermal trends: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_thermal_load(self) -> float:
        """Calculate overall system thermal load."""
        if not self.thermal_zones:
            return 0.0
        
        # Weight zones by importance for thermal load calculation
        zone_weights = {
            'cpu': 0.4,
            'gpu': 0.4,
            'memory': 0.1,
            'storage': 0.05,
            'motherboard': 0.03,
            'ambient': 0.02
        }
        
        total_load = 0.0
        total_weight = 0.0
        
        for zone in self.thermal_zones.values():
            weight = zone_weights.get(zone.sensor_type, 0.01)
            # Normalize temperature to load (0-100 scale)
            temp_load = (zone.current_temp_c / zone.critical_temp_c) * 100
            total_load += temp_load * weight
            total_weight += weight
        
        return round(total_load / max(total_weight, 1), 2)
    
    def _calculate_thermal_trends(self) -> Dict[str, Any]:
        """Calculate thermal trends from historical data."""
        if len(self.thermal_history) < 3:
            return {}
        
        recent_data = list(self.thermal_history)[-10:]  # Last 10 measurements
        trends = {}
        
        # Overall thermal load trend
        thermal_loads = [snapshot['overall_thermal_load'] for snapshot in recent_data]
        if len(thermal_loads) >= 3:
            # Simple linear trend
            load_changes = [thermal_loads[i] - thermal_loads[i-1] for i in range(1, len(thermal_loads))]
            avg_change = statistics.mean(load_changes)
            
            if avg_change > 1.0:
                trend_direction = 'increasing'
            elif avg_change < -1.0:
                trend_direction = 'decreasing'
            else:
                trend_direction = 'stable'
            
            trends['overall_thermal_load'] = {
                'current': thermal_loads[-1],
                'trend_direction': trend_direction,
                'rate_of_change': round(avg_change, 2),
                'stability': 'stable' if abs(avg_change) < 0.5 else 'unstable'
            }
        
        # Individual zone trends
        zone_trends = {}
        for zone_id in self.thermal_zones.keys():
            zone_temps = []
            for snapshot in recent_data:
                if zone_id in snapshot['thermal_zones']:
                    zone_temps.append(snapshot['thermal_zones'][zone_id])
            
            if len(zone_temps) >= 3:
                temp_changes = [zone_temps[i] - zone_temps[i-1] for i in range(1, len(zone_temps))]
                avg_temp_change = statistics.mean(temp_changes)
                
                zone_trends[zone_id] = {
                    'current_temp': zone_temps[-1],
                    'temp_trend': 'rising' if avg_temp_change > 0.5 else 'falling' if avg_temp_change < -0.5 else 'stable',
                    'rate_of_change_per_minute': round(avg_temp_change, 2)
                }
        
        trends['zone_trends'] = zone_trends
        
        return trends
    
    def _identify_temperature_patterns(self) -> Dict[str, Any]:
        """Identify temperature patterns and cycles."""
        patterns = {
            'cyclic_behavior_detected': False,
            'temperature_cycles': [],
            'peak_patterns': {},
            'baseline_stability': 'unknown'
        }
        
        if len(self.thermal_history) < 20:  # Need more data for pattern analysis
            return patterns
        
        # Analyze thermal load cycles
        thermal_loads = [snapshot['overall_thermal_load'] for snapshot in self.thermal_history]
        
        # Simple peak detection
        peaks = []
        for i in range(1, len(thermal_loads) - 1):
            if thermal_loads[i] > thermal_loads[i-1] and thermal_loads[i] > thermal_loads[i+1]:
                peaks.append({'index': i, 'value': thermal_loads[i]})
        
        if len(peaks) > 2:
            patterns['cyclic_behavior_detected'] = True
            # Calculate average time between peaks
            peak_intervals = [peaks[i]['index'] - peaks[i-1]['index'] for i in range(1, len(peaks))]
            if peak_intervals:
                patterns['temperature_cycles'] = {
                    'average_cycle_length_minutes': round(statistics.mean(peak_intervals), 2),
                    'peak_count': len(peaks),
                    'max_peak_temperature': max([p['value'] for p in peaks])
                }
        
        return patterns
    
    def _assess_thermal_stability(self) -> str:
        """Assess overall thermal stability."""
        if len(self.thermal_history) < 10:
            return 'insufficient_data'
        
        recent_loads = [snapshot['overall_thermal_load'] for snapshot in list(self.thermal_history)[-10:]]
        
        # Calculate coefficient of variation
        if statistics.mean(recent_loads) > 0:
            cv = statistics.stdev(recent_loads) / statistics.mean(recent_loads)
            
            if cv < 0.1:
                return 'very_stable'
            elif cv < 0.2:
                return 'stable'
            elif cv < 0.3:
                return 'moderately_stable'
            else:
                return 'unstable'
        
        return 'unknown'
    
    async def _analyze_workload_thermal_correlation(self) -> Dict[str, Any]:
        """Analyze correlation between workloads and thermal behavior."""
        try:
            correlation_analysis = {
                'current_workload_impact': {},
                'workload_thermal_patterns': {},
                'ai_workload_sustainability': {},
                'thermal_bottlenecks': []
            }
            
            # Analyze current workload impact
            current_workload_impact = await self._assess_current_workload_thermal_impact()
            correlation_analysis['current_workload_impact'] = current_workload_impact
            
            # AI workload sustainability analysis
            ai_sustainability = await self._assess_ai_workload_sustainability()
            correlation_analysis['ai_workload_sustainability'] = ai_sustainability
            
            # Identify thermal bottlenecks
            thermal_bottlenecks = self._identify_thermal_bottlenecks()
            correlation_analysis['thermal_bottlenecks'] = thermal_bottlenecks
            
            return correlation_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing workload-thermal correlation: {e}")
            return {'error': str(e)}
    
    async def _assess_current_workload_thermal_impact(self) -> Dict[str, Any]:
        """Assess thermal impact of current workload."""
        workload_impact = {
            'workload_type': 'unknown',
            'thermal_load_increase': 0.0,
            'hotspots': [],
            'cooling_demand': 'normal'
        }
        
        try:
            # Get current CPU and GPU utilization
            cpu_util = psutil.cpu_percent(interval=1)
            
            # Estimate workload type
            if cpu_util > 80:
                workload_impact['workload_type'] = 'cpu_intensive'
            elif cpu_util > 30:
                workload_impact['workload_type'] = 'moderate_compute'
            else:
                workload_impact['workload_type'] = 'light_load'
            
            # Identify thermal hotspots
            for zone_id, zone in self.thermal_zones.items():
                if zone.thermal_state in ['warning', 'critical', 'throttling']:
                    workload_impact['hotspots'].append({
                        'zone_id': zone_id,
                        'zone_name': zone.zone_name,
                        'temperature': zone.current_temp_c,
                        'thermal_state': zone.thermal_state
                    })
            
            # Assess cooling demand
            max_zone_temp = max([zone.current_temp_c for zone in self.thermal_zones.values()]) if self.thermal_zones else 0
            
            if max_zone_temp > 85:
                workload_impact['cooling_demand'] = 'high'
            elif max_zone_temp > 75:
                workload_impact['cooling_demand'] = 'elevated'
            else:
                workload_impact['cooling_demand'] = 'normal'
            
        except Exception as e:
            logger.warning(f"Error assessing workload thermal impact: {e}")
        
        return workload_impact
    
    async def _assess_ai_workload_sustainability(self) -> Dict[str, Any]:
        """Assess sustainability of AI workloads under current thermal conditions."""
        sustainability = {
            'sustainable_for_continuous_inference': False,
            'estimated_max_runtime_hours': 0,
            'thermal_limiting_factors': [],
            'optimization_recommendations': []
        }
        
        try:
            # Check if current temperatures are within sustained operation range
            cpu_zones = [zone for zone in self.thermal_zones.values() if zone.sensor_type == 'cpu']
            gpu_zones = [zone for zone in self.thermal_zones.values() if zone.sensor_type == 'gpu']
            
            max_cpu_temp = max([zone.current_temp_c for zone in cpu_zones]) if cpu_zones else 0
            max_gpu_temp = max([zone.current_temp_c for zone in gpu_zones]) if gpu_zones else 0
            
            # Sustainability assessment
            if max_cpu_temp <= self.thresholds['sustained_load_temp'] and max_gpu_temp <= self.thresholds['sustained_load_temp']:
                sustainability['sustainable_for_continuous_inference'] = True
                sustainability['estimated_max_runtime_hours'] = 24  # Can run continuously
            elif max_cpu_temp <= 85 and max_gpu_temp <= 85:
                sustainability['sustainable_for_continuous_inference'] = True
                sustainability['estimated_max_runtime_hours'] = 8  # Limited continuous operation
            else:
                sustainability['sustainable_for_continuous_inference'] = False
                sustainability['estimated_max_runtime_hours'] = 2  # Short burst only
                
                # Identify limiting factors
                if max_cpu_temp > self.thresholds['sustained_load_temp']:
                    sustainability['thermal_limiting_factors'].append({
                        'component': 'CPU',
                        'current_temp': max_cpu_temp,
                        'sustainable_temp': self.thresholds['sustained_load_temp'],
                        'temp_margin': max_cpu_temp - self.thresholds['sustained_load_temp']
                    })
                
                if max_gpu_temp > self.thresholds['sustained_load_temp']:
                    sustainability['thermal_limiting_factors'].append({
                        'component': 'GPU',
                        'current_temp': max_gpu_temp,
                        'sustainable_temp': self.thresholds['sustained_load_temp'],
                        'temp_margin': max_gpu_temp - self.thresholds['sustained_load_temp']
                    })
            
            # Generate optimization recommendations
            if not sustainability['sustainable_for_continuous_inference']:
                sustainability['optimization_recommendations'].extend([
                    'Increase case ventilation and airflow',
                    'Consider undervolting CPU/GPU for lower thermal output',
                    'Implement dynamic workload scheduling with thermal breaks',
                    'Optimize AI model parameters to reduce computational load'
                ])
            
        except Exception as e:
            logger.warning(f"Error assessing AI workload sustainability: {e}")
        
        return sustainability
    
    def _identify_thermal_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify thermal bottlenecks in the system."""
        bottlenecks = []
        
        try:
            # Check for zones approaching thermal limits
            for zone_id, zone in self.thermal_zones.items():
                temp_margin = zone.critical_temp_c - zone.current_temp_c
                
                if temp_margin < 10:  # Less than 10°C margin
                    bottlenecks.append({
                        'type': 'thermal_zone_bottleneck',
                        'zone_id': zone_id,
                        'zone_name': zone.zone_name,
                        'current_temp': zone.current_temp_c,
                        'critical_temp': zone.critical_temp_c,
                        'temp_margin': temp_margin,
                        'severity': 'critical' if temp_margin < 5 else 'high'
                    })
            
            # Check for inadequate cooling
            overall_load = self._calculate_overall_thermal_load()
            if overall_load > 80:
                bottlenecks.append({
                    'type': 'cooling_system_bottleneck',
                    'description': 'Overall thermal load exceeds 80%',
                    'thermal_load_percent': overall_load,
                    'severity': 'high'
                })
            
        except Exception as e:
            logger.warning(f"Error identifying thermal bottlenecks: {e}")
        
        return bottlenecks
    
    async def _generate_thermal_predictions(self) -> List[Dict[str, Any]]:
        """Generate thermal predictions using built-in models."""
        predictions = []
        
        try:
            for prediction_type, prediction_model in self.prediction_models.items():
                prediction = prediction_model()
                if prediction:
                    predictions.append(prediction)
        
        except Exception as e:
            logger.error(f"Error generating thermal predictions: {e}")
        
        return predictions
    
    def _predict_temperature_trend(self) -> Optional[Dict[str, Any]]:
        """Predict temperature trends."""
        if len(self.thermal_history) < 5:
            return None
        
        # Simple linear prediction based on recent trend
        recent_loads = [snapshot['overall_thermal_load'] for snapshot in list(self.thermal_history)[-5:]]
        
        if len(recent_loads) >= 3:
            # Calculate trend
            load_changes = [recent_loads[i] - recent_loads[i-1] for i in range(1, len(recent_loads))]
            avg_change = statistics.mean(load_changes)
            
            # Predict temperature in next 10 minutes
            predicted_load = recent_loads[-1] + (avg_change * 10)  # 10 intervals ahead
            
            return {
                'prediction_type': 'temperature_trend',
                'confidence_level': 0.7,
                'time_horizon_minutes': 10,
                'predicted_thermal_load': round(predicted_load, 2),
                'trend_direction': 'increasing' if avg_change > 0.5 else 'decreasing' if avg_change < -0.5 else 'stable',
                'recommended_actions': ['Monitor thermal conditions'] if abs(avg_change) < 1 else ['Consider thermal management actions']
            }
        
        return None
    
    def _predict_throttle_risk(self) -> Optional[Dict[str, Any]]:
        """Predict thermal throttling risk."""
        if not self.thermal_zones:
            return None
        
        # Check zones approaching throttle temperatures
        high_risk_zones = []
        for zone_id, zone in self.thermal_zones.items():
            if zone.sensor_type in ['cpu', 'gpu']:
                throttle_temp = self.thresholds.get(f'{zone.sensor_type}_throttle', 90)
                temp_margin = throttle_temp - zone.current_temp_c
                
                if temp_margin < 10:
                    risk_level = 'high' if temp_margin < 5 else 'medium'
                    high_risk_zones.append({
                        'zone_id': zone_id,
                        'zone_name': zone.zone_name,
                        'temp_margin': temp_margin,
                        'risk_level': risk_level
                    })
        
        if high_risk_zones:
            overall_risk = 'high' if any(z['risk_level'] == 'high' for z in high_risk_zones) else 'medium'
            
            return {
                'prediction_type': 'throttle_risk',
                'confidence_level': 0.8,
                'time_horizon_minutes': 5,
                'risk_level': overall_risk,
                'at_risk_zones': high_risk_zones,
                'recommended_actions': [
                    'Reduce workload intensity',
                    'Increase cooling performance',
                    'Monitor critical zones closely'
                ]
            }
        
        return None
    
    def _predict_cooling_requirement(self) -> Optional[Dict[str, Any]]:
        """Predict cooling requirements."""
        if not self.thermal_zones:
            return None
        
        overall_load = self._calculate_overall_thermal_load()
        
        if overall_load > 70:
            cooling_requirement = 'high' if overall_load > 85 else 'increased'
            
            return {
                'prediction_type': 'cooling_requirement',
                'confidence_level': 0.7,
                'time_horizon_minutes': 15,
                'required_cooling_level': cooling_requirement,
                'current_thermal_load': overall_load,
                'recommended_actions': [
                    'Increase fan speeds',
                    'Optimize airflow patterns',
                    'Consider workload distribution'
                ] if cooling_requirement == 'increased' else [
                    'Maximize cooling performance',
                    'Implement emergency cooling measures',
                    'Consider workload reduction'
                ]
            }
        
        return None
    
    async def _assess_thermal_health(self) -> Dict[str, Any]:
        """Assess overall thermal health of the system."""
        try:
            thermal_health = {
                'overall_health_score': 0.0,
                'health_status': 'unknown',
                'critical_issues': [],
                'warnings': [],
                'recommendations': [],
                'component_health': {}
            }
            
            health_score = 100.0
            critical_issues = []
            warnings = []
            
            # Assess each thermal zone
            for zone_id, zone in self.thermal_zones.items():
                component_health = {'status': 'good', 'temperature': zone.current_temp_c, 'issues': []}
                
                if zone.thermal_state == 'throttling':
                    health_score -= 30
                    critical_issues.append(f"{zone.zone_name} is throttling")
                    component_health['status'] = 'critical'
                    component_health['issues'].append('throttling_active')
                elif zone.thermal_state == 'critical':
                    health_score -= 20
                    critical_issues.append(f"{zone.zone_name} at critical temperature")
                    component_health['status'] = 'critical'
                    component_health['issues'].append('critical_temperature')
                elif zone.thermal_state == 'warning':
                    health_score -= 10
                    warnings.append(f"{zone.zone_name} temperature elevated")
                    component_health['status'] = 'warning'
                    component_health['issues'].append('elevated_temperature')
                
                thermal_health['component_health'][zone_id] = component_health
            
            # Overall assessment
            thermal_health['overall_health_score'] = max(0, health_score)
            
            if health_score >= 90:
                thermal_health['health_status'] = 'excellent'
            elif health_score >= 75:
                thermal_health['health_status'] = 'good'
            elif health_score >= 60:
                thermal_health['health_status'] = 'fair'
            elif health_score >= 40:
                thermal_health['health_status'] = 'poor'
            else:
                thermal_health['health_status'] = 'critical'
            
            thermal_health['critical_issues'] = critical_issues
            thermal_health['warnings'] = warnings
            
            # Generate recommendations
            recommendations = []
            if critical_issues:
                recommendations.extend([
                    'Immediate action required - reduce system load',
                    'Check cooling system functionality',
                    'Consider emergency thermal management'
                ])
            elif warnings:
                recommendations.extend([
                    'Monitor thermal conditions closely',
                    'Consider increasing cooling performance',
                    'Review system airflow optimization'
                ])
            else:
                recommendations.append('Thermal conditions are optimal')
            
            thermal_health['recommendations'] = recommendations
            
            return thermal_health
            
        except Exception as e:
            logger.error(f"Error assessing thermal health: {e}")
            return {'error': str(e)}
    
    async def _analyze_environmental_factors(self) -> Dict[str, Any]:
        """Analyze environmental factors affecting thermal performance."""
        try:
            environmental_analysis = {
                'ambient_temperature': 'unknown',
                'case_airflow_efficiency': 'unknown',
                'environmental_impact_score': 0.0,
                'seasonal_considerations': {},
                'optimization_opportunities': []
            }
            
            # Try to get ambient temperature from thermal zones
            ambient_zones = [zone for zone in self.thermal_zones.values() if zone.sensor_type == 'ambient']
            
            if ambient_zones:
                ambient_temp = statistics.mean([zone.current_temp_c for zone in ambient_zones])
                environmental_analysis['ambient_temperature'] = round(ambient_temp, 2)
                
                # Environmental impact assessment
                if ambient_temp > self.thresholds['ambient_critical']:
                    environmental_analysis['environmental_impact_score'] = 20  # High impact
                    environmental_analysis['optimization_opportunities'].append({
                        'type': 'ambient_temperature',
                        'description': f'High ambient temperature: {ambient_temp}°C',
                        'recommendation': 'Improve room cooling or case ventilation'
                    })
                elif ambient_temp > self.thresholds['ambient_high']:
                    environmental_analysis['environmental_impact_score'] = 50  # Medium impact
                else:
                    environmental_analysis['environmental_impact_score'] = 80  # Low impact
            
            # Case airflow efficiency estimation
            if self.thermal_zones:
                # Estimate airflow efficiency based on temperature differentials
                cpu_zones = [zone for zone in self.thermal_zones.values() if zone.sensor_type == 'cpu']
                case_zones = [zone for zone in self.thermal_zones.values() if zone.sensor_type in ['ambient', 'case']]
                
                if cpu_zones and case_zones:
                    max_cpu_temp = max([zone.current_temp_c for zone in cpu_zones])
                    min_case_temp = min([zone.current_temp_c for zone in case_zones])
                    
                    temp_delta = max_cpu_temp - min_case_temp
                    
                    if temp_delta < 30:
                        environmental_analysis['case_airflow_efficiency'] = 'excellent'
                    elif temp_delta < 40:
                        environmental_analysis['case_airflow_efficiency'] = 'good'
                    elif temp_delta < 50:
                        environmental_analysis['case_airflow_efficiency'] = 'fair'
                    else:
                        environmental_analysis['case_airflow_efficiency'] = 'poor'
                        environmental_analysis['optimization_opportunities'].append({
                            'type': 'airflow_efficiency',
                            'description': f'Large temperature delta: {temp_delta}°C',
                            'recommendation': 'Optimize case airflow and fan placement'
                        })
            
            return environmental_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing environmental factors: {e}")
            return {'error': str(e)}
    
    async def _generate_thermal_optimization_recommendations(self,
                                                           thermal_zones: Dict[str, Any],
                                                           cooling_analysis: Dict[str, Any],
                                                           workload_correlation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate comprehensive thermal optimization recommendations."""
        recommendations = []
        
        try:
            # Critical thermal issues
            critical_zones = thermal_zones.get('critical_zones', [])
            if critical_zones:
                recommendations.append({
                    'category': 'critical_thermal_management',
                    'priority': 'critical',
                    'title': 'Critical Thermal Conditions Detected',
                    'description': f'{len(critical_zones)} zones in critical state',
                    'actions': [
                        'Reduce system workload immediately',
                        'Check cooling system functionality',
                        'Implement emergency thermal management protocol'
                    ]
                })
            
            # Cooling optimization
            cooling_effectiveness = cooling_analysis.get('cooling_effectiveness', 0)
            if cooling_effectiveness < 70:
                recommendations.append({
                    'category': 'cooling_optimization',
                    'priority': 'high',
                    'title': 'Cooling System Optimization Required',
                    'description': f'Cooling effectiveness at {cooling_effectiveness}%',
                    'actions': [
                        'Review and optimize fan curves',
                        'Check for dust accumulation in fans and heatsinks',
                        'Consider additional case ventilation',
                        'Verify thermal paste application on CPU/GPU'
                    ]
                })
            
            # AI workload sustainability
            ai_sustainability = workload_correlation.get('ai_workload_sustainability', {})
            if not ai_sustainability.get('sustainable_for_continuous_inference', True):
                recommendations.append({
                    'category': 'ai_workload_optimization',
                    'priority': 'medium',
                    'title': 'AI Workload Thermal Sustainability',
                    'description': 'Current thermal conditions limit continuous AI inference',
                    'actions': [
                        'Implement dynamic workload scheduling with thermal breaks',
                        'Consider model optimization for reduced thermal output',
                        'Optimize CPU/GPU power management settings',
                        'Review container resource allocation and core pinning'
                    ]
                })
            
            # Environmental optimization
            thermal_overview = thermal_zones.get('thermal_overview', {})
            avg_temp = thermal_overview.get('average_temperature_c', 0)
            if avg_temp > 70:
                recommendations.append({
                    'category': 'environmental_optimization',
                    'priority': 'medium',
                    'title': 'Environmental Thermal Management',
                    'description': f'Average system temperature: {avg_temp}°C',
                    'actions': [
                        'Improve room ambient temperature control',
                        'Optimize case placement for better airflow',
                        'Consider case modifications for improved ventilation',
                        'Review cable management for airflow optimization'
                    ]
                })
            
            # Preventive maintenance
            recommendations.append({
                'category': 'preventive_maintenance',
                'priority': 'low',
                'title': 'Thermal System Maintenance',
                'description': 'Regular maintenance for optimal thermal performance',
                'actions': [
                    'Schedule monthly dust cleaning of fans and filters',
                    'Monitor thermal paste condition (replace every 2-3 years)',
                    'Verify fan operation and replace failing fans promptly',
                    'Update system BIOS/firmware for improved thermal management'
                ]
            })
            
        except Exception as e:
            logger.warning(f"Error generating thermal recommendations: {e}")
        
        return recommendations
    
    async def _analyze_emergency_thermal_response(self) -> Dict[str, Any]:
        """Analyze emergency thermal response capabilities."""
        try:
            emergency_response = {
                'emergency_protocols_available': [],
                'automatic_protection_active': False,
                'manual_intervention_options': [],
                'thermal_shutdown_risk': 'low',
                'emergency_contact_points': []
            }
            
            # Check for critical thermal conditions
            critical_zones = [zone for zone in self.thermal_zones.values() 
                            if zone.thermal_state in ['critical', 'throttling']]
            
            if critical_zones:
                emergency_response['thermal_shutdown_risk'] = 'high'
                emergency_response['emergency_protocols_available'].extend([
                    'immediate_workload_reduction',
                    'emergency_cooling_activation',
                    'graceful_system_shutdown_preparation'
                ])
            
            # Manual intervention options
            emergency_response['manual_intervention_options'] = [
                'Manually increase fan speeds to maximum',
                'Reduce CPU/GPU frequencies and voltages',
                'Pause non-critical AI inference services',
                'Open case panels for emergency cooling',
                'Redirect workload to cooler system components'
            ]
            
            # Automatic protection status
            thermal_protection_active = any(zone.thermal_state == 'throttling' 
                                          for zone in self.thermal_zones.values())
            emergency_response['automatic_protection_active'] = thermal_protection_active
            
            return emergency_response
            
        except Exception as e:
            logger.error(f"Error analyzing emergency thermal response: {e}")
            return {'error': str(e)}
    
    async def detect_changes(self, old_data: Dict[str, Any], new_data: Dict[str, Any]) -> List[SystemChange]:
        """Detect changes in thermal intelligence state."""
        changes = []
        
        if 'thermal_zones' not in old_data or 'thermal_zones' not in new_data:
            return changes
        
        old_thermal = old_data['thermal_zones']
        new_thermal = new_data['thermal_zones']
        
        # Overall thermal health changes
        if 'thermal_health_assessment' in old_data and 'thermal_health_assessment' in new_data:
            old_health = old_data['thermal_health_assessment'].get('overall_health_score', 0)
            new_health = new_data['thermal_health_assessment'].get('overall_health_score', 0)
            
            health_delta = abs(new_health - old_health)
            if health_delta > 15:  # 15 point health score change
                changes.append(SystemChange(
                    category='thermal_intelligence',
                    change_type=ChangeType.MODIFIED,
                    entity_id='thermal_health_score',
                    old_value=old_health,
                    new_value=new_health,
                    significance=0.8,
                    metadata={
                        'change_type': 'thermal_health_change',
                        'health_delta': new_health - old_health,
                        'health_trend': 'improving' if new_health > old_health else 'degrading'
                    },
                    timestamp=datetime.now()
                ))
        
        # Critical thermal zone changes
        old_critical = old_thermal.get('critical_zones', [])
        new_critical = new_thermal.get('critical_zones', [])
        
        if len(new_critical) > len(old_critical):
            changes.append(SystemChange(
                category='thermal_intelligence',
                change_type=ChangeType.THRESHOLD_CROSSED,
                entity_id='critical_thermal_zones',
                old_value=len(old_critical),
                new_value=len(new_critical),
                significance=1.0,
                metadata={
                    'change_type': 'critical_thermal_zones_increased',
                    'new_critical_zones': new_critical,
                    'immediate_action_required': True
                },
                timestamp=datetime.now()
            ))
        
        # Thermal overview changes
        if ('thermal_overview' in old_thermal and 'thermal_overview' in new_thermal):
            old_overview = old_thermal['thermal_overview']
            new_overview = new_thermal['thermal_overview']
            
            old_max_temp = old_overview.get('max_temperature_c', 0)
            new_max_temp = new_overview.get('max_temperature_c', 0)
            
            temp_delta = abs(new_max_temp - old_max_temp)
            if temp_delta > 5:  # 5°C change in max temperature
                changes.append(SystemChange(
                    category='thermal_intelligence',
                    change_type=ChangeType.MODIFIED,
                    entity_id='max_system_temperature',
                    old_value=old_max_temp,
                    new_value=new_max_temp,
                    significance=0.7,
                    metadata={
                        'change_type': 'max_temperature_change',
                        'temperature_delta': new_max_temp - old_max_temp,
                        'trend': 'heating' if new_max_temp > old_max_temp else 'cooling'
                    },
                    timestamp=datetime.now()
                ))
        
        # Cooling system effectiveness changes
        if ('cooling_system_analysis' in old_data and 'cooling_system_analysis' in new_data):
            old_cooling = old_data['cooling_system_analysis']
            new_cooling = new_data['cooling_system_analysis']
            
            old_effectiveness = old_cooling.get('cooling_effectiveness', 0)
            new_effectiveness = new_cooling.get('cooling_effectiveness', 0)
            
            effectiveness_delta = abs(new_effectiveness - old_effectiveness)
            if effectiveness_delta > 10:  # 10% effectiveness change
                changes.append(SystemChange(
                    category='thermal_intelligence',
                    change_type=ChangeType.MODIFIED,
                    entity_id='cooling_effectiveness',
                    old_value=old_effectiveness,
                    new_value=new_effectiveness,
                    significance=0.6,
                    metadata={
                        'change_type': 'cooling_effectiveness_change',
                        'effectiveness_delta': new_effectiveness - old_effectiveness,
                        'trend': 'improving' if new_effectiveness > old_effectiveness else 'degrading'
                    },
                    timestamp=datetime.now()
                ))
        
        # AI workload sustainability changes
        if ('workload_thermal_correlation' in old_data and 'workload_thermal_correlation' in new_data):
            old_ai_sustainability = old_data['workload_thermal_correlation'].get('ai_workload_sustainability', {})
            new_ai_sustainability = new_data['workload_thermal_correlation'].get('ai_workload_sustainability', {})
            
            old_sustainable = old_ai_sustainability.get('sustainable_for_continuous_inference', True)
            new_sustainable = new_ai_sustainability.get('sustainable_for_continuous_inference', True)
            
            if old_sustainable != new_sustainable:
                changes.append(SystemChange(
                    category='thermal_intelligence',
                    change_type=ChangeType.THRESHOLD_CROSSED,
                    entity_id='ai_workload_sustainability',
                    old_value=old_sustainable,
                    new_value=new_sustainable,
                    significance=0.9,
                    metadata={
                        'change_type': 'ai_sustainability_change',
                        'sustainability_status': 'gained' if new_sustainable else 'lost',
                        'workload_impact': 'high'
                    },
                    timestamp=datetime.now()
                ))
        
        return changes