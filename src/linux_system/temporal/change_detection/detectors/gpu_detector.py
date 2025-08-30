"""
RTX 5090 GPU Change Detector
=============================

Specialized change detector for NVIDIA RTX 5090 GPU with deep understanding
of thermal behavior, memory patterns, and compute workload characteristics.

This detector is specifically tuned for the RTX 5090's performance envelope:
- 32GB GDDR6X memory
- 600W+ power consumption capabilities
- Advanced thermal management
- ML/AI workload optimization
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from ..base_detector import BaseChangeDetector
from ...types import SystemChange, ChangeType
from ...config import GPUDetectorConfig


class RTX5090ThermalThresholds:
    """RTX 5090-specific thermal threshold definitions."""
    
    # Temperature thresholds (°C)
    IDLE_MAX = 35           # Above this when idle is concerning
    NORMAL_LOAD_MAX = 75    # Normal operating temperature under load
    WARNING_THRESHOLD = 80  # Start watching closely
    CRITICAL_THRESHOLD = 85 # Performance impact likely
    THROTTLING_IMMINENT = 88 # Throttling very likely
    EMERGENCY_THRESHOLD = 90 # Emergency throttling
    
    # Temperature change rates (°C per collection interval)
    RAPID_RISE = 8.0        # Rapid temperature increase
    RAPID_DROP = 10.0       # Rapid temperature decrease (cooling)
    
    @classmethod
    def get_threshold_type(cls, temperature: float) -> str:
        """Get threshold type for a given temperature."""
        if temperature >= cls.EMERGENCY_THRESHOLD:
            return "emergency"
        elif temperature >= cls.THROTTLING_IMMINENT:
            return "throttling_imminent" 
        elif temperature >= cls.CRITICAL_THRESHOLD:
            return "critical"
        elif temperature >= cls.WARNING_THRESHOLD:
            return "warning"
        elif temperature >= cls.NORMAL_LOAD_MAX:
            return "normal_load"
        elif temperature >= cls.IDLE_MAX:
            return "elevated_idle"
        else:
            return "normal"


class GPUMemoryAnalyzer:
    """Analyzes RTX 5090's 32GB memory usage patterns."""
    
    def __init__(self):
        self.total_memory_gb = 32  # RTX 5090 has 32GB
        self.total_memory_mb = self.total_memory_gb * 1024
    
    def analyze_memory_change(self, old_memory: Dict, new_memory: Dict) -> List[Dict[str, Any]]:
        """Analyze memory usage changes."""
        insights = []
        
        old_used_mb = self._extract_memory_used_mb(old_memory)
        new_used_mb = self._extract_memory_used_mb(new_memory)
        
        if old_used_mb is None or new_used_mb is None:
            return insights
        
        usage_change_mb = new_used_mb - old_used_mb
        old_usage_percent = (old_used_mb / self.total_memory_mb) * 100
        new_usage_percent = (new_used_mb / self.total_memory_mb) * 100
        
        # Significant memory allocation/deallocation
        if abs(usage_change_mb) > 512:  # > 512MB change
            insights.append({
                'type': 'significant_memory_change',
                'change_mb': usage_change_mb,
                'old_usage_percent': old_usage_percent,
                'new_usage_percent': new_usage_percent,
                'severity': 'high' if abs(usage_change_mb) > 2048 else 'medium'
            })
        
        # Memory pressure levels
        if new_usage_percent > 95:
            insights.append({
                'type': 'critical_memory_pressure',
                'usage_percent': new_usage_percent,
                'available_mb': self.total_memory_mb - new_used_mb
            })
        elif new_usage_percent > 85:
            insights.append({
                'type': 'high_memory_pressure',
                'usage_percent': new_usage_percent,
                'trend': 'increasing' if usage_change_mb > 0 else 'stable'
            })
        
        # Memory leakage detection (steady increase pattern would be detected over time)
        if usage_change_mb > 100 and new_usage_percent > 70:
            insights.append({
                'type': 'potential_memory_leak',
                'change_mb': usage_change_mb,
                'usage_percent': new_usage_percent
            })
        
        return insights
    
    def _extract_memory_used_mb(self, memory_data: Dict) -> Optional[float]:
        """Extract used memory in MB from memory data."""
        try:
            # Try different possible locations for memory data in SystemCollector output
            if isinstance(memory_data, dict):
                # Look for basic metrics format
                if 'basic_metrics' in memory_data:
                    # Parse CSV format from nvidia-smi
                    metrics_str = memory_data['basic_metrics']
                    if isinstance(metrics_str, str) and 'memory.used' in metrics_str:
                        # Extract memory.used value from CSV
                        match = re.search(r'memory\.used,\s*(\d+)', metrics_str)
                        if match:
                            return float(match.group(1))
                
                # Look for detailed XML format
                if 'detailed_xml' in memory_data:
                    xml_str = memory_data['detailed_xml']
                    if isinstance(xml_str, str):
                        # Parse memory usage from XML
                        match = re.search(r'<memory_usage>.*?<used>(\d+)\s*MiB</used>', xml_str, re.DOTALL)
                        if match:
                            return float(match.group(1))
                
                # Direct memory value
                if 'memory_used_mb' in memory_data:
                    return float(memory_data['memory_used_mb'])
            
            return None
            
        except (ValueError, AttributeError, KeyError):
            return None


class GPUProcessTracker:
    """Tracks GPU process spawning and termination."""
    
    def analyze_process_changes(self, old_processes: Dict, new_processes: Dict) -> List[Dict[str, Any]]:
        """Analyze changes in GPU processes."""
        insights = []
        
        old_pids = self._extract_process_pids(old_processes)
        new_pids = self._extract_process_pids(new_processes)
        
        # New processes
        new_process_pids = new_pids - old_pids
        for pid in new_process_pids:
            process_info = self._get_process_info(new_processes, pid)
            insights.append({
                'type': 'gpu_process_started',
                'pid': pid,
                'process_info': process_info,
                'memory_usage': process_info.get('memory_usage', 0)
            })
        
        # Terminated processes  
        terminated_pids = old_pids - new_pids
        for pid in terminated_pids:
            process_info = self._get_process_info(old_processes, pid)
            insights.append({
                'type': 'gpu_process_terminated',
                'pid': pid,
                'process_info': process_info,
                'memory_freed': process_info.get('memory_usage', 0)
            })
        
        # Memory usage changes for existing processes
        common_pids = old_pids.intersection(new_pids)
        for pid in common_pids:
            old_info = self._get_process_info(old_processes, pid)
            new_info = self._get_process_info(new_processes, pid)
            
            old_memory = old_info.get('memory_usage', 0)
            new_memory = new_info.get('memory_usage', 0)
            
            if abs(new_memory - old_memory) > 256:  # > 256MB change
                insights.append({
                    'type': 'gpu_process_memory_change',
                    'pid': pid,
                    'process_name': new_info.get('process_name', 'unknown'),
                    'old_memory_mb': old_memory,
                    'new_memory_mb': new_memory,
                    'change_mb': new_memory - old_memory
                })
        
        return insights
    
    def _extract_process_pids(self, process_data: Dict) -> set:
        """Extract PIDs of GPU processes."""
        pids = set()
        
        try:
            # Look for compute processes
            if 'compute_processes' in process_data:
                compute_str = process_data['compute_processes']
                if isinstance(compute_str, str):
                    # Parse CSV format
                    for line in compute_str.split('\n'):
                        if line.strip() and not line.startswith('pid'):
                            parts = line.split(',')
                            if len(parts) > 0 and parts[0].strip().isdigit():
                                pids.add(int(parts[0].strip()))
            
            # Look for graphics processes
            if 'graphics_processes' in process_data:
                graphics_str = process_data['graphics_processes']
                if isinstance(graphics_str, str):
                    for line in graphics_str.split('\n'):
                        if line.strip() and not line.startswith('pid'):
                            parts = line.split(',')
                            if len(parts) > 0 and parts[0].strip().isdigit():
                                pids.add(int(parts[0].strip()))
                                
        except (ValueError, AttributeError):
            pass
        
        return pids
    
    def _get_process_info(self, process_data: Dict, pid: int) -> Dict[str, Any]:
        """Get information about a specific GPU process."""
        info = {'pid': pid}
        
        try:
            # Search in compute processes
            if 'compute_processes' in process_data:
                compute_str = process_data['compute_processes']
                if isinstance(compute_str, str):
                    for line in compute_str.split('\n'):
                        if line.strip() and str(pid) in line:
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 5:
                                info.update({
                                    'process_name': parts[1],
                                    'gpu_uuid': parts[2],
                                    'gpu_name': parts[3],
                                    'memory_usage': self._parse_memory_usage(parts[4])
                                })
                                break
            
            # Search in graphics processes if not found
            if 'process_name' not in info and 'graphics_processes' in process_data:
                graphics_str = process_data['graphics_processes']
                if isinstance(graphics_str, str):
                    for line in graphics_str.split('\n'):
                        if line.strip() and str(pid) in line:
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 5:
                                info.update({
                                    'process_name': parts[1],
                                    'gpu_uuid': parts[2], 
                                    'gpu_name': parts[3],
                                    'memory_usage': self._parse_memory_usage(parts[4])
                                })
                                break
                                
        except (ValueError, AttributeError, IndexError):
            pass
        
        return info
    
    def _parse_memory_usage(self, memory_str: str) -> int:
        """Parse memory usage string to MB integer."""
        try:
            # Remove units and convert to int
            memory_clean = re.sub(r'[^\d]', '', str(memory_str))
            return int(memory_clean) if memory_clean else 0
        except (ValueError, AttributeError):
            return 0


class GPUChangeDetector(BaseChangeDetector):
    """
    RTX 5090-specialized GPU change detector with thermal intelligence.
    
    Features:
    - RTX 5090-specific thermal threshold monitoring
    - 32GB memory usage pattern analysis
    - GPU process lifecycle tracking
    - Performance state transitions
    - Power consumption monitoring
    - CUDA workload detection
    """
    
    def __init__(self, config: GPUDetectorConfig, category: str = "nvidia_gpu"):
        """Initialize RTX 5090 GPU change detector."""
        super().__init__(config, category)
        
        self.thermal_thresholds = RTX5090ThermalThresholds()
        self.memory_analyzer = GPUMemoryAnalyzer()
        self.process_tracker = GPUProcessTracker()
        
        # Track thermal history for trend analysis
        self.thermal_history = []
        self.max_thermal_history = 10  # Keep last 10 readings
        
        self.logger.info("GPUChangeDetector initialized for RTX 5090")
    
    def detect_changes(self, old_data: Dict[str, Any], new_data: Dict[str, Any]) -> List[SystemChange]:
        """Detect RTX 5090-specific GPU changes."""
        changes = []
        
        try:
            # Thermal monitoring (highest priority for RTX 5090)
            changes.extend(self._detect_thermal_changes(old_data, new_data))
            
            # Memory usage analysis
            changes.extend(self._detect_memory_changes(old_data, new_data))
            
            # GPU process monitoring
            changes.extend(self._detect_process_changes(old_data, new_data))
            
            # Performance state changes
            changes.extend(self._detect_performance_changes(old_data, new_data))
            
            # Power consumption monitoring
            changes.extend(self._detect_power_changes(old_data, new_data))
            
            # Clock speed changes
            changes.extend(self._detect_clock_changes(old_data, new_data))
            
        except Exception as e:
            self.logger.error(f"Error in GPU change detection: {e}")
            raise
        
        return changes
    
    def _detect_thermal_changes(self, old_data: Dict, new_data: Dict) -> List[SystemChange]:
        """Detect thermal-related changes with RTX 5090 awareness."""
        changes = []
        
        old_temp = self._extract_gpu_temperature(old_data)
        new_temp = self._extract_gpu_temperature(new_data)
        
        if old_temp is None or new_temp is None:
            return changes
        
        # Update thermal history
        self.thermal_history.append(new_temp)
        if len(self.thermal_history) > self.max_thermal_history:
            self.thermal_history.pop(0)
        
        temp_delta = new_temp - old_temp
        
        # Threshold crossing detection
        old_threshold_type = self.thermal_thresholds.get_threshold_type(old_temp)
        new_threshold_type = self.thermal_thresholds.get_threshold_type(new_temp)
        
        if old_threshold_type != new_threshold_type:
            significance = self._calculate_thermal_significance(old_temp, new_temp, new_threshold_type)
            
            changes.append(self._create_change(
                change_type=ChangeType.THRESHOLD_CROSSED,
                entity_id="gpu:temperature",
                old_value=old_temp,
                new_value=new_temp,
                significance=significance,
                metadata={
                    'old_threshold': old_threshold_type,
                    'new_threshold': new_threshold_type,
                    'temp_delta': temp_delta,
                    'trend': self._calculate_thermal_trend()
                }
            ))
        
        # Rapid temperature changes (even without threshold crossing)
        if abs(temp_delta) >= self.config.rapid_temp_change_threshold:
            change_type = ChangeType.ANOMALY_DETECTED if abs(temp_delta) > 10 else ChangeType.MODIFIED
            significance = min(0.9, abs(temp_delta) / 15.0)  # Scale with temperature change
            
            changes.append(self._create_change(
                change_type=change_type,
                entity_id="gpu:temperature_trend",
                old_value=old_temp,
                new_value=new_temp,
                significance=significance,
                metadata={
                    'temp_delta': temp_delta,
                    'change_rate': abs(temp_delta),
                    'direction': 'rising' if temp_delta > 0 else 'falling',
                    'thermal_trend': self._calculate_thermal_trend()
                }
            ))
        
        return changes
    
    def _detect_memory_changes(self, old_data: Dict, new_data: Dict) -> List[SystemChange]:
        """Detect memory usage changes.""" 
        changes = []
        
        memory_insights = self.memory_analyzer.analyze_memory_change(old_data, new_data)
        
        for insight in memory_insights:
            insight_type = insight['type']
            significance = self._calculate_memory_significance(insight)
            
            if insight_type == 'significant_memory_change':
                changes.append(self._create_change(
                    change_type=ChangeType.MODIFIED,
                    entity_id="gpu:memory_usage",
                    old_value=f"{insight['old_usage_percent']:.1f}%",
                    new_value=f"{insight['new_usage_percent']:.1f}%",
                    significance=significance,
                    metadata=insight
                ))
            
            elif insight_type in ['critical_memory_pressure', 'high_memory_pressure']:
                changes.append(self._create_change(
                    change_type=ChangeType.THRESHOLD_CROSSED,
                    entity_id="gpu:memory_pressure",
                    old_value="normal",
                    new_value=insight_type,
                    significance=0.9 if 'critical' in insight_type else 0.7,
                    metadata=insight
                ))
            
            elif insight_type == 'potential_memory_leak':
                changes.append(self._create_change(
                    change_type=ChangeType.ANOMALY_DETECTED,
                    entity_id="gpu:memory_leak_indicator",
                    old_value="stable",
                    new_value="increasing",
                    significance=0.8,
                    metadata=insight
                ))
        
        return changes
    
    def _detect_process_changes(self, old_data: Dict, new_data: Dict) -> List[SystemChange]:
        """Detect GPU process changes."""
        changes = []
        
        process_insights = self.process_tracker.analyze_process_changes(old_data, new_data)
        
        for insight in process_insights:
            insight_type = insight['type']
            pid = insight['pid']
            
            if insight_type == 'gpu_process_started':
                changes.append(self._create_change(
                    change_type=ChangeType.ADDED,
                    entity_id=f"gpu_process:{pid}",
                    old_value=None,
                    new_value=insight['process_info'],
                    significance=0.7,
                    metadata=insight
                ))
            
            elif insight_type == 'gpu_process_terminated':
                changes.append(self._create_change(
                    change_type=ChangeType.REMOVED,
                    entity_id=f"gpu_process:{pid}",
                    old_value=insight['process_info'],
                    new_value=None,
                    significance=0.6,
                    metadata=insight
                ))
            
            elif insight_type == 'gpu_process_memory_change':
                significance = min(0.8, abs(insight['change_mb']) / 1024.0)  # Scale with memory change
                changes.append(self._create_change(
                    change_type=ChangeType.MODIFIED,
                    entity_id=f"gpu_process:{pid}:memory",
                    old_value=insight['old_memory_mb'],
                    new_value=insight['new_memory_mb'],
                    significance=significance,
                    metadata=insight
                ))
        
        return changes
    
    def _detect_performance_changes(self, old_data: Dict, new_data: Dict) -> List[SystemChange]:
        """Detect GPU performance state changes."""
        changes = []
        
        # GPU utilization changes
        old_util = self._extract_gpu_utilization(old_data)
        new_util = self._extract_gpu_utilization(new_data)
        
        if old_util is not None and new_util is not None:
            util_delta = abs(new_util - old_util)
            
            if util_delta > 20:  # Significant utilization change
                changes.append(self._create_change(
                    change_type=ChangeType.MODIFIED,
                    entity_id="gpu:utilization",
                    old_value=old_util,
                    new_value=new_util,
                    significance=min(0.8, util_delta / 100.0),
                    metadata={
                        'utilization_delta': new_util - old_util,
                        'utilization_change': util_delta
                    }
                ))
        
        return changes
    
    def _detect_power_changes(self, old_data: Dict, new_data: Dict) -> List[SystemChange]:
        """Detect power consumption changes."""
        changes = []
        
        old_power = self._extract_power_draw(old_data)
        new_power = self._extract_power_draw(new_data)
        
        if old_power is not None and new_power is not None:
            power_delta = abs(new_power - old_power)
            
            if power_delta > self.config.power_significant_change_w:
                changes.append(self._create_change(
                    change_type=ChangeType.MODIFIED,
                    entity_id="gpu:power_draw",
                    old_value=old_power,
                    new_value=new_power,
                    significance=min(0.8, power_delta / 100.0),
                    metadata={
                        'power_delta_w': new_power - old_power,
                        'power_change_w': power_delta
                    }
                ))
            
            # High power consumption warning
            if new_power > self.config.power_critical_threshold_w:
                changes.append(self._create_change(
                    change_type=ChangeType.THRESHOLD_CROSSED,
                    entity_id="gpu:power_threshold",
                    old_value="normal",
                    new_value="high_power",
                    significance=0.8,
                    metadata={'power_w': new_power, 'threshold_w': self.config.power_critical_threshold_w}
                ))
        
        return changes
    
    def _detect_clock_changes(self, old_data: Dict, new_data: Dict) -> List[SystemChange]:
        """Detect GPU clock speed changes."""
        changes = []
        
        # Graphics clock changes
        old_graphics_clock = self._extract_graphics_clock(old_data)
        new_graphics_clock = self._extract_graphics_clock(new_data)
        
        if old_graphics_clock is not None and new_graphics_clock is not None:
            clock_delta = abs(new_graphics_clock - old_graphics_clock)
            
            if clock_delta > 50:  # > 50 MHz change
                changes.append(self._create_change(
                    change_type=ChangeType.MODIFIED,
                    entity_id="gpu:graphics_clock",
                    old_value=old_graphics_clock,
                    new_value=new_graphics_clock,
                    significance=min(0.6, clock_delta / 500.0),
                    metadata={
                        'clock_delta_mhz': new_graphics_clock - old_graphics_clock,
                        'clock_change_mhz': clock_delta
                    }
                ))
        
        return changes
    
    def _calculate_thermal_significance(self, old_temp: float, new_temp: float, threshold_type: str) -> float:
        """Calculate significance of thermal changes."""
        base_significance = 0.5
        
        # Higher significance for dangerous temperatures
        if threshold_type in ['emergency', 'throttling_imminent']:
            base_significance = 0.95
        elif threshold_type == 'critical':
            base_significance = 0.85
        elif threshold_type == 'warning':
            base_significance = 0.7
        
        # Adjust for temperature delta
        temp_delta = abs(new_temp - old_temp)
        delta_factor = min(0.3, temp_delta / 20.0)
        
        return min(1.0, base_significance + delta_factor)
    
    def _calculate_memory_significance(self, insight: Dict[str, Any]) -> float:
        """Calculate significance of memory changes."""
        insight_type = insight['type']
        
        if insight_type == 'critical_memory_pressure':
            return 0.9
        elif insight_type == 'high_memory_pressure':
            return 0.7
        elif insight_type == 'potential_memory_leak':
            return 0.8
        elif insight_type == 'significant_memory_change':
            # Scale with change magnitude
            change_mb = abs(insight.get('change_mb', 0))
            return min(0.8, change_mb / 2048.0)  # Scale up to 2GB
        
        return 0.5
    
    def _calculate_thermal_trend(self) -> str:
        """Calculate thermal trend from recent history."""
        if len(self.thermal_history) < 3:
            return "insufficient_data"
        
        recent_temps = self.thermal_history[-3:]
        
        if recent_temps[-1] > recent_temps[-2] > recent_temps[-3]:
            return "rising"
        elif recent_temps[-1] < recent_temps[-2] < recent_temps[-3]:
            return "falling"
        else:
            return "stable"
    
    # Data extraction utilities
    
    def _extract_gpu_temperature(self, data: Dict) -> Optional[float]:
        """Extract GPU temperature from data."""
        return self._extract_numeric_value(data, 'basic_metrics.temperature.gpu', None) or \
               self._parse_csv_value(data.get('basic_metrics', ''), 'temperature.gpu')
    
    def _extract_gpu_utilization(self, data: Dict) -> Optional[float]:
        """Extract GPU utilization percentage."""
        return self._extract_numeric_value(data, 'basic_metrics.utilization.gpu', None) or \
               self._parse_csv_value(data.get('basic_metrics', ''), 'utilization.gpu')
    
    def _extract_power_draw(self, data: Dict) -> Optional[float]:
        """Extract power draw in watts.""" 
        return self._extract_numeric_value(data, 'basic_metrics.power.draw', None) or \
               self._parse_csv_value(data.get('basic_metrics', ''), 'power.draw')
    
    def _extract_graphics_clock(self, data: Dict) -> Optional[float]:
        """Extract graphics clock speed in MHz."""
        return self._extract_numeric_value(data, 'clock_speeds.graphics', None) or \
               self._parse_csv_value(data.get('clock_speeds', ''), 'clocks.current.graphics')
    
    def _parse_csv_value(self, csv_string: str, field_name: str) -> Optional[float]:
        """Parse value from nvidia-smi CSV output."""
        if not csv_string or not isinstance(csv_string, str):
            return None
        
        try:
            # Look for field in CSV header and extract corresponding value
            lines = csv_string.strip().split('\n')
            if len(lines) < 2:
                return None
            
            header_line = lines[0]
            data_line = lines[1]
            
            headers = [h.strip() for h in header_line.split(',')]
            values = [v.strip() for v in data_line.split(',')]
            
            if field_name in headers and len(values) > headers.index(field_name):
                value_str = values[headers.index(field_name)]
                # Remove units and extract number
                number_match = re.search(r'(\d+\.?\d*)', value_str)
                if number_match:
                    return float(number_match.group(1))
            
            return None
            
        except (ValueError, IndexError, AttributeError):
            return None