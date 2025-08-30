"""
Mock System Event Factory
==========================

Factory for creating realistic SystemEvent objects for testing
event extraction and correlation logic.
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.linux_system.temporal.types import SystemEvent, SystemChange, EventSeverity
from .mock_change_factory import MockSystemChangeFactory


class MockSystemEventFactory:
    """Factory for creating mock SystemEvent objects."""
    
    @staticmethod
    def gpu_thermal_event(
        temperature: float = 84.0,
        severity: EventSeverity = EventSeverity.WARNING,
        confidence: float = 0.9
    ) -> SystemEvent:
        """Create a GPU thermal event."""
        cause_change = MockSystemChangeFactory.gpu_thermal_threshold_crossed(
            old_temp=75.0, new_temp=temperature
        )
        
        return SystemEvent(
            event_type="gpu_thermal_event",
            entity="rtx_5090",
            description=f"GPU temperature crossed thermal threshold: 75.0°C → {temperature}°C",
            severity=severity,
            timestamp=datetime.now(),
            causes=[cause_change],
            predicted_effects=["gpu_throttling_imminent", "performance_degradation"],
            context={
                'temperature_delta': temperature - 75.0,
                'throttling_threshold': 88,
                'current_threshold_type': 'warning'
            },
            confidence=confidence
        )
    
    @staticmethod 
    def gpu_process_started_event(
        pid: int = 12345,
        process_name: str = "python",
        memory_mb: int = 2048,
        severity: EventSeverity = EventSeverity.INFO
    ) -> SystemEvent:
        """Create a GPU process started event."""
        cause_change = MockSystemChangeFactory.gpu_process_started(pid, process_name, memory_mb)
        
        return SystemEvent(
            event_type="gpu_process_started",
            entity=f"gpu_process:{pid}",
            description=f"New GPU process started: {process_name} (PID: {pid})",
            severity=severity,
            timestamp=datetime.now(),
            causes=[cause_change],
            predicted_effects=["gpu_memory_increase", "gpu_utilization_increase"],
            context={
                'process_name': process_name,
                'memory_usage_mb': memory_mb,
                'gpu_uuid': 'GPU-a1b2c3d4-e5f6-7890-1234-567890abcdef'
            },
            confidence=0.95
        )
    
    @staticmethod
    def memory_pressure_event(
        usage_percent: float = 96.0,
        severity: EventSeverity = EventSeverity.CRITICAL
    ) -> SystemEvent:
        """Create a memory pressure event.""" 
        cause_change = MockSystemChangeFactory.gpu_memory_pressure(
            old_usage_percent=60.0, new_usage_percent=usage_percent
        )
        
        return SystemEvent(
            event_type="gpu_memory_pressure",
            entity="rtx_5090_memory",
            description=f"Critical GPU memory pressure detected: {usage_percent:.1f}% used",
            severity=severity,
            timestamp=datetime.now(),
            causes=[cause_change],
            predicted_effects=["out_of_memory_risk", "process_termination_likely", "performance_degradation"],
            context={
                'usage_percent': usage_percent,
                'available_mb': int(32768 * (100 - usage_percent) / 100),
                'pressure_level': 'critical'
            },
            confidence=0.9
        )
    
    @staticmethod
    def package_installation_event(
        package_name: str = "tensorflow-gpu",
        version: str = "2.14.0"
    ) -> SystemEvent:
        """Create a package installation event."""
        cause_change = MockSystemChangeFactory.python_package_installed(package_name, version)
        
        return SystemEvent(
            event_type="python_package_installed",
            entity=f"package:{package_name}",
            description=f"Python package installed: {package_name}=={version}",
            severity=EventSeverity.INFO,
            timestamp=datetime.now(),
            causes=[cause_change],
            predicted_effects=["environment_change", "dependency_updates_possible"],
            context={
                'package_name': package_name,
                'version': version,
                'installation_method': 'pip',
                'is_ml_framework': package_name in ['torch', 'tensorflow', 'transformers']
            },
            confidence=0.95
        )
    
    @staticmethod
    def service_restart_event(
        service_name: str = "docker",
        severity: EventSeverity = EventSeverity.WARNING
    ) -> SystemEvent:
        """Create a service restart event."""
        return SystemEvent(
            event_type="service_restarted",
            entity=f"service:{service_name}",
            description=f"System service restarted: {service_name}",
            severity=severity,
            timestamp=datetime.now(),
            causes=[],  # Would be filled by actual change detection
            predicted_effects=["container_disruption", "network_reset"],
            context={
                'service_name': service_name,
                'restart_reason': 'automatic',
                'downtime_seconds': 3
            },
            confidence=0.85
        )
    
    @staticmethod
    def create_thermal_escalation_sequence() -> List[SystemEvent]:
        """Create a sequence of escalating thermal events."""
        events = []
        
        # Initial warning
        events.append(MockSystemEventFactory.gpu_thermal_event(
            temperature=81.0,
            severity=EventSeverity.WARNING,
            confidence=0.8
        ))
        
        # Critical level
        events.append(MockSystemEventFactory.gpu_thermal_event(
            temperature=87.0,
            severity=EventSeverity.CRITICAL,
            confidence=0.95
        ))
        
        # Emergency level
        emergency_event = MockSystemEventFactory.gpu_thermal_event(
            temperature=91.0,
            severity=EventSeverity.CRITICAL,
            confidence=0.99
        )
        emergency_event.predicted_effects.append("emergency_throttling")
        emergency_event.context['emergency_threshold_exceeded'] = True
        events.append(emergency_event)
        
        # Set appropriate timestamps
        base_time = datetime.now() - timedelta(minutes=5)
        for i, event in enumerate(events):
            event.timestamp = base_time + timedelta(minutes=i * 2)
        
        return events
    
    @staticmethod
    def create_ml_training_workflow_events() -> List[SystemEvent]:
        """Create events that simulate an ML training workflow starting."""
        events = []
        base_time = datetime.now() - timedelta(minutes=10)
        
        # 1. Package installation
        package_event = MockSystemEventFactory.package_installation_event("torch", "2.1.0")
        package_event.timestamp = base_time
        events.append(package_event)
        
        # 2. GPU process starts
        process_event = MockSystemEventFactory.gpu_process_started_event(
            pid=12345, process_name="pytorch_training", memory_mb=4096
        )
        process_event.timestamp = base_time + timedelta(minutes=2)
        events.append(process_event)
        
        # 3. Memory pressure builds
        memory_event = MockSystemEventFactory.memory_pressure_event(usage_percent=85.0)
        memory_event.severity = EventSeverity.WARNING  # Not critical yet
        memory_event.timestamp = base_time + timedelta(minutes=5)
        events.append(memory_event)
        
        # 4. Thermal increase
        thermal_event = MockSystemEventFactory.gpu_thermal_event(temperature=83.0)
        thermal_event.timestamp = base_time + timedelta(minutes=7)
        events.append(thermal_event)
        
        return events
    
    @staticmethod
    def create_events_with_correlations(
        correlation_type: str = "causal_chain",
        event_count: int = 3
    ) -> List[SystemEvent]:
        """Create events designed to show specific correlation patterns."""
        events = []
        base_time = datetime.now() - timedelta(minutes=event_count * 2)
        
        if correlation_type == "causal_chain":
            # Process spawn → Memory increase → Thermal increase
            events = [
                MockSystemEventFactory.gpu_process_started_event(),
                MockSystemEventFactory.memory_pressure_event(usage_percent=78.0),
                MockSystemEventFactory.gpu_thermal_event(temperature=82.0)
            ]
        
        elif correlation_type == "temporal_sequence":
            # Regular pattern of events
            for i in range(event_count):
                events.append(MockSystemEventFactory.gpu_thermal_event(
                    temperature=75.0 + i * 2,
                    confidence=0.8 + i * 0.05
                ))
        
        elif correlation_type == "anomaly_cluster":
            # Multiple unusual events in short timeframe
            events = [
                MockSystemEventFactory.memory_pressure_event(),
                MockSystemEventFactory.service_restart_event("nvidia-persistenced"),
                MockSystemEventFactory.gpu_thermal_event(temperature=89.0)
            ]
        
        # Set appropriate timestamps
        for i, event in enumerate(events):
            event.timestamp = base_time + timedelta(minutes=i * 2)
        
        return events