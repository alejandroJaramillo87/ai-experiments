"""
Mock System Change Factory
===========================

Factory for creating realistic SystemChange objects for testing
change detection logic and event correlation.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from src.linux_system.temporal.types import SystemChange, ChangeType


class MockSystemChangeFactory:
    """Factory for creating mock SystemChange objects."""
    
    @staticmethod
    def gpu_thermal_threshold_crossed(
        old_temp: float = 75.0,
        new_temp: float = 84.0,
        significance: float = 0.85
    ) -> SystemChange:
        """Create a GPU thermal threshold crossing change."""
        return SystemChange(
            category="nvidia_gpu",
            change_type=ChangeType.THRESHOLD_CROSSED,
            entity_id="gpu:temperature",
            old_value=old_temp,
            new_value=new_temp,
            significance=significance,
            metadata={
                'old_threshold': 'normal_load',
                'new_threshold': 'warning',
                'temp_delta': new_temp - old_temp,
                'trend': 'rising'
            },
            timestamp=datetime.now()
        )
    
    @staticmethod
    def gpu_process_started(
        pid: int = 12345,
        process_name: str = "python",
        memory_mb: int = 2048,
        significance: float = 0.7
    ) -> SystemChange:
        """Create a GPU process started change."""
        return SystemChange(
            category="nvidia_gpu",
            change_type=ChangeType.ADDED,
            entity_id=f"gpu_process:{pid}",
            old_value=None,
            new_value={
                'pid': pid,
                'process_name': process_name,
                'memory_usage': memory_mb,
                'gpu_uuid': 'GPU-a1b2c3d4-e5f6-7890-1234-567890abcdef'
            },
            significance=significance,
            metadata={
                'type': 'gpu_process_started',
                'process_info': {
                    'process_name': process_name,
                    'memory_usage': memory_mb
                }
            },
            timestamp=datetime.now()
        )
    
    @staticmethod
    def gpu_memory_pressure(
        old_usage_percent: float = 60.0,
        new_usage_percent: float = 96.0,
        significance: float = 0.9
    ) -> SystemChange:
        """Create a GPU memory pressure change."""
        return SystemChange(
            category="nvidia_gpu", 
            change_type=ChangeType.THRESHOLD_CROSSED,
            entity_id="gpu:memory_pressure",
            old_value="normal",
            new_value="critical_memory_pressure",
            significance=significance,
            metadata={
                'type': 'critical_memory_pressure',
                'old_usage_percent': old_usage_percent,
                'new_usage_percent': new_usage_percent,
                'available_mb': int(32768 * (100 - new_usage_percent) / 100)
            },
            timestamp=datetime.now()
        )
    
    @staticmethod
    def python_package_installed(
        package_name: str = "tensorflow-gpu",
        version: str = "2.14.0",
        significance: float = 0.6
    ) -> SystemChange:
        """Create a Python package installation change."""
        return SystemChange(
            category="python_env",
            change_type=ChangeType.ADDED,
            entity_id=f"package:{package_name}",
            old_value=None,
            new_value=f"{package_name}=={version}",
            significance=significance,
            metadata={
                'package_name': package_name,
                'version': version,
                'installation_method': 'pip'
            },
            timestamp=datetime.now()
        )
    
    @staticmethod
    def process_spawned(
        pid: int = 23456,
        process_name: str = "/usr/bin/python3",
        parent_pid: int = 1234,
        significance: float = 0.5
    ) -> SystemChange:
        """Create a process spawned change."""
        return SystemChange(
            category="processes",
            change_type=ChangeType.ADDED,
            entity_id=f"process:{pid}",
            old_value=None,
            new_value={
                'pid': pid,
                'command': process_name,
                'parent_pid': parent_pid,
                'status': 'running'
            },
            significance=significance,
            metadata={
                'parent_pid': parent_pid,
                'process_type': 'user_process'
            },
            timestamp=datetime.now()
        )
    
    @staticmethod
    def memory_usage_spike(
        old_usage_percent: float = 45.0,
        new_usage_percent: float = 85.0,
        significance: float = 0.75
    ) -> SystemChange:
        """Create a system memory usage spike change."""
        return SystemChange(
            category="memory",
            change_type=ChangeType.MODIFIED,
            entity_id="system:memory_usage",
            old_value=f"{old_usage_percent}%",
            new_value=f"{new_usage_percent}%",
            significance=significance,
            metadata={
                'usage_delta': new_usage_percent - old_usage_percent,
                'memory_type': 'system_ram'
            },
            timestamp=datetime.now()
        )
    
    @staticmethod
    def create_thermal_sequence(
        base_temp: float = 70.0,
        temp_increases: List[float] = None,
        time_intervals_seconds: List[int] = None
    ) -> List[SystemChange]:
        """Create a sequence of thermal changes over time."""
        if temp_increases is None:
            temp_increases = [5.0, 8.0, 12.0]  # Progressive heating
        if time_intervals_seconds is None:
            time_intervals_seconds = [60, 120, 180]  # 1, 2, 3 minutes apart
        
        changes = []
        current_temp = base_temp
        base_time = datetime.now() - timedelta(seconds=sum(time_intervals_seconds))
        
        for i, (temp_increase, interval) in enumerate(zip(temp_increases, time_intervals_seconds)):
            old_temp = current_temp
            new_temp = current_temp + temp_increase
            current_temp = new_temp
            
            change_time = base_time + timedelta(seconds=sum(time_intervals_seconds[:i+1]))
            
            change = SystemChange(
                category="nvidia_gpu",
                change_type=ChangeType.THRESHOLD_CROSSED if new_temp > 80 else ChangeType.MODIFIED,
                entity_id="gpu:temperature",
                old_value=old_temp,
                new_value=new_temp,
                significance=min(0.9, (new_temp - 70.0) / 20.0),  # Increases with temperature
                metadata={
                    'temp_delta': temp_increase,
                    'sequence_position': i,
                    'trend': 'rising'
                },
                timestamp=change_time
            )
            changes.append(change)
        
        return changes
    
    @staticmethod
    def create_batch_with_categories(categories: List[str], count_per_category: int = 3) -> List[SystemChange]:
        """Create a batch of changes across multiple categories."""
        changes = []
        
        for category in categories:
            for i in range(count_per_category):
                if category == "nvidia_gpu":
                    changes.append(MockSystemChangeFactory.gpu_thermal_threshold_crossed(
                        old_temp=70.0 + i * 2,
                        new_temp=80.0 + i * 3
                    ))
                elif category == "processes":
                    changes.append(MockSystemChangeFactory.process_spawned(
                        pid=10000 + i,
                        process_name=f"process_{i}"
                    ))
                elif category == "python_env":
                    packages = ["torch", "tensorflow", "transformers"]
                    changes.append(MockSystemChangeFactory.python_package_installed(
                        package_name=packages[i % len(packages)],
                        version=f"1.{i}.0"
                    ))
                elif category == "memory":
                    changes.append(MockSystemChangeFactory.memory_usage_spike(
                        old_usage_percent=50.0 + i * 5,
                        new_usage_percent=70.0 + i * 5
                    ))
        
        return changes