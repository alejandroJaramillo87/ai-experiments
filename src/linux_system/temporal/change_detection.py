"""
Change Detection Engine
======================

Orchestrates all system change detectors to provide comprehensive
change detection across all system components.
"""

import logging
from typing import Dict, List, Optional, Any, Type
from datetime import datetime

from .types import SystemSnapshot, SystemChange
from .change_detection.detectors import (
    GPUChangeDetector,
    ProcessChangeDetector, 
    PythonEnvChangeDetector,
    MemoryChangeDetector,
    StorageChangeDetector,
    SecurityChangeDetector,
    NetworkChangeDetector
)

logger = logging.getLogger(__name__)


class ChangeDetectionEngine:
    """
    Engine that orchestrates all change detection components.
    
    Manages multiple specialized detectors and provides unified
    interface for detecting system changes.
    """
    
    def __init__(self, config: Any):
        """
        Initialize change detection engine.
        
        Args:
            config: Configuration object with detector settings
        """
        self.config = config
        self.detectors = {}
        
        # Initialize available detectors based on config
        if getattr(config, 'enable_gpu_monitoring', True):
            self.detectors['gpu'] = GPUChangeDetector()
            
        if getattr(config, 'enable_process_monitoring', True):
            self.detectors['process'] = ProcessChangeDetector()
            
        if getattr(config, 'enable_python_env_monitoring', True):
            self.detectors['python_env'] = PythonEnvChangeDetector()
        
        if getattr(config, 'enable_memory_monitoring', True):
            self.detectors['memory'] = MemoryChangeDetector()
            
        if getattr(config, 'enable_storage_monitoring', True):
            self.detectors['storage'] = StorageChangeDetector()
            
        if getattr(config, 'enable_network_monitoring', True):
            self.detectors['network'] = NetworkChangeDetector()
            
        if getattr(config, 'enable_security_monitoring', True):
            self.detectors['security'] = SecurityChangeDetector()
        
        logger.info(f"Initialized change detection engine with {len(self.detectors)} detectors: {list(self.detectors.keys())}")
    
    def detect_changes(self, old_snapshot: SystemSnapshot, 
                      new_snapshot: SystemSnapshot) -> List[SystemChange]:
        """
        Detect changes between system snapshots.
        
        Args:
            old_snapshot: Previous system state
            new_snapshot: Current system state
            
        Returns:
            List of detected changes
        """
        all_changes = []
        
        for detector_name, detector in self.detectors.items():
            try:
                changes = detector.detect_changes(old_snapshot, new_snapshot)
                all_changes.extend(changes)
                logger.debug(f"Detector {detector_name} found {len(changes)} changes")
            except Exception as e:
                logger.error(f"Error in {detector_name} detector: {e}")
        
        logger.debug(f"Total changes detected: {len(all_changes)}")
        return all_changes
    
    def get_detector_status(self) -> Dict[str, Any]:
        """Get status of all detectors."""
        status = {}
        
        for name, detector in self.detectors.items():
            try:
                if hasattr(detector, 'get_status'):
                    detector_status = detector.get_status()
                else:
                    detector_status = {
                        "enabled": True,
                        "healthy": True,
                        "last_check": datetime.now().isoformat()
                    }
                
                status[name] = detector_status
                
            except Exception as e:
                status[name] = {
                    "enabled": False,
                    "healthy": False,
                    "error": str(e),
                    "last_check": datetime.now().isoformat()
                }
        
        return status
    
    def add_detector(self, name: str, detector: Any) -> None:
        """Add a new detector to the engine."""
        self.detectors[name] = detector
        logger.info(f"Added detector: {name}")
    
    def remove_detector(self, name: str) -> bool:
        """Remove a detector from the engine."""
        if name in self.detectors:
            del self.detectors[name]
            logger.info(f"Removed detector: {name}")
            return True
        return False
    
    def get_available_detectors(self) -> List[str]:
        """Get list of available detector names."""
        return list(self.detectors.keys())