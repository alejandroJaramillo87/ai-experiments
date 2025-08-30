"""
Change Detector Registry
========================

Plugin system for registering and managing change detectors.
Provides a clean way to add new detectors and configure the detection pipeline.
"""

import logging
from typing import Dict, List, Type, Optional, Any
from ..config import ChangeDetectorConfig
from .base_detector import BaseChangeDetector


class ChangeDetectorRegistry:
    """
    Registry for managing change detector plugins.
    
    Provides a centralized way to register, configure, and instantiate
    change detectors for different system categories.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("temporal.change_detection.registry")
        
        # Registry of detector classes
        self._detector_classes: Dict[str, Type[BaseChangeDetector]] = {}
        
        # Default configurations for each category
        self._default_configs: Dict[str, ChangeDetectorConfig] = {}
        
        # Instantiated detectors
        self._detectors: Dict[str, BaseChangeDetector] = {}
        
        # Categories that are disabled
        self._disabled_categories: set = set()
    
    def register_detector(self, 
                         category: str, 
                         detector_class: Type[BaseChangeDetector],
                         default_config: Optional[ChangeDetectorConfig] = None):
        """
        Register a change detector class for a category.
        
        Args:
            category: Category name (e.g., 'nvidia_gpu', 'processes')
            detector_class: Detector class that extends BaseChangeDetector
            default_config: Default configuration for this detector
        """
        if not issubclass(detector_class, BaseChangeDetector):
            raise ValueError(f"Detector class must extend BaseChangeDetector")
        
        self._detector_classes[category] = detector_class
        
        if default_config:
            self._default_configs[category] = default_config
        else:
            self._default_configs[category] = ChangeDetectorConfig()
        
        self.logger.info(f"Registered change detector for category: {category}")
    
    def unregister_detector(self, category: str):
        """Unregister a detector for a category."""
        if category in self._detector_classes:
            del self._detector_classes[category]
            
        if category in self._default_configs:
            del self._default_configs[category]
            
        if category in self._detectors:
            del self._detectors[category]
            
        self.logger.info(f"Unregistered change detector for category: {category}")
    
    def get_detector(self, 
                    category: str, 
                    config: Optional[ChangeDetectorConfig] = None) -> BaseChangeDetector:
        """
        Get or create a detector instance for a category.
        
        Args:
            category: Category name
            config: Configuration to use (uses default if None)
            
        Returns:
            Configured detector instance
            
        Raises:
            ValueError: If category is not registered or disabled
        """
        if category in self._disabled_categories:
            raise ValueError(f"Category '{category}' is disabled")
            
        if category not in self._detector_classes:
            raise ValueError(f"No detector registered for category '{category}'")
        
        # Use cached detector if available and config matches
        if category in self._detectors:
            detector = self._detectors[category]
            # Check if we need to update config
            if config and detector.config != config:
                # Re-create detector with new config
                del self._detectors[category]
            else:
                return detector
        
        # Create new detector instance
        detector_class = self._detector_classes[category]
        detector_config = config or self._default_configs[category]
        
        try:
            detector = detector_class(detector_config, category)
            self._detectors[category] = detector
            
            self.logger.debug(f"Created detector instance for category: {category}")
            return detector
            
        except Exception as e:
            self.logger.error(f"Failed to create detector for {category}: {e}")
            raise
    
    def get_all_detectors(self, 
                         configs: Optional[Dict[str, ChangeDetectorConfig]] = None) -> Dict[str, BaseChangeDetector]:
        """
        Get detector instances for all registered categories.
        
        Args:
            configs: Optional configurations for specific categories
            
        Returns:
            Dictionary mapping categories to detector instances
        """
        detectors = {}
        configs = configs or {}
        
        for category in self.get_registered_categories():
            if category not in self._disabled_categories:
                try:
                    config = configs.get(category)
                    detectors[category] = self.get_detector(category, config)
                except Exception as e:
                    self.logger.error(f"Failed to get detector for {category}: {e}")
                    # Continue with other detectors
        
        return detectors
    
    def get_registered_categories(self) -> List[str]:
        """Get list of all registered categories."""
        return list(self._detector_classes.keys())
    
    def is_registered(self, category: str) -> bool:
        """Check if a category has a registered detector."""
        return category in self._detector_classes
    
    def is_disabled(self, category: str) -> bool:
        """Check if a category is disabled."""
        return category in self._disabled_categories
    
    def disable_category(self, category: str):
        """Disable detection for a category."""
        self._disabled_categories.add(category)
        
        # Remove cached detector if exists
        if category in self._detectors:
            del self._detectors[category]
            
        self.logger.info(f"Disabled change detection for category: {category}")
    
    def enable_category(self, category: str):
        """Enable detection for a category."""
        self._disabled_categories.discard(category)
        self.logger.info(f"Enabled change detection for category: {category}")
    
    def get_detector_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all instantiated detectors."""
        stats = {}
        
        for category, detector in self._detectors.items():
            stats[category] = detector.get_statistics()
        
        return stats
    
    def get_healthy_detectors(self) -> List[str]:
        """Get list of categories with healthy detectors."""
        healthy = []
        
        for category, detector in self._detectors.items():
            if detector.is_healthy():
                healthy.append(category)
        
        return healthy
    
    def get_unhealthy_detectors(self) -> List[str]:
        """Get list of categories with unhealthy detectors."""
        unhealthy = []
        
        for category, detector in self._detectors.items():
            if not detector.is_healthy():
                unhealthy.append(category)
        
        return unhealthy
    
    def reset_detector_errors(self, category: Optional[str] = None):
        """Reset error counters for detectors."""
        if category:
            if category in self._detectors:
                self._detectors[category].reset_error_count()
                self.logger.info(f"Reset error count for detector: {category}")
        else:
            # Reset all detectors
            for detector in self._detectors.values():
                detector.reset_error_count()
            self.logger.info("Reset error counts for all detectors")
    
    def clear_cache(self):
        """Clear all cached detector instances."""
        self._detectors.clear()
        self.logger.info("Cleared detector instance cache")
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get overall registry status."""
        return {
            'registered_categories': self.get_registered_categories(),
            'disabled_categories': list(self._disabled_categories),
            'instantiated_detectors': list(self._detectors.keys()),
            'healthy_detectors': self.get_healthy_detectors(),
            'unhealthy_detectors': self.get_unhealthy_detectors(),
            'total_registered': len(self._detector_classes),
            'total_disabled': len(self._disabled_categories),
            'total_instantiated': len(self._detectors)
        }


# Global registry instance
_global_registry = ChangeDetectorRegistry()


def get_registry() -> ChangeDetectorRegistry:
    """Get the global detector registry instance."""
    return _global_registry


def register_detector(category: str, 
                     detector_class: Type[BaseChangeDetector],
                     default_config: Optional[ChangeDetectorConfig] = None):
    """Convenience function to register a detector globally."""
    _global_registry.register_detector(category, detector_class, default_config)