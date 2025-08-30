"""
Storage Change Detector
======================

Detects changes in storage systems, disk usage, I/O patterns, and filesystem activity.
Monitors disk space, I/O operations, filesystem mounts, and storage performance.
"""

import psutil
import os
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from pathlib import Path

from ..detector_base import SystemChangeDetectorBase
from ..types import SystemChange, ChangeType, SystemSnapshot


logger = logging.getLogger(__name__)


class StorageChangeDetector(SystemChangeDetectorBase):
    """
    Detector for storage-related system changes.
    
    Monitors:
    - Disk space usage and availability
    - Filesystem mount/unmount events
    - I/O operations and patterns
    - Storage device health and performance
    - File system changes and metadata
    - Storage bandwidth and latency
    """
    
    def __init__(self, disk_usage_threshold: float = 0.05, io_threshold: float = 0.1):
        """
        Initialize storage change detector.
        
        Args:
            disk_usage_threshold: Minimum disk usage change to consider significant (5% default)
            io_threshold: Minimum I/O change to consider significant (10% default)
        """
        super().__init__()
        self.disk_usage_threshold = disk_usage_threshold
        self.io_threshold = io_threshold
        
        # Storage tracking state
        self._last_disk_stats = {}
        self._last_io_stats = {}
        self._last_filesystem_list = set()
        self._io_history = []
        
        # Important paths to monitor
        self.monitored_paths = {
            '/': 'root_filesystem',
            '/home': 'home_filesystem',
            '/tmp': 'temp_filesystem',
            '/var': 'var_filesystem',
            '/usr': 'usr_filesystem'
        }
        
        # I/O thresholds
        self.high_io_threshold = 1024 * 1024 * 100  # 100MB/s
        self.disk_full_threshold = 0.95  # 95% usage
        self.disk_warning_threshold = 0.85  # 85% usage
        
    def detect_changes(self, old_snapshot: SystemSnapshot, new_snapshot: SystemSnapshot) -> List[SystemChange]:
        """
        Detect storage-related changes between snapshots.
        
        Args:
            old_snapshot: Previous system state
            new_snapshot: Current system state
            
        Returns:
            List of detected storage changes
        """
        changes = []
        timestamp = datetime.now()
        
        # Extract storage data from snapshots
        old_storage = self._extract_storage_data(old_snapshot)
        new_storage = self._extract_storage_data(new_snapshot)
        
        # Disk usage changes
        changes.extend(self._detect_disk_usage_changes(old_storage, new_storage, timestamp))
        
        # Filesystem mount/unmount changes
        changes.extend(self._detect_filesystem_changes(old_storage, new_storage, timestamp))
        
        # I/O activity changes
        changes.extend(self._detect_io_changes(old_storage, new_storage, timestamp))
        
        # Storage device changes
        changes.extend(self._detect_storage_device_changes(old_storage, new_storage, timestamp))
        
        # Storage health and performance
        changes.extend(self._detect_storage_health_changes(old_storage, new_storage, timestamp))
        
        return changes
    
    def get_current_storage_state(self) -> Dict[str, Any]:
        """Get current storage state for diagnostics."""
        try:
            storage_state = {
                'disk_usage': {},
                'filesystems': [],
                'io_counters': {},
                'storage_devices': []
            }
            
            # Disk usage for all mount points
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    storage_state['disk_usage'][partition.mountpoint] = {
                        'total_gb': usage.total / (1024**3),
                        'used_gb': usage.used / (1024**3),
                        'free_gb': usage.free / (1024**3),
                        'percent': (usage.used / usage.total) * 100 if usage.total > 0 else 0,
                        'filesystem': partition.fstype,
                        'device': partition.device
                    }
                except (PermissionError, OSError):
                    continue
            
            # Filesystem information
            for partition in psutil.disk_partitions():
                storage_state['filesystems'].append({
                    'device': partition.device,
                    'mountpoint': partition.mountpoint,
                    'fstype': partition.fstype,
                    'opts': partition.opts
                })
            
            # I/O counters
            io_counters = psutil.disk_io_counters(perdisk=True)
            if io_counters:
                for device, counters in io_counters.items():
                    storage_state['io_counters'][device] = {
                        'read_count': counters.read_count,
                        'write_count': counters.write_count,
                        'read_bytes': counters.read_bytes,
                        'write_bytes': counters.write_bytes,
                        'read_time': counters.read_time,
                        'write_time': counters.write_time
                    }
            
            return storage_state
            
        except Exception as e:
            logger.error(f"Error getting storage state: {e}")
            return {}
    
    def _extract_storage_data(self, snapshot: SystemSnapshot) -> Dict[str, Any]:
        """Extract storage-related data from system snapshot."""
        if not snapshot or not hasattr(snapshot, 'system_stats'):
            return {}
        
        try:
            storage_data = {}
            
            # Disk usage information
            if 'disk_usage' in snapshot.system_stats:
                storage_data['disk_usage'] = snapshot.system_stats['disk_usage']
            else:
                storage_data['disk_usage'] = {}
                for partition in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        storage_data['disk_usage'][partition.mountpoint] = {
                            'total': usage.total,
                            'used': usage.used,
                            'free': usage.free,
                            'percent': (usage.used / usage.total) * 100 if usage.total > 0 else 0
                        }
                    except (PermissionError, OSError):
                        continue
            
            # Filesystem information
            if 'filesystems' in snapshot.system_stats:
                storage_data['filesystems'] = snapshot.system_stats['filesystems']
            else:
                storage_data['filesystems'] = []
                for partition in psutil.disk_partitions():
                    storage_data['filesystems'].append({
                        'device': partition.device,
                        'mountpoint': partition.mountpoint,
                        'fstype': partition.fstype,
                        'opts': partition.opts
                    })
            
            # I/O counters
            if 'disk_io' in snapshot.system_stats:
                storage_data['io_counters'] = snapshot.system_stats['disk_io']
            else:
                io_counters = psutil.disk_io_counters(perdisk=True)
                storage_data['io_counters'] = {}
                if io_counters:
                    for device, counters in io_counters.items():
                        storage_data['io_counters'][device] = {
                            'read_count': counters.read_count,
                            'write_count': counters.write_count,
                            'read_bytes': counters.read_bytes,
                            'write_bytes': counters.write_bytes,
                            'read_time': counters.read_time,
                            'write_time': counters.write_time
                        }
            
            return storage_data
            
        except Exception as e:
            logger.error(f"Error extracting storage data: {e}")
            return {}
    
    def _detect_disk_usage_changes(self, old_storage: Dict[str, Any], 
                                  new_storage: Dict[str, Any], timestamp: datetime) -> List[SystemChange]:
        """Detect changes in disk usage across filesystems."""
        changes = []
        
        old_disk_usage = old_storage.get('disk_usage', {})
        new_disk_usage = new_storage.get('disk_usage', {})
        
        if not old_disk_usage or not new_disk_usage:
            return changes
        
        # Check each filesystem
        for mountpoint in set(old_disk_usage.keys()) & set(new_disk_usage.keys()):
            old_usage = old_disk_usage[mountpoint]
            new_usage = new_disk_usage[mountpoint]
            
            old_percent = old_usage.get('percent', 0)
            new_percent = new_usage.get('percent', 0)
            
            # Calculate change in usage percentage
            percent_change = abs(new_percent - old_percent)
            
            if percent_change >= self.disk_usage_threshold * 100:  # Convert to percentage
                significance = min(percent_change / 50.0, 1.0)  # Scale to 0-1
                
                # Special handling for critical disk space
                if new_percent >= self.disk_full_threshold * 100:
                    significance = max(significance, 0.95)
                elif new_percent >= self.disk_warning_threshold * 100:
                    significance = max(significance, 0.7)
                
                changes.append(SystemChange(
                    category="storage",
                    change_type=ChangeType.MODIFIED,
                    entity_id=f"disk_usage_{mountpoint.replace('/', '_root' if mountpoint == '/' else '_')}",
                    old_value=old_percent,
                    new_value=new_percent,
                    significance=significance,
                    metadata={
                        'mountpoint': mountpoint,
                        'change_percent': percent_change,
                        'direction': 'increased' if new_percent > old_percent else 'decreased',
                        'free_gb': new_usage.get('free', 0) / (1024**3),
                        'used_gb': new_usage.get('used', 0) / (1024**3),
                        'total_gb': new_usage.get('total', 0) / (1024**3),
                        'warning_level': 'critical' if new_percent >= self.disk_full_threshold * 100
                                       else 'warning' if new_percent >= self.disk_warning_threshold * 100
                                       else 'normal'
                    },
                    timestamp=timestamp
                ))
            
            # Detect significant absolute space changes (useful for large disks)
            old_free_gb = old_usage.get('free', 0) / (1024**3)
            new_free_gb = new_usage.get('free', 0) / (1024**3)
            free_change_gb = abs(new_free_gb - old_free_gb)
            
            if free_change_gb >= 10:  # 10GB change
                significance = min(free_change_gb / 100.0, 1.0)  # Scale by 100GB
                
                changes.append(SystemChange(
                    category="storage",
                    change_type=ChangeType.MODIFIED,
                    entity_id=f"disk_space_{mountpoint.replace('/', '_root' if mountpoint == '/' else '_')}",
                    old_value=old_free_gb,
                    new_value=new_free_gb,
                    significance=significance,
                    metadata={
                        'mountpoint': mountpoint,
                        'change_gb': free_change_gb,
                        'change_type': 'space_freed' if new_free_gb > old_free_gb else 'space_used'
                    },
                    timestamp=timestamp
                ))
        
        return changes
    
    def _detect_filesystem_changes(self, old_storage: Dict[str, Any], 
                                  new_storage: Dict[str, Any], timestamp: datetime) -> List[SystemChange]:
        """Detect filesystem mount/unmount events."""
        changes = []
        
        old_filesystems = {fs['mountpoint']: fs for fs in old_storage.get('filesystems', [])}
        new_filesystems = {fs['mountpoint']: fs for fs in new_storage.get('filesystems', [])}
        
        # Detect new mounts
        new_mounts = set(new_filesystems.keys()) - set(old_filesystems.keys())
        for mountpoint in new_mounts:
            fs_info = new_filesystems[mountpoint]
            
            changes.append(SystemChange(
                category="storage",
                change_type=ChangeType.ADDED,
                entity_id=f"filesystem_{mountpoint.replace('/', '_root' if mountpoint == '/' else '_')}",
                old_value=None,
                new_value=mountpoint,
                significance=0.8,  # Filesystem changes are significant
                metadata={
                    'event_type': 'mount',
                    'mountpoint': mountpoint,
                    'device': fs_info.get('device', 'unknown'),
                    'fstype': fs_info.get('fstype', 'unknown'),
                    'options': fs_info.get('opts', '')
                },
                timestamp=timestamp
            ))
        
        # Detect unmounts
        removed_mounts = set(old_filesystems.keys()) - set(new_filesystems.keys())
        for mountpoint in removed_mounts:
            fs_info = old_filesystems[mountpoint]
            
            changes.append(SystemChange(
                category="storage",
                change_type=ChangeType.REMOVED,
                entity_id=f"filesystem_{mountpoint.replace('/', '_root' if mountpoint == '/' else '_')}",
                old_value=mountpoint,
                new_value=None,
                significance=0.9,  # Unmounts are very significant
                metadata={
                    'event_type': 'unmount',
                    'mountpoint': mountpoint,
                    'device': fs_info.get('device', 'unknown'),
                    'fstype': fs_info.get('fstype', 'unknown')
                },
                timestamp=timestamp
            ))
        
        return changes
    
    def _detect_io_changes(self, old_storage: Dict[str, Any], 
                          new_storage: Dict[str, Any], timestamp: datetime) -> List[SystemChange]:
        """Detect changes in I/O activity patterns."""
        changes = []
        
        old_io = old_storage.get('io_counters', {})
        new_io = new_storage.get('io_counters', {})
        
        if not old_io or not new_io:
            return changes
        
        # Analyze I/O changes per device
        for device in set(old_io.keys()) & set(new_io.keys()):
            old_counters = old_io[device]
            new_counters = new_io[device]
            
            # Calculate I/O rates (assumes 1-minute interval between snapshots)
            time_delta = 60  # seconds - this should be calculated from actual timestamps
            
            # Read I/O changes
            read_bytes_delta = new_counters.get('read_bytes', 0) - old_counters.get('read_bytes', 0)
            write_bytes_delta = new_counters.get('write_bytes', 0) - old_counters.get('write_bytes', 0)
            
            read_rate = read_bytes_delta / time_delta if time_delta > 0 else 0
            write_rate = write_bytes_delta / time_delta if time_delta > 0 else 0
            
            # Detect high I/O activity
            if read_rate >= self.high_io_threshold or write_rate >= self.high_io_threshold:
                total_rate = read_rate + write_rate
                significance = min(total_rate / (self.high_io_threshold * 10), 1.0)
                
                changes.append(SystemChange(
                    category="storage",
                    change_type=ChangeType.MODIFIED,
                    entity_id=f"io_activity_{device}",
                    old_value=0,  # Previous rate not easily calculable
                    new_value=total_rate / (1024 * 1024),  # MB/s
                    significance=significance,
                    metadata={
                        'device': device,
                        'activity_type': 'high_io',
                        'read_rate_mbs': read_rate / (1024 * 1024),
                        'write_rate_mbs': write_rate / (1024 * 1024),
                        'total_rate_mbs': total_rate / (1024 * 1024),
                        'read_bytes_delta': read_bytes_delta,
                        'write_bytes_delta': write_bytes_delta
                    },
                    timestamp=timestamp
                ))
            
            # Detect I/O operation count changes
            read_ops_delta = new_counters.get('read_count', 0) - old_counters.get('read_count', 0)
            write_ops_delta = new_counters.get('write_count', 0) - old_counters.get('write_count', 0)
            
            total_ops_rate = (read_ops_delta + write_ops_delta) / time_delta
            
            # High IOPS detection
            if total_ops_rate >= 1000:  # 1000 IOPS threshold
                significance = min(total_ops_rate / 10000, 1.0)  # Scale by 10K IOPS
                
                changes.append(SystemChange(
                    category="storage",
                    change_type=ChangeType.MODIFIED,
                    entity_id=f"iops_{device}",
                    old_value=0,
                    new_value=total_ops_rate,
                    significance=significance,
                    metadata={
                        'device': device,
                        'activity_type': 'high_iops',
                        'read_ops_rate': read_ops_delta / time_delta,
                        'write_ops_rate': write_ops_delta / time_delta,
                        'total_ops_rate': total_ops_rate
                    },
                    timestamp=timestamp
                ))
            
            # Track I/O history
            self._io_history.append({
                'timestamp': timestamp,
                'device': device,
                'read_rate_mbs': read_rate / (1024 * 1024),
                'write_rate_mbs': write_rate / (1024 * 1024),
                'ops_rate': total_ops_rate
            })
        
        # Keep only last 100 I/O history entries
        if len(self._io_history) > 100:
            self._io_history = self._io_history[-100:]
        
        return changes
    
    def _detect_storage_device_changes(self, old_storage: Dict[str, Any], 
                                      new_storage: Dict[str, Any], timestamp: datetime) -> List[SystemChange]:
        """Detect storage device addition/removal."""
        changes = []
        
        old_devices = set(old_storage.get('io_counters', {}).keys())
        new_devices = set(new_storage.get('io_counters', {}).keys())
        
        # New storage devices
        added_devices = new_devices - old_devices
        for device in added_devices:
            changes.append(SystemChange(
                category="storage",
                change_type=ChangeType.ADDED,
                entity_id=f"storage_device_{device}",
                old_value=None,
                new_value=device,
                significance=0.9,  # Device changes are highly significant
                metadata={
                    'event_type': 'device_added',
                    'device': device,
                    'device_type': 'storage_device'
                },
                timestamp=timestamp
            ))
        
        # Removed storage devices
        removed_devices = old_devices - new_devices
        for device in removed_devices:
            changes.append(SystemChange(
                category="storage",
                change_type=ChangeType.REMOVED,
                entity_id=f"storage_device_{device}",
                old_value=device,
                new_value=None,
                significance=1.0,  # Device removal is critical
                metadata={
                    'event_type': 'device_removed',
                    'device': device,
                    'device_type': 'storage_device'
                },
                timestamp=timestamp
            ))
        
        return changes
    
    def _detect_storage_health_changes(self, old_storage: Dict[str, Any], 
                                      new_storage: Dict[str, Any], timestamp: datetime) -> List[SystemChange]:
        """Detect storage health and performance changes."""
        changes = []
        
        # This would require SMART data or other health indicators
        # For now, we'll monitor I/O timing changes as a proxy for health
        
        old_io = old_storage.get('io_counters', {})
        new_io = new_storage.get('io_counters', {})
        
        for device in set(old_io.keys()) & set(new_io.keys()):
            old_counters = old_io[device]
            new_counters = new_io[device]
            
            # Calculate average I/O time changes
            old_read_time = old_counters.get('read_time', 0)
            new_read_time = new_counters.get('read_time', 0)
            old_write_time = old_counters.get('write_time', 0)
            new_write_time = new_counters.get('write_time', 0)
            
            old_read_count = old_counters.get('read_count', 1)
            new_read_count = new_counters.get('read_count', 1)
            old_write_count = old_counters.get('write_count', 1)
            new_write_count = new_counters.get('write_count', 1)
            
            # Calculate average times
            old_avg_read_time = old_read_time / old_read_count if old_read_count > 0 else 0
            new_avg_read_time = new_read_time / new_read_count if new_read_count > 0 else 0
            old_avg_write_time = old_write_time / old_write_count if old_write_count > 0 else 0
            new_avg_write_time = new_write_time / new_write_count if new_write_count > 0 else 0
            
            # Detect significant latency changes
            read_time_change = abs(new_avg_read_time - old_avg_read_time) if old_avg_read_time > 0 else 0
            write_time_change = abs(new_avg_write_time - old_avg_write_time) if old_avg_write_time > 0 else 0
            
            if read_time_change > 50 or write_time_change > 50:  # 50ms latency change
                total_change = read_time_change + write_time_change
                significance = min(total_change / 1000, 1.0)  # Scale by 1 second
                
                changes.append(SystemChange(
                    category="storage",
                    change_type=ChangeType.MODIFIED,
                    entity_id=f"storage_latency_{device}",
                    old_value=(old_avg_read_time + old_avg_write_time) / 2,
                    new_value=(new_avg_read_time + new_avg_write_time) / 2,
                    significance=significance,
                    metadata={
                        'device': device,
                        'performance_change': 'degraded' if (new_avg_read_time + new_avg_write_time) > 
                                                          (old_avg_read_time + old_avg_write_time) else 'improved',
                        'read_latency_ms': new_avg_read_time,
                        'write_latency_ms': new_avg_write_time,
                        'read_latency_change_ms': read_time_change,
                        'write_latency_change_ms': write_time_change
                    },
                    timestamp=timestamp
                ))
        
        return changes
    
    def get_io_history(self) -> List[Dict[str, Any]]:
        """Get historical I/O activity data."""
        return self._io_history.copy()
    
    def get_disk_health_summary(self) -> Dict[str, Any]:
        """Get disk health summary."""
        try:
            health_summary = {}
            current_state = self.get_current_storage_state()
            
            for mountpoint, usage_info in current_state.get('disk_usage', {}).items():
                percent_used = usage_info['percent']
                
                if percent_used >= self.disk_full_threshold * 100:
                    health_level = 'critical'
                elif percent_used >= self.disk_warning_threshold * 100:
                    health_level = 'warning'
                else:
                    health_level = 'good'
                
                health_summary[mountpoint] = {
                    'health_level': health_level,
                    'usage_percent': percent_used,
                    'free_gb': usage_info['free_gb'],
                    'recommendation': self._get_disk_recommendation(percent_used)
                }
            
            return health_summary
            
        except Exception as e:
            logger.error(f"Error getting disk health summary: {e}")
            return {}
    
    def _get_disk_recommendation(self, usage_percent: float) -> str:
        """Get recommendation based on disk usage."""
        if usage_percent >= 95:
            return "URGENT: Disk almost full - immediate cleanup required"
        elif usage_percent >= 90:
            return "WARNING: Disk usage high - consider cleanup"
        elif usage_percent >= 80:
            return "INFO: Monitor disk usage - may need cleanup soon"
        else:
            return "OK: Disk usage normal"
    
    def get_detector_stats(self) -> Dict[str, Any]:
        """Get detector statistics and current state."""
        current_state = self.get_current_storage_state()
        health_summary = self.get_disk_health_summary()
        
        return {
            'detector_type': 'storage',
            'current_storage_state': current_state,
            'disk_health_summary': health_summary,
            'thresholds': {
                'disk_usage_threshold': self.disk_usage_threshold,
                'io_threshold': self.io_threshold,
                'high_io_threshold_mbs': self.high_io_threshold / (1024 * 1024),
                'disk_full_threshold': self.disk_full_threshold,
                'disk_warning_threshold': self.disk_warning_threshold
            },
            'monitoring': {
                'monitored_paths': self.monitored_paths,
                'io_history_entries': len(self._io_history)
            }
        }