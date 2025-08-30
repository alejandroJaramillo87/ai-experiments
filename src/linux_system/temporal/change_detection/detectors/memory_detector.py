"""
Memory Change Detector
=====================

Detects changes in system memory usage, allocation patterns, and memory pressure.
Monitors RAM, swap, memory mapping, and OOM events.
"""

import psutil
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from pathlib import Path

from ..detector_base import SystemChangeDetectorBase
from ..types import SystemChange, ChangeType, SystemSnapshot


logger = logging.getLogger(__name__)


class MemoryChangeDetector(SystemChangeDetectorBase):
    """
    Detector for memory-related system changes.
    
    Monitors:
    - Physical memory usage and pressure
    - Virtual memory and swap activity  
    - Process memory consumption changes
    - Memory allocation patterns
    - OOM (Out of Memory) events
    - Memory mapping changes
    """
    
    def __init__(self, significance_threshold: float = 0.05):
        """
        Initialize memory change detector.
        
        Args:
            significance_threshold: Minimum change to consider significant (5% default)
        """
        super().__init__()
        self.significance_threshold = significance_threshold
        
        # Memory tracking state
        self._last_memory_stats = {}
        self._last_process_memory = {}
        self._memory_pressure_history = []
        self._swap_activity_history = []
        
        # OOM detection
        self._last_oom_check = datetime.now()
        self._known_oom_events = set()
        
        # Memory threshold configurations
        self.memory_pressure_threshold = 0.85  # 85% usage considered pressure
        self.swap_activity_threshold = 1024 * 1024  # 1MB swap activity is significant
        self.process_memory_threshold = 0.1  # 10% change in process memory
        
    def detect_changes(self, old_snapshot: SystemSnapshot, new_snapshot: SystemSnapshot) -> List[SystemChange]:
        """
        Detect memory-related changes between snapshots.
        
        Args:
            old_snapshot: Previous system state
            new_snapshot: Current system state
            
        Returns:
            List of detected memory changes
        """
        changes = []
        timestamp = datetime.now()
        
        # Extract memory data from snapshots
        old_memory = self._extract_memory_data(old_snapshot)
        new_memory = self._extract_memory_data(new_snapshot)
        
        # Physical memory changes
        changes.extend(self._detect_physical_memory_changes(old_memory, new_memory, timestamp))
        
        # Virtual memory and swap changes
        changes.extend(self._detect_virtual_memory_changes(old_memory, new_memory, timestamp))
        
        # Process memory changes
        changes.extend(self._detect_process_memory_changes(old_memory, new_memory, timestamp))
        
        # Memory pressure detection
        changes.extend(self._detect_memory_pressure_changes(old_memory, new_memory, timestamp))
        
        # OOM event detection
        changes.extend(self._detect_oom_events(timestamp))
        
        # Memory mapping changes
        changes.extend(self._detect_memory_mapping_changes(old_memory, new_memory, timestamp))
        
        return changes
    
    def get_current_memory_state(self) -> Dict[str, Any]:
        """Get current memory state for diagnostics."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'physical_memory': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_percent': memory.percent,
                    'free_gb': memory.free / (1024**3),
                    'buffers_gb': getattr(memory, 'buffers', 0) / (1024**3),
                    'cached_gb': getattr(memory, 'cached', 0) / (1024**3)
                },
                'swap_memory': {
                    'total_gb': swap.total / (1024**3),
                    'used_gb': swap.used / (1024**3),
                    'used_percent': swap.percent,
                    'free_gb': swap.free / (1024**3)
                },
                'memory_pressure': memory.percent >= self.memory_pressure_threshold,
                'swap_active': swap.percent > 1.0
            }
        except Exception as e:
            logger.error(f"Error getting memory state: {e}")
            return {}
    
    def _extract_memory_data(self, snapshot: SystemSnapshot) -> Dict[str, Any]:
        """Extract memory-related data from system snapshot."""
        if not snapshot or not hasattr(snapshot, 'system_stats'):
            return {}
        
        try:
            # Get memory stats from snapshot
            memory_data = {}
            
            # Physical memory
            if 'memory' in snapshot.system_stats:
                memory_data['physical'] = snapshot.system_stats['memory']
            else:
                memory = psutil.virtual_memory()
                memory_data['physical'] = {
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used,
                    'percent': memory.percent,
                    'free': memory.free,
                    'buffers': getattr(memory, 'buffers', 0),
                    'cached': getattr(memory, 'cached', 0)
                }
            
            # Swap memory
            if 'swap' in snapshot.system_stats:
                memory_data['swap'] = snapshot.system_stats['swap']
            else:
                swap = psutil.swap_memory()
                memory_data['swap'] = {
                    'total': swap.total,
                    'used': swap.used,
                    'free': swap.free,
                    'percent': swap.percent
                }
            
            # Process memory information
            if hasattr(snapshot, 'processes') and snapshot.processes:
                memory_data['processes'] = {}
                for proc in snapshot.processes:
                    if hasattr(proc, 'memory_info') and hasattr(proc, 'pid'):
                        try:
                            mem_info = proc.memory_info()
                            memory_data['processes'][proc.pid] = {
                                'rss': mem_info.rss,  # Resident Set Size
                                'vms': mem_info.vms,  # Virtual Memory Size
                                'percent': getattr(proc, 'memory_percent', lambda: 0)(),
                                'name': getattr(proc, 'name', lambda: 'unknown')()
                            }
                        except:
                            continue
            
            return memory_data
            
        except Exception as e:
            logger.error(f"Error extracting memory data: {e}")
            return {}
    
    def _detect_physical_memory_changes(self, old_memory: Dict[str, Any], 
                                       new_memory: Dict[str, Any], timestamp: datetime) -> List[SystemChange]:
        """Detect changes in physical memory usage."""
        changes = []
        
        if not old_memory.get('physical') or not new_memory.get('physical'):
            return changes
        
        old_phys = old_memory['physical']
        new_phys = new_memory['physical']
        
        # Memory usage percentage change
        old_percent = old_phys.get('percent', 0)
        new_percent = new_phys.get('percent', 0)
        percent_change = abs(new_percent - old_percent)
        
        if percent_change >= self.significance_threshold * 100:  # Convert to percentage
            significance = min(percent_change / 50.0, 1.0)  # Scale to 0-1
            
            changes.append(SystemChange(
                category="memory",
                change_type=ChangeType.MODIFIED,
                entity_id="physical_memory_usage",
                old_value=old_percent,
                new_value=new_percent,
                significance=significance,
                metadata={
                    'change_percent': percent_change,
                    'available_gb': new_phys.get('available', 0) / (1024**3),
                    'used_gb': new_phys.get('used', 0) / (1024**3),
                    'total_gb': new_phys.get('total', 0) / (1024**3)
                },
                timestamp=timestamp
            ))
        
        # Available memory change
        old_available = old_phys.get('available', 0)
        new_available = new_phys.get('available', 0)
        available_change_gb = abs(new_available - old_available) / (1024**3)
        
        if available_change_gb >= 0.5:  # 500MB change
            significance = min(available_change_gb / 4.0, 1.0)  # Scale based on 4GB being max significance
            
            changes.append(SystemChange(
                category="memory",
                change_type=ChangeType.MODIFIED,
                entity_id="available_memory",
                old_value=old_available / (1024**3),
                new_value=new_available / (1024**3),
                significance=significance,
                metadata={
                    'change_gb': available_change_gb,
                    'direction': 'increased' if new_available > old_available else 'decreased'
                },
                timestamp=timestamp
            ))
        
        # Cache and buffer changes
        old_cached = old_phys.get('cached', 0)
        new_cached = new_phys.get('cached', 0)
        cache_change_gb = abs(new_cached - old_cached) / (1024**3)
        
        if cache_change_gb >= 0.2:  # 200MB cache change
            changes.append(SystemChange(
                category="memory",
                change_type=ChangeType.MODIFIED,
                entity_id="memory_cache",
                old_value=old_cached / (1024**3),
                new_value=new_cached / (1024**3),
                significance=min(cache_change_gb / 2.0, 1.0),
                metadata={'change_gb': cache_change_gb},
                timestamp=timestamp
            ))
        
        return changes
    
    def _detect_virtual_memory_changes(self, old_memory: Dict[str, Any], 
                                      new_memory: Dict[str, Any], timestamp: datetime) -> List[SystemChange]:
        """Detect changes in virtual memory and swap usage."""
        changes = []
        
        if not old_memory.get('swap') or not new_memory.get('swap'):
            return changes
        
        old_swap = old_memory['swap']
        new_swap = new_memory['swap']
        
        # Swap usage percentage change
        old_swap_percent = old_swap.get('percent', 0)
        new_swap_percent = new_swap.get('percent', 0)
        swap_percent_change = abs(new_swap_percent - old_swap_percent)
        
        if swap_percent_change >= 1.0:  # 1% swap change is significant
            significance = min(swap_percent_change / 20.0, 1.0)
            
            changes.append(SystemChange(
                category="memory",
                change_type=ChangeType.MODIFIED,
                entity_id="swap_usage",
                old_value=old_swap_percent,
                new_value=new_swap_percent,
                significance=significance,
                metadata={
                    'swap_used_gb': new_swap.get('used', 0) / (1024**3),
                    'swap_total_gb': new_swap.get('total', 0) / (1024**3)
                },
                timestamp=timestamp
            ))
        
        # Swap activity detection (new swap usage)
        old_swap_used = old_swap.get('used', 0)
        new_swap_used = new_swap.get('used', 0)
        swap_activity = new_swap_used - old_swap_used
        
        if abs(swap_activity) >= self.swap_activity_threshold:
            significance = min(abs(swap_activity) / (100 * 1024 * 1024), 1.0)  # Scale by 100MB
            
            activity_type = "swap_in" if swap_activity > 0 else "swap_out"
            
            changes.append(SystemChange(
                category="memory",
                change_type=ChangeType.MODIFIED,
                entity_id=activity_type,
                old_value=old_swap_used,
                new_value=new_swap_used,
                significance=significance,
                metadata={
                    'activity_mb': abs(swap_activity) / (1024 * 1024),
                    'direction': 'in' if swap_activity > 0 else 'out'
                },
                timestamp=timestamp
            ))
        
        return changes
    
    def _detect_process_memory_changes(self, old_memory: Dict[str, Any], 
                                     new_memory: Dict[str, Any], timestamp: datetime) -> List[SystemChange]:
        """Detect significant process memory changes."""
        changes = []
        
        old_processes = old_memory.get('processes', {})
        new_processes = new_memory.get('processes', {})
        
        if not old_processes or not new_processes:
            return changes
        
        # Track memory changes per process
        for pid in set(old_processes.keys()) & set(new_processes.keys()):
            old_proc = old_processes[pid]
            new_proc = new_processes[pid]
            
            old_rss = old_proc.get('rss', 0)
            new_rss = new_proc.get('rss', 0)
            
            if old_rss == 0:
                continue
            
            # Calculate percentage change in RSS
            rss_change_percent = abs(new_rss - old_rss) / old_rss
            
            if rss_change_percent >= self.process_memory_threshold:
                significance = min(rss_change_percent, 1.0)
                
                changes.append(SystemChange(
                    category="memory", 
                    change_type=ChangeType.MODIFIED,
                    entity_id=f"process_{pid}_memory",
                    old_value=old_rss / (1024 * 1024),  # MB
                    new_value=new_rss / (1024 * 1024),  # MB
                    significance=significance,
                    metadata={
                        'pid': pid,
                        'process_name': new_proc.get('name', 'unknown'),
                        'change_mb': abs(new_rss - old_rss) / (1024 * 1024),
                        'change_percent': rss_change_percent * 100
                    },
                    timestamp=timestamp
                ))
        
        # Detect new high-memory processes
        new_pids = set(new_processes.keys()) - set(old_processes.keys())
        for pid in new_pids:
            proc = new_processes[pid]
            rss = proc.get('rss', 0)
            
            # Consider processes using more than 100MB significant
            if rss > 100 * 1024 * 1024:
                significance = min(rss / (1024**3), 1.0)  # Scale by 1GB
                
                changes.append(SystemChange(
                    category="memory",
                    change_type=ChangeType.ADDED,
                    entity_id=f"process_{pid}_memory",
                    old_value=0,
                    new_value=rss / (1024 * 1024),  # MB
                    significance=significance,
                    metadata={
                        'pid': pid,
                        'process_name': proc.get('name', 'unknown'),
                        'rss_mb': rss / (1024 * 1024)
                    },
                    timestamp=timestamp
                ))
        
        return changes
    
    def _detect_memory_pressure_changes(self, old_memory: Dict[str, Any], 
                                       new_memory: Dict[str, Any], timestamp: datetime) -> List[SystemChange]:
        """Detect memory pressure state changes."""
        changes = []
        
        old_phys = old_memory.get('physical', {})
        new_phys = new_memory.get('physical', {})
        
        if not old_phys or not new_phys:
            return changes
        
        old_percent = old_phys.get('percent', 0)
        new_percent = new_phys.get('percent', 0)
        
        old_pressure = old_percent >= self.memory_pressure_threshold * 100
        new_pressure = new_percent >= self.memory_pressure_threshold * 100
        
        # Memory pressure state change
        if old_pressure != new_pressure:
            changes.append(SystemChange(
                category="memory",
                change_type=ChangeType.MODIFIED,
                entity_id="memory_pressure_state",
                old_value=old_pressure,
                new_value=new_pressure,
                significance=0.8,  # High significance for pressure changes
                metadata={
                    'old_usage_percent': old_percent,
                    'new_usage_percent': new_percent,
                    'pressure_threshold': self.memory_pressure_threshold * 100,
                    'entered_pressure': new_pressure and not old_pressure,
                    'exited_pressure': old_pressure and not new_pressure
                },
                timestamp=timestamp
            ))
        
        # Track memory pressure history
        self._memory_pressure_history.append({
            'timestamp': timestamp,
            'usage_percent': new_percent,
            'in_pressure': new_pressure
        })
        
        # Keep only last 100 entries
        if len(self._memory_pressure_history) > 100:
            self._memory_pressure_history = self._memory_pressure_history[-100:]
        
        return changes
    
    def _detect_oom_events(self, timestamp: datetime) -> List[SystemChange]:
        """Detect Out of Memory (OOM) events from kernel logs."""
        changes = []
        
        try:
            # Check kernel messages for OOM events
            dmesg_path = Path("/var/log/dmesg")
            kern_log_path = Path("/var/log/kern.log")
            
            # Try different log sources
            log_paths = [kern_log_path, dmesg_path]
            
            for log_path in log_paths:
                if log_path.exists():
                    changes.extend(self._parse_oom_from_log(log_path, timestamp))
                    break
            
        except Exception as e:
            logger.error(f"Error checking for OOM events: {e}")
        
        return changes
    
    def _parse_oom_from_log(self, log_path: Path, timestamp: datetime) -> List[SystemChange]:
        """Parse OOM events from kernel log."""
        changes = []
        
        try:
            # Only check recent entries (last 5 minutes)
            cutoff_time = timestamp - timedelta(minutes=5)
            
            with open(log_path, 'r') as f:
                lines = f.readlines()
                
            # Look for OOM killer messages in recent lines
            recent_lines = lines[-1000:]  # Check last 1000 lines
            
            for line in recent_lines:
                if "Out of memory:" in line or "oom-killer:" in line:
                    # Extract process information
                    oom_signature = hash(line) % (2**32)  # Simple signature
                    
                    if oom_signature not in self._known_oom_events:
                        self._known_oom_events.add(oom_signature)
                        
                        # Extract killed process name
                        killed_process = "unknown"
                        if "Killed process" in line:
                            parts = line.split("Killed process")
                            if len(parts) > 1:
                                # Extract process name from the line
                                process_part = parts[1].strip()
                                process_name_match = process_part.split("(")
                                if len(process_name_match) > 1:
                                    killed_process = process_name_match[1].split(")")[0]
                        
                        changes.append(SystemChange(
                            category="memory",
                            change_type=ChangeType.REMOVED,
                            entity_id=f"oom_killed_{killed_process}",
                            old_value=True,
                            new_value=False,
                            significance=1.0,  # OOM events are always critical
                            metadata={
                                'event_type': 'oom_kill',
                                'killed_process': killed_process,
                                'log_line': line.strip(),
                                'oom_signature': oom_signature
                            },
                            timestamp=timestamp
                        ))
                        
        except Exception as e:
            logger.error(f"Error parsing OOM events from {log_path}: {e}")
        
        return changes
    
    def _detect_memory_mapping_changes(self, old_memory: Dict[str, Any], 
                                      new_memory: Dict[str, Any], timestamp: datetime) -> List[SystemChange]:
        """Detect changes in memory mappings and shared memory."""
        changes = []
        
        # This would require more detailed memory mapping information
        # For now, we'll detect changes in virtual memory size of processes
        
        old_processes = old_memory.get('processes', {})
        new_processes = new_memory.get('processes', {})
        
        for pid in set(old_processes.keys()) & set(new_processes.keys()):
            old_proc = old_processes[pid]
            new_proc = new_processes[pid]
            
            old_vms = old_proc.get('vms', 0)
            new_vms = new_proc.get('vms', 0)
            
            if old_vms == 0:
                continue
            
            # Significant VMS change indicates memory mapping changes
            vms_change_percent = abs(new_vms - old_vms) / old_vms
            
            if vms_change_percent >= 0.2:  # 20% VMS change
                significance = min(vms_change_percent, 1.0)
                
                changes.append(SystemChange(
                    category="memory",
                    change_type=ChangeType.MODIFIED,
                    entity_id=f"process_{pid}_virtual_memory",
                    old_value=old_vms / (1024 * 1024),  # MB
                    new_value=new_vms / (1024 * 1024),  # MB
                    significance=significance,
                    metadata={
                        'pid': pid,
                        'process_name': new_proc.get('name', 'unknown'),
                        'vms_change_mb': abs(new_vms - old_vms) / (1024 * 1024),
                        'change_type': 'memory_mapping_change'
                    },
                    timestamp=timestamp
                ))
        
        return changes
    
    def get_memory_pressure_history(self) -> List[Dict[str, Any]]:
        """Get historical memory pressure data."""
        return self._memory_pressure_history.copy()
    
    def get_detector_stats(self) -> Dict[str, Any]:
        """Get detector statistics and current state."""
        current_state = self.get_current_memory_state()
        
        return {
            'detector_type': 'memory',
            'current_memory_state': current_state,
            'thresholds': {
                'significance_threshold': self.significance_threshold,
                'memory_pressure_threshold': self.memory_pressure_threshold,
                'swap_activity_threshold_mb': self.swap_activity_threshold / (1024 * 1024),
                'process_memory_threshold': self.process_memory_threshold
            },
            'history_stats': {
                'pressure_events_tracked': len(self._memory_pressure_history),
                'oom_events_seen': len(self._known_oom_events)
            },
            'last_oom_check': self._last_oom_check.isoformat()
        }