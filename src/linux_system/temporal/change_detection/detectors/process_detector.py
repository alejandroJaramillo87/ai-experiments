"""
Process Change Detector
======================

Detects and analyzes process lifecycle changes including spawn, termination,
resource usage changes, and state transitions. Provides deep insights into
system process behavior for temporal intelligence.
"""

import re
import psutil
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass

from ..base_detector import BaseChangeDetector
from ...types import SystemChange, ChangeType
from ...config import ProcessDetectorConfig


@dataclass
class ProcessInfo:
    """Comprehensive process information structure."""
    pid: int
    name: str
    cmdline: List[str]
    ppid: int
    status: str
    create_time: float
    memory_info: Dict[str, int]
    cpu_percent: float
    num_threads: int
    username: str
    cwd: Optional[str]
    exe: Optional[str]
    connections: List[Dict[str, Any]]
    open_files: List[str]
    environ: Dict[str, str]


@dataclass
class ProcessResourceSnapshot:
    """Process resource usage snapshot for trend analysis."""
    timestamp: float
    memory_rss: int
    memory_vms: int
    cpu_percent: float
    num_threads: int
    num_fds: int


class ProcessResourceTracker:
    """Tracks process resource usage trends and detects anomalies."""
    
    def __init__(self, history_window: int = 10):
        self.history_window = history_window
        self.process_history: Dict[int, List[ProcessResourceSnapshot]] = {}
    
    def add_snapshot(self, pid: int, snapshot: ProcessResourceSnapshot) -> None:
        """Add resource snapshot for process."""
        if pid not in self.process_history:
            self.process_history[pid] = []
        
        self.process_history[pid].append(snapshot)
        
        # Maintain window size
        if len(self.process_history[pid]) > self.history_window:
            self.process_history[pid] = self.process_history[pid][-self.history_window:]
    
    def get_resource_trends(self, pid: int) -> Dict[str, Any]:
        """Analyze resource usage trends for process."""
        if pid not in self.process_history or len(self.process_history[pid]) < 2:
            return {}
        
        snapshots = self.process_history[pid]
        trends = {}
        
        # Memory trend analysis
        memory_values = [s.memory_rss for s in snapshots]
        memory_trend = self._calculate_trend(memory_values)
        trends['memory_trend'] = {
            'direction': memory_trend,
            'current_mb': memory_values[-1] / 1024 / 1024,
            'change_rate': self._calculate_rate_of_change(memory_values),
            'volatility': self._calculate_volatility(memory_values)
        }
        
        # CPU trend analysis  
        cpu_values = [s.cpu_percent for s in snapshots]
        cpu_trend = self._calculate_trend(cpu_values)
        trends['cpu_trend'] = {
            'direction': cpu_trend,
            'current_percent': cpu_values[-1],
            'average': sum(cpu_values) / len(cpu_values),
            'peak': max(cpu_values)
        }
        
        # Thread count analysis
        thread_values = [s.num_threads for s in snapshots]
        if len(set(thread_values)) > 1:  # Only if there's variation
            trends['thread_trend'] = {
                'direction': self._calculate_trend(thread_values),
                'current': thread_values[-1],
                'max_observed': max(thread_values)
            }
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear regression slope
        n = len(values)
        x = list(range(n))
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_rate_of_change(self, values: List[float]) -> float:
        """Calculate rate of change per time unit."""
        if len(values) < 2:
            return 0.0
        
        return (values[-1] - values[0]) / len(values)
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility (standard deviation) of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5


class ProcessPatternAnalyzer:
    """Analyzes process patterns and behaviors."""
    
    def __init__(self):
        self.ml_frameworks = {
            'torch', 'pytorch', 'tensorflow', 'keras', 'transformers',
            'huggingface', 'accelerate', 'deepspeed', 'lightning'
        }
        
        self.development_tools = {
            'code', 'vim', 'nvim', 'emacs', 'pycharm', 'jupyter',
            'vscode', 'sublime', 'atom'
        }
        
        self.system_services = {
            'systemd', 'dbus', 'networkd', 'resolved', 'udev',
            'pulseaudio', 'pipewire', 'docker', 'containerd'
        }
    
    def categorize_process(self, process_info: ProcessInfo) -> Dict[str, Any]:
        """Categorize process and extract behavioral insights."""
        name_lower = process_info.name.lower()
        cmdline_str = ' '.join(process_info.cmdline).lower()
        
        categories = []
        insights = {}
        
        # ML/AI Framework Detection
        if any(fw in name_lower or fw in cmdline_str for fw in self.ml_frameworks):
            categories.append('ml_framework')
            insights['ml_context'] = self._analyze_ml_context(process_info)
        
        # Development Environment Detection
        if any(tool in name_lower or tool in cmdline_str for tool in self.development_tools):
            categories.append('development')
            insights['dev_context'] = self._analyze_dev_context(process_info)
        
        # System Service Detection
        if any(service in name_lower for service in self.system_services):
            categories.append('system_service')
            insights['service_context'] = self._analyze_service_context(process_info)
        
        # GPU Process Detection
        if self._is_gpu_process(process_info):
            categories.append('gpu_accelerated')
            insights['gpu_context'] = self._analyze_gpu_context(process_info)
        
        # High Resource Consumer Detection
        if self._is_resource_intensive(process_info):
            categories.append('resource_intensive')
            insights['resource_context'] = self._analyze_resource_context(process_info)
        
        return {
            'categories': categories,
            'insights': insights,
            'risk_level': self._assess_risk_level(categories, process_info),
            'monitoring_priority': self._calculate_monitoring_priority(categories, insights)
        }
    
    def _analyze_ml_context(self, process_info: ProcessInfo) -> Dict[str, Any]:
        """Analyze ML framework process context."""
        context = {'framework_type': 'unknown'}
        
        cmdline_str = ' '.join(process_info.cmdline).lower()
        
        if 'torch' in cmdline_str or 'pytorch' in cmdline_str:
            context['framework_type'] = 'pytorch'
        elif 'tensorflow' in cmdline_str or 'tf.' in cmdline_str:
            context['framework_type'] = 'tensorflow'
        elif 'transformers' in cmdline_str:
            context['framework_type'] = 'huggingface'
        
        # Detect training vs inference
        if any(keyword in cmdline_str for keyword in ['train', 'fit', 'epoch']):
            context['task_type'] = 'training'
        elif any(keyword in cmdline_str for keyword in ['inference', 'predict', 'eval']):
            context['task_type'] = 'inference'
        
        # Model size estimation based on memory usage
        memory_gb = process_info.memory_info.get('rss', 0) / 1024 / 1024 / 1024
        if memory_gb > 20:
            context['model_scale'] = 'large'
        elif memory_gb > 5:
            context['model_scale'] = 'medium'
        else:
            context['model_scale'] = 'small'
        
        return context
    
    def _is_gpu_process(self, process_info: ProcessInfo) -> bool:
        """Check if process is likely using GPU resources."""
        # Check environment variables for CUDA
        cuda_vars = ['CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES']
        has_cuda_env = any(var in process_info.environ for var in cuda_vars)
        
        # Check command line for GPU-related terms
        cmdline_str = ' '.join(process_info.cmdline).lower()
        gpu_terms = ['cuda', 'gpu', 'nvidia', 'torch.cuda', 'tensorflow-gpu']
        has_gpu_terms = any(term in cmdline_str for term in gpu_terms)
        
        return has_cuda_env or has_gpu_terms
    
    def _is_resource_intensive(self, process_info: ProcessInfo) -> bool:
        """Check if process is resource intensive."""
        memory_gb = process_info.memory_info.get('rss', 0) / 1024 / 1024 / 1024
        cpu_high = process_info.cpu_percent > 50
        memory_high = memory_gb > 2.0
        thread_intensive = process_info.num_threads > 10
        
        return cpu_high or memory_high or thread_intensive
    
    def _assess_risk_level(self, categories: List[str], process_info: ProcessInfo) -> str:
        """Assess process risk level."""
        risk_score = 0
        
        # Category-based risk
        if 'system_service' in categories:
            risk_score += 3
        if 'resource_intensive' in categories:
            risk_score += 2
        if 'gpu_accelerated' in categories:
            risk_score += 1
        
        # Resource-based risk
        memory_gb = process_info.memory_info.get('rss', 0) / 1024 / 1024 / 1024
        if memory_gb > 10:
            risk_score += 2
        elif memory_gb > 5:
            risk_score += 1
        
        if process_info.cpu_percent > 80:
            risk_score += 2
        elif process_info.cpu_percent > 50:
            risk_score += 1
        
        if risk_score >= 5:
            return 'high'
        elif risk_score >= 3:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_monitoring_priority(self, categories: List[str], insights: Dict[str, Any]) -> int:
        """Calculate monitoring priority (1-10)."""
        priority = 1
        
        if 'ml_framework' in categories:
            priority += 3
        if 'gpu_accelerated' in categories:
            priority += 2
        if 'system_service' in categories:
            priority += 2
        if 'resource_intensive' in categories:
            priority += 2
        
        return min(10, priority)


class ProcessChangeDetector(BaseChangeDetector):
    """Detects and analyzes process lifecycle and resource changes."""
    
    def __init__(self, config: ProcessDetectorConfig = None):
        super().__init__()
        self.config = config or ProcessDetectorConfig()
        self.resource_tracker = ProcessResourceTracker()
        self.pattern_analyzer = ProcessPatternAnalyzer()
        self.previous_processes: Dict[int, ProcessInfo] = {}
        self.process_birth_times: Dict[int, datetime] = {}
        self.monitored_pids: Set[int] = set()
    
    def detect_changes(self, old_snapshot: Dict[str, Any], 
                      new_snapshot: Dict[str, Any]) -> List[SystemChange]:
        """Detect process-related changes between snapshots."""
        changes = []
        
        try:
            old_processes = self._extract_process_data(old_snapshot)
            new_processes = self._extract_process_data(new_snapshot)
            
            # Detect new processes
            changes.extend(self._detect_spawned_processes(old_processes, new_processes))
            
            # Detect terminated processes
            changes.extend(self._detect_terminated_processes(old_processes, new_processes))
            
            # Detect resource changes
            changes.extend(self._detect_resource_changes(old_processes, new_processes))
            
            # Detect state changes
            changes.extend(self._detect_state_changes(old_processes, new_processes))
            
            # Update tracking data
            self._update_tracking_data(new_processes)
            
        except Exception as e:
            self._log_error(f"Error detecting process changes: {e}")
        
        return changes
    
    def _extract_process_data(self, snapshot: Dict[str, Any]) -> Dict[int, ProcessInfo]:
        """Extract and structure process data from snapshot."""
        processes = {}
        
        if 'processes' not in snapshot:
            return processes
        
        for proc_data in snapshot['processes']:
            try:
                pid = proc_data.get('pid')
                if not pid:
                    continue
                
                # Build comprehensive process info
                process_info = ProcessInfo(
                    pid=pid,
                    name=proc_data.get('name', ''),
                    cmdline=proc_data.get('cmdline', []),
                    ppid=proc_data.get('ppid', 0),
                    status=proc_data.get('status', 'unknown'),
                    create_time=proc_data.get('create_time', 0),
                    memory_info=proc_data.get('memory_info', {}),
                    cpu_percent=proc_data.get('cpu_percent', 0.0),
                    num_threads=proc_data.get('num_threads', 1),
                    username=proc_data.get('username', ''),
                    cwd=proc_data.get('cwd'),
                    exe=proc_data.get('exe'),
                    connections=proc_data.get('connections', []),
                    open_files=proc_data.get('open_files', []),
                    environ=proc_data.get('environ', {})
                )
                
                processes[pid] = process_info
                
            except Exception as e:
                self._log_error(f"Error processing process {pid}: {e}")
                continue
        
        return processes
    
    def _detect_spawned_processes(self, old_processes: Dict[int, ProcessInfo], 
                                 new_processes: Dict[int, ProcessInfo]) -> List[SystemChange]:
        """Detect newly spawned processes."""
        changes = []
        
        for pid, process_info in new_processes.items():
            if pid not in old_processes:
                # New process detected
                self.process_birth_times[pid] = datetime.now()
                
                # Analyze process characteristics
                analysis = self.pattern_analyzer.categorize_process(process_info)
                
                significance = self._calculate_spawn_significance(process_info, analysis)
                
                change = SystemChange(
                    category="processes",
                    change_type=ChangeType.ADDED,
                    entity_id=f"process:{pid}",
                    old_value=None,
                    new_value={
                        'pid': pid,
                        'name': process_info.name,
                        'cmdline': process_info.cmdline,
                        'ppid': process_info.ppid,
                        'create_time': process_info.create_time,
                        'memory_mb': process_info.memory_info.get('rss', 0) / 1024 / 1024,
                        'categories': analysis['categories']
                    },
                    significance=significance,
                    metadata={
                        'spawn_type': 'new_process',
                        'parent_pid': process_info.ppid,
                        'analysis': analysis,
                        'resource_footprint': self._calculate_resource_footprint(process_info),
                        'startup_context': self._analyze_startup_context(process_info)
                    }
                )
                changes.append(change)
                
                # Track high-priority processes
                if analysis['monitoring_priority'] >= 7:
                    self.monitored_pids.add(pid)
        
        return changes
    
    def _detect_terminated_processes(self, old_processes: Dict[int, ProcessInfo], 
                                   new_processes: Dict[int, ProcessInfo]) -> List[SystemChange]:
        """Detect terminated processes."""
        changes = []
        
        for pid, process_info in old_processes.items():
            if pid not in new_processes:
                # Process terminated
                significance = self._calculate_termination_significance(pid, process_info)
                
                # Calculate process lifetime
                birth_time = self.process_birth_times.get(pid)
                lifetime = None
                if birth_time:
                    lifetime = (datetime.now() - birth_time).total_seconds()
                
                change = SystemChange(
                    category="processes",
                    change_type=ChangeType.REMOVED,
                    entity_id=f"process:{pid}",
                    old_value={
                        'pid': pid,
                        'name': process_info.name,
                        'cmdline': process_info.cmdline
                    },
                    new_value=None,
                    significance=significance,
                    metadata={
                        'termination_type': 'process_exit',
                        'lifetime_seconds': lifetime,
                        'final_memory_mb': process_info.memory_info.get('rss', 0) / 1024 / 1024,
                        'was_monitored': pid in self.monitored_pids
                    }
                )
                changes.append(change)
                
                # Cleanup tracking data
                self.monitored_pids.discard(pid)
                if pid in self.process_birth_times:
                    del self.process_birth_times[pid]
                if pid in self.resource_tracker.process_history:
                    del self.resource_tracker.process_history[pid]
        
        return changes
    
    def _detect_resource_changes(self, old_processes: Dict[int, ProcessInfo], 
                               new_processes: Dict[int, ProcessInfo]) -> List[SystemChange]:
        """Detect significant resource usage changes."""
        changes = []
        
        for pid, new_proc in new_processes.items():
            if pid in old_processes:
                old_proc = old_processes[pid]
                
                # Update resource tracking
                snapshot = ProcessResourceSnapshot(
                    timestamp=datetime.now().timestamp(),
                    memory_rss=new_proc.memory_info.get('rss', 0),
                    memory_vms=new_proc.memory_info.get('vms', 0),
                    cpu_percent=new_proc.cpu_percent,
                    num_threads=new_proc.num_threads,
                    num_fds=len(new_proc.open_files)
                )
                self.resource_tracker.add_snapshot(pid, snapshot)
                
                # Detect significant memory changes
                memory_change = self._detect_memory_change(old_proc, new_proc)
                if memory_change:
                    changes.append(memory_change)
                
                # Detect significant CPU changes
                cpu_change = self._detect_cpu_change(old_proc, new_proc)
                if cpu_change:
                    changes.append(cpu_change)
                
                # Detect thread count changes
                thread_change = self._detect_thread_change(old_proc, new_proc)
                if thread_change:
                    changes.append(thread_change)
        
        return changes
    
    def _calculate_spawn_significance(self, process_info: ProcessInfo, 
                                    analysis: Dict[str, Any]) -> float:
        """Calculate significance of process spawn."""
        significance = 0.3  # Base significance
        
        # Category-based adjustments
        categories = analysis['categories']
        
        if 'ml_framework' in categories:
            significance += 0.4
        if 'gpu_accelerated' in categories:
            significance += 0.3
        if 'system_service' in categories:
            significance += 0.2
        if 'resource_intensive' in categories:
            significance += 0.2
        
        # Memory footprint impact
        memory_gb = process_info.memory_info.get('rss', 0) / 1024 / 1024 / 1024
        if memory_gb > 10:
            significance += 0.3
        elif memory_gb > 2:
            significance += 0.1
        
        # Parent process context
        if process_info.ppid == 1:  # Init process child
            significance += 0.1
        
        return min(1.0, significance)
    
    def _calculate_termination_significance(self, pid: int, 
                                          process_info: ProcessInfo) -> float:
        """Calculate significance of process termination."""
        significance = 0.2  # Base significance
        
        # Was it a monitored high-priority process?
        if pid in self.monitored_pids:
            significance += 0.4
        
        # Resource impact
        memory_gb = process_info.memory_info.get('rss', 0) / 1024 / 1024 / 1024
        if memory_gb > 5:
            significance += 0.2
        
        # Lifetime consideration
        birth_time = self.process_birth_times.get(pid)
        if birth_time:
            lifetime = (datetime.now() - birth_time).total_seconds()
            if lifetime < 60:  # Short-lived process
                significance += 0.2
            elif lifetime > 3600:  # Long-running process
                significance += 0.3
        
        return min(1.0, significance)
    
    def _detect_memory_change(self, old_proc: ProcessInfo, 
                            new_proc: ProcessInfo) -> Optional[SystemChange]:
        """Detect significant memory usage changes."""
        old_rss = old_proc.memory_info.get('rss', 0)
        new_rss = new_proc.memory_info.get('rss', 0)
        
        if old_rss == 0:
            return None
        
        change_ratio = abs(new_rss - old_rss) / old_rss
        
        # Only report significant changes (>20% or >1GB absolute)
        absolute_change_gb = abs(new_rss - old_rss) / 1024 / 1024 / 1024
        
        if change_ratio > 0.2 or absolute_change_gb > 1.0:
            trends = self.resource_tracker.get_resource_trends(new_proc.pid)
            
            return SystemChange(
                category="processes",
                change_type=ChangeType.MODIFIED,
                entity_id=f"process:{new_proc.pid}:memory",
                old_value=f"{old_rss / 1024 / 1024:.1f}MB",
                new_value=f"{new_rss / 1024 / 1024:.1f}MB",
                significance=min(0.9, change_ratio + 0.1),
                metadata={
                    'change_type': 'memory_usage',
                    'process_name': new_proc.name,
                    'change_ratio': change_ratio,
                    'absolute_change_gb': absolute_change_gb,
                    'trends': trends.get('memory_trend', {})
                }
            )
        
        return None
    
    def _detect_cpu_change(self, old_proc: ProcessInfo, 
                         new_proc: ProcessInfo) -> Optional[SystemChange]:
        """Detect significant CPU usage changes."""
        old_cpu = old_proc.cpu_percent
        new_cpu = new_proc.cpu_percent
        
        change = abs(new_cpu - old_cpu)
        
        # Only report significant CPU changes (>30% change)
        if change > 30:
            trends = self.resource_tracker.get_resource_trends(new_proc.pid)
            
            return SystemChange(
                category="processes",
                change_type=ChangeType.MODIFIED,
                entity_id=f"process:{new_proc.pid}:cpu",
                old_value=f"{old_cpu:.1f}%",
                new_value=f"{new_cpu:.1f}%",
                significance=min(0.8, change / 100),
                metadata={
                    'change_type': 'cpu_usage',
                    'process_name': new_proc.name,
                    'cpu_delta': new_cpu - old_cpu,
                    'trends': trends.get('cpu_trend', {})
                }
            )
        
        return None
    
    def _detect_thread_change(self, old_proc: ProcessInfo, 
                            new_proc: ProcessInfo) -> Optional[SystemChange]:
        """Detect significant thread count changes."""
        old_threads = old_proc.num_threads
        new_threads = new_proc.num_threads
        
        change = abs(new_threads - old_threads)
        
        # Only report significant thread changes (>5 threads or >50% change)
        if change > 5 or (old_threads > 0 and change / old_threads > 0.5):
            return SystemChange(
                category="processes",
                change_type=ChangeType.MODIFIED,
                entity_id=f"process:{new_proc.pid}:threads",
                old_value=old_threads,
                new_value=new_threads,
                significance=min(0.7, change / 20),
                metadata={
                    'change_type': 'thread_count',
                    'process_name': new_proc.name,
                    'thread_delta': new_threads - old_threads
                }
            )
        
        return None
    
    def _calculate_resource_footprint(self, process_info: ProcessInfo) -> Dict[str, Any]:
        """Calculate process resource footprint."""
        return {
            'memory_mb': process_info.memory_info.get('rss', 0) / 1024 / 1024,
            'cpu_percent': process_info.cpu_percent,
            'num_threads': process_info.num_threads,
            'num_open_files': len(process_info.open_files),
            'num_connections': len(process_info.connections),
            'has_children': False  # Would need to analyze process tree
        }
    
    def _analyze_startup_context(self, process_info: ProcessInfo) -> Dict[str, Any]:
        """Analyze process startup context."""
        context = {}
        
        # Working directory analysis
        if process_info.cwd:
            if 'home' in process_info.cwd:
                context['launch_location'] = 'user_home'
            elif 'tmp' in process_info.cwd:
                context['launch_location'] = 'temporary'
            else:
                context['launch_location'] = 'system'
        
        # Environment variables analysis
        interesting_env_vars = ['CUDA_VISIBLE_DEVICES', 'PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH']
        context['environment'] = {
            var: process_info.environ.get(var)
            for var in interesting_env_vars
            if var in process_info.environ
        }
        
        return context