"""
Process Change Detector Tests
============================

Tests for ProcessChangeDetector focusing on process lifecycle detection,
resource monitoring, and behavioral analysis.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.linux_system.temporal.change_detection.detectors.process_detector import (
    ProcessChangeDetector, ProcessInfo, ProcessPatternAnalyzer, ProcessResourceTracker
)
from src.linux_system.temporal.types import ChangeType
from src.linux_system.temporal.config import ProcessDetectorConfig
from tests.temporal.fixtures import TemporalAssertions


class TestProcessChangeDetector:
    """Test ProcessChangeDetector functionality."""
    
    @pytest.fixture
    def detector(self):
        """Create ProcessChangeDetector instance."""
        config = ProcessDetectorConfig()
        return ProcessChangeDetector(config)
    
    @pytest.fixture
    def sample_process_info(self):
        """Create sample process information."""
        return ProcessInfo(
            pid=12345,
            name="python3",
            cmdline=["python3", "-m", "torch.distributed.launch", "train.py"],
            ppid=1234,
            status="running",
            create_time=datetime.now().timestamp() - 300,  # 5 minutes ago
            memory_info={'rss': 2048 * 1024 * 1024, 'vms': 4096 * 1024 * 1024},
            cpu_percent=45.5,
            num_threads=8,
            username="alejandro",
            cwd="/home/alejandro/ml-projects/training",
            exe="/usr/bin/python3",
            connections=[],
            open_files=["/dev/nvidia0", "/tmp/training.log"],
            environ={
                'CUDA_VISIBLE_DEVICES': '0',
                'PYTHONPATH': '/home/alejandro/ml-projects',
                'PATH': '/usr/local/cuda/bin:/usr/bin'
            }
        )
    
    def test_process_spawn_detection(self, detector):
        """Test detection of new process spawning."""
        old_snapshot = {'processes': []}
        
        new_snapshot = {
            'processes': [{
                'pid': 12345,
                'name': 'python3',
                'cmdline': ['python3', 'script.py'],
                'ppid': 1234,
                'status': 'running',
                'create_time': datetime.now().timestamp(),
                'memory_info': {'rss': 100 * 1024 * 1024},  # 100MB
                'cpu_percent': 15.0,
                'num_threads': 4,
                'username': 'alejandro',
                'cwd': '/home/alejandro',
                'exe': '/usr/bin/python3',
                'connections': [],
                'open_files': [],
                'environ': {}
            }]
        }
        
        changes = detector.detect_changes(old_snapshot, new_snapshot)
        
        # Should detect process spawn
        TemporalAssertions.assert_change_detected(
            changes, "processes", ChangeType.ADDED, "process:12345"
        )
    
    def test_process_termination_detection(self, detector):
        """Test detection of process termination."""
        old_snapshot = {
            'processes': [{
                'pid': 12345,
                'name': 'python3',
                'cmdline': ['python3', 'script.py'],
                'ppid': 1234,
                'status': 'running',
                'create_time': datetime.now().timestamp() - 3600,  # 1 hour ago
                'memory_info': {'rss': 500 * 1024 * 1024},  # 500MB
                'cpu_percent': 25.0,
                'num_threads': 6,
                'username': 'alejandro',
                'cwd': '/home/alejandro',
                'exe': '/usr/bin/python3',
                'connections': [],
                'open_files': [],
                'environ': {}
            }]
        }
        
        new_snapshot = {'processes': []}
        
        changes = detector.detect_changes(old_snapshot, new_snapshot)
        
        # Should detect process termination
        TemporalAssertions.assert_change_detected(
            changes, "processes", ChangeType.REMOVED, "process:12345"
        )
    
    def test_memory_usage_change_detection(self, detector):
        """Test detection of significant memory usage changes."""
        base_memory = 1024 * 1024 * 1024  # 1GB
        
        old_snapshot = {
            'processes': [{
                'pid': 12345,
                'name': 'python3',
                'cmdline': ['python3', 'train.py'],
                'memory_info': {'rss': base_memory},
                'cpu_percent': 50.0,
                'num_threads': 8
            }]
        }
        
        new_snapshot = {
            'processes': [{
                'pid': 12345,
                'name': 'python3', 
                'cmdline': ['python3', 'train.py'],
                'memory_info': {'rss': base_memory + 2 * 1024 * 1024 * 1024},  # +2GB
                'cpu_percent': 50.0,
                'num_threads': 8
            }]
        }
        
        changes = detector.detect_changes(old_snapshot, new_snapshot)
        
        # Should detect memory change
        memory_changes = [c for c in changes if 'memory' in c.entity_id]
        assert len(memory_changes) > 0
        assert memory_changes[0].significance > 0.5  # Significant change
    
    def test_cpu_usage_change_detection(self, detector):
        """Test detection of significant CPU usage changes."""
        old_snapshot = {
            'processes': [{
                'pid': 12345,
                'name': 'training_job',
                'cpu_percent': 10.0,
                'memory_info': {'rss': 1024 * 1024 * 1024},
                'num_threads': 4
            }]
        }
        
        new_snapshot = {
            'processes': [{
                'pid': 12345,
                'name': 'training_job',
                'cpu_percent': 85.0,  # Significant CPU increase
                'memory_info': {'rss': 1024 * 1024 * 1024},
                'num_threads': 4
            }]
        }
        
        changes = detector.detect_changes(old_snapshot, new_snapshot)
        
        # Should detect CPU change
        cpu_changes = [c for c in changes if 'cpu' in c.entity_id]
        assert len(cpu_changes) > 0


class TestProcessPatternAnalyzer:
    """Test ProcessPatternAnalyzer functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create ProcessPatternAnalyzer instance."""
        return ProcessPatternAnalyzer()
    
    def test_ml_framework_detection(self, analyzer, sample_process_info):
        """Test detection of ML framework processes."""
        analysis = analyzer.categorize_process(sample_process_info)
        
        assert 'ml_framework' in analysis['categories']
        assert 'gpu_accelerated' in analysis['categories']
        assert analysis['insights']['ml_context']['framework_type'] == 'pytorch'
        assert analysis['monitoring_priority'] >= 7  # High priority for ML
    
    def test_development_tool_detection(self, analyzer):
        """Test detection of development environment processes."""
        dev_process = ProcessInfo(
            pid=54321,
            name="code",
            cmdline=["code", "/home/alejandro/projects"],
            ppid=1000,
            status="running",
            create_time=datetime.now().timestamp(),
            memory_info={'rss': 500 * 1024 * 1024},  # 500MB
            cpu_percent=5.0,
            num_threads=12,
            username="alejandro",
            cwd="/home/alejandro",
            exe="/usr/share/code/code",
            connections=[],
            open_files=[],
            environ={}
        )
        
        analysis = analyzer.categorize_process(dev_process)
        
        assert 'development' in analysis['categories']
        assert analysis['risk_level'] == 'low'
    
    def test_gpu_process_detection(self, analyzer, sample_process_info):
        """Test GPU process detection logic."""
        assert analyzer._is_gpu_process(sample_process_info) == True
        
        # Test process without GPU context
        non_gpu_process = ProcessInfo(
            pid=99999,
            name="vim",
            cmdline=["vim", "README.md"],
            ppid=1000,
            status="running",
            create_time=datetime.now().timestamp(),
            memory_info={'rss': 50 * 1024 * 1024},  # 50MB
            cpu_percent=0.1,
            num_threads=1,
            username="alejandro",
            cwd="/home/alejandro",
            exe="/usr/bin/vim",
            connections=[],
            open_files=[],
            environ={'TERM': 'xterm-256color'}
        )
        
        assert analyzer._is_gpu_process(non_gpu_process) == False
    
    def test_risk_assessment(self, analyzer, sample_process_info):
        """Test process risk level assessment."""
        analysis = analyzer.categorize_process(sample_process_info)
        
        # ML + GPU + Resource intensive = higher risk
        assert analysis['risk_level'] in ['medium', 'high']
    
    def test_system_service_detection(self, analyzer):
        """Test system service process detection."""
        service_process = ProcessInfo(
            pid=1001,
            name="systemd-resolved",
            cmdline=["systemd-resolved"],
            ppid=1,
            status="running",
            create_time=datetime.now().timestamp() - 86400,  # Running for a day
            memory_info={'rss': 20 * 1024 * 1024},  # 20MB
            cpu_percent=0.1,
            num_threads=1,
            username="systemd-resolve",
            cwd="/",
            exe="/lib/systemd/systemd-resolved",
            connections=[],
            open_files=[],
            environ={}
        )
        
        analysis = analyzer.categorize_process(service_process)
        
        assert 'system_service' in analysis['categories']
        assert analysis['risk_level'] == 'medium'  # System services are medium risk


class TestProcessResourceTracker:
    """Test ProcessResourceTracker functionality."""
    
    @pytest.fixture
    def tracker(self):
        """Create ProcessResourceTracker instance."""
        return ProcessResourceTracker(history_window=5)
    
    def test_resource_trend_analysis(self, tracker):
        """Test resource usage trend analysis."""
        from src.linux_system.temporal.change_detection.detectors.process_detector import ProcessResourceSnapshot
        
        pid = 12345
        base_time = datetime.now().timestamp()
        
        # Add snapshots showing increasing memory usage
        snapshots = [
            ProcessResourceSnapshot(
                timestamp=base_time + i * 60,
                memory_rss=1024 * 1024 * 1024 + i * 500 * 1024 * 1024,  # Growing memory
                memory_vms=2048 * 1024 * 1024,
                cpu_percent=50.0 + i * 5,  # Growing CPU
                num_threads=4 + i,
                num_fds=10 + i
            )
            for i in range(5)
        ]
        
        for snapshot in snapshots:
            tracker.add_snapshot(pid, snapshot)
        
        trends = tracker.get_resource_trends(pid)
        
        assert 'memory_trend' in trends
        assert 'cpu_trend' in trends
        assert trends['memory_trend']['direction'] == 'increasing'
        assert trends['cpu_trend']['direction'] == 'increasing'
    
    def test_stable_resource_trend(self, tracker):
        """Test stable resource usage detection."""
        from src.linux_system.temporal.change_detection.detectors.process_detector import ProcessResourceSnapshot
        
        pid = 54321
        base_time = datetime.now().timestamp()
        
        # Add snapshots with stable resource usage
        snapshots = [
            ProcessResourceSnapshot(
                timestamp=base_time + i * 60,
                memory_rss=1024 * 1024 * 1024,  # Stable memory
                memory_vms=2048 * 1024 * 1024,
                cpu_percent=25.0,  # Stable CPU
                num_threads=4,
                num_fds=10
            )
            for i in range(5)
        ]
        
        for snapshot in snapshots:
            tracker.add_snapshot(pid, snapshot)
        
        trends = tracker.get_resource_trends(pid)
        
        assert trends['memory_trend']['direction'] == 'stable'
        assert trends['cpu_trend']['direction'] == 'stable'


class TestProcessDetectorIntegration:
    """Integration tests for ProcessChangeDetector."""
    
    def test_ml_training_workflow_detection(self):
        """Test detection of ML training workflow patterns."""
        detector = ProcessChangeDetector()
        
        # Simulate ML training process spawning
        empty_snapshot = {'processes': []}
        
        ml_training_snapshot = {
            'processes': [{
                'pid': 98765,
                'name': 'python3',
                'cmdline': ['python3', '-m', 'transformers.training', '--model=llama'],
                'ppid': 1000,
                'status': 'running',
                'create_time': datetime.now().timestamp(),
                'memory_info': {'rss': 16 * 1024 * 1024 * 1024},  # 16GB - large model
                'cpu_percent': 80.0,
                'num_threads': 16,
                'username': 'alejandro',
                'cwd': '/home/alejandro/llm-training',
                'exe': '/usr/bin/python3',
                'connections': [],
                'open_files': ['/dev/nvidia0', '/dev/nvidia1'],
                'environ': {
                    'CUDA_VISIBLE_DEVICES': '0,1',
                    'TRANSFORMERS_CACHE': '/home/alejandro/.cache/transformers',
                    'HF_DATASETS_CACHE': '/home/alejandro/.cache/huggingface'
                }
            }]
        }
        
        changes = detector.detect_changes(empty_snapshot, ml_training_snapshot)
        
        # Should detect high-significance process spawn
        spawn_changes = [c for c in changes if c.change_type == ChangeType.ADDED]
        assert len(spawn_changes) > 0
        
        spawn_change = spawn_changes[0]
        assert spawn_change.significance > 0.7  # High significance for large ML process
        assert 'ml_framework' in spawn_change.metadata['analysis']['categories']
        assert 'gpu_accelerated' in spawn_change.metadata['analysis']['categories']
    
    def test_process_lifecycle_tracking(self):
        """Test complete process lifecycle tracking."""
        detector = ProcessChangeDetector()
        
        # Stage 1: Empty system
        stage1 = {'processes': []}
        
        # Stage 2: Process spawns
        stage2 = {
            'processes': [{
                'pid': 11111,
                'name': 'training_job',
                'memory_info': {'rss': 2 * 1024 * 1024 * 1024},  # 2GB
                'cpu_percent': 60.0,
                'num_threads': 8
            }]
        }
        
        # Stage 3: Process grows memory usage
        stage3 = {
            'processes': [{
                'pid': 11111,
                'name': 'training_job',
                'memory_info': {'rss': 8 * 1024 * 1024 * 1024},  # 8GB - significant growth
                'cpu_percent': 90.0,
                'num_threads': 12
            }]
        }
        
        # Stage 4: Process terminates
        stage4 = {'processes': []}
        
        # Test transitions
        changes_spawn = detector.detect_changes(stage1, stage2)
        changes_growth = detector.detect_changes(stage2, stage3)  
        changes_termination = detector.detect_changes(stage3, stage4)
        
        # Verify spawn detection
        TemporalAssertions.assert_change_detected(
            changes_spawn, "processes", ChangeType.ADDED, "process:11111"
        )
        
        # Verify resource growth detection
        memory_changes = [c for c in changes_growth if 'memory' in c.entity_id]
        assert len(memory_changes) > 0
        assert memory_changes[0].significance > 0.5
        
        # Verify termination detection
        TemporalAssertions.assert_change_detected(
            changes_termination, "processes", ChangeType.REMOVED, "process:11111"
        )