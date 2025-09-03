#!/usr/bin/env python3
"""
Unit Tests for ResourceManager

Tests the resource management system including memory tracking, optimization,
garbage collection strategies, and resource monitoring.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import threading
import time
import sys
from pathlib import Path

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "core"))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from core.resource_manager import (
    ResourceManager, ResourceTracker, MemoryOptimizer, 
    ResourceMetrics, ResourceLimits
)


class TestResourceMetrics(unittest.TestCase):
    """Test ResourceMetrics data structure"""
    
    def test_resource_metrics_creation(self):
        """Test ResourceMetrics creation with all fields"""
        metrics = ResourceMetrics(
            timestamp=1234567890.0,
            memory_mb=1024.5,
            cpu_percent=75.2,
            peak_memory_mb=1500.0,
            gc_collections=150,
            active_threads=8,
            open_files=25
        )
        
        self.assertEqual(metrics.timestamp, 1234567890.0)
        self.assertEqual(metrics.memory_mb, 1024.5)
        self.assertEqual(metrics.cpu_percent, 75.2)
        self.assertEqual(metrics.peak_memory_mb, 1500.0)
        self.assertEqual(metrics.gc_collections, 150)
        self.assertEqual(metrics.active_threads, 8)
        self.assertEqual(metrics.open_files, 25)


class TestResourceLimits(unittest.TestCase):
    """Test ResourceLimits configuration"""
    
    def test_resource_limits_defaults(self):
        """Test default ResourceLimits values"""
        limits = ResourceLimits()
        
        self.assertEqual(limits.max_memory_mb, 14000)
        self.assertEqual(limits.cleanup_threshold_mb, 10000)
        self.assertEqual(limits.critical_threshold_mb, 12000)
        self.assertEqual(limits.max_open_files, 1000)
        self.assertEqual(limits.max_threads, 50)
        self.assertEqual(limits.gc_frequency, 100)
    
    def test_resource_limits_custom_values(self):
        """Test ResourceLimits with custom values"""
        limits = ResourceLimits(
            max_memory_mb=8000,
            cleanup_threshold_mb=6000,
            critical_threshold_mb=7000,
            max_open_files=500,
            max_threads=25
        )
        
        self.assertEqual(limits.max_memory_mb, 8000)
        self.assertEqual(limits.cleanup_threshold_mb, 6000)
        self.assertEqual(limits.critical_threshold_mb, 7000)
        self.assertEqual(limits.max_open_files, 500)
        self.assertEqual(limits.max_threads, 25)


class TestResourceTracker(unittest.TestCase):
    """Test ResourceTracker functionality"""
    
    @patch('core.resource_manager.psutil.Process')
    @patch('core.resource_manager.gc.get_stats')
    @patch('core.resource_manager.threading.active_count')
    @patch('core.resource_manager.time.time')
    def test_capture_snapshot_success(self, mock_time, mock_active_count, mock_gc_stats, mock_process_class):
        """Test successful resource snapshot capture"""
        # Mock time
        mock_time.return_value = 1234567890.0
        
        # Mock psutil Process
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 1024  # 1GB in bytes
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.cpu_percent.return_value = 45.5
        mock_process.open_files.return_value = ['file1', 'file2', 'file3']
        mock_process_class.return_value = mock_process
        
        # Mock GC stats
        mock_gc_stats.return_value = [
            {'collections': 50}, {'collections': 30}, {'collections': 20}
        ]
        
        # Mock threading
        mock_active_count.return_value = 12
        
        # Create tracker and capture snapshot
        tracker = ResourceTracker(history_size=10)
        snapshot = tracker.capture_snapshot()
        
        # Verify snapshot data
        self.assertEqual(snapshot.timestamp, 1234567890.0)
        self.assertEqual(snapshot.memory_mb, 1024.0)  # 1GB / 1024
        self.assertEqual(snapshot.cpu_percent, 45.5)
        self.assertEqual(snapshot.peak_memory_mb, 1024.0)
        self.assertEqual(snapshot.gc_collections, 100)  # 50+30+20
        self.assertEqual(snapshot.active_threads, 12)
        self.assertEqual(snapshot.open_files, 3)
        
        # Verify history tracking
        self.assertEqual(len(tracker.history), 1)
        self.assertEqual(tracker.peak_memory, 1024.0)
    
    @patch('core.resource_manager.psutil.Process')
    def test_capture_snapshot_exception_handling(self, mock_process_class):
        """Test snapshot capture with psutil exception"""
        # Create a mock process that raises exceptions on method calls
        mock_process = Mock()
        mock_process.memory_info.side_effect = Exception("psutil error")
        mock_process_class.return_value = mock_process
        
        # Create tracker and capture snapshot
        tracker = ResourceTracker()
        snapshot = tracker.capture_snapshot()
        
        # Should return default metrics on error
        self.assertEqual(snapshot.memory_mb, 0)
        self.assertEqual(snapshot.cpu_percent, 0)
        self.assertEqual(snapshot.gc_collections, 0)
    
    def test_history_size_limit(self):
        """Test history size limiting"""
        # Create properly configured mock process
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 1000  # 1000 MB in bytes
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.cpu_percent.return_value = 50.0
        mock_process.open_files.return_value = [Mock()] * 10  # 10 open files
        
        with patch('core.resource_manager.psutil.Process', return_value=mock_process), \
             patch('core.resource_manager.gc.get_stats', return_value=[]), \
             patch('core.resource_manager.threading.active_count', return_value=1):
            
            tracker = ResourceTracker(history_size=3)
            
            # Add more snapshots than history size
            for i in range(5):
                with patch('core.resource_manager.time.time', return_value=float(i)):
                    tracker.capture_snapshot()
            
            # Should only keep last 3 snapshots
            self.assertEqual(len(tracker.history), 3)
            # Check timestamps to verify correct snapshots kept
            timestamps = [s.timestamp for s in tracker.history]
            self.assertEqual(timestamps, [2.0, 3.0, 4.0])
    
    def test_get_recent_metrics(self):
        """Test filtering metrics by time window"""
        # Create properly configured mock process
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 1000  # 1000 MB in bytes
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.cpu_percent.return_value = 50.0
        mock_process.open_files.return_value = [Mock()] * 10  # 10 open files
        
        with patch('core.resource_manager.psutil.Process', return_value=mock_process), \
             patch('core.resource_manager.gc.get_stats', return_value=[]), \
             patch('core.resource_manager.threading.active_count', return_value=1):
            
            tracker = ResourceTracker()
            current_time = 1000.0
            
            # Add snapshots with different timestamps
            for i, timestamp in enumerate([950.0, 970.0, 990.0, 995.0, 999.0]):
                with patch('core.resource_manager.time.time', return_value=timestamp):
                    tracker.capture_snapshot()
            
            # Get recent metrics (last 20 seconds from time 1000)
            with patch('core.resource_manager.time.time', return_value=current_time):
                recent = tracker.get_recent_metrics(seconds=20)
            
            # Should get metrics from 980.0 onwards (990.0, 995.0, 999.0)
            self.assertEqual(len(recent), 3)
            timestamps = [m.timestamp for m in recent]
            self.assertEqual(timestamps, [990.0, 995.0, 999.0])
    
    def test_analyze_memory_trend(self):
        """Test memory trend analysis"""
        with patch('core.resource_manager.psutil.Process'), \
             patch('core.resource_manager.gc.get_stats', return_value=[]), \
             patch('core.resource_manager.threading.active_count', return_value=1):
            
            tracker = ResourceTracker()
            
            # Mock memory info to return different memory values  
            # Rate: (1050 - 1000) / (240 - 0) * 60 = 50 / 240 * 60 = 12.5 MB/min (> 10)
            memory_values = [1000, 1010, 1025, 1035, 1050]  # Increasing trend
            
            for i, memory_mb in enumerate(memory_values):
                mock_memory_info = Mock()
                mock_memory_info.rss = memory_mb * 1024 * 1024  # Convert to bytes
                
                with patch('core.resource_manager.time.time', return_value=float(i * 60)), \
                     patch.object(tracker.process, 'memory_info', return_value=mock_memory_info), \
                     patch.object(tracker.process, 'cpu_percent', return_value=50.0), \
                     patch.object(tracker.process, 'open_files', return_value=[]):
                    tracker.capture_snapshot()
            
            # Analyze trend
            with patch('core.resource_manager.time.time', return_value=300.0):
                trend_analysis = tracker.analyze_memory_trend(window_seconds=300)
            
            # Should detect positive trend (memory increasing)
            self.assertEqual(trend_analysis['trend'], 1.0)
            self.assertGreater(trend_analysis['rate_mb_per_min'], 0)
            self.assertEqual(trend_analysis['current_mb'], 1050.0)
            self.assertEqual(trend_analysis['peak_mb'], 1050.0)
    
    def test_analyze_memory_trend_insufficient_data(self):
        """Test memory trend analysis with insufficient data"""
        tracker = ResourceTracker()
        
        # No data points
        trend_analysis = tracker.analyze_memory_trend()
        expected = {'trend': 0.0, 'rate_mb_per_min': 0.0, 'stability': 1.0}
        self.assertEqual(trend_analysis, expected)


class TestMemoryOptimizer(unittest.TestCase):
    """Test MemoryOptimizer functionality"""
    
    def test_memory_optimizer_initialization(self):
        """Test MemoryOptimizer initialization"""
        limits = ResourceLimits(max_memory_mb=8000)
        optimizer = MemoryOptimizer(limits)
        
        self.assertEqual(optimizer.limits, limits)
        self.assertEqual(len(optimizer.cleanup_callbacks), 0)
        self.assertEqual(optimizer.cleanup_count, 0)
    
    def test_register_cleanup_callback(self):
        """Test registering cleanup callbacks"""
        optimizer = MemoryOptimizer()
        
        callback1 = Mock(return_value=1024)
        callback2 = Mock(return_value=2048)
        
        optimizer.register_cleanup_callback(callback1)
        optimizer.register_cleanup_callback(callback2)
        
        self.assertEqual(len(optimizer.cleanup_callbacks), 2)
        self.assertIn(callback1, optimizer.cleanup_callbacks)
        self.assertIn(callback2, optimizer.cleanup_callbacks)
    
    @patch('core.resource_manager.gc.collect')
    def test_force_garbage_collection(self, mock_gc_collect):
        """Test forced garbage collection"""
        # Mock gc.collect to return different values for different calls
        mock_gc_collect.side_effect = [10, 5, 3, 8]  # 3 generation calls + final call
        
        optimizer = MemoryOptimizer()
        collected = optimizer.force_garbage_collection()
        
        # Should call gc.collect 4 times (3 generations + final)
        self.assertEqual(mock_gc_collect.call_count, 4)
        
        # Verify generation-specific calls
        expected_calls = [call(0), call(1), call(2), call()]
        mock_gc_collect.assert_has_calls(expected_calls)
        
        # Should return total from generation collections (10+5+3 = 18)
        self.assertEqual(collected, 18)
    
    def test_cleanup_large_objects(self):
        """Test large object cleanup with callbacks"""
        optimizer = MemoryOptimizer()
        
        # Register mock callbacks
        callback1 = Mock(return_value=1024)
        callback2 = Mock(return_value=2048)
        callback_error = Mock(side_effect=Exception("Cleanup error"))
        
        optimizer.register_cleanup_callback(callback1)
        optimizer.register_cleanup_callback(callback2)
        optimizer.register_cleanup_callback(callback_error)
        
        # Run cleanup
        freed_bytes = optimizer.cleanup_large_objects()
        
        # Should call all callbacks
        callback1.assert_called_once()
        callback2.assert_called_once()
        callback_error.assert_called_once()
        
        # Should return sum of successful callbacks (error callback ignored)
        self.assertEqual(freed_bytes, 3072)  # 1024 + 2048
    
    @patch('core.resource_manager.time.time')
    def test_optimize_memory_usage_no_action(self, mock_time):
        """Test memory optimization when no action needed"""
        mock_time.return_value = 1000.0
        
        optimizer = MemoryOptimizer()
        result = optimizer.optimize_memory_usage(current_mb=5000.0)  # Below cleanup threshold
        
        expected = {
            'level': 'none',
            'actions': [],
            'bytes_freed': 0,
            'memory_mb': 5000.0
        }
        self.assertEqual(result, expected)
    
    @patch('core.resource_manager.time.time')
    @patch.object(MemoryOptimizer, 'force_garbage_collection')
    def test_optimize_memory_usage_moderate_cleanup(self, mock_gc, mock_time):
        """Test moderate memory optimization"""
        mock_time.return_value = 1000.0
        mock_gc.return_value = 50
        
        optimizer = MemoryOptimizer()
        result = optimizer.optimize_memory_usage(current_mb=10500.0)  # Above cleanup threshold
        
        # Should perform garbage collection but not large object cleanup
        mock_gc.assert_called_once()
        self.assertEqual(result['level'], 'moderate')
        self.assertIn('gc_collected_50_objects', result['actions'])
        self.assertEqual(len(result['actions']), 1)
    
    @patch('core.resource_manager.time.time')
    @patch.object(MemoryOptimizer, 'force_garbage_collection')
    @patch.object(MemoryOptimizer, 'cleanup_large_objects')
    def test_optimize_memory_usage_critical_cleanup(self, mock_cleanup, mock_gc, mock_time):
        """Test critical memory optimization"""
        mock_time.return_value = 1000.0
        mock_gc.return_value = 75
        mock_cleanup.return_value = 512000
        
        optimizer = MemoryOptimizer()
        result = optimizer.optimize_memory_usage(current_mb=12500.0)  # Above critical threshold
        
        # Should perform both garbage collection and large object cleanup
        mock_gc.assert_called_once()
        mock_cleanup.assert_called_once()
        
        self.assertEqual(result['level'], 'critical')
        self.assertIn('gc_collected_75_objects', result['actions'])
        self.assertIn('cache_cleanup_512000_bytes', result['actions'])
        self.assertEqual(result['bytes_freed'], 512000)
        self.assertEqual(len(result['actions']), 2)
    
    @patch('core.resource_manager.time.time')
    def test_optimize_memory_usage_recent_cleanup_skip(self, mock_time):
        """Test skipping optimization if recently cleaned"""
        optimizer = MemoryOptimizer()
        
        # First optimization
        mock_time.return_value = 1000.0
        with patch.object(optimizer, 'force_garbage_collection', return_value=10):
            result1 = optimizer.optimize_memory_usage(current_mb=10500.0)
        
        # Second optimization within 30 seconds
        mock_time.return_value = 1020.0  # 20 seconds later
        result2 = optimizer.optimize_memory_usage(current_mb=10500.0)
        
        # First should work, second should be skipped
        self.assertEqual(result1['level'], 'moderate')
        self.assertEqual(result2, {'actions': [], 'bytes_freed': 0, 'memory_mb': 10500.0})


class TestResourceManager(unittest.TestCase):
    """Test ResourceManager main controller"""
    
    @patch('core.resource_manager.ResourceTracker')
    @patch('core.resource_manager.MemoryOptimizer')
    def test_resource_manager_initialization(self, mock_optimizer_class, mock_tracker_class):
        """Test ResourceManager initialization"""
        limits = ResourceLimits(max_memory_mb=8000)
        
        manager = ResourceManager(limits)
        
        # Should initialize with provided limits
        self.assertEqual(manager.limits, limits)
        self.assertFalse(manager.monitoring)
        self.assertIsNone(manager.monitor_thread)
        self.assertEqual(manager.operation_count, 0)
        
        # Should create tracker and optimizer
        mock_tracker_class.assert_called_once()
        mock_optimizer_class.assert_called_once_with(limits)
    
    @patch('core.resource_manager.threading.Thread')
    def test_start_monitoring(self, mock_thread_class):
        """Test starting resource monitoring"""
        mock_thread = Mock()
        mock_thread_class.return_value = mock_thread
        
        manager = ResourceManager()
        manager.start_monitoring(interval_seconds=10)
        
        # Should start monitoring
        self.assertTrue(manager.monitoring)
        self.assertEqual(manager.monitor_thread, mock_thread)
        
        # Should create and start thread
        mock_thread_class.assert_called_once()
        args, kwargs = mock_thread_class.call_args
        self.assertEqual(kwargs['target'], manager._monitor_loop)
        self.assertEqual(kwargs['args'], (10,))
        mock_thread.start.assert_called_once()
    
    def test_start_monitoring_already_active(self):
        """Test starting monitoring when already active"""
        manager = ResourceManager()
        manager.monitoring = True
        
        with patch('core.resource_manager.threading.Thread') as mock_thread_class:
            manager.start_monitoring()
            
            # Should not create new thread
            mock_thread_class.assert_not_called()
    
    def test_stop_monitoring(self):
        """Test stopping resource monitoring"""
        manager = ResourceManager()
        mock_thread = Mock()
        
        # Set up active monitoring
        manager.monitoring = True
        manager.monitor_thread = mock_thread
        
        manager.stop_monitoring()
        
        # Should stop monitoring
        self.assertFalse(manager.monitoring)
        mock_thread.join.assert_called_once_with(timeout=5.0)
    
    def test_stop_monitoring_not_active(self):
        """Test stopping monitoring when not active"""
        manager = ResourceManager()
        
        # Should handle gracefully when not monitoring
        manager.stop_monitoring()
        self.assertFalse(manager.monitoring)
    
    @patch('core.resource_manager.time.sleep')
    def test_monitor_loop(self, mock_sleep):
        """Test monitoring loop functionality"""
        manager = ResourceManager()
        
        # Mock tracker and optimizer
        manager.tracker = Mock()
        manager.optimizer = Mock()
        
        # Create real ResourceMetrics object with memory usage above cleanup threshold (10000 MB)
        from core.resource_manager import ResourceMetrics
        import time
        real_snapshot = ResourceMetrics(
            timestamp=time.time(),
            memory_mb=11000.0,  # Above cleanup threshold of 10000 MB
            cpu_percent=50.0,
            peak_memory_mb=11500.0,
            gc_collections=5,
            active_threads=10,
            open_files=25
        )
        manager.tracker.capture_snapshot.return_value = real_snapshot
        
        # Mock optimization result
        mock_optimization = {'level': 'moderate', 'actions': ['gc'], 'bytes_freed': 1024}
        manager.optimizer.optimize_memory_usage.return_value = mock_optimization
        
        # Set up monitoring state
        manager.monitoring = True
        
        # Mock sleep to stop after first iteration
        def stop_monitoring(*args):
            manager.monitoring = False
        mock_sleep.side_effect = stop_monitoring
        
        # Run monitor loop
        manager._monitor_loop(interval_seconds=5)
        
        # Should capture snapshot and optimize
        manager.tracker.capture_snapshot.assert_called_once()
        manager.optimizer.optimize_memory_usage.assert_called_once_with(11000.0)
        mock_sleep.assert_called_once_with(5)
    
    def test_get_current_metrics(self):
        """Test getting current resource metrics"""
        manager = ResourceManager()
        
        # Mock tracker
        manager.tracker = Mock()
        mock_snapshot = Mock()
        mock_snapshot.memory_mb = 6000.0
        mock_snapshot.cpu_percent = 45.0
        manager.tracker.capture_snapshot.return_value = mock_snapshot
        
        # Get metrics
        metrics = manager.get_current_metrics()
        
        # Should return snapshot from tracker
        self.assertEqual(metrics, mock_snapshot)
        manager.tracker.capture_snapshot.assert_called_once()
    
    def test_analyze_resource_usage(self):
        """Test resource usage analysis"""
        manager = ResourceManager()
        
        # Mock tracker
        manager.tracker = Mock()
        mock_trend = {
            'trend': 1.0,
            'rate_mb_per_min': 15.5,
            'stability': 0.8,
            'current_mb': 9000.0,
            'peak_mb': 9500.0
        }
        manager.tracker.analyze_memory_trend.return_value = mock_trend
        
        # Analyze usage
        analysis = manager.analyze_resource_usage()
        
        # Should include trend analysis and status
        self.assertEqual(analysis['memory_trend'], mock_trend)
        self.assertIn('status', analysis)
        self.assertIn('recommendations', analysis)
        
        manager.tracker.analyze_memory_trend.assert_called_once_with(300)
    
    def test_force_optimization(self):
        """Test forcing memory optimization"""
        manager = ResourceManager()
        
        # Mock optimizer
        manager.optimizer = Mock()
        mock_result = {'level': 'critical', 'actions': ['gc', 'cleanup'], 'bytes_freed': 2048}
        manager.optimizer.optimize_memory_usage.return_value = mock_result
        
        # Force optimization
        result = manager.force_optimization()
        
        # Should call optimizer with force=True
        manager.optimizer.optimize_memory_usage.assert_called_once()
        args = manager.optimizer.optimize_memory_usage.call_args[0]
        kwargs = manager.optimizer.optimize_memory_usage.call_args[1]
        self.assertTrue(kwargs.get('force', False))
        
        self.assertEqual(result, mock_result)


if __name__ == '__main__':
    unittest.main()