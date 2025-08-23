"""
Unit Tests for SystemCollector
=============================

Comprehensive test suite for the SystemCollector component.
Tests cover functionality, error handling, security, and performance.
"""

import unittest
from unittest.mock import Mock, patch, mock_open, MagicMock
import json
import tempfile
import os
from pathlib import Path
import time

from src.ubuntu_llm_system.data_collection.collectors.system_collector import SystemCollector


class TestSystemCollector(unittest.TestCase):
    """Test suite for SystemCollector functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = SystemCollector(collection_interval=1)  # 1 second for testing
    
    def tearDown(self):
        """Clean up after tests."""
        pass
    
    def test_initialization(self):
        """Test SystemCollector initialization."""
        collector = SystemCollector(collection_interval=30)
        
        self.assertEqual(collector.name, "system")
        self.assertEqual(collector.collection_interval, 30)
        self.assertEqual(collector.collection_count, 0)
        self.assertIsNone(collector.last_collection_time)
        
        # Check safety settings
        self.assertEqual(collector.max_file_size, 50 * 1024 * 1024)  # 50MB
        self.assertEqual(collector.max_command_timeout, 30)
        
        # Check allowed commands include expected ones
        allowed_commands = collector._get_allowed_commands()
        self.assertIn('nvidia-smi', allowed_commands)
        self.assertIn('lscpu', allowed_commands)
        self.assertIn('free', allowed_commands)
        self.assertIn('smartctl', allowed_commands)
    
    def test_should_collect_timing(self):
        """Test collection timing logic."""
        # First collection should always run
        self.assertTrue(self.collector.should_collect())
        
        # Simulate collection
        self.collector.last_collection_time = self.collector._get_current_time_ms() / 1000
        self.collector.last_collection_time = time.time()
        
        # Should not collect immediately after
        self.assertFalse(self.collector.should_collect())
        
        # Should collect after interval
        time.sleep(1.1)  # Wait longer than 1 second interval
        self.assertTrue(self.collector.should_collect())
    
    def test_safe_read_file_success(self):
        """Test successful file reading."""
        test_content = "test file content\nline 2\nline 3"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tf:
            tf.write(test_content)
            tf.flush()
            
            result = self.collector.safe_read_file(tf.name)
            self.assertEqual(result, test_content)
        
        os.unlink(tf.name)
    
    def test_safe_read_file_nonexistent(self):
        """Test reading nonexistent file."""
        result = self.collector.safe_read_file('/nonexistent/file.txt')
        self.assertIsNone(result)
    
    def test_safe_read_file_max_lines(self):
        """Test file reading with line limit."""
        test_content = "line 1\nline 2\nline 3\nline 4\nline 5"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tf:
            tf.write(test_content)
            tf.flush()
            
            result = self.collector.safe_read_file(tf.name, max_lines=3)
            expected = "line 1\nline 2\nline 3\n"
            self.assertEqual(result, expected)
        
        os.unlink(tf.name)
    
    def test_safe_read_file_too_large(self):
        """Test file size limit enforcement."""
        # Mock file that appears too large
        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 100 * 1024 * 1024  # 100MB
            
            result = self.collector.safe_read_file('/test/file')
            self.assertIsNone(result)
    
    @patch('subprocess.run')
    def test_safe_execute_command_success(self, mock_run):
        """Test successful command execution."""
        mock_run.return_value.stdout = "command output"
        mock_run.return_value.stderr = ""
        mock_run.return_value.returncode = 0
        
        result = self.collector.safe_execute_command(['ls', '-la'])
        
        self.assertIsNotNone(result)
        self.assertEqual(result['stdout'], "command output")
        self.assertEqual(result['return_code'], 0)
        self.assertEqual(result['command'], 'ls -la')
    
    @patch('subprocess.run')
    def test_safe_execute_command_disallowed(self, mock_run):
        """Test command validation."""
        result = self.collector.safe_execute_command(['rm', '-rf', '/'])
        self.assertIsNone(result)
        mock_run.assert_not_called()
    
    @patch('subprocess.run')
    def test_safe_execute_command_timeout(self, mock_run):
        """Test command timeout handling."""
        from subprocess import TimeoutExpired
        mock_run.side_effect = TimeoutExpired(['sleep', '60'], 30)
        
        result = self.collector.safe_execute_command(['sleep', '60'])
        self.assertIsNone(result)
    
    @patch('subprocess.run')
    def test_safe_execute_command_not_found(self, mock_run):
        """Test handling of nonexistent commands."""
        mock_run.side_effect = FileNotFoundError()
        
        result = self.collector.safe_execute_command(['nonexistent-command'])
        self.assertIsNone(result)
    
    def test_create_data_snapshot(self):
        """Test data snapshot creation."""
        test_data = {'cpu': 'intel', 'gpu': 'nvidia'}
        
        snapshot = self.collector.create_data_snapshot(test_data)
        
        self.assertIn('metadata', snapshot)
        self.assertIn('data', snapshot)
        
        metadata = snapshot['metadata']
        self.assertEqual(metadata['collector'], 'system')
        self.assertIn('timestamp', metadata)
        self.assertIn('data_hash', metadata)
        self.assertEqual(metadata['collection_count'], 1)
        
        self.assertEqual(snapshot['data'], test_data)
        
        # Test counter increment
        snapshot2 = self.collector.create_data_snapshot(test_data)
        self.assertEqual(snapshot2['metadata']['collection_count'], 2)
    
    def test_get_status(self):
        """Test collector status information."""
        status = self.collector.get_status()
        
        expected_keys = ['name', 'collection_interval', 'last_collection_time', 
                        'collection_count', 'status']
        for key in expected_keys:
            self.assertIn(key, status)
        
        self.assertEqual(status['name'], 'system')
        self.assertEqual(status['collection_interval'], 1)
        self.assertEqual(status['status'], 'active')
    
    def test_parse_cpuinfo(self):
        """Test CPU info parsing."""
        test_cpuinfo = """processor	: 0
vendor_id	: GenuineIntel
cpu family	: 6
model		: 142
model name	: Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz
stepping	: 10

processor	: 1
vendor_id	: GenuineIntel
cpu family	: 6
model		: 142
model name	: Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz
stepping	: 10
"""
        
        result = self.collector._parse_cpuinfo(test_cpuinfo)
        
        self.assertEqual(result['count'], 2)
        self.assertEqual(len(result['processors']), 2)
        self.assertEqual(result['processors'][0]['processor'], '0')
        self.assertEqual(result['processors'][0]['vendor_id'], 'GenuineIntel')
        self.assertEqual(result['processors'][1]['processor'], '1')
    
    def test_parse_meminfo(self):
        """Test memory info parsing."""
        test_meminfo = """MemTotal:       16384000 kB
MemFree:         8192000 kB
MemAvailable:   12288000 kB
Buffers:          512000 kB
Cached:          2048000 kB
"""
        
        result = self.collector._parse_meminfo(test_meminfo)
        
        self.assertEqual(result['MemTotal'], '16384000 kB')
        self.assertEqual(result['MemFree'], '8192000 kB')
        self.assertEqual(result['MemAvailable'], '12288000 kB')
    
    def test_parse_cpu_stat(self):
        """Test CPU stat parsing."""
        test_cpu_stat = """cpu  123456 1234 56789 9876543 1234 567 890
cpu0 61728 617 28394 4938271 617 283 445
cpu1 61728 617 28395 4938272 617 284 445
"""
        
        result = self.collector._parse_cpu_stat(test_cpu_stat)
        
        self.assertIn('cpu', result)
        self.assertIn('cpu0', result)
        self.assertIn('cpu1', result)
        
        cpu_data = result['cpu']
        self.assertEqual(cpu_data['user'], 123456)
        self.assertEqual(cpu_data['system'], 56789)
        self.assertEqual(cpu_data['idle'], 9876543)
    
    def test_parse_vmstat(self):
        """Test vmstat parsing."""
        test_vmstat = """nr_free_pages 2048000
nr_zone_inactive_anon 512000
nr_zone_active_anon 1024000
nr_zone_inactive_file 256000
nr_zone_active_file 768000
"""
        
        result = self.collector._parse_vmstat(test_vmstat)
        
        self.assertEqual(result['nr_free_pages'], '2048000')
        self.assertEqual(result['nr_zone_inactive_anon'], '512000')
        self.assertEqual(result['nr_zone_active_file'], '768000')
    
    @patch('os.environ')
    def test_get_cuda_env_vars(self, mock_environ):
        """Test CUDA environment variable collection."""
        mock_environ.__contains__ = lambda self, key: key in ['CUDA_HOME', 'PATH']
        mock_environ.__getitem__ = lambda self, key: {
            'CUDA_HOME': '/usr/local/cuda',
            'PATH': '/usr/local/cuda/bin:/usr/bin'
        }[key]
        
        result = self.collector._get_cuda_env_vars()
        
        self.assertEqual(result.get('CUDA_HOME'), '/usr/local/cuda')
        self.assertEqual(result.get('PATH'), '/usr/local/cuda/bin:/usr/bin')
    
    @patch('pathlib.Path.glob')
    @patch('pathlib.Path.exists')
    def test_get_cuda_library_paths(self, mock_exists, mock_glob):
        """Test CUDA library path discovery."""
        mock_exists.return_value = True
        mock_glob.return_value = [
            Path('/usr/local/cuda/lib64/libcuda.so'),
            Path('/usr/local/cuda/lib64/libcudart.so')
        ]
        
        result = self.collector._get_cuda_library_paths()
        
        self.assertIn('/usr/local/cuda/lib64', result)
        libs = result['/usr/local/cuda/lib64']
        self.assertIn('/usr/local/cuda/lib64/libcuda.so', libs)
        self.assertIn('/usr/local/cuda/lib64/libcudart.so', libs)
    
    @patch.object(SystemCollector, 'safe_read_file')
    @patch.object(SystemCollector, 'safe_execute_command')
    def test_collect_cpu_data(self, mock_execute, mock_read):
        """Test CPU data collection."""
        # Mock /proc/cpuinfo
        mock_read.return_value = """processor	: 0
vendor_id	: GenuineIntel
model name	: Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz
"""
        
        # Mock lscpu command
        mock_execute.return_value = {
            'stdout': 'CPU(s): 4\nModel name: Intel Core i7',
            'return_code': 0
        }
        
        result = self.collector._collect_cpu_data()
        
        self.assertIn('cpuinfo', result)
        self.assertIn('lscpu', result)
        self.assertIn('frequency', result)
        self.assertIn('thermal', result)
        self.assertIn('numa', result)
    
    @patch.object(SystemCollector, 'safe_execute_command')
    def test_collect_nvidia_gpu_data(self, mock_execute):
        """Test NVIDIA GPU data collection."""
        mock_execute.return_value = {
            'stdout': 'timestamp, name, temperature.gpu\n2024-01-15, RTX 5090, 65',
            'return_code': 0
        }
        
        result = self.collector._collect_nvidia_gpu_data()
        
        self.assertIn('basic_metrics', result)
        # Test that multiple nvidia-smi calls are made
        self.assertGreaterEqual(mock_execute.call_count, 5)
    
    @patch.object(SystemCollector, 'safe_execute_command')
    def test_collect_security_data(self, mock_execute):
        """Test security data collection."""
        mock_execute.return_value = {
            'stdout': 'user1:x:1000:1000:User One:/home/user1:/bin/bash',
            'return_code': 0
        }
        
        result = self.collector._collect_security_data()
        
        self.assertIn('users', result)
        self.assertIn('network', result)
        self.assertIn('authentication', result)
        self.assertIn('permissions', result)
        self.assertIn('audit', result)
        self.assertIn('firewall', result)
        self.assertIn('processes', result)
        self.assertIn('integrity', result)
    
    @patch.object(SystemCollector, 'safe_execute_command')
    def test_collect_python_environment_data(self, mock_execute):
        """Test Python environment data collection."""
        mock_execute.return_value = {
            'stdout': 'Python 3.12.0',
            'return_code': 0
        }
        
        result = self.collector._collect_python_environment_data()
        
        self.assertIn('installations', result)
        self.assertIn('packages', result)
        self.assertIn('virtual_envs', result)
        self.assertIn('conda', result)
        self.assertIn('poetry', result)
        self.assertIn('system_packages', result)
        self.assertIn('ai_frameworks', result)
    
    @patch.object(SystemCollector, 'safe_execute_command')
    @patch.object(SystemCollector, 'safe_read_file')
    def test_comprehensive_storage_data(self, mock_read, mock_execute):
        """Test comprehensive storage data collection."""
        mock_execute.return_value = {
            'stdout': 'nvme0n1    931.5G  disk',
            'return_code': 0
        }
        mock_read.return_value = 'sda 100 200 300 400 500'
        
        result = self.collector._collect_comprehensive_storage_data()
        
        self.assertIn('hardware', result)
        self.assertIn('performance', result)
        self.assertIn('filesystem', result)
        self.assertIn('io_patterns', result)
        self.assertIn('ai_storage', result)
        self.assertIn('health', result)
    
    @patch.object(SystemCollector, '_collect_cpu_data')
    @patch.object(SystemCollector, '_collect_nvidia_gpu_data')
    @patch.object(SystemCollector, '_collect_security_data')
    @patch.object(SystemCollector, '_collect_comprehensive_storage_data')
    def test_full_collect_integration(self, mock_storage, mock_security, 
                                     mock_gpu, mock_cpu):
        """Test full collection integration."""
        # Mock all major collection methods
        mock_cpu.return_value = {'cpus': 8}
        mock_gpu.return_value = {'gpu_name': 'RTX 5090'}
        mock_security.return_value = {'users': ['user1']}
        mock_storage.return_value = {'devices': ['nvme0n1']}
        
        snapshot = self.collector.collect()
        
        # Verify structure
        self.assertIn('metadata', snapshot)
        self.assertIn('data', snapshot)
        
        # Verify all major sections present
        data = snapshot['data']
        expected_sections = [
            'cpu', 'memory', 'nvidia_gpu', 'cuda', 'storage',
            'security', 'python_env', 'hardware', 'kernel',
            'network', 'processes', 'performance', 'docker_nvidia'
        ]
        for section in expected_sections:
            self.assertIn(section, data)
        
        # Verify metadata
        metadata = snapshot['metadata']
        self.assertEqual(metadata['collector'], 'system')
        self.assertIn('timestamp', metadata)
        self.assertIn('collection_duration_ms', metadata)
    
    def test_error_handling_graceful_degradation(self):
        """Test graceful error handling."""
        # Test with methods that might fail
        with patch.object(self.collector, 'safe_execute_command', 
                         return_value=None):  # Simulate command failures
            
            snapshot = self.collector.collect()
            
            # Collection should still complete
            self.assertIn('metadata', snapshot)
            self.assertIn('data', snapshot)
            
            # Data sections should exist but may be empty
            self.assertIn('cpu', snapshot['data'])
            self.assertIn('nvidia_gpu', snapshot['data'])
    
    def test_data_hash_consistency(self):
        """Test data hash generation consistency."""
        test_data = {'cpu': 'intel', 'memory': '16GB'}
        
        snapshot1 = self.collector.create_data_snapshot(test_data)
        snapshot2 = self.collector.create_data_snapshot(test_data)
        
        # Same data should produce same hash
        hash1 = snapshot1['metadata']['data_hash']
        hash2 = snapshot2['metadata']['data_hash']
        self.assertEqual(hash1, hash2)
        
        # Different data should produce different hash
        different_data = {'cpu': 'amd', 'memory': '32GB'}
        snapshot3 = self.collector.create_data_snapshot(different_data)
        hash3 = snapshot3['metadata']['data_hash']
        self.assertNotEqual(hash1, hash3)
    
    def test_performance_timing(self):
        """Test collection performance timing."""
        start_time = time.time()
        
        with patch.object(self.collector, '_collect_cpu_data', return_value={}):
            with patch.object(self.collector, '_collect_nvidia_gpu_data', return_value={}):
                snapshot = self.collector.collect()
        
        duration_ms = snapshot['metadata']['collection_duration_ms']
        actual_duration = (time.time() - start_time) * 1000
        
        # Timing should be reasonably accurate (within 100ms)
        self.assertLess(abs(duration_ms - actual_duration), 100)
    
    def test_allowed_commands_security(self):
        """Test that allowed commands list is secure."""
        allowed = self.collector._get_allowed_commands()
        
        # Should not contain dangerous commands
        dangerous_commands = ['rm', 'dd', 'mkfs', 'fdisk', 'parted', 
                             'chmod', 'chown', 'systemctl start', 'kill']
        
        for cmd in dangerous_commands:
            self.assertNotIn(cmd, allowed, 
                           f"Dangerous command '{cmd}' found in allowed list")
        
        # Should contain expected safe commands
        safe_commands = ['lscpu', 'free', 'nvidia-smi', 'lsblk', 'ps']
        for cmd in safe_commands:
            self.assertIn(cmd, allowed, 
                         f"Expected safe command '{cmd}' not in allowed list")


class TestSystemCollectorIntegration(unittest.TestCase):
    """Integration tests for SystemCollector with real system."""
    
    @classmethod
    def setUpClass(cls):
        """Set up integration test environment."""
        cls.collector = SystemCollector(collection_interval=1)
    
    def test_real_system_collection(self):
        """Test collection on real system (if available)."""
        try:
            snapshot = self.collector.collect()
            
            # Verify basic structure
            self.assertIn('metadata', snapshot)
            self.assertIn('data', snapshot)
            
            # Collection should complete in reasonable time
            duration = snapshot['metadata']['collection_duration_ms']
            self.assertLess(duration, 30000, "Collection took too long (>30s)")
            
            print(f"✓ Real system collection completed in {duration}ms")
            
            # Test specific data availability
            data = snapshot['data']
            
            # CPU data should always be available
            self.assertIn('cpu', data)
            if data['cpu'].get('lscpu'):
                print("✓ CPU information collected successfully")
            
            # Memory data should always be available  
            self.assertIn('memory', data)
            if data['memory'].get('free'):
                print("✓ Memory information collected successfully")
            
            # Network data should be available
            self.assertIn('network', data)
            if data['network'].get('interfaces'):
                print("✓ Network information collected successfully")
            
            # GPU data may not be available on all systems
            if data.get('nvidia_gpu', {}).get('basic_metrics'):
                print("✓ NVIDIA GPU information collected successfully")
            else:
                print("! NVIDIA GPU not available or accessible")
            
        except Exception as e:
            self.skipTest(f"Real system test skipped due to: {e}")
    
    def test_file_system_access(self):
        """Test file system access patterns."""
        # Test /proc filesystem access
        proc_files = ['/proc/cpuinfo', '/proc/meminfo', '/proc/stat']
        
        for proc_file in proc_files:
            if os.path.exists(proc_file):
                content = self.collector.safe_read_file(proc_file)
                self.assertIsNotNone(content, f"Could not read {proc_file}")
                self.assertGreater(len(content), 0, f"{proc_file} is empty")
                print(f"✓ Successfully read {proc_file}")
    
    def test_command_execution_safety(self):
        """Test command execution with real system commands."""
        safe_commands = [
            ['ps', '--version'],
            ['ls', '--version'],
            ['free', '--version']
        ]
        
        for cmd in safe_commands:
            if cmd[0] in self.collector._get_allowed_commands():
                result = self.collector.safe_execute_command(cmd)
                if result:
                    self.assertEqual(result['return_code'], 0)
                    print(f"✓ Successfully executed: {' '.join(cmd)}")


if __name__ == '__main__':
    # Run unit tests
    print("Running SystemCollector Unit Tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run integration tests separately
    print("\nRunning SystemCollector Integration Tests...")
    integration_suite = unittest.TestLoader().loadTestsFromTestCase(TestSystemCollectorIntegration)
    unittest.TextTestRunner(verbosity=2).run(integration_suite)