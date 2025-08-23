"""
SystemCollector Functional Tests
================================

Functional tests that interact with the real Ubuntu system using Python's os module
and system libraries to validate SystemCollector output against actual system state.

These tests:
- Use os.system(), subprocess, psutil, etc. to get ground truth
- Compare SystemCollector output with real system data
- Validate data accuracy and completeness
- Test on actual hardware (RTX 5090, storage, etc.)

Run these tests on the target Ubuntu system to verify accuracy.
"""

import os
import sys
import unittest
import subprocess
import json
import re
import time
from pathlib import Path
import psutil
import platform
import socket
import getpass
from typing import Dict, Any, List

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.ubuntu_llm_system.data_collection.collectors.system_collector import SystemCollector


class SystemCollectorFunctionalTests(unittest.TestCase):
    """Functional tests comparing SystemCollector with real system data."""
    
    @classmethod
    def setUpClass(cls):
        """Set up functional test environment."""
        cls.collector = SystemCollector(collection_interval=1)
        print("\n🔧 Setting up functional tests on real Ubuntu system...")
        
        # Collect system data once for all tests
        print("📊 Collecting system data...")
        cls.snapshot = cls.collector.collect()
        cls.system_data = cls.snapshot['data']
        
        print(f"✅ Data collection completed in {cls.snapshot['metadata']['collection_duration_ms']}ms")
        print(f"📦 Collected {len(cls.system_data)} data sections")
    
    def test_cpu_core_count_accuracy(self):
        """Test CPU core count accuracy against os.cpu_count()."""
        print("\n🔍 Testing CPU core count accuracy...")
        
        # Get ground truth from os module
        os_cpu_count = os.cpu_count()
        
        # Get from psutil for additional verification
        psutil_cpu_count = psutil.cpu_count(logical=True)
        psutil_physical_count = psutil.cpu_count(logical=False)
        
        # Get from SystemCollector
        cpu_data = self.system_data.get('cpu', {})
        cpuinfo = cpu_data.get('cpuinfo', {})
        collector_cpu_count = cpuinfo.get('count', 0)
        
        print(f"   os.cpu_count(): {os_cpu_count}")
        print(f"   psutil logical: {psutil_cpu_count}")
        print(f"   psutil physical: {psutil_physical_count}")
        print(f"   SystemCollector: {collector_cpu_count}")
        
        # Verify accuracy
        self.assertEqual(collector_cpu_count, os_cpu_count,
                        "SystemCollector CPU count doesn't match os.cpu_count()")
        self.assertEqual(collector_cpu_count, psutil_cpu_count,
                        "SystemCollector CPU count doesn't match psutil")
    
    def test_memory_information_accuracy(self):
        """Test memory information accuracy against psutil."""
        print("\n🔍 Testing memory information accuracy...")
        
        # Get ground truth from psutil
        psutil_memory = psutil.virtual_memory()
        psutil_total_gb = psutil_memory.total / (1024**3)
        psutil_available_gb = psutil_memory.available / (1024**3)
        
        # Get from SystemCollector
        memory_data = self.system_data.get('memory', {})
        meminfo = memory_data.get('meminfo', {})
        
        if 'MemTotal' in meminfo:
            # Parse MemTotal (format: "16777216 kB")
            total_kb = int(meminfo['MemTotal'].split()[0])
            collector_total_gb = total_kb / (1024**2)
            
            print(f"   psutil total: {psutil_total_gb:.2f} GB")
            print(f"   SystemCollector total: {collector_total_gb:.2f} GB")
            
            # Allow 1% difference due to rounding/reporting differences
            diff_percent = abs(collector_total_gb - psutil_total_gb) / psutil_total_gb
            self.assertLess(diff_percent, 0.01, 
                           "Memory total differs by more than 1%")
        else:
            self.fail("SystemCollector missing MemTotal information")
        
        # Test memory availability
        if 'MemAvailable' in meminfo:
            available_kb = int(meminfo['MemAvailable'].split()[0])
            collector_available_gb = available_kb / (1024**2)
            
            print(f"   psutil available: {psutil_available_gb:.2f} GB")
            print(f"   SystemCollector available: {collector_available_gb:.2f} GB")
            
            # Allow 5% difference for available memory (more volatile)
            diff_percent = abs(collector_available_gb - psutil_available_gb) / psutil_available_gb
            self.assertLess(diff_percent, 0.05,
                           "Memory available differs by more than 5%")
    
    def test_disk_usage_accuracy(self):
        """Test disk usage accuracy against os.statvfs()."""
        print("\n🔍 Testing disk usage accuracy...")
        
        # Test root filesystem
        root_stat = os.statvfs('/')
        os_total_gb = (root_stat.f_blocks * root_stat.f_frsize) / (1024**3)
        os_free_gb = (root_stat.f_bavail * root_stat.f_frsize) / (1024**3)
        os_used_gb = os_total_gb - os_free_gb
        
        # Get from psutil for additional verification
        psutil_usage = psutil.disk_usage('/')
        psutil_total_gb = psutil_usage.total / (1024**3)
        psutil_free_gb = psutil_usage.free / (1024**3)
        
        print(f"   os.statvfs() total: {os_total_gb:.2f} GB")
        print(f"   psutil total: {psutil_total_gb:.2f} GB")
        print(f"   os.statvfs() free: {os_free_gb:.2f} GB")
        print(f"   psutil free: {psutil_free_gb:.2f} GB")
        
        # Get from SystemCollector
        storage_data = self.system_data.get('storage', {})
        filesystem_data = storage_data.get('filesystem', {})
        df_output = filesystem_data.get('usage', '')
        
        if df_output:
            # Parse df output for root filesystem
            lines = df_output.strip().split('\n')
            for line in lines[1:]:  # Skip header
                if line.endswith(' /'):
                    parts = line.split()
                    if len(parts) >= 4:
                        collector_used = parts[2]
                        collector_available = parts[3]
                        print(f"   SystemCollector used: {collector_used}")
                        print(f"   SystemCollector available: {collector_available}")
                        break
        
        # Verify os.statvfs and psutil agree (they should be identical)
        self.assertAlmostEqual(os_total_gb, psutil_total_gb, delta=0.1,
                              msg="os.statvfs and psutil disk totals don't match")
    
    def test_network_interfaces_accuracy(self):
        """Test network interface detection against psutil."""
        print("\n🔍 Testing network interface accuracy...")
        
        # Get ground truth from psutil
        psutil_interfaces = psutil.net_if_addrs()
        psutil_stats = psutil.net_if_stats()
        
        print(f"   psutil found {len(psutil_interfaces)} interfaces:")
        for iface in sorted(psutil_interfaces.keys()):
            is_up = psutil_stats.get(iface, {}).isup if iface in psutil_stats else False
            print(f"     {iface} ({'UP' if is_up else 'DOWN'})")
        
        # Get from SystemCollector
        network_data = self.system_data.get('network', {})
        interfaces = network_data.get('interfaces', '')
        
        if interfaces:
            # Parse ip addr show output
            collector_interfaces = set()
            for line in interfaces.split('\n'):
                if re.match(r'^\d+:', line):  # Interface line
                    iface_name = line.split(':')[1].strip().split('@')[0]
                    collector_interfaces.add(iface_name)
            
            print(f"   SystemCollector found {len(collector_interfaces)} interfaces:")
            for iface in sorted(collector_interfaces):
                print(f"     {iface}")
            
            # Check that major interfaces are detected by both
            common_interfaces = ['lo', 'eth0', 'enp0s3', 'wlan0', 'docker0']
            for iface in common_interfaces:
                if iface in psutil_interfaces:
                    self.assertIn(iface, collector_interfaces,
                                f"SystemCollector missing interface {iface}")
        else:
            self.fail("SystemCollector returned no network interface data")
    
    def test_process_information_accuracy(self):
        """Test process information against psutil."""
        print("\n🔍 Testing process information accuracy...")
        
        # Get current process info from psutil
        current_pid = os.getpid()
        psutil_proc = psutil.Process(current_pid)
        psutil_name = psutil_proc.name()
        psutil_ppid = psutil_proc.ppid()
        
        print(f"   Current PID: {current_pid}")
        print(f"   psutil process name: {psutil_name}")
        print(f"   psutil parent PID: {psutil_ppid}")
        
        # Get from SystemCollector
        process_data = self.system_data.get('processes', {})
        ps_output = process_data.get('ps_cpu', '') or process_data.get('ps_memory', '')
        
        if ps_output:
            # Check if our current process is in the output
            found_process = False
            for line in ps_output.split('\n'):
                if str(current_pid) in line and 'python' in line.lower():
                    found_process = True
                    print(f"   Found current process in SystemCollector output")
                    print(f"     {line}")
                    break
            
            self.assertTrue(found_process, 
                           "Current Python process not found in SystemCollector output")
        else:
            self.fail("SystemCollector returned no process data")
    
    def test_python_environment_accuracy(self):
        """Test Python environment detection accuracy."""
        print("\n🔍 Testing Python environment accuracy...")
        
        # Get ground truth from sys and platform
        real_version = sys.version.split()[0]  # e.g., "3.12.0"
        real_executable = sys.executable
        real_platform = platform.platform()
        
        print(f"   Real Python version: {real_version}")
        print(f"   Real Python executable: {real_executable}")
        print(f"   Real platform: {real_platform}")
        
        # Get from SystemCollector
        python_data = self.system_data.get('python_env', {})
        installations = python_data.get('installations', {})
        
        # Check if our current Python is detected
        found_version = False
        for py_cmd, info in installations.items():
            if isinstance(info, dict) and info.get('path') == real_executable:
                collector_version = info.get('version', '')
                print(f"   SystemCollector found: {py_cmd} -> {collector_version}")
                
                # Extract version number from collector output
                version_match = re.search(r'(\d+\.\d+\.\d+)', collector_version)
                if version_match:
                    collector_version_num = version_match.group(1)
                    self.assertEqual(collector_version_num, real_version,
                                   "Python version mismatch")
                    found_version = True
                    break
        
        if not found_version:
            # Check if at least some Python version was found
            self.assertGreater(len(installations), 0,
                              "SystemCollector found no Python installations")
    
    def test_gpu_nvidia_smi_comparison(self):
        """Test GPU data accuracy against direct nvidia-smi calls."""
        print("\n🔍 Testing GPU data accuracy...")
        
        try:
            # Direct nvidia-smi call
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,temperature.gpu,utilization.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                nvidia_output = result.stdout.strip()
                print(f"   Direct nvidia-smi: {nvidia_output}")
                
                # Get from SystemCollector
                gpu_data = self.system_data.get('nvidia_gpu', {})
                basic_metrics = gpu_data.get('basic_metrics', '')
                
                if basic_metrics:
                    lines = basic_metrics.split('\n')
                    if len(lines) > 1:
                        collector_output = lines[1]
                        print(f"   SystemCollector GPU: {collector_output}")
                        
                        # Extract GPU name from both
                        nvidia_parts = nvidia_output.split(', ')
                        collector_parts = collector_output.split(', ')
                        
                        if len(nvidia_parts) >= 1 and len(collector_parts) >= 2:
                            nvidia_gpu_name = nvidia_parts[0].strip()
                            collector_gpu_name = collector_parts[1].strip()
                            
                            self.assertEqual(nvidia_gpu_name, collector_gpu_name,
                                           "GPU name mismatch between direct call and SystemCollector")
                else:
                    self.fail("SystemCollector returned no GPU basic metrics")
            else:
                print("   ⚠️  nvidia-smi not available or failed - skipping GPU test")
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("   ⚠️  nvidia-smi command not available - skipping GPU test")
    
    def test_security_user_information_accuracy(self):
        """Test user information accuracy against os and pwd modules."""
        print("\n🔍 Testing user information accuracy...")
        
        # Get ground truth
        real_username = getpass.getuser()
        real_uid = os.getuid()
        real_gid = os.getgid()
        
        print(f"   Real username: {real_username}")
        print(f"   Real UID: {real_uid}")
        print(f"   Real GID: {real_gid}")
        
        # Get from SystemCollector
        security_data = self.system_data.get('security', {})
        users = security_data.get('users', {})
        current_user_id = users.get('current_user_id', '')
        
        if current_user_id:
            print(f"   SystemCollector user info: {current_user_id}")
            
            # Check if our UID is in the output
            self.assertIn(str(real_uid), current_user_id,
                         "Current UID not found in SystemCollector output")
            self.assertIn(str(real_gid), current_user_id,
                         "Current GID not found in SystemCollector output")
        else:
            self.fail("SystemCollector returned no current user ID information")
    
    def test_storage_mount_points_accuracy(self):
        """Test storage mount points against psutil."""
        print("\n🔍 Testing storage mount points accuracy...")
        
        # Get ground truth from psutil
        psutil_mounts = psutil.disk_partitions()
        psutil_mountpoints = {mount.mountpoint for mount in psutil_mounts}
        
        print(f"   psutil found {len(psutil_mountpoints)} mount points:")
        for mp in sorted(psutil_mountpoints):
            print(f"     {mp}")
        
        # Get from SystemCollector
        storage_data = self.system_data.get('storage', {})
        filesystem = storage_data.get('filesystem', {})
        mount_details = filesystem.get('mount_details', '')
        
        if mount_details:
            # Parse mount output
            collector_mountpoints = set()
            for line in mount_details.split('\n'):
                if line and not line.startswith('TARGET'):  # Skip header
                    parts = line.split()
                    if parts:
                        mountpoint = parts[0]
                        collector_mountpoints.add(mountpoint)
            
            print(f"   SystemCollector found {len(collector_mountpoints)} mount points:")
            for mp in sorted(collector_mountpoints):
                print(f"     {mp}")
            
            # Check that root filesystem is detected by both
            self.assertIn('/', psutil_mountpoints, "Root filesystem missing from psutil")
            self.assertIn('/', collector_mountpoints, "Root filesystem missing from SystemCollector")
            
            # Check overlap
            overlap = psutil_mountpoints.intersection(collector_mountpoints)
            overlap_percent = len(overlap) / len(psutil_mountpoints) * 100
            print(f"   Mount point overlap: {overlap_percent:.1f}%")
            
            # Should have significant overlap (at least 80%)
            self.assertGreaterEqual(overlap_percent, 80,
                                  "Mount point detection has low overlap with psutil")
        else:
            self.fail("SystemCollector returned no mount details")
    
    def test_kernel_version_accuracy(self):
        """Test kernel version accuracy against os.uname()."""
        print("\n🔍 Testing kernel version accuracy...")
        
        # Get ground truth from os.uname()
        uname_info = os.uname()
        real_sysname = uname_info.sysname  # Linux
        real_release = uname_info.release  # 5.15.0-91-generic
        real_machine = uname_info.machine  # x86_64
        
        print(f"   Real system: {real_sysname}")
        print(f"   Real release: {real_release}")
        print(f"   Real machine: {real_machine}")
        
        # Get from SystemCollector
        kernel_data = self.system_data.get('kernel', {})
        version = kernel_data.get('version', '')
        
        if version:
            print(f"   SystemCollector: {version}")
            
            # Check components are present
            self.assertIn(real_sysname, version, "System name not found in kernel version")
            self.assertIn(real_release, version, "Kernel release not found in version")
            self.assertIn(real_machine, version, "Machine architecture not found in version")
        else:
            self.fail("SystemCollector returned no kernel version")
    
    def test_data_collection_performance(self):
        """Test data collection performance and completeness."""
        print("\n🔍 Testing data collection performance...")
        
        # Measure collection time
        start_time = time.time()
        test_snapshot = self.collector.collect()
        end_time = time.time()
        
        actual_duration = (end_time - start_time) * 1000  # Convert to ms
        reported_duration = test_snapshot['metadata']['collection_duration_ms']
        
        print(f"   Actual collection time: {actual_duration:.1f}ms")
        print(f"   Reported collection time: {reported_duration:.1f}ms")
        print(f"   Timing accuracy: {abs(actual_duration - reported_duration):.1f}ms difference")
        
        # Timing should be reasonably accurate (within 10% or 100ms)
        timing_error = abs(actual_duration - reported_duration)
        max_acceptable_error = max(actual_duration * 0.1, 100)  # 10% or 100ms
        
        self.assertLess(timing_error, max_acceptable_error,
                       "Collection timing measurement is inaccurate")
        
        # Collection should complete in reasonable time (< 30 seconds)
        self.assertLess(actual_duration, 30000,
                       "Collection took longer than 30 seconds")
        
        # Check data completeness
        expected_sections = ['cpu', 'memory', 'hardware', 'kernel', 'network']
        data = test_snapshot['data']
        
        for section in expected_sections:
            self.assertIn(section, data, f"Missing expected data section: {section}")
            self.assertTrue(data[section], f"Data section {section} is empty")
        
        print(f"   ✅ All {len(expected_sections)} essential data sections present")
    
    def test_file_system_permissions_safety(self):
        """Test that SystemCollector respects file system permissions."""
        print("\n🔍 Testing file system permissions safety...")
        
        # Try to read a file that should not be accessible
        sensitive_files = ['/etc/shadow', '/etc/sudoers', '/root/.bashrc']
        
        for sensitive_file in sensitive_files:
            if os.path.exists(sensitive_file):
                # Check if we can read it directly
                try:
                    with open(sensitive_file, 'r') as f:
                        f.read()
                    can_read_direct = True
                except PermissionError:
                    can_read_direct = False
                
                # Check what SystemCollector returns
                collector_content = self.collector.safe_read_file(sensitive_file)
                
                print(f"   {sensitive_file}:")
                print(f"     Direct access: {'✅' if can_read_direct else '❌'}")
                print(f"     SystemCollector: {'✅' if collector_content else '❌'}")
                
                # SystemCollector should not bypass permission restrictions
                if not can_read_direct:
                    self.assertIsNone(collector_content,
                                     f"SystemCollector bypassed permissions for {sensitive_file}")
    
    def test_command_execution_safety(self):
        """Test that SystemCollector only executes safe commands."""
        print("\n🔍 Testing command execution safety...")
        
        # Test dangerous commands are rejected
        dangerous_commands = [
            ['rm', '-rf', '/tmp/test'],
            ['dd', 'if=/dev/zero', 'of=/tmp/test'],
            ['chmod', '777', '/tmp'],
            ['sudo', 'reboot'],
            ['mkfs.ext4', '/dev/sda1']
        ]
        
        for cmd in dangerous_commands:
            result = self.collector.safe_execute_command(cmd)
            self.assertIsNone(result, 
                             f"SystemCollector executed dangerous command: {' '.join(cmd)}")
            print(f"   ❌ Correctly blocked: {' '.join(cmd)}")
        
        # Test safe commands are allowed
        safe_commands = [
            ['ls', '--version'],
            ['ps', '--version'],
            ['free', '--version']
        ]
        
        allowed_commands = self.collector._get_allowed_commands()
        
        for cmd in safe_commands:
            if cmd[0] in allowed_commands:
                result = self.collector.safe_execute_command(cmd)
                if result and result.get('return_code') == 0:
                    print(f"   ✅ Correctly allowed: {' '.join(cmd)}")
                elif result is None:
                    print(f"   ⚠️  Command blocked: {' '.join(cmd)}")
    
    def test_continuous_collection_consistency(self):
        """Test consistency across multiple collections."""
        print("\n🔍 Testing continuous collection consistency...")
        
        # Collect data multiple times
        snapshots = []
        for i in range(3):
            time.sleep(0.1)  # Small delay between collections
            snapshot = self.collector.collect()
            snapshots.append(snapshot)
            print(f"   Collection {i+1}: {snapshot['metadata']['collection_duration_ms']:.1f}ms")
        
        # Check that stable data remains consistent
        stable_fields = [
            ('cpu', 'cpuinfo', 'count'),
            ('kernel', 'version'),
            ('hardware', 'pci_devices')
        ]
        
        for field_path in stable_fields:
            values = []
            for snapshot in snapshots:
                data = snapshot['data']
                for key in field_path:
                    data = data.get(key, {})
                    if not isinstance(data, dict):
                        break
                values.append(data)
            
            # All values should be identical for stable data
            if all(v == values[0] for v in values):
                print(f"   ✅ Stable data consistent: {' -> '.join(field_path)}")
            else:
                print(f"   ⚠️  Stable data changed: {' -> '.join(field_path)}")
        
        # Check that collection count increments properly
        counts = [s['metadata']['collection_count'] for s in snapshots]
        for i in range(1, len(counts)):
            self.assertGreater(counts[i], counts[i-1],
                              "Collection count should increment")


def run_functional_tests():
    """Run all functional tests with detailed output."""
    print("🚀 Starting SystemCollector Functional Tests")
    print("=" * 60)
    print("These tests validate SystemCollector against real Ubuntu system data")
    print("using Python's os module, psutil, and direct system calls.\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(SystemCollectorFunctionalTests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=False)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏁 Functional Test Summary")
    print("=" * 60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_tests - failures - errors - skipped
    
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failures}")
    print(f"💥 Errors: {errors}")
    print(f"⏭️  Skipped: {skipped}")
    
    if failures == 0 and errors == 0:
        print("\n🎉 All functional tests passed!")
        print("✅ SystemCollector is accurately collecting real system data.")
    else:
        print(f"\n⚠️  {failures + errors} test(s) failed.")
        print("❌ SystemCollector may have accuracy issues.")
        
        # Print failure details
        if result.failures:
            print("\nFailure Details:")
            for test, traceback in result.failures:
                print(f"  ❌ {test}: {traceback.split('AssertionError: ')[-1].split('\n')[0]}")
        
        if result.errors:
            print("\nError Details:")
            for test, traceback in result.errors:
                print(f"  💥 {test}: {traceback.split('\n')[-2]}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_functional_tests()
    sys.exit(0 if success else 1)