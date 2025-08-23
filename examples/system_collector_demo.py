#!/usr/bin/env python3
"""
SystemCollector Demo Script
===========================

Demonstrates practical usage of the SystemCollector for Ubuntu LLM system monitoring.
This script shows various collection patterns and analysis techniques.

Usage:
    python examples/system_collector_demo.py [--mode MODE] [--output FORMAT]
    
Modes:
    - basic: Basic system information collection
    - gpu: Focus on RTX 5090 GPU analysis
    - storage: Storage performance and health analysis  
    - security: Security posture assessment
    - python: Python environment analysis
    - continuous: Continuous monitoring demo
    - all: Complete system analysis (default)
    
Output formats:
    - console: Human-readable console output (default)
    - json: JSON format output
    - summary: Brief summary format
"""

import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.ubuntu_llm_system.data_collection.collectors.system_collector import SystemCollector


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n--- {title} ---")


def format_bytes(bytes_value: str) -> str:
    """Format bytes string to human readable."""
    try:
        if 'kB' in bytes_value:
            kb = int(bytes_value.replace('kB', '').strip())
            return f"{kb/1024/1024:.2f} GB"
        return bytes_value
    except:
        return bytes_value


def demo_basic_collection():
    """Demonstrate basic system information collection."""
    print_header("Basic System Collection Demo")
    
    collector = SystemCollector(collection_interval=30)
    
    print("Collecting system information...")
    snapshot = collector.collect()
    
    # Display metadata
    print_section("Collection Metadata")
    metadata = snapshot['metadata']
    print(f"Collector: {metadata['collector']}")
    print(f"Timestamp: {metadata['timestamp']}")
    print(f"Collection Duration: {metadata['collection_duration_ms']}ms")
    print(f"Data Hash: {metadata['data_hash']}")
    
    # Basic system info
    data = snapshot['data']
    
    print_section("System Overview")
    if 'kernel' in data and data['kernel'].get('version'):
        print(f"Kernel: {data['kernel']['version']}")
    
    if 'cpu' in data and data['cpu'].get('lscpu'):
        lscpu_lines = data['cpu']['lscpu'].split('\n')[:3]
        for line in lscpu_lines:
            if line.strip():
                print(f"CPU: {line.strip()}")
    
    if 'memory' in data and data['memory'].get('free'):
        free_lines = data['memory']['free'].split('\n')[:2]
        for line in free_lines:
            if line.strip() and not line.startswith('Mem:'):
                print(f"Memory: {line.strip()}")
    
    return snapshot


def demo_gpu_analysis(snapshot):
    """Demonstrate RTX 5090 GPU analysis."""
    print_header("RTX 5090 GPU Analysis Demo")
    
    gpu_data = snapshot['data'].get('nvidia_gpu', {})
    cuda_data = snapshot['data'].get('cuda', {})
    
    if not gpu_data.get('basic_metrics'):
        print("❌ No NVIDIA GPU data available")
        print("   - Check if NVIDIA drivers are installed")
        print("   - Verify nvidia-smi command works")
        print("   - Ensure user has access to GPU")
        return
    
    print_section("GPU Hardware Information")
    basic_metrics = gpu_data.get('basic_metrics', '')
    if basic_metrics:
        lines = basic_metrics.split('\n')
        if len(lines) > 1:  # Skip header
            gpu_info = lines[1].split(', ')
            if len(gpu_info) >= 3:
                print(f"📊 GPU: {gpu_info[1]}")  # GPU name
                print(f"🌡️  Temperature: {gpu_info[7]} °C")
                print(f"⚡ Power: {gpu_info[13]} / {gpu_info[14]}")
    
    print_section("CUDA Environment")
    if cuda_data.get('nvcc_version'):
        nvcc_lines = cuda_data['nvcc_version'].split('\n')
        for line in nvcc_lines:
            if 'release' in line.lower():
                print(f"🔧 NVCC: {line.strip()}")
                break
    
    if cuda_data.get('compute_capability'):
        print(f"💻 Compute Capability: {cuda_data['compute_capability']}")
    
    print_section("GPU Processes")
    if gpu_data.get('compute_processes'):
        processes = gpu_data['compute_processes']
        if processes.strip():
            print("🔄 Active GPU Processes:")
            for line in processes.split('\n'):
                if line.strip() and not line.startswith('pid'):
                    print(f"   {line}")
        else:
            print("✅ No active GPU processes")
    
    print_section("Performance Samples")
    if gpu_data.get('utilization_samples'):
        print("📈 Recent GPU Utilization:")
        samples = gpu_data['utilization_samples'].split('\n')[1:]  # Skip header
        for i, sample in enumerate(samples[:3]):  # Show first 3 samples
            if sample.strip():
                parts = sample.split(', ')
                if len(parts) >= 5:
                    print(f"   Sample {i+1}: GPU {parts[0]}%, Memory {parts[1]}%, Temp {parts[3]}°C")


def demo_storage_analysis(snapshot):
    """Demonstrate comprehensive storage analysis."""
    print_header("Storage Performance & Health Analysis")
    
    storage_data = snapshot['data'].get('storage', {})
    
    print_section("Storage Hardware")
    hardware = storage_data.get('hardware', {})
    if hardware.get('block_devices_detailed'):
        print("💾 Block Devices:")
        lines = hardware['block_devices_detailed'].split('\n')[:6]  # First few lines
        for line in lines:
            if line.strip():
                print(f"   {line}")
    
    if hardware.get('nvme_devices'):
        print("\n⚡ NVMe Devices:")
        lines = hardware['nvme_devices'].split('\n')[:3]
        for line in lines:
            if line.strip() and 'nvme' in line:
                print(f"   {line}")
    
    print_section("Storage Performance")
    performance = storage_data.get('performance', {})
    if performance.get('iostat_extended'):
        print("📊 I/O Statistics (Recent):")
        lines = performance['iostat_extended'].split('\n')
        for line in lines[-10:]:  # Show last 10 lines
            if line.strip() and any(x in line for x in ['nvme', 'sda', 'Device']):
                print(f"   {line}")
    
    print_section("AI/LLM Storage Analysis")
    ai_storage = storage_data.get('ai_storage', {})
    if ai_storage.get('model_storage'):
        model_storage = ai_storage['model_storage']
        for path, info in model_storage.items():
            if 'large_files' in path and info.strip():
                print(f"🤖 Large Model Files in {path.split('_')[0]}:")
                lines = info.split('\n')[:5]  # First 5 files
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 9:
                            print(f"   {parts[8]} ({parts[4]})")
    
    print_section("Storage Health")
    health = storage_data.get('health', {})
    if health.get('smart_data'):
        smart_data = health['smart_data']
        devices_with_smart = [k for k in smart_data.keys() if not k.endswith('_health')]
        print(f"🔍 SMART monitoring active on {len(devices_with_smart)} devices")
        
        for device in devices_with_smart[:3]:  # First 3 devices
            if smart_data[device]:
                print(f"   {device}: SMART data available")


def demo_security_analysis(snapshot):
    """Demonstrate security posture analysis."""
    print_header("Security Posture Analysis")
    
    security_data = snapshot['data'].get('security', {})
    
    print_section("User Security")
    users = security_data.get('users', {})
    if users.get('current_users'):
        print("👥 Currently Logged In:")
        lines = users['current_users'].split('\n')[:5]
        for line in lines:
            if line.strip():
                print(f"   {line}")
    
    print_section("Network Security")
    network = security_data.get('network', {})
    if network.get('listening_ports'):
        print("🔌 Listening Ports:")
        lines = network['listening_ports'].split('\n')[:10]
        for line in lines:
            if line.strip() and 'LISTEN' in line:
                print(f"   {line}")
    
    print_section("Firewall Status")
    firewall = security_data.get('firewall', {})
    if firewall.get('ufw_status'):
        print("🛡️  UFW Firewall:")
        lines = firewall['ufw_status'].split('\n')[:5]
        for line in lines:
            if line.strip():
                print(f"   {line}")
    
    print_section("File Permissions")
    permissions = security_data.get('permissions', {})
    if permissions.get('suid_files'):
        suid_count = len([l for l in permissions['suid_files'].split('\n') if l.strip()])
        print(f"⚠️  SUID Files Found: {suid_count}")
    
    if permissions.get('world_writable'):
        writable_count = len([l for l in permissions['world_writable'].split('\n') if l.strip()])
        print(f"⚠️  World-Writable Files: {writable_count}")


def demo_python_analysis(snapshot):
    """Demonstrate Python environment analysis."""
    print_header("Python Environment Analysis")
    
    python_data = snapshot['data'].get('python_env', {})
    
    print_section("Python Installations")
    installations = python_data.get('installations', {})
    for py_cmd, info in installations.items():
        if isinstance(info, dict) and info.get('version'):
            print(f"🐍 {py_cmd}: {info['version']} at {info['path']}")
    
    print_section("AI/ML Frameworks")
    frameworks = python_data.get('ai_frameworks', {})
    if frameworks:
        print("🤖 AI/ML Framework Versions:")
        key_frameworks = ['torch', 'transformers', 'datasets', 'accelerate', 'peft']
        for fw in key_frameworks:
            version = frameworks.get(fw, 'Not installed')
            status = "✅" if version != 'Not installed' else "❌"
            print(f"   {status} {fw}: {version}")
    
    print_section("CUDA Integration")
    if frameworks.get('torch_cuda'):
        print("⚡ PyTorch CUDA Status:")
        lines = frameworks['torch_cuda'].split('\n')
        for line in lines:
            if line.strip():
                print(f"   {line}")
    
    print_section("Virtual Environments")
    venvs = python_data.get('virtual_envs', {})
    if venvs.get('virtual_environments'):
        env_count = len(venvs['virtual_environments'])
        print(f"📦 Virtual Environments Found: {env_count}")
        for env_path in list(venvs['virtual_environments'].keys())[:3]:
            print(f"   {env_path}")
    
    # Package security check
    packages = python_data.get('packages', {})
    if packages.get('pip_security'):
        security_lines = packages['pip_security'].split('\n')
        if any('vulnerability' in line.lower() for line in security_lines):
            print("⚠️  Security vulnerabilities found in packages!")
        else:
            print("✅ No known security vulnerabilities in packages")


def demo_continuous_monitoring():
    """Demonstrate continuous monitoring pattern."""
    print_header("Continuous Monitoring Demo")
    
    collector = SystemCollector(collection_interval=5)  # 5 second intervals
    
    print("Starting continuous monitoring (press Ctrl+C to stop)...")
    print("Collecting system metrics every 5 seconds...\n")
    
    try:
        iteration = 0
        while True:
            iteration += 1
            
            if collector.should_collect():
                snapshot = collector.collect()
                data = snapshot['data']
                
                # Extract key metrics
                timestamp = datetime.now().strftime("%H:%M:%S")
                duration = snapshot['metadata']['collection_duration_ms']
                
                # CPU load
                cpu_load = "N/A"
                if data.get('cpu', {}).get('loadavg'):
                    cpu_load = data['cpu']['loadavg'][0]
                
                # Memory usage
                mem_usage = "N/A"
                if data.get('memory', {}).get('meminfo'):
                    meminfo = data['memory']['meminfo']
                    if 'MemTotal' in meminfo and 'MemAvailable' in meminfo:
                        total = int(meminfo['MemTotal'].split()[0])
                        available = int(meminfo['MemAvailable'].split()[0])
                        used_pct = ((total - available) / total) * 100
                        mem_usage = f"{used_pct:.1f}%"
                
                # GPU temperature
                gpu_temp = "N/A"
                if data.get('nvidia_gpu', {}).get('basic_metrics'):
                    lines = data['nvidia_gpu']['basic_metrics'].split('\n')
                    if len(lines) > 1:
                        parts = lines[1].split(', ')
                        if len(parts) >= 8:
                            gpu_temp = f"{parts[7]}°C"
                
                print(f"[{timestamp}] Iteration {iteration:2d} | "
                      f"Collection: {duration:4.0f}ms | "
                      f"CPU Load: {cpu_load:>6s} | "
                      f"Memory: {mem_usage:>6s} | "
                      f"GPU: {gpu_temp:>6s}")
            
            time.sleep(1)  # Check every second
            
    except KeyboardInterrupt:
        print(f"\n\nContinuous monitoring stopped after {iteration} iterations.")
        print("✅ Monitoring completed successfully!")


def output_json(snapshot, output_file=None):
    """Output snapshot data in JSON format."""
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)
        print(f"✅ JSON output saved to: {output_file}")
    else:
        print(json.dumps(snapshot, indent=2, default=str))


def output_summary(snapshot):
    """Output a brief summary of the collected data."""
    print_header("System Summary")
    
    data = snapshot['data']
    metadata = snapshot['metadata']
    
    print(f"📊 Collection completed in {metadata['collection_duration_ms']}ms")
    print(f"🕒 Timestamp: {metadata['timestamp']}")
    
    # Count available data sections
    available_sections = [k for k, v in data.items() if v]
    print(f"📦 Data sections collected: {len(available_sections)}")
    
    # Key metrics
    if data.get('cpu', {}).get('cpuinfo', {}).get('count'):
        print(f"🔧 CPU Cores: {data['cpu']['cpuinfo']['count']}")
    
    if data.get('memory', {}).get('meminfo', {}).get('MemTotal'):
        total_mem = format_bytes(data['memory']['meminfo']['MemTotal'])
        print(f"💾 Total Memory: {total_mem}")
    
    if data.get('nvidia_gpu', {}).get('basic_metrics'):
        lines = data['nvidia_gpu']['basic_metrics'].split('\n')
        if len(lines) > 1:
            parts = lines[1].split(', ')
            if len(parts) >= 2:
                print(f"🎮 GPU: {parts[1]}")
    
    if data.get('security', {}).get('users', {}).get('current_users'):
        user_count = len([l for l in data['security']['users']['current_users'].split('\n') if l.strip()])
        print(f"👥 Active Users: {user_count}")
    
    if data.get('python_env', {}).get('ai_frameworks', {}).get('torch'):
        torch_version = data['python_env']['ai_frameworks']['torch']
        print(f"🤖 PyTorch: {torch_version}")


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description='SystemCollector Demo Script')
    parser.add_argument('--mode', choices=['basic', 'gpu', 'storage', 'security', 
                                          'python', 'continuous', 'all'], 
                       default='all', help='Demo mode to run')
    parser.add_argument('--output', choices=['console', 'json', 'summary'], 
                       default='console', help='Output format')
    parser.add_argument('--output-file', help='Output file for JSON format')
    
    args = parser.parse_args()
    
    print("🚀 SystemCollector Demo Script")
    print(f"Mode: {args.mode} | Output: {args.output}")
    
    if args.mode == 'continuous':
        demo_continuous_monitoring()
        return
    
    # Collect data once for most demos
    print("\n🔍 Collecting system data...")
    snapshot = None
    
    try:
        if args.mode == 'basic':
            snapshot = demo_basic_collection()
        else:
            collector = SystemCollector(collection_interval=30)
            snapshot = collector.collect()
            print(f"✅ Collection completed in {snapshot['metadata']['collection_duration_ms']}ms")
            
            if args.mode in ['all', 'gpu']:
                demo_gpu_analysis(snapshot)
            
            if args.mode in ['all', 'storage']:
                demo_storage_analysis(snapshot)
            
            if args.mode in ['all', 'security']:
                demo_security_analysis(snapshot)
            
            if args.mode in ['all', 'python']:
                demo_python_analysis(snapshot)
    
    except KeyboardInterrupt:
        print("\n❌ Demo interrupted by user")
        return
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Handle output format
    if snapshot:
        if args.output == 'json':
            output_json(snapshot, args.output_file)
        elif args.output == 'summary':
            output_summary(snapshot)
    
    print("\n✅ Demo completed successfully!")


if __name__ == '__main__':
    main()