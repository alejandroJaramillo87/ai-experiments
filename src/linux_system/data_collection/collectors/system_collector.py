"""
System Collector
===============

Collects comprehensive Ubuntu system state data for LLM optimization insights:

- CPU information, topology, and real-time usage
- Memory usage, NUMA topology, and allocation patterns  
- NVIDIA RTX 5090 GPU detailed metrics and compute utilization
- Comprehensive storage I/O, performance, and intelligence
- Hardware information and thermal data
- Kernel parameters and system limits
- Network interfaces and routing
- Filesystem usage and mount points
- Docker NVIDIA runtime integration

Particularly focused on data useful for LLM inference optimization.
"""

import re
import json
import subprocess
from typing import Dict, Any, List, Optional
from pathlib import Path
from .base_collector import BaseCollector


class SystemCollector(BaseCollector):
    """Collects system state data from /proc, /sys, and system commands with RTX 5090 focus."""
    
    def __init__(self, collection_interval: int = 30):
        """Initialize system collector with 30-second default interval."""
        super().__init__("system", collection_interval)
        
    def _get_allowed_commands(self) -> List[str]:
        """Return allowed read-only system commands."""
        return [
            'lscpu', 'free', 'df', 'lsblk', 'ip', 'ss', 'ps', 'top',
            'uptime', 'uname', 'lshw', 'lspci', 'lsusb', 'dmidecode',
            'numactl', 'vmstat', 'iostat', 'sar', 'sensors', 'systemctl', 
            'mount', 'fdisk',
            # NVIDIA RTX 5090 specific commands
            'nvidia-smi', 'nvtop', 'nvidia-settings', 'nvcc', 'nvidia-ml-py',
            'nvidia-persistenced', 'nvidia-modprobe', 'nvidia-bug-report.sh',
            'nsight-compute', 'nsight-systems', 'docker',
            # Enhanced storage and I/O commands
            'smartctl', 'hdparm', 'sdparm', 'lsscsi', 'blkid',
            'iotop', 'dstat', 'fio', 'tune2fs', 'dumpe2fs',
            'nvme', 'sg_inq', 'sg_logs', 'parted', 'gdisk', 'sfdisk',
            'findmnt', 'du', 'find', 'stat', 'file', 'ls', 'lsattr',
            # Security monitoring commands
            'ss', 'netstat', 'who', 'w', 'last', 'lastlog', 'faillog',
            'chage', 'passwd', 'getent', 'id', 'groups', 'sudo', 'visudo',
            'ufw', 'iptables', 'fail2ban-client', 'systemctl', 'journalctl',
            'ausearch', 'aureport', 'semanage', 'sestatus', 'aa-status',
            'lynis', 'rkhunter', 'chkrootkit', 'clamav', 'freshclam',
            # Python environment commands  
            'python', 'python3', 'pip', 'pip3', 'conda', 'pyenv', 'pipenv',
            'poetry', 'virtualenv', 'which', 'whereis', 'locate', 'updatedb'
        ]
    
    def collect(self) -> Dict[str, Any]:
        """
        Collect comprehensive system state data with RTX 5090 and storage focus.
        
        Returns:
            Dictionary with CPU, memory, GPU, storage, hardware, and system information
        """
        start_time = self._get_current_time_ms()
        
        data = {
            'cpu': self._collect_cpu_data(),
            'memory': self._collect_memory_data(),
            'nvidia_gpu': self._collect_nvidia_gpu_data(),
            'cuda': self._collect_cuda_environment(),
            'storage': self._collect_comprehensive_storage_data(),
            'security': self._collect_security_data(),              # Security monitoring
            'python_env': self._collect_python_environment_data(),  # Python environments
            'hardware': self._collect_hardware_data(),
            'kernel': self._collect_kernel_data(),
            'network': self._collect_network_data(),
            'processes': self._collect_process_data(),
            'performance': self._collect_performance_data(),
            'docker_nvidia': self._collect_docker_nvidia_data(),
        }
        
        # Add collection timing
        end_time = self._get_current_time_ms()
        
        snapshot = self.create_data_snapshot(data)
        snapshot['metadata']['collection_duration_ms'] = end_time - start_time
        
        return snapshot
    
    def _collect_comprehensive_storage_data(self) -> Dict[str, Any]:
        """
        Collect comprehensive storage and I/O data for LLM optimization analysis.
        
        Returns:
            Dictionary with storage hardware, performance, and intelligence data
        """
        storage_data = {
            'hardware': self._collect_storage_hardware(),
            'performance': self._collect_storage_performance(),
            'filesystem': self._collect_filesystem_detailed(),
            'io_patterns': self._collect_io_patterns(),
            'ai_storage': self._collect_ai_specific_storage(),
            'health': self._collect_storage_health(),
        }
        
        return storage_data
    
    def _collect_storage_hardware(self) -> Dict[str, Any]:
        """Collect storage hardware information and capabilities."""
        hardware_data = {}
        
        # Block device information with detailed specs
        lsblk_detailed = self.safe_execute_command(['lsblk', '-o', 'NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,UUID,MODEL,SERIAL,ROTA,DISC-GRAN,DISC-MAX,WSAME'])
        if lsblk_detailed and lsblk_detailed['return_code'] == 0:
            hardware_data['block_devices_detailed'] = lsblk_detailed['stdout']
        
        # NVMe device information
        nvme_list = self.safe_execute_command(['nvme', 'list'])
        if nvme_list and nvme_list['return_code'] == 0:
            hardware_data['nvme_devices'] = nvme_list['stdout']
        
        # SCSI devices
        lsscsi_result = self.safe_execute_command(['lsscsi'])
        if lsscsi_result and lsscsi_result['return_code'] == 0:
            hardware_data['scsi_devices'] = lsscsi_result['stdout']
        
        # Storage controller information from PCI
        lspci_storage = self.safe_execute_command(['lspci', '-v', '-s', '$(lspci | grep -i storage | cut -d" " -f1)'])
        if lspci_storage and lspci_storage['return_code'] == 0:
            hardware_data['storage_controllers'] = lspci_storage['stdout']
        
        # Device identifiers and UUIDs
        blkid_result = self.safe_execute_command(['blkid'])
        if blkid_result and blkid_result['return_code'] == 0:
            hardware_data['device_uuids'] = blkid_result['stdout']
        
        # Partition information
        parted_list = self.safe_execute_command(['parted', '-l'])
        if parted_list and parted_list['return_code'] == 0:
            hardware_data['partition_tables'] = parted_list['stdout']
        
        return hardware_data
    
    def _collect_storage_performance(self) -> Dict[str, Any]:
        """Collect real-time storage I/O performance metrics."""
        performance_data = {}
        
        # Enhanced I/O statistics with extended metrics
        iostat_extended = self.safe_execute_command(['iostat', '-x', '-d', '1', '3'])
        if iostat_extended and iostat_extended['return_code'] == 0:
            performance_data['iostat_extended'] = iostat_extended['stdout']
        
        # I/O statistics from /proc/diskstats
        diskstats = self.safe_read_file('/proc/diskstats')
        if diskstats:
            performance_data['diskstats'] = diskstats
        
        # Per-device I/O scheduler information
        io_schedulers = {}
        for device_path in Path('/sys/block').glob('sd*'):
            scheduler_file = device_path / 'queue' / 'scheduler'
            if scheduler_file.exists():
                scheduler = self.safe_read_file(str(scheduler_file))
                if scheduler:
                    io_schedulers[device_path.name] = scheduler.strip()
        performance_data['io_schedulers'] = io_schedulers
        
        # NVMe device performance and health
        nvme_performance = self._collect_nvme_performance()
        if nvme_performance:
            performance_data['nvme_performance'] = nvme_performance
        
        # Storage device temperatures
        storage_temps = self._collect_storage_temperatures()
        if storage_temps:
            performance_data['storage_temperatures'] = storage_temps
        
        # I/O wait and system impact
        io_pressure = self.safe_read_file('/proc/pressure/io')
        if io_pressure:
            performance_data['io_pressure'] = io_pressure
        
        return performance_data
    
    def _collect_filesystem_detailed(self) -> Dict[str, Any]:
        """Collect detailed filesystem usage and configuration."""
        fs_data = {}
        
        # Extended disk usage with inodes
        df_inodes = self.safe_execute_command(['df', '-i'])
        if df_inodes and df_inodes['return_code'] == 0:
            fs_data['inode_usage'] = df_inodes['stdout']
        
        # Detailed mount information
        findmnt_detailed = self.safe_execute_command(['findmnt', '-D'])
        if findmnt_detailed and findmnt_detailed['return_code'] == 0:
            fs_data['mount_details'] = findmnt_detailed['stdout']
        
        # Filesystem-specific information
        fs_info = {}
        
        # Get all mounted ext4/xfs filesystems
        mounts_data = self.safe_read_file('/proc/mounts')
        if mounts_data:
            for line in mounts_data.split('\n'):
                if 'ext4' in line or 'xfs' in line:
                    parts = line.split()
                    if len(parts) >= 3:
                        device, mountpoint, fstype = parts[0], parts[1], parts[2]
                        if fstype == 'ext4':
                            tune2fs_result = self.safe_execute_command(['tune2fs', '-l', device])
                            if tune2fs_result and tune2fs_result['return_code'] == 0:
                                fs_info[f"{device}_ext4"] = tune2fs_result['stdout']
        
        fs_data['filesystem_info'] = fs_info
        
        # Directory usage for AI/LLM relevant paths
        ai_directories = [
            '/mnt/ai-data/models/',
            '/mnt/ai-data/',
            '/var/lib/docker/',
            '/tmp/',
            '/home',
            '/'
        ]
        
        directory_usage = {}
        for dir_path in ai_directories:
            if Path(dir_path).exists():
                du_result = self.safe_execute_command(['du', '-sh', dir_path])
                if du_result and du_result['return_code'] == 0:
                    directory_usage[dir_path] = du_result['stdout'].strip()
        
        fs_data['directory_usage'] = directory_usage
        
        return fs_data
    
    def _collect_io_patterns(self) -> Dict[str, Any]:
        """Collect I/O patterns and access information."""
        io_data = {}
        
        # Process I/O statistics
        proc_io = {}
        proc_path = Path('/proc')
        
        # Get I/O stats for top processes
        for pid_dir in list(proc_path.glob('[0-9]*'))[:50]:  # Limit to top 50
            io_file = pid_dir / 'io'
            if io_file.exists():
                io_content = self.safe_read_file(str(io_file))
                if io_content:
                    proc_io[pid_dir.name] = io_content
        
        io_data['process_io'] = proc_io
        
        # System-wide I/O counters
        vmstat_io = self.safe_read_file('/proc/vmstat')
        if vmstat_io:
            # Extract I/O related counters
            io_counters = {}
            for line in vmstat_io.split('\n'):
                if any(keyword in line for keyword in ['pgpgin', 'pgpgout', 'pswpin', 'pswpout', 'pgfault', 'pgmajfault']):
                    parts = line.split()
                    if len(parts) >= 2:
                        io_counters[parts[0]] = parts[1]
            io_data['io_counters'] = io_counters
        
        return io_data
    
    def _collect_ai_specific_storage(self) -> Dict[str, Any]:
        """Collect AI/LLM specific storage information."""
        ai_storage = {}
        
        # Model storage analysis
        model_paths = ['/mnt/ai-data/models/', '/models/', '/app/models/']
        model_storage = {}
        
        for model_path in model_paths:
            path_obj = Path(model_path)
            if path_obj.exists():
                # Find large model files (>100MB)
                large_files = self.safe_execute_command(['find', str(path_obj), '-type', 'f', '-size', '+100M', '-exec', 'ls', '-lh', '{}', ';'])
                if large_files and large_files['return_code'] == 0:
                    model_storage[f"{model_path}_large_files"] = large_files['stdout']
                
                # Recent model access patterns
                recent_access = self.safe_execute_command(['find', str(path_obj), '-type', 'f', '-atime', '-7', '-exec', 'ls', '-ltu', '{}', ';'])
                if recent_access and recent_access['return_code'] == 0:
                    model_storage[f"{model_path}_recent_access"] = recent_access['stdout']
        
        ai_storage['model_storage'] = model_storage
        
        # Docker volume analysis
        docker_volumes = self.safe_execute_command(['docker', 'system', 'df', '-v'])
        if docker_volumes and docker_volumes['return_code'] == 0:
            ai_storage['docker_storage'] = docker_volumes['stdout']
        
        # Cache directory analysis
        cache_dirs = ['/tmp/', '/var/tmp/', '/var/cache/', '~/.cache/']
        cache_usage = {}
        
        for cache_dir in cache_dirs:
            cache_path = Path(cache_dir).expanduser()
            if cache_path.exists():
                du_result = self.safe_execute_command(['du', '-sh', str(cache_path)])
                if du_result and du_result['return_code'] == 0:
                    cache_usage[cache_dir] = du_result['stdout'].strip()
        
        ai_storage['cache_usage'] = cache_usage
        
        # Large file identification across the system
        large_files_system = self.safe_execute_command(['find', '/', '-type', 'f', '-size', '+1G', '-exec', 'ls', '-lh', '{}', ';', '2>/dev/null'])
        if large_files_system and large_files_system['return_code'] == 0:
            ai_storage['large_files_system'] = large_files_system['stdout']
        
        return ai_storage
    
    def _collect_storage_health(self) -> Dict[str, Any]:
        """Collect storage device health and SMART data."""
        health_data = {}
        
        # SMART data for all drives
        smart_data = {}
        
        # Get list of storage devices
        lsblk_result = self.safe_execute_command(['lsblk', '-d', '-n', '-o', 'NAME'])
        if lsblk_result and lsblk_result['return_code'] == 0:
            devices = lsblk_result['stdout'].strip().split('\n')
            
            for device in devices:
                device = device.strip()
                if device:
                    # Get SMART data
                    smart_result = self.safe_execute_command(['smartctl', '-A', f'/dev/{device}'])
                    if smart_result and smart_result['return_code'] == 0:
                        smart_data[device] = smart_result['stdout']
                    
                    # Get device health summary
                    smart_health = self.safe_execute_command(['smartctl', '-H', f'/dev/{device}'])
                    if smart_health and smart_health['return_code'] == 0:
                        smart_data[f"{device}_health"] = smart_health['stdout']
        
        health_data['smart_data'] = smart_data
        
        # NVMe specific health information
        nvme_health = self._collect_nvme_health()
        if nvme_health:
            health_data['nvme_health'] = nvme_health
        
        # Filesystem errors and issues
        dmesg_storage = self.safe_execute_command(['dmesg', '--level=err,warn', '--facility=kern', '--since=yesterday'])
        if dmesg_storage and dmesg_storage['return_code'] == 0:
            # Filter for storage-related errors
            storage_errors = []
            for line in dmesg_storage['stdout'].split('\n'):
                if any(keyword in line.lower() for keyword in ['ata', 'scsi', 'nvme', 'disk', 'storage', 'i/o', 'sector']):
                    storage_errors.append(line)
            health_data['storage_errors'] = '\n'.join(storage_errors)
        
        return health_data
    
    def _collect_nvme_performance(self) -> Dict[str, Any]:
        """Collect NVMe-specific performance metrics."""
        nvme_data = {}
        
        # List NVMe devices
        nvme_list = self.safe_execute_command(['nvme', 'list'])
        if nvme_list and nvme_list['return_code'] == 0:
            # Extract NVMe device paths
            for line in nvme_list['stdout'].split('\n'):
                if '/dev/nvme' in line:
                    device = line.split()[0]
                    
                    # Get NVMe ID information
                    nvme_id = self.safe_execute_command(['nvme', 'id-ctrl', device])
                    if nvme_id and nvme_id['return_code'] == 0:
                        nvme_data[f"{device}_id"] = nvme_id['stdout']
                    
                    # Get smart log
                    nvme_smart = self.safe_execute_command(['nvme', 'smart-log', device])
                    if nvme_smart and nvme_smart['return_code'] == 0:
                        nvme_data[f"{device}_smart"] = nvme_smart['stdout']
        
        return nvme_data
    
    def _collect_nvme_health(self) -> Dict[str, Any]:
        """Collect NVMe health and wear information."""
        nvme_health = {}
        
        # Get NVMe device list
        nvme_list = self.safe_execute_command(['nvme', 'list', '-o', 'json'])
        if nvme_list and nvme_list['return_code'] == 0:
            try:
                devices_data = json.loads(nvme_list['stdout'])
                if 'Devices' in devices_data:
                    for device_info in devices_data['Devices']:
                        device_path = device_info.get('DevicePath', '')
                        
                        # Get detailed smart information
                        smart_detailed = self.safe_execute_command(['nvme', 'smart-log', device_path, '--output-format=json'])
                        if smart_detailed and smart_detailed['return_code'] == 0:
                            nvme_health[f"{device_path}_smart"] = smart_detailed['stdout']
                        
                        # Get wear leveling information if available
                        wear_info = self.safe_execute_command(['nvme', 'get-log', device_path, '--log-id=0xca'])
                        if wear_info and wear_info['return_code'] == 0:
                            nvme_health[f"{device_path}_wear"] = wear_info['stdout']
                            
            except json.JSONDecodeError:
                pass
        
        return nvme_health
    
    def _collect_storage_temperatures(self) -> Dict[str, Any]:
        """Collect storage device temperature information."""
        temp_data = {}
        
        # HDD/SSD temperatures via hddtemp (if available)
        hddtemp_result = self.safe_execute_command(['hddtemp', '/dev/sd*'])
        if hddtemp_result:  # May fail if not installed, that's OK
            temp_data['hdd_temperatures'] = hddtemp_result['stdout']
        
        # NVMe temperatures via smartctl
        lsblk_result = self.safe_execute_command(['lsblk', '-d', '-n', '-o', 'NAME'])
        if lsblk_result and lsblk_result['return_code'] == 0:
            devices = lsblk_result['stdout'].strip().split('\n')
            
            device_temps = {}
            for device in devices:
                device = device.strip()
                if device.startswith('nvme'):
                    smart_temp = self.safe_execute_command(['smartctl', '-A', f'/dev/{device}', '--log=temperature'])
                    if smart_temp and smart_temp['return_code'] == 0:
                        device_temps[device] = smart_temp['stdout']
            
            temp_data['device_temperatures'] = device_temps
        
        return temp_data

    def _collect_security_data(self) -> Dict[str, Any]:
        """
        Collect comprehensive system security information and posture analysis.
        
        Returns:
            Dictionary with user accounts, permissions, network security, and audit data
        """
        security_data = {
            'users': self._collect_user_security(),
            'network': self._collect_network_security(),
            'authentication': self._collect_authentication_data(),
            'permissions': self._collect_permissions_data(),
            'audit': self._collect_audit_logs(),
            'firewall': self._collect_firewall_data(),
            'processes': self._collect_security_processes(),
            'integrity': self._collect_system_integrity(),
        }
        
        return security_data
    
    def _collect_user_security(self) -> Dict[str, Any]:
        """Collect user account and access information."""
        user_data = {}
        
        # Current logged in users
        who_result = self.safe_execute_command(['who', '-a'])
        if who_result and who_result['return_code'] == 0:
            user_data['current_users'] = who_result['stdout']
        
        # User login history
        last_result = self.safe_execute_command(['last', '-n', '50'])
        if last_result and last_result['return_code'] == 0:
            user_data['login_history'] = last_result['stdout']
        
        # Failed login attempts
        lastlog_result = self.safe_execute_command(['lastlog'])
        if lastlog_result and lastlog_result['return_code'] == 0:
            user_data['last_logins'] = lastlog_result['stdout']
        
        # User account information
        passwd_data = self.safe_read_file('/etc/passwd')
        if passwd_data:
            user_data['user_accounts'] = passwd_data
        
        # Group information
        group_data = self.safe_read_file('/etc/group')
        if group_data:
            user_data['groups'] = group_data
        
        # Shadow file info (just basic stats, not contents)
        shadow_path = Path('/etc/shadow')
        if shadow_path.exists():
            shadow_stat = self.safe_execute_command(['stat', '/etc/shadow'])
            if shadow_stat and shadow_stat['return_code'] == 0:
                user_data['shadow_permissions'] = shadow_stat['stdout']
        
        # Sudo configuration
        sudoers_data = self.safe_read_file('/etc/sudoers')
        if sudoers_data:
            user_data['sudoers'] = sudoers_data
        
        # User ID and group memberships for current user
        id_result = self.safe_execute_command(['id'])
        if id_result and id_result['return_code'] == 0:
            user_data['current_user_id'] = id_result['stdout']
        
        return user_data
    
    def _collect_network_security(self) -> Dict[str, Any]:
        """Collect network security configuration and active connections."""
        network_security = {}
        
        # Active network connections with process info
        ss_detailed = self.safe_execute_command(['ss', '-tulpn'])
        if ss_detailed and ss_detailed['return_code'] == 0:
            network_security['active_connections'] = ss_detailed['stdout']
        
        # Network statistics
        netstat_stats = self.safe_execute_command(['ss', '-s'])
        if netstat_stats and netstat_stats['return_code'] == 0:
            network_security['connection_stats'] = netstat_stats['stdout']
        
        # Open ports
        open_ports = self.safe_execute_command(['ss', '-ln'])
        if open_ports and open_ports['return_code'] == 0:
            network_security['listening_ports'] = open_ports['stdout']
        
        # Network configuration files
        network_configs = {}
        
        # hosts file
        hosts_data = self.safe_read_file('/etc/hosts')
        if hosts_data:
            network_configs['hosts'] = hosts_data
        
        # Network interfaces config
        interfaces_data = self.safe_read_file('/etc/network/interfaces')
        if interfaces_data:
            network_configs['interfaces'] = interfaces_data
        
        # DNS configuration
        resolv_data = self.safe_read_file('/etc/resolv.conf')
        if resolv_data:
            network_configs['dns'] = resolv_data
        
        network_security['configs'] = network_configs
        
        return network_security
    
    def _collect_authentication_data(self) -> Dict[str, Any]:
        """Collect authentication and access control information."""
        auth_data = {}
        
        # PAM configuration
        pam_configs = {}
        pam_dir = Path('/etc/pam.d')
        if pam_dir.exists():
            for pam_file in ['common-auth', 'common-password', 'sshd', 'sudo']:
                pam_content = self.safe_read_file(pam_dir / pam_file)
                if pam_content:
                    pam_configs[pam_file] = pam_content
        
        auth_data['pam_config'] = pam_configs
        
        # SSH configuration
        ssh_configs = {}
        
        sshd_config = self.safe_read_file('/etc/ssh/sshd_config')
        if sshd_config:
            ssh_configs['sshd_config'] = sshd_config
        
        ssh_config = self.safe_read_file('/etc/ssh/ssh_config')
        if ssh_config:
            ssh_configs['ssh_config'] = ssh_config
        
        auth_data['ssh_config'] = ssh_configs
        
        # Login definitions
        login_defs = self.safe_read_file('/etc/login.defs')
        if login_defs:
            auth_data['login_defs'] = login_defs
        
        # Security limits
        limits_conf = self.safe_read_file('/etc/security/limits.conf')
        if limits_conf:
            auth_data['limits'] = limits_conf
        
        return auth_data
    
    def _collect_permissions_data(self) -> Dict[str, Any]:
        """Collect file permissions and access control data."""
        permissions_data = {}
        
        # Critical system file permissions
        critical_files = [
            '/etc/passwd', '/etc/shadow', '/etc/group', '/etc/sudoers',
            '/etc/ssh/sshd_config', '/etc/crontab', '/etc/fstab',
            '/boot', '/root', '/home'
        ]
        
        file_permissions = {}
        for file_path in critical_files:
            if Path(file_path).exists():
                ls_result = self.safe_execute_command(['ls', '-ld', file_path])
                if ls_result and ls_result['return_code'] == 0:
                    file_permissions[file_path] = ls_result['stdout']
        
        permissions_data['critical_files'] = file_permissions
        
        # SUID and SGID files (potential security risks)
        suid_files = self.safe_execute_command(['find', '/', '-type', 'f', '-perm', '-4000', '-ls', '2>/dev/null'])
        if suid_files and suid_files['return_code'] == 0:
            permissions_data['suid_files'] = suid_files['stdout']
        
        sgid_files = self.safe_execute_command(['find', '/', '-type', 'f', '-perm', '-2000', '-ls', '2>/dev/null'])
        if sgid_files and sgid_files['return_code'] == 0:
            permissions_data['sgid_files'] = sgid_files['stdout']
        
        # World writable files (security concern)
        world_writable = self.safe_execute_command(['find', '/', '-type', 'f', '-perm', '-002', '-ls', '2>/dev/null'])
        if world_writable and world_writable['return_code'] == 0:
            permissions_data['world_writable'] = world_writable['stdout']
        
        # Sticky bit directories
        sticky_dirs = self.safe_execute_command(['find', '/', '-type', 'd', '-perm', '-1000', '-ls', '2>/dev/null'])
        if sticky_dirs and sticky_dirs['return_code'] == 0:
            permissions_data['sticky_directories'] = sticky_dirs['stdout']
        
        return permissions_data
    
    def _collect_audit_logs(self) -> Dict[str, Any]:
        """Collect system audit and security logs."""
        audit_data = {}
        
        # System authentication logs
        auth_log = self.safe_read_file('/var/log/auth.log', max_lines=1000)
        if auth_log:
            audit_data['auth_log'] = auth_log
        
        # Secure log (RHEL/CentOS style)
        secure_log = self.safe_read_file('/var/log/secure', max_lines=1000)
        if secure_log:
            audit_data['secure_log'] = secure_log
        
        # Failed login attempts
        faillog_result = self.safe_execute_command(['faillog'])
        if faillog_result and faillog_result['return_code'] == 0:
            audit_data['failed_logins'] = faillog_result['stdout']
        
        # Journal security events
        journal_security = self.safe_execute_command(['journalctl', '--since', 'yesterday', '--priority=warning', '--no-pager'])
        if journal_security and journal_security['return_code'] == 0:
            audit_data['journal_warnings'] = journal_security['stdout']
        
        # Audit daemon logs (if auditd is running)
        audit_log_file = self.safe_read_file('/var/log/audit/audit.log', max_lines=500)
        if audit_log_file:
            audit_data['audit_daemon'] = audit_log_file
        
        return audit_data
    
    def _collect_firewall_data(self) -> Dict[str, Any]:
        """Collect firewall configuration and status."""
        firewall_data = {}
        
        # UFW status (Ubuntu Firewall)
        ufw_status = self.safe_execute_command(['ufw', 'status', 'verbose'])
        if ufw_status and ufw_status['return_code'] == 0:
            firewall_data['ufw_status'] = ufw_status['stdout']
        
        # iptables rules
        iptables_list = self.safe_execute_command(['iptables', '-L', '-n', '-v'])
        if iptables_list and iptables_list['return_code'] == 0:
            firewall_data['iptables_rules'] = iptables_list['stdout']
        
        # iptables NAT rules
        iptables_nat = self.safe_execute_command(['iptables', '-t', 'nat', '-L', '-n', '-v'])
        if iptables_nat and iptables_nat['return_code'] == 0:
            firewall_data['iptables_nat'] = iptables_nat['stdout']
        
        # fail2ban status (if installed)
        fail2ban_status = self.safe_execute_command(['fail2ban-client', 'status'])
        if fail2ban_status and fail2ban_status['return_code'] == 0:
            firewall_data['fail2ban_status'] = fail2ban_status['stdout']
        
        return firewall_data
    
    def _collect_security_processes(self) -> Dict[str, Any]:
        """Collect information about security-related processes."""
        security_processes = {}
        
        # Security services status
        security_services = ['ufw', 'fail2ban', 'auditd', 'apparmor', 'clamav-daemon', 'rkhunter']
        
        service_status = {}
        for service in security_services:
            systemctl_result = self.safe_execute_command(['systemctl', 'is-active', service])
            if systemctl_result:
                service_status[service] = systemctl_result['stdout'].strip()
        
        security_processes['service_status'] = service_status
        
        # AppArmor status (if available)
        apparmor_status = self.safe_execute_command(['aa-status'])
        if apparmor_status and apparmor_status['return_code'] == 0:
            security_processes['apparmor'] = apparmor_status['stdout']
        
        # SELinux status (if available)
        selinux_status = self.safe_execute_command(['sestatus'])
        if selinux_status and selinux_status['return_code'] == 0:
            security_processes['selinux'] = selinux_status['stdout']
        
        return security_processes
    
    def _collect_system_integrity(self) -> Dict[str, Any]:
        """Collect system integrity and security scanning information."""
        integrity_data = {}
        
        # Package integrity (dpkg verification)
        dpkg_verify = self.safe_execute_command(['dpkg', '--verify'])
        if dpkg_verify:  # Will return non-zero if issues found
            integrity_data['dpkg_verify'] = dpkg_verify['stdout']
        
        # System file checksums (if AIDE or similar is installed)
        aide_check = self.safe_execute_command(['aide', '--check'])
        if aide_check:
            integrity_data['aide_check'] = aide_check['stdout']
        
        # Check for common security tools installation
        security_tools = ['rkhunter', 'chkrootkit', 'clamav', 'lynis', 'aide']
        installed_tools = {}
        
        for tool in security_tools:
            which_result = self.safe_execute_command(['which', tool])
            if which_result and which_result['return_code'] == 0:
                installed_tools[tool] = which_result['stdout'].strip()
        
        integrity_data['security_tools'] = installed_tools
        
        return integrity_data

    def _collect_python_environment_data(self) -> Dict[str, Any]:
        """
        Collect comprehensive Python environment information for ML/AI development.
        
        Returns:
            Dictionary with Python installations, packages, and virtual environments
        """
        python_data = {
            'installations': self._collect_python_installations(),
            'packages': self._collect_python_packages(),
            'virtual_envs': self._collect_virtual_environments(),
            'conda': self._collect_conda_environment(),
            'poetry': self._collect_poetry_environment(),
            'system_packages': self._collect_system_python_packages(),
            'ai_frameworks': self._collect_ai_ml_frameworks(),
        }
        
        return python_data
    
    def _collect_python_installations(self) -> Dict[str, Any]:
        """Collect information about Python installations."""
        python_installs = {}
        
        # Find all Python executables
        python_commands = ['python', 'python3', 'python2', 'pypy', 'pypy3']
        
        for py_cmd in python_commands:
            which_result = self.safe_execute_command(['which', py_cmd])
            if which_result and which_result['return_code'] == 0:
                py_path = which_result['stdout'].strip()
                
                # Get version information
                version_result = self.safe_execute_command([py_cmd, '--version'])
                if version_result and version_result['return_code'] == 0:
                    version = version_result['stdout'].strip()
                else:
                    version = version_result['stderr'].strip() if version_result else 'Unknown'
                
                # Get detailed sys info
                sys_info = self.safe_execute_command([py_cmd, '-c', 
                    'import sys; print("Path:", sys.executable); print("Version:", sys.version); print("Platform:", sys.platform)'])
                if sys_info and sys_info['return_code'] == 0:
                    python_installs[py_cmd] = {
                        'path': py_path,
                        'version': version,
                        'sys_info': sys_info['stdout']
                    }
        
        # Check for pyenv installations
        pyenv_versions = self.safe_execute_command(['pyenv', 'versions'])
        if pyenv_versions and pyenv_versions['return_code'] == 0:
            python_installs['pyenv_versions'] = pyenv_versions['stdout']
        
        return python_installs
    
    def _collect_python_packages(self) -> Dict[str, Any]:
        """Collect installed Python packages across different environments."""
        packages_data = {}
        
        # pip packages for different Python versions
        pip_commands = ['pip', 'pip3', 'pip2']
        
        for pip_cmd in pip_commands:
            which_pip = self.safe_execute_command(['which', pip_cmd])
            if which_pip and which_pip['return_code'] == 0:
                # List installed packages
                pip_list = self.safe_execute_command([pip_cmd, 'list'])
                if pip_list and pip_list['return_code'] == 0:
                    packages_data[f'{pip_cmd}_list'] = pip_list['stdout']
                
                # Show outdated packages
                pip_outdated = self.safe_execute_command([pip_cmd, 'list', '--outdated'])
                if pip_outdated and pip_outdated['return_code'] == 0:
                    packages_data[f'{pip_cmd}_outdated'] = pip_outdated['stdout']
                
                # Check for security vulnerabilities (if safety is installed)
                safety_check = self.safe_execute_command(['safety', 'check'])
                if safety_check:
                    packages_data[f'{pip_cmd}_security'] = safety_check['stdout']
        
        return packages_data
    
    def _collect_virtual_environments(self) -> Dict[str, Any]:
        """Collect virtual environment information."""
        venv_data = {}
        
        # Common virtual environment locations
        venv_paths = [
            Path.home() / '.virtualenvs',
            Path.home() / 'venv',
            Path.home() / 'envs',
            Path('/opt/venv'),
            Path('/usr/local/venv'),
        ]
        
        found_venvs = {}
        for venv_path in venv_paths:
            if venv_path.exists():
                venv_dirs = [d for d in venv_path.iterdir() if d.is_dir()]
                for venv_dir in venv_dirs:
                    pip_freeze = self.safe_execute_command([str(venv_dir / 'bin' / 'pip'), 'freeze'])
                    if pip_freeze and pip_freeze['return_code'] == 0:
                        found_venvs[str(venv_dir)] = pip_freeze['stdout']
        
        venv_data['virtual_environments'] = found_venvs
        
        # pipenv environments
        pipenv_graph = self.safe_execute_command(['pipenv', 'graph'])
        if pipenv_graph and pipenv_graph['return_code'] == 0:
            venv_data['pipenv_dependencies'] = pipenv_graph['stdout']
        
        return venv_data
    
    def _collect_conda_environment(self) -> Dict[str, Any]:
        """Collect Conda environment information."""
        conda_data = {}
        
        # Conda installation info
        conda_info = self.safe_execute_command(['conda', 'info'])
        if conda_info and conda_info['return_code'] == 0:
            conda_data['conda_info'] = conda_info['stdout']
        
        # List conda environments
        conda_envs = self.safe_execute_command(['conda', 'env', 'list'])
        if conda_envs and conda_envs['return_code'] == 0:
            conda_data['environments'] = conda_envs['stdout']
        
        # List packages in base environment
        conda_packages = self.safe_execute_command(['conda', 'list'])
        if conda_packages and conda_packages['return_code'] == 0:
            conda_data['base_packages'] = conda_packages['stdout']
        
        # Check for mamba (faster conda alternative)
        mamba_info = self.safe_execute_command(['mamba', 'info'])
        if mamba_info and mamba_info['return_code'] == 0:
            conda_data['mamba_info'] = mamba_info['stdout']
        
        return conda_data
    
    def _collect_poetry_environment(self) -> Dict[str, Any]:
        """Collect Poetry dependency management information."""
        poetry_data = {}
        
        # Poetry version and config
        poetry_version = self.safe_execute_command(['poetry', '--version'])
        if poetry_version and poetry_version['return_code'] == 0:
            poetry_data['version'] = poetry_version['stdout']
        
        # Poetry configuration
        poetry_config = self.safe_execute_command(['poetry', 'config', '--list'])
        if poetry_config and poetry_config['return_code'] == 0:
            poetry_data['config'] = poetry_config['stdout']
        
        # Check if current directory has poetry project
        if Path('pyproject.toml').exists():
            poetry_show = self.safe_execute_command(['poetry', 'show'])
            if poetry_show and poetry_show['return_code'] == 0:
                poetry_data['current_project_deps'] = poetry_show['stdout']
            
            poetry_env_info = self.safe_execute_command(['poetry', 'env', 'info'])
            if poetry_env_info and poetry_env_info['return_code'] == 0:
                poetry_data['current_env_info'] = poetry_env_info['stdout']
        
        return poetry_data
    
    def _collect_system_python_packages(self) -> Dict[str, Any]:
        """Collect system-wide Python package installations."""
        system_packages = {}
        
        # Debian/Ubuntu python packages
        dpkg_python = self.safe_execute_command(['dpkg', '-l', '*python*'])
        if dpkg_python and dpkg_python['return_code'] == 0:
            system_packages['dpkg_python'] = dpkg_python['stdout']
        
        # Python site-packages directories
        python_paths = self.safe_execute_command(['python3', '-c', 'import site; print(site.getsitepackages())'])
        if python_paths and python_paths['return_code'] == 0:
            system_packages['site_packages'] = python_paths['stdout']
        
        # System Python path
        python_path = self.safe_execute_command(['python3', '-c', 'import sys; print("\\n".join(sys.path))'])
        if python_path and python_path['return_code'] == 0:
            system_packages['python_path'] = python_path['stdout']
        
        return system_packages
    
    def _collect_ai_ml_frameworks(self) -> Dict[str, Any]:
        """Collect AI/ML framework versions and configurations."""
        ai_frameworks = {}
        
        # Key AI/ML frameworks to check
        frameworks = {
            'torch': 'import torch; print(torch.__version__)',
            'tensorflow': 'import tensorflow as tf; print(tf.__version__)',
            'transformers': 'import transformers; print(transformers.__version__)',
            'datasets': 'import datasets; print(datasets.__version__)',
            'accelerate': 'import accelerate; print(accelerate.__version__)',
            'peft': 'import peft; print(peft.__version__)',
            'deepspeed': 'import deepspeed; print(deepspeed.__version__)',
            'bitsandbytes': 'import bitsandbytes; print(bitsandbytes.__version__)',
            'numpy': 'import numpy; print(numpy.__version__)',
            'pandas': 'import pandas; print(pandas.__version__)',
            'scikit-learn': 'import sklearn; print(sklearn.__version__)',
            'jupyter': 'import jupyter; print(jupyter.__version__)',
            'matplotlib': 'import matplotlib; print(matplotlib.__version__)',
            'wandb': 'import wandb; print(wandb.__version__)',
            'mlflow': 'import mlflow; print(mlflow.__version__)'
        }
        
        for framework, check_cmd in frameworks.items():
            version_check = self.safe_execute_command(['python3', '-c', check_cmd])
            if version_check and version_check['return_code'] == 0:
                ai_frameworks[framework] = version_check['stdout'].strip()
            else:
                ai_frameworks[framework] = 'Not installed'
        
        # PyTorch CUDA availability
        torch_cuda = self.safe_execute_command(['python3', '-c', 
            'import torch; print("CUDA available:", torch.cuda.is_available()); print("CUDA devices:", torch.cuda.device_count())'])
        if torch_cuda and torch_cuda['return_code'] == 0:
            ai_frameworks['torch_cuda'] = torch_cuda['stdout']
        
        # TensorFlow GPU check
        tf_gpu = self.safe_execute_command(['python3', '-c',
            'import tensorflow as tf; print("GPU devices:", len(tf.config.list_physical_devices("GPU")))'])
        if tf_gpu and tf_gpu['return_code'] == 0:
            ai_frameworks['tensorflow_gpu'] = tf_gpu['stdout']
        
        return ai_frameworks

    def _collect_nvidia_gpu_data(self) -> Dict[str, Any]:
        """Collect comprehensive RTX 5090 GPU data for LLM optimization."""
        gpu_data = {}
        
        # Basic nvidia-smi output
        nvidia_basic = self.safe_execute_command(['nvidia-smi', '--query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,fan.speed,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,power.limit', '--format=csv'])
        if nvidia_basic and nvidia_basic['return_code'] == 0:
            gpu_data['basic_metrics'] = nvidia_basic['stdout']
        
        # Detailed XML output for comprehensive data
        nvidia_xml = self.safe_execute_command(['nvidia-smi', '-q', '-x'])
        if nvidia_xml and nvidia_xml['return_code'] == 0:
            gpu_data['detailed_xml'] = nvidia_xml['stdout']
        
        # GPU processes and memory usage
        nvidia_processes = self.safe_execute_command(['nvidia-smi', '--query-compute-apps=pid,process_name,gpu_uuid,gpu_name,used_memory', '--format=csv'])
        if nvidia_processes and nvidia_processes['return_code'] == 0:
            gpu_data['compute_processes'] = nvidia_processes['stdout']
        
        # Graphics processes
        nvidia_graphics = self.safe_execute_command(['nvidia-smi', '--query-graphics-apps=pid,process_name,gpu_uuid,gpu_name,used_memory', '--format=csv'])
        if nvidia_graphics and nvidia_graphics['return_code'] == 0:
            gpu_data['graphics_processes'] = nvidia_graphics['stdout']
        
        # Clock speeds and performance states
        nvidia_clocks = self.safe_execute_command(['nvidia-smi', '--query-gpu=clocks.current.graphics,clocks.current.sm,clocks.current.memory,clocks.current.video', '--format=csv'])
        if nvidia_clocks and nvidia_clocks['return_code'] == 0:
            gpu_data['clock_speeds'] = nvidia_clocks['stdout']
        
        # GPU topology and NVLink (RTX 5090 may have NVLink)
        nvidia_topo = self.safe_execute_command(['nvidia-smi', 'topo', '-m'])
        if nvidia_topo and nvidia_topo['return_code'] == 0:
            gpu_data['topology'] = nvidia_topo['stdout']
        
        # GPU utilization samples over time (for trend analysis)
        nvidia_samples = self.safe_execute_command(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw', '--format=csv', '-l', '1', '-c', '5'])
        if nvidia_samples and nvidia_samples['return_code'] == 0:
            gpu_data['utilization_samples'] = nvidia_samples['stdout']
        
        return gpu_data
    
    def _collect_cuda_environment(self) -> Dict[str, Any]:
        """Collect CUDA runtime and development environment information."""
        cuda_data = {}
        
        # CUDA compiler version
        nvcc_version = self.safe_execute_command(['nvcc', '--version'])
        if nvcc_version and nvcc_version['return_code'] == 0:
            cuda_data['nvcc_version'] = nvcc_version['stdout']
        
        # CUDA runtime version via nvidia-smi
        cuda_version = self.safe_execute_command(['nvidia-smi', '--query-gpu=driver_version,cuda_version', '--format=csv'])
        if cuda_version and cuda_version['return_code'] == 0:
            cuda_data['runtime_version'] = cuda_version['stdout']
        
        # Device capabilities
        cuda_caps = self.safe_execute_command(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv'])
        if cuda_caps and cuda_caps['return_code'] == 0:
            cuda_data['compute_capability'] = cuda_caps['stdout']
        
        # CUDA libraries and paths
        cuda_data['environment'] = {
            'cuda_home': self.safe_read_file('/usr/local/cuda/version.txt'),
            'library_paths': self._get_cuda_library_paths(),
            'environment_vars': self._get_cuda_env_vars()
        }
        
        return cuda_data
    
    def _collect_docker_nvidia_data(self) -> Dict[str, Any]:
        """Collect Docker NVIDIA Container Toolkit and GPU container information."""
        docker_nvidia_data = {}
        
        # Docker runtime info
        docker_info = self.safe_execute_command(['docker', 'info'])
        if docker_info and docker_info['return_code'] == 0:
            docker_nvidia_data['docker_info'] = docker_info['stdout']
        
        # NVIDIA Container Toolkit version
        nvidia_ctk = self.safe_execute_command(['nvidia-ctk', '--version'])
        if nvidia_ctk and nvidia_ctk['return_code'] == 0:
            docker_nvidia_data['nvidia_ctk_version'] = nvidia_ctk['stdout']
        
        # Running containers with GPU access
        gpu_containers = self.safe_execute_command(['docker', 'ps', '--filter', 'label=com.nvidia.volumes.needed=nvidia_driver', '--format', 'table {{.ID}}\\t{{.Image}}\\t{{.Names}}\\t{{.Status}}'])
        if gpu_containers and gpu_containers['return_code'] == 0:
            docker_nvidia_data['gpu_containers'] = gpu_containers['stdout']
        
        # Docker compose services status (matching your setup)
        compose_ps = self.safe_execute_command(['docker', 'compose', 'ps', '--format', 'json'])
        if compose_ps and compose_ps['return_code'] == 0:
            docker_nvidia_data['compose_services'] = compose_ps['stdout']
        
        return docker_nvidia_data
    
    def _collect_cpu_data(self) -> Dict[str, Any]:
        """Collect comprehensive CPU information and usage data."""
        cpu_data = {}
        
        # CPU topology and info from /proc/cpuinfo
        cpuinfo = self.safe_read_file('/proc/cpuinfo')
        if cpuinfo:
            cpu_data['cpuinfo'] = self._parse_cpuinfo(cpuinfo)
        
        # CPU usage from /proc/stat
        cpu_stat = self.safe_read_file('/proc/stat')
        if cpu_stat:
            cpu_data['usage'] = self._parse_cpu_stat(cpu_stat)
        
        # CPU frequency scaling
        cpu_data['frequency'] = self._collect_cpu_frequency()
        
        # Thermal data
        cpu_data['thermal'] = self._collect_thermal_data()
        
        # NUMA topology
        cpu_data['numa'] = self._collect_numa_data()
        
        # Load averages
        loadavg = self.safe_read_file('/proc/loadavg')
        if loadavg:
            cpu_data['loadavg'] = loadavg.strip().split()[:3]
        
        # lscpu command output for additional details
        lscpu_result = self.safe_execute_command(['lscpu'])
        if lscpu_result and lscpu_result['return_code'] == 0:
            cpu_data['lscpu'] = lscpu_result['stdout']
        
        return cpu_data
    
    def _collect_memory_data(self) -> Dict[str, Any]:
        """Collect memory usage and allocation information."""
        memory_data = {}
        
        # Memory info from /proc/meminfo
        meminfo = self.safe_read_file('/proc/meminfo')
        if meminfo:
            memory_data['meminfo'] = self._parse_meminfo(meminfo)
        
        # Virtual memory statistics
        vmstat = self.safe_read_file('/proc/vmstat')
        if vmstat:
            memory_data['vmstat'] = self._parse_vmstat(vmstat)
        
        # Swap usage
        swaps = self.safe_read_file('/proc/swaps')
        if swaps:
            memory_data['swaps'] = swaps
        
        # Memory maps for analysis
        memory_data['numa_maps'] = self._collect_numa_memory_maps()
        
        # free command output
        free_result = self.safe_execute_command(['free', '-h'])
        if free_result and free_result['return_code'] == 0:
            memory_data['free'] = free_result['stdout']
        
        return memory_data
    
    def _collect_hardware_data(self) -> Dict[str, Any]:
        """Collect hardware information."""
        hardware_data = {}
        
        # PCI devices
        lspci_result = self.safe_execute_command(['lspci'])
        if lspci_result and lspci_result['return_code'] == 0:
            hardware_data['pci_devices'] = lspci_result['stdout']
        
        # PCI tree view for GPU placement
        lspci_tree = self.safe_execute_command(['lspci', '-tv'])
        if lspci_tree and lspci_tree['return_code'] == 0:
            hardware_data['pci_tree'] = lspci_tree['stdout']
        
        # USB devices  
        lsusb_result = self.safe_execute_command(['lsusb'])
        if lsusb_result and lsusb_result['return_code'] == 0:
            hardware_data['usb_devices'] = lsusb_result['stdout']
        
        # Hardware sensors
        sensors_result = self.safe_execute_command(['sensors'])
        if sensors_result and sensors_result['return_code'] == 0:
            hardware_data['sensors'] = sensors_result['stdout']
        
        # DMI/SMBIOS information
        dmidecode_result = self.safe_execute_command(['dmidecode', '-t', 'system'])
        if dmidecode_result and dmidecode_result['return_code'] == 0:
            hardware_data['system_info'] = dmidecode_result['stdout']
        
        return hardware_data
    
    def _collect_kernel_data(self) -> Dict[str, Any]:
        """Collect kernel and system parameters."""
        kernel_data = {}
        
        # Kernel version
        uname_result = self.safe_execute_command(['uname', '-a'])
        if uname_result and uname_result['return_code'] == 0:
            kernel_data['version'] = uname_result['stdout'].strip()
        
        # Boot time
        uptime = self.safe_read_file('/proc/uptime')
        if uptime:
            kernel_data['uptime'] = uptime.strip()
        
        # Kernel parameters
        cmdline = self.safe_read_file('/proc/cmdline')
        if cmdline:
            kernel_data['cmdline'] = cmdline.strip()
        
        # NVIDIA kernel modules
        nvidia_modules = self.safe_execute_command(['lsmod'])
        if nvidia_modules and nvidia_modules['return_code'] == 0:
            # Filter for NVIDIA modules
            nvidia_lines = [line for line in nvidia_modules['stdout'].split('\n') if 'nvidia' in line.lower()]
            kernel_data['nvidia_modules'] = '\n'.join(nvidia_lines)
        
        # System limits
        limits_data = {}
        for limit_file in ['limits.conf']:
            content = self.safe_read_file(f'/etc/security/{limit_file}')
            if content:
                limits_data[limit_file] = content
        kernel_data['limits'] = limits_data
        
        return kernel_data
    
    def _collect_network_data(self) -> Dict[str, Any]:
        """Collect network interface and connection information."""
        network_data = {}
        
        # Network interfaces
        ip_result = self.safe_execute_command(['ip', 'addr', 'show'])
        if ip_result and ip_result['return_code'] == 0:
            network_data['interfaces'] = ip_result['stdout']
        
        # Routing table
        route_result = self.safe_execute_command(['ip', 'route', 'show'])
        if route_result and route_result['return_code'] == 0:
            network_data['routing'] = route_result['stdout']
        
        # Active connections
        ss_result = self.safe_execute_command(['ss', '-tuln'])
        if ss_result and ss_result['return_code'] == 0:
            network_data['connections'] = ss_result['stdout']
        
        # Network statistics
        net_dev = self.safe_read_file('/proc/net/dev')
        if net_dev:
            network_data['statistics'] = net_dev
        
        return network_data
    
    def _collect_process_data(self) -> Dict[str, Any]:
        """Collect process information for analysis."""
        process_data = {}
        
        # Process list with resource usage
        ps_result = self.safe_execute_command(['ps', 'aux', '--sort=-%cpu'])
        if ps_result and ps_result['return_code'] == 0:
            process_data['ps_cpu'] = ps_result['stdout']
        
        # Process list sorted by memory
        ps_mem_result = self.safe_execute_command(['ps', 'aux', '--sort=-%mem'])
        if ps_mem_result and ps_mem_result['return_code'] == 0:
            process_data['ps_memory'] = ps_mem_result['stdout']
        
        # Process tree
        pstree_result = self.safe_execute_command(['ps', 'axjf'])
        if pstree_result and pstree_result['return_code'] == 0:
            process_data['process_tree'] = pstree_result['stdout']
        
        # NVIDIA processes specifically
        nvidia_processes = self._collect_nvidia_processes()
        if nvidia_processes:
            process_data['nvidia_processes'] = nvidia_processes
        
        return process_data
    
    def _collect_performance_data(self) -> Dict[str, Any]:
        """Collect performance metrics and statistics."""
        performance_data = {}
        
        # I/O statistics
        iostat_result = self.safe_execute_command(['iostat', '-x', '1', '1'])
        if iostat_result and iostat_result['return_code'] == 0:
            performance_data['iostat'] = iostat_result['stdout']
        
        # VM statistics  
        vmstat_result = self.safe_execute_command(['vmstat', '1', '2'])
        if vmstat_result and vmstat_result['return_code'] == 0:
            performance_data['vmstat'] = vmstat_result['stdout']
        
        return performance_data
    
    def _collect_nvidia_processes(self) -> Dict[str, Any]:
        """Collect detailed information about NVIDIA GPU processes."""
        nvidia_proc_data = {}
        
        # Get all processes using GPU
        gpu_processes = self.safe_execute_command(['nvidia-smi', '--query-compute-apps=pid,process_name,gpu_uuid,used_memory', '--format=csv,noheader,nounits'])
        if gpu_processes and gpu_processes['return_code'] == 0:
            nvidia_proc_data['compute_apps'] = gpu_processes['stdout']
        
        # Get process details for each GPU process
        if gpu_processes and gpu_processes['stdout'].strip():
            process_details = {}
            for line in gpu_processes['stdout'].strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 1:
                        pid = parts[0].strip()
                        # Get detailed process info
                        proc_info = self.safe_read_file(f'/proc/{pid}/status')
                        if proc_info:
                            process_details[pid] = proc_info
            nvidia_proc_data['process_details'] = process_details
        
        return nvidia_proc_data
    
    def _get_cuda_library_paths(self) -> Dict[str, Any]:
        """Get CUDA library paths and versions."""
        cuda_libs = {}
        
        # Common CUDA library locations
        cuda_paths = ['/usr/local/cuda/lib64', '/usr/lib/x86_64-linux-gnu']
        
        for path in cuda_paths:
            path_obj = Path(path)
            if path_obj.exists():
                cuda_files = list(path_obj.glob('*cuda*'))
                cuda_libs[path] = [str(f) for f in cuda_files[:10]]  # Limit results
        
        return cuda_libs
    
    def _get_cuda_env_vars(self) -> Dict[str, str]:
        """Get CUDA-related environment variables."""
        import os
        
        cuda_env_vars = {}
        cuda_var_names = ['CUDA_HOME', 'CUDA_ROOT', 'CUDA_PATH', 'LD_LIBRARY_PATH', 
                         'PATH', 'NVIDIA_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES']
        
        for var_name in cuda_var_names:
            if var_name in os.environ:
                cuda_env_vars[var_name] = os.environ[var_name]
        
        return cuda_env_vars
    
    # Helper methods for parsing specific formats
    
    def _parse_cpuinfo(self, cpuinfo: str) -> Dict[str, Any]:
        """Parse /proc/cpuinfo into structured data."""
        cpus = []
        current_cpu = {}
        
        for line in cpuinfo.split('\n'):
            if not line.strip():
                if current_cpu:
                    cpus.append(current_cpu)
                    current_cpu = {}
                continue
                
            if ':' in line:
                key, value = line.split(':', 1)
                current_cpu[key.strip()] = value.strip()
        
        if current_cpu:
            cpus.append(current_cpu)
        
        return {'processors': cpus, 'count': len(cpus)}
    
    def _parse_meminfo(self, meminfo: str) -> Dict[str, str]:
        """Parse /proc/meminfo into key-value pairs."""
        memory_info = {}
        
        for line in meminfo.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                memory_info[key.strip()] = value.strip()
        
        return memory_info
    
    def _parse_cpu_stat(self, cpu_stat: str) -> Dict[str, Any]:
        """Parse /proc/stat CPU usage data."""
        cpu_usage = {}
        
        for line in cpu_stat.split('\n'):
            if line.startswith('cpu'):
                parts = line.split()
                cpu_name = parts[0]
                if len(parts) >= 8:
                    cpu_usage[cpu_name] = {
                        'user': int(parts[1]),
                        'nice': int(parts[2]),  
                        'system': int(parts[3]),
                        'idle': int(parts[4]),
                        'iowait': int(parts[5]),
                        'irq': int(parts[6]),
                        'softirq': int(parts[7])
                    }
        
        return cpu_usage
    
    def _parse_vmstat(self, vmstat: str) -> Dict[str, str]:
        """Parse /proc/vmstat into key-value pairs."""
        vm_stats = {}
        
        for line in vmstat.split('\n'):
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    vm_stats[parts[0]] = parts[1]
        
        return vm_stats
    
    def _collect_cpu_frequency(self) -> Dict[str, Any]:
        """Collect CPU frequency scaling information."""
        freq_data = {}
        
        # Current frequencies
        freq_data['current'] = {}
        cpufreq_path = Path('/sys/devices/system/cpu')
        
        for cpu_dir in cpufreq_path.glob('cpu*/cpufreq'):
            cpu_num = cpu_dir.parent.name
            
            current_freq = self.safe_read_file(cpu_dir / 'scaling_cur_freq')
            if current_freq:
                freq_data['current'][cpu_num] = current_freq.strip()
        
        return freq_data
    
    def _collect_thermal_data(self) -> Dict[str, Any]:
        """Collect thermal/temperature data."""
        thermal_data = {}
        
        thermal_path = Path('/sys/class/thermal')
        if thermal_path.exists():
            for zone in thermal_path.glob('thermal_zone*'):
                zone_name = zone.name
                temp_file = zone / 'temp'
                
                if temp_file.exists():
                    temp = self.safe_read_file(str(temp_file))
                    if temp:
                        thermal_data[zone_name] = temp.strip()
        
        return thermal_data
    
    def _collect_numa_data(self) -> Dict[str, Any]:
        """Collect NUMA topology information."""
        numa_data = {}
        
        # NUMA nodes
        numactl_result = self.safe_execute_command(['numactl', '--show'])
        if numactl_result and numactl_result['return_code'] == 0:
            numa_data['topology'] = numactl_result['stdout']
        
        # NUMA hardware info
        numa_hw_result = self.safe_execute_command(['numactl', '--hardware'])
        if numa_hw_result and numa_hw_result['return_code'] == 0:
            numa_data['hardware'] = numa_hw_result['stdout']
        
        return numa_data
    
    def _collect_numa_memory_maps(self) -> Dict[str, str]:
        """Collect NUMA memory mapping information."""
        numa_maps = {}
        
        # Read NUMA maps for key processes
        proc_path = Path('/proc')
        for pid_dir in list(proc_path.glob('[0-9]*'))[:20]:  # Limit to top 20 processes
            numa_maps_file = pid_dir / 'numa_maps'
            if numa_maps_file.exists():
                content = self.safe_read_file(str(numa_maps_file))
                if content and len(content) < 10000:  # Limit size
                    numa_maps[pid_dir.name] = content[:1000]  # Sample
        
        return numa_maps
    
    def _get_current_time_ms(self) -> int:
        """Get current time in milliseconds."""
        import time
        return int(time.time() * 1000)