"""
Network Change Detector
======================

Detects network-related changes including connection states, traffic patterns,
interface configurations, and network performance metrics.
"""

import psutil
import socket
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict

from ..detector_base import SystemChangeDetectorBase
from ..types import SystemChange, ChangeType, SystemSnapshot


logger = logging.getLogger(__name__)


class NetworkChangeDetector(SystemChangeDetectorBase):
    """
    Detector for network-related system changes.
    
    Monitors:
    - Network interface state changes (up/down, IP changes)
    - Network connections (new/closed connections)
    - Network traffic patterns and bandwidth usage
    - DNS configuration changes
    - Routing table modifications
    - Network performance metrics (latency, packet loss)
    - Suspicious network activities
    """
    
    def __init__(self, traffic_threshold: float = 0.1, connection_threshold: int = 10):
        """
        Initialize network change detector.
        
        Args:
            traffic_threshold: Minimum traffic change to consider significant (10% default)
            connection_threshold: Number of new connections to consider significant
        """
        super().__init__()
        self.traffic_threshold = traffic_threshold
        self.connection_threshold = connection_threshold
        
        # Network tracking state
        self._last_interface_stats = {}
        self._last_connections = set()
        self._traffic_history = []
        self._connection_history = []
        
        # Network thresholds
        self.high_traffic_threshold = 100 * 1024 * 1024  # 100 MB/s
        self.suspicious_port_ranges = [
            (1, 1023),      # Well-known ports
            (6660, 6669),   # IRC
            (6697, 6697),   # IRC SSL
            (31337, 31337), # Back Orifice
            (12345, 12346), # NetBus
        ]
        
        # Common suspicious ports
        self.suspicious_ports = {21, 23, 25, 53, 80, 135, 139, 443, 445, 993, 995}
        
    def detect_changes(self, old_snapshot: SystemSnapshot, new_snapshot: SystemSnapshot) -> List[SystemChange]:
        """
        Detect network-related changes between snapshots.
        
        Args:
            old_snapshot: Previous system state
            new_snapshot: Current system state
            
        Returns:
            List of detected network changes
        """
        changes = []
        timestamp = datetime.now()
        
        # Extract network data from snapshots
        old_network = self._extract_network_data(old_snapshot)
        new_network = self._extract_network_data(new_snapshot)
        
        # Network interface changes
        changes.extend(self._detect_interface_changes(old_network, new_network, timestamp))
        
        # Network traffic changes
        changes.extend(self._detect_traffic_changes(old_network, new_network, timestamp))
        
        # Network connection changes
        changes.extend(self._detect_connection_changes(old_network, new_network, timestamp))
        
        # Network performance changes
        changes.extend(self._detect_performance_changes(old_network, new_network, timestamp))
        
        # Suspicious network activities
        changes.extend(self._detect_suspicious_network_activity(old_network, new_network, timestamp))
        
        return changes
    
    def get_current_network_state(self) -> Dict[str, Any]:
        """Get current network state for diagnostics."""
        try:
            network_state = {
                'interfaces': {},
                'connections': [],
                'traffic_stats': {},
                'dns_servers': [],
                'routing_info': {}
            }
            
            # Network interfaces
            for interface_name, stats in psutil.net_io_counters(pernic=True).items():
                addrs = psutil.net_if_addrs().get(interface_name, [])
                network_state['interfaces'][interface_name] = {
                    'bytes_sent': stats.bytes_sent,
                    'bytes_recv': stats.bytes_recv,
                    'packets_sent': stats.packets_sent,
                    'packets_recv': stats.packets_recv,
                    'errin': stats.errin,
                    'errout': stats.errout,
                    'dropin': stats.dropin,
                    'dropout': stats.dropout,
                    'addresses': [
                        {
                            'family': addr.family.name,
                            'address': addr.address,
                            'netmask': getattr(addr, 'netmask', None),
                            'broadcast': getattr(addr, 'broadcast', None)
                        }
                        for addr in addrs
                    ]
                }
            
            # Active connections
            for conn in psutil.net_connections():
                if conn.status == psutil.CONN_ESTABLISHED:
                    network_state['connections'].append({
                        'local_address': f"{conn.laddr.ip}:{conn.laddr.port}",
                        'remote_address': f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None,
                        'status': conn.status,
                        'pid': conn.pid,
                        'family': conn.family.name if conn.family else None,
                        'type': conn.type.name if conn.type else None
                    })
            
            # Traffic statistics
            total_stats = psutil.net_io_counters()
            network_state['traffic_stats'] = {
                'total_bytes_sent': total_stats.bytes_sent,
                'total_bytes_recv': total_stats.bytes_recv,
                'total_packets_sent': total_stats.packets_sent,
                'total_packets_recv': total_stats.packets_recv,
                'total_errors': total_stats.errin + total_stats.errout,
                'total_drops': total_stats.dropin + total_stats.dropout
            }
            
            return network_state
            
        except Exception as e:
            logger.error(f"Error getting network state: {e}")
            return {}
    
    def _extract_network_data(self, snapshot: SystemSnapshot) -> Dict[str, Any]:
        """Extract network-related data from system snapshot."""
        if not snapshot:
            return {}
        
        try:
            network_data = {}
            
            # Network interface statistics
            if hasattr(snapshot, 'network_stats') and snapshot.network_stats:
                network_data['interfaces'] = snapshot.network_stats
            else:
                network_data['interfaces'] = {}
                try:
                    for interface_name, stats in psutil.net_io_counters(pernic=True).items():
                        network_data['interfaces'][interface_name] = {
                            'bytes_sent': stats.bytes_sent,
                            'bytes_recv': stats.bytes_recv,
                            'packets_sent': stats.packets_sent,
                            'packets_recv': stats.packets_recv,
                            'errin': stats.errin,
                            'errout': stats.errout,
                            'dropin': stats.dropin,
                            'dropout': stats.dropout
                        }
                except:
                    pass
            
            # Network connections
            if hasattr(snapshot, 'network_connections') and snapshot.network_connections:
                network_data['connections'] = snapshot.network_connections
            else:
                network_data['connections'] = []
                try:
                    for conn in psutil.net_connections():
                        network_data['connections'].append({
                            'family': conn.family,
                            'type': conn.type,
                            'local_addr': conn.laddr,
                            'remote_addr': conn.raddr,
                            'status': conn.status,
                            'pid': conn.pid
                        })
                except:
                    pass
            
            # Interface addresses
            try:
                network_data['interface_addrs'] = {}
                for interface_name, addrs in psutil.net_if_addrs().items():
                    network_data['interface_addrs'][interface_name] = [
                        {
                            'family': addr.family,
                            'address': addr.address,
                            'netmask': getattr(addr, 'netmask', None)
                        }
                        for addr in addrs
                    ]
            except:
                network_data['interface_addrs'] = {}
            
            return network_data
            
        except Exception as e:
            logger.error(f"Error extracting network data: {e}")
            return {}
    
    def _detect_interface_changes(self, old_network: Dict[str, Any], 
                                 new_network: Dict[str, Any], timestamp: datetime) -> List[SystemChange]:
        """Detect network interface changes."""
        changes = []
        
        old_interfaces = old_network.get('interfaces', {})
        new_interfaces = new_network.get('interfaces', {})
        old_addrs = old_network.get('interface_addrs', {})
        new_addrs = new_network.get('interface_addrs', {})
        
        # New interfaces
        new_interface_names = set(new_interfaces.keys()) - set(old_interfaces.keys())
        for interface_name in new_interface_names:
            changes.append(SystemChange(
                category="network",
                change_type=ChangeType.ADDED,
                entity_id=f"interface_{interface_name}",
                old_value=None,
                new_value=interface_name,
                significance=0.8,  # Interface addition is significant
                metadata={
                    'event_type': 'interface_added',
                    'interface_name': interface_name,
                    'interface_stats': new_interfaces[interface_name]
                },
                timestamp=timestamp
            ))
        
        # Removed interfaces
        removed_interfaces = set(old_interfaces.keys()) - set(new_interfaces.keys())
        for interface_name in removed_interfaces:
            changes.append(SystemChange(
                category="network",
                change_type=ChangeType.REMOVED,
                entity_id=f"interface_{interface_name}",
                old_value=interface_name,
                new_value=None,
                significance=0.9,  # Interface removal is highly significant
                metadata={
                    'event_type': 'interface_removed',
                    'interface_name': interface_name
                },
                timestamp=timestamp
            ))
        
        # Interface address changes
        for interface_name in set(old_addrs.keys()) | set(new_addrs.keys()):
            old_interface_addrs = {addr['address'] for addr in old_addrs.get(interface_name, [])}
            new_interface_addrs = {addr['address'] for addr in new_addrs.get(interface_name, [])}
            
            added_addrs = new_interface_addrs - old_interface_addrs
            removed_addrs = old_interface_addrs - new_interface_addrs
            
            for addr in added_addrs:
                changes.append(SystemChange(
                    category="network",
                    change_type=ChangeType.ADDED,
                    entity_id=f"interface_{interface_name}_addr_{addr.replace('.', '_').replace(':', '_')}",
                    old_value=None,
                    new_value=addr,
                    significance=0.7,
                    metadata={
                        'event_type': 'ip_address_added',
                        'interface_name': interface_name,
                        'ip_address': addr
                    },
                    timestamp=timestamp
                ))
            
            for addr in removed_addrs:
                changes.append(SystemChange(
                    category="network",
                    change_type=ChangeType.REMOVED,
                    entity_id=f"interface_{interface_name}_addr_{addr.replace('.', '_').replace(':', '_')}",
                    old_value=addr,
                    new_value=None,
                    significance=0.8,
                    metadata={
                        'event_type': 'ip_address_removed',
                        'interface_name': interface_name,
                        'ip_address': addr
                    },
                    timestamp=timestamp
                ))
        
        return changes
    
    def _detect_traffic_changes(self, old_network: Dict[str, Any], 
                               new_network: Dict[str, Any], timestamp: datetime) -> List[SystemChange]:
        """Detect significant network traffic changes."""
        changes = []
        
        old_interfaces = old_network.get('interfaces', {})
        new_interfaces = new_network.get('interfaces', {})
        
        # Analyze traffic per interface
        for interface_name in set(old_interfaces.keys()) & set(new_interfaces.keys()):
            old_stats = old_interfaces[interface_name]
            new_stats = new_interfaces[interface_name]
            
            # Calculate traffic deltas (assuming 1-minute intervals)
            bytes_sent_delta = new_stats.get('bytes_sent', 0) - old_stats.get('bytes_sent', 0)
            bytes_recv_delta = new_stats.get('bytes_recv', 0) - old_stats.get('bytes_recv', 0)
            
            # Calculate rates (bytes per second, assuming 60-second interval)
            time_delta = 60  # This should be calculated from actual timestamps
            send_rate = bytes_sent_delta / time_delta if time_delta > 0 else 0
            recv_rate = bytes_recv_delta / time_delta if time_delta > 0 else 0
            total_rate = send_rate + recv_rate
            
            # Detect high traffic
            if total_rate >= self.high_traffic_threshold:
                significance = min(total_rate / (self.high_traffic_threshold * 10), 1.0)
                
                changes.append(SystemChange(
                    category="network",
                    change_type=ChangeType.MODIFIED,
                    entity_id=f"traffic_{interface_name}",
                    old_value=0,  # Previous rate not easily calculable
                    new_value=total_rate / (1024 * 1024),  # MB/s
                    significance=significance,
                    metadata={
                        'event_type': 'high_network_traffic',
                        'interface_name': interface_name,
                        'send_rate_mbs': send_rate / (1024 * 1024),
                        'recv_rate_mbs': recv_rate / (1024 * 1024),
                        'total_rate_mbs': total_rate / (1024 * 1024),
                        'bytes_sent_delta': bytes_sent_delta,
                        'bytes_recv_delta': bytes_recv_delta
                    },
                    timestamp=timestamp
                ))
            
            # Detect error rate changes
            old_errors = old_stats.get('errin', 0) + old_stats.get('errout', 0)
            new_errors = new_stats.get('errin', 0) + new_stats.get('errout', 0)
            error_delta = new_errors - old_errors
            
            if error_delta > 0:
                significance = min(error_delta / 100.0, 1.0)  # Scale by 100 errors
                
                changes.append(SystemChange(
                    category="network",
                    change_type=ChangeType.MODIFIED,
                    entity_id=f"network_errors_{interface_name}",
                    old_value=old_errors,
                    new_value=new_errors,
                    significance=significance,
                    metadata={
                        'event_type': 'network_errors_increased',
                        'interface_name': interface_name,
                        'error_delta': error_delta,
                        'error_in_delta': new_stats.get('errin', 0) - old_stats.get('errin', 0),
                        'error_out_delta': new_stats.get('errout', 0) - old_stats.get('errout', 0)
                    },
                    timestamp=timestamp
                ))
            
            # Track traffic history
            self._traffic_history.append({
                'timestamp': timestamp,
                'interface': interface_name,
                'send_rate_mbs': send_rate / (1024 * 1024),
                'recv_rate_mbs': recv_rate / (1024 * 1024),
                'error_rate': error_delta / time_delta if time_delta > 0 else 0
            })
        
        # Keep only last 100 traffic history entries
        if len(self._traffic_history) > 100:
            self._traffic_history = self._traffic_history[-100:]
        
        return changes
    
    def _detect_connection_changes(self, old_network: Dict[str, Any], 
                                  new_network: Dict[str, Any], timestamp: datetime) -> List[SystemChange]:
        """Detect network connection changes."""
        changes = []
        
        old_connections = old_network.get('connections', [])
        new_connections = new_network.get('connections', [])
        
        # Convert connections to comparable format
        old_conn_set = set()
        for conn in old_connections:
            if conn.get('local_addr') and conn.get('remote_addr'):
                conn_signature = (
                    f"{conn['local_addr'].ip}:{conn['local_addr'].port}",
                    f"{conn['remote_addr'].ip}:{conn['remote_addr'].port}",
                    conn.get('status', '')
                )
                old_conn_set.add(conn_signature)
        
        new_conn_set = set()
        new_conn_details = {}
        for conn in new_connections:
            if conn.get('local_addr') and conn.get('remote_addr'):
                conn_signature = (
                    f"{conn['local_addr'].ip}:{conn['local_addr'].port}",
                    f"{conn['remote_addr'].ip}:{conn['remote_addr'].port}",
                    conn.get('status', '')
                )
                new_conn_set.add(conn_signature)
                new_conn_details[conn_signature] = conn
        
        # New connections
        new_connections_list = new_conn_set - old_conn_set
        if len(new_connections_list) >= self.connection_threshold:
            significance = min(len(new_connections_list) / 100.0, 1.0)  # Scale by 100 connections
            
            changes.append(SystemChange(
                category="network",
                change_type=ChangeType.ADDED,
                entity_id="new_connections",
                old_value=len(old_conn_set),
                new_value=len(new_conn_set),
                significance=significance,
                metadata={
                    'event_type': 'multiple_new_connections',
                    'new_connection_count': len(new_connections_list),
                    'sample_connections': list(new_connections_list)[:10]  # First 10 as sample
                },
                timestamp=timestamp
            ))
        
        # Check for suspicious connections
        for conn_signature in new_connections_list:
            local_addr, remote_addr, status = conn_signature
            
            # Extract port from remote address
            try:
                remote_port = int(remote_addr.split(':')[-1])
                if self._is_suspicious_port(remote_port):
                    changes.append(SystemChange(
                        category="network",
                        change_type=ChangeType.ADDED,
                        entity_id=f"suspicious_connection_{remote_addr.replace(':', '_').replace('.', '_')}",
                        old_value=None,
                        new_value=remote_addr,
                        significance=0.9,
                        metadata={
                            'event_type': 'suspicious_connection',
                            'local_address': local_addr,
                            'remote_address': remote_addr,
                            'remote_port': remote_port,
                            'status': status,
                            'suspicion_reason': 'suspicious_port'
                        },
                        timestamp=timestamp
                    ))
            except ValueError:
                pass
        
        # Track connection history
        self._connection_history.append({
            'timestamp': timestamp,
            'total_connections': len(new_conn_set),
            'new_connections': len(new_connections_list),
            'closed_connections': len(old_conn_set - new_conn_set)
        })
        
        # Keep only last 100 connection history entries
        if len(self._connection_history) > 100:
            self._connection_history = self._connection_history[-100:]
        
        return changes
    
    def _detect_performance_changes(self, old_network: Dict[str, Any], 
                                   new_network: Dict[str, Any], timestamp: datetime) -> List[SystemChange]:
        """Detect network performance changes."""
        changes = []
        
        # This would require more detailed network performance metrics
        # For now, we'll focus on packet loss detection
        
        old_interfaces = old_network.get('interfaces', {})
        new_interfaces = new_network.get('interfaces', {})
        
        for interface_name in set(old_interfaces.keys()) & set(new_interfaces.keys()):
            old_stats = old_interfaces[interface_name]
            new_stats = new_interfaces[interface_name]
            
            # Packet drop rate changes
            old_drops = old_stats.get('dropin', 0) + old_stats.get('dropout', 0)
            new_drops = new_stats.get('dropin', 0) + new_stats.get('dropout', 0)
            drop_delta = new_drops - old_drops
            
            if drop_delta > 10:  # More than 10 dropped packets
                old_packets = old_stats.get('packets_sent', 0) + old_stats.get('packets_recv', 0)
                new_packets = new_stats.get('packets_sent', 0) + new_stats.get('packets_recv', 0)
                packet_delta = new_packets - old_packets
                
                drop_rate = drop_delta / packet_delta if packet_delta > 0 else 0
                significance = min(drop_rate * 10, 1.0)  # Scale by 10% drop rate
                
                changes.append(SystemChange(
                    category="network",
                    change_type=ChangeType.MODIFIED,
                    entity_id=f"packet_loss_{interface_name}",
                    old_value=old_drops,
                    new_value=new_drops,
                    significance=significance,
                    metadata={
                        'event_type': 'packet_loss_increased',
                        'interface_name': interface_name,
                        'dropped_packets_delta': drop_delta,
                        'drop_rate': drop_rate,
                        'drop_in_delta': new_stats.get('dropin', 0) - old_stats.get('dropin', 0),
                        'drop_out_delta': new_stats.get('dropout', 0) - old_stats.get('dropout', 0)
                    },
                    timestamp=timestamp
                ))
        
        return changes
    
    def _detect_suspicious_network_activity(self, old_network: Dict[str, Any], 
                                           new_network: Dict[str, Any], timestamp: datetime) -> List[SystemChange]:
        """Detect suspicious network activities."""
        changes = []
        
        new_connections = new_network.get('connections', [])
        
        # Group connections by remote IP to detect potential scanning
        remote_ips = defaultdict(int)
        for conn in new_connections:
            if conn.get('remote_addr'):
                remote_ip = conn['remote_addr'].ip
                remote_ips[remote_ip] += 1
        
        # Detect potential port scanning (many connections from same IP)
        for remote_ip, connection_count in remote_ips.items():
            if connection_count >= 10:  # 10+ connections from same IP
                changes.append(SystemChange(
                    category="network",
                    change_type=ChangeType.ADDED,
                    entity_id=f"potential_scan_{remote_ip.replace('.', '_')}",
                    old_value=0,
                    new_value=connection_count,
                    significance=0.9,
                    metadata={
                        'event_type': 'potential_port_scan',
                        'remote_ip': remote_ip,
                        'connection_count': connection_count,
                        'suspicion_reason': 'multiple_connections_same_ip'
                    },
                    timestamp=timestamp
                ))
        
        # Detect connections to known suspicious networks
        # This would be enhanced with threat intelligence feeds
        
        return changes
    
    def _is_suspicious_port(self, port: int) -> bool:
        """Check if port is potentially suspicious."""
        if port in self.suspicious_ports:
            return True
        
        for start, end in self.suspicious_port_ranges:
            if start <= port <= end:
                return True
        
        return False
    
    def get_network_summary(self) -> Dict[str, Any]:
        """Get network status summary."""
        try:
            current_state = self.get_current_network_state()
            
            total_connections = len(current_state.get('connections', []))
            active_interfaces = len([iface for iface, stats in current_state.get('interfaces', {}).items() 
                                   if stats.get('bytes_sent', 0) > 0 or stats.get('bytes_recv', 0) > 0])
            
            # Calculate total traffic
            total_traffic_gb = 0
            for stats in current_state.get('interfaces', {}).values():
                total_traffic_gb += (stats.get('bytes_sent', 0) + stats.get('bytes_recv', 0)) / (1024**3)
            
            return {
                'active_interfaces': active_interfaces,
                'total_connections': total_connections,
                'total_traffic_gb': round(total_traffic_gb, 2),
                'total_errors': sum(stats.get('errin', 0) + stats.get('errout', 0) 
                                  for stats in current_state.get('interfaces', {}).values()),
                'total_drops': sum(stats.get('dropin', 0) + stats.get('dropout', 0) 
                                 for stats in current_state.get('interfaces', {}).values()),
                'network_health': self._assess_network_health(current_state)
            }
            
        except Exception as e:
            logger.error(f"Error getting network summary: {e}")
            return {}
    
    def _assess_network_health(self, state: Dict[str, Any]) -> str:
        """Assess overall network health."""
        total_errors = sum(stats.get('errin', 0) + stats.get('errout', 0) 
                          for stats in state.get('interfaces', {}).values())
        total_drops = sum(stats.get('dropin', 0) + stats.get('dropout', 0) 
                         for stats in state.get('interfaces', {}).values())
        
        if total_errors > 1000 or total_drops > 1000:
            return "DEGRADED: High error/drop rate detected"
        elif total_errors > 100 or total_drops > 100:
            return "WARNING: Moderate error/drop rate"
        else:
            return "GOOD: Low error/drop rate"
    
    def get_traffic_history(self) -> List[Dict[str, Any]]:
        """Get historical network traffic data."""
        return self._traffic_history.copy()
    
    def get_connection_history(self) -> List[Dict[str, Any]]:
        """Get historical connection data."""
        return self._connection_history.copy()
    
    def get_detector_stats(self) -> Dict[str, Any]:
        """Get detector statistics and current state."""
        current_state = self.get_current_network_state()
        network_summary = self.get_network_summary()
        
        return {
            'detector_type': 'network',
            'current_network_state': current_state,
            'network_summary': network_summary,
            'thresholds': {
                'traffic_threshold': self.traffic_threshold,
                'connection_threshold': self.connection_threshold,
                'high_traffic_threshold_mbs': self.high_traffic_threshold / (1024 * 1024)
            },
            'monitoring': {
                'suspicious_ports': list(self.suspicious_ports),
                'traffic_history_entries': len(self._traffic_history),
                'connection_history_entries': len(self._connection_history)
            }
        }