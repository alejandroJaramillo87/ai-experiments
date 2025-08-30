"""
Security Change Detector
=======================

Detects security-related system changes including authentication events,
permission changes, security policy modifications, and suspicious activities.
"""

import os
import pwd
import grp
import stat
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from pathlib import Path

from ..detector_base import SystemChangeDetectorBase
from ..types import SystemChange, ChangeType, SystemSnapshot


logger = logging.getLogger(__name__)


class SecurityChangeDetector(SystemChangeDetectorBase):
    """
    Detector for security-related system changes.
    
    Monitors:
    - Authentication events (login/logout, failed attempts)
    - File permission and ownership changes
    - User and group account modifications
    - Security policy changes (sudoers, PAM, etc.)
    - Network security events
    - Process privilege escalations
    - Suspicious file system activities
    """
    
    def __init__(self):
        """Initialize security change detector."""
        super().__init__()
        
        # Security monitoring state
        self._last_auth_check = datetime.now()
        self._known_users = set()
        self._known_groups = set()
        self._monitored_security_files = {
            '/etc/passwd': 'user_accounts',
            '/etc/group': 'group_accounts',
            '/etc/shadow': 'password_hashes',
            '/etc/sudoers': 'sudo_permissions',
            '/etc/ssh/sshd_config': 'ssh_configuration',
            '/etc/pam.conf': 'pam_configuration',
            '/etc/security/limits.conf': 'security_limits'
        }
        
        # Authentication log paths
        self._auth_log_paths = [
            '/var/log/auth.log',
            '/var/log/secure',
            '/var/log/messages'
        ]
        
        # Suspicious activity patterns
        self._suspicious_patterns = {
            'failed_login': ['Failed password', 'authentication failure', 'FAILED LOGIN'],
            'privilege_escalation': ['sudo', 'su:', 'gained root'],
            'account_creation': ['new user', 'useradd', 'adduser'],
            'account_modification': ['usermod', 'chage'],
            'suspicious_commands': ['nc ', 'netcat', 'wget http', 'curl http', 'base64 -d']
        }
        
        # File permission tracking
        self._permission_cache = {}
        self._sensitive_paths = {
            '/etc/passwd',
            '/etc/shadow', 
            '/etc/sudoers',
            '/etc/ssh/',
            '/home/',
            '/root/',
            '/var/log/',
            '/etc/crontab'
        }
        
    def detect_changes(self, old_snapshot: SystemSnapshot, new_snapshot: SystemSnapshot) -> List[SystemChange]:
        """
        Detect security-related changes between snapshots.
        
        Args:
            old_snapshot: Previous system state
            new_snapshot: Current system state
            
        Returns:
            List of detected security changes
        """
        changes = []
        timestamp = datetime.now()
        
        # Extract security data from snapshots
        old_security = self._extract_security_data(old_snapshot)
        new_security = self._extract_security_data(new_snapshot)
        
        # User and group changes
        changes.extend(self._detect_user_group_changes(old_security, new_security, timestamp))
        
        # Authentication events
        changes.extend(self._detect_authentication_events(timestamp))
        
        # File permission changes
        changes.extend(self._detect_permission_changes(old_security, new_security, timestamp))
        
        # Security configuration changes
        changes.extend(self._detect_security_config_changes(old_security, new_security, timestamp))
        
        # Process privilege changes
        changes.extend(self._detect_privilege_changes(old_security, new_security, timestamp))
        
        # Suspicious activities
        changes.extend(self._detect_suspicious_activities(timestamp))
        
        return changes
    
    def get_current_security_state(self) -> Dict[str, Any]:
        """Get current security state for diagnostics."""
        try:
            security_state = {
                'users': [],
                'groups': [],
                'login_sessions': [],
                'security_files': {},
                'failed_logins_recent': 0,
                'sudo_activities_recent': 0
            }
            
            # Current users
            for user in pwd.getpwall():
                security_state['users'].append({
                    'name': user.pw_name,
                    'uid': user.pw_uid,
                    'gid': user.pw_gid,
                    'home': user.pw_dir,
                    'shell': user.pw_shell
                })
            
            # Current groups
            for group in grp.getgrall():
                security_state['groups'].append({
                    'name': group.gr_name,
                    'gid': group.gr_gid,
                    'members': group.gr_mem
                })
            
            # Security file status
            for file_path in self._monitored_security_files:
                if os.path.exists(file_path):
                    stat_info = os.stat(file_path)
                    security_state['security_files'][file_path] = {
                        'exists': True,
                        'mode': oct(stat_info.st_mode),
                        'uid': stat_info.st_uid,
                        'gid': stat_info.st_gid,
                        'mtime': datetime.fromtimestamp(stat_info.st_mtime).isoformat()
                    }
                else:
                    security_state['security_files'][file_path] = {'exists': False}
            
            # Recent authentication events summary
            security_state['failed_logins_recent'] = self._count_recent_auth_events('failed_login')
            security_state['sudo_activities_recent'] = self._count_recent_auth_events('privilege_escalation')
            
            return security_state
            
        except Exception as e:
            logger.error(f"Error getting security state: {e}")
            return {}
    
    def _extract_security_data(self, snapshot: SystemSnapshot) -> Dict[str, Any]:
        """Extract security-related data from system snapshot."""
        if not snapshot:
            return {}
        
        try:
            security_data = {
                'users': {},
                'groups': {},
                'processes': [],
                'security_files': {}
            }
            
            # User information
            for user in pwd.getpwall():
                security_data['users'][user.pw_name] = {
                    'uid': user.pw_uid,
                    'gid': user.pw_gid,
                    'home': user.pw_dir,
                    'shell': user.pw_shell
                }
            
            # Group information
            for group in grp.getgrall():
                security_data['groups'][group.gr_name] = {
                    'gid': group.gr_gid,
                    'members': group.gr_mem
                }
            
            # Process security context
            if hasattr(snapshot, 'processes') and snapshot.processes:
                for proc in snapshot.processes:
                    try:
                        if hasattr(proc, 'uids') and hasattr(proc, 'gids'):
                            uids = proc.uids()
                            gids = proc.gids()
                            security_data['processes'].append({
                                'pid': proc.pid,
                                'name': proc.name(),
                                'real_uid': uids.real,
                                'effective_uid': uids.effective,
                                'saved_uid': uids.saved,
                                'real_gid': gids.real,
                                'effective_gid': gids.effective,
                                'saved_gid': gids.saved
                            })
                    except:
                        continue
            
            # Security file status
            for file_path in self._monitored_security_files:
                try:
                    if os.path.exists(file_path):
                        stat_info = os.stat(file_path)
                        security_data['security_files'][file_path] = {
                            'mode': stat_info.st_mode,
                            'uid': stat_info.st_uid,
                            'gid': stat_info.st_gid,
                            'mtime': stat_info.st_mtime
                        }
                except:
                    continue
            
            return security_data
            
        except Exception as e:
            logger.error(f"Error extracting security data: {e}")
            return {}
    
    def _detect_user_group_changes(self, old_security: Dict[str, Any], 
                                  new_security: Dict[str, Any], timestamp: datetime) -> List[SystemChange]:
        """Detect changes in user and group accounts."""
        changes = []
        
        if not old_security or not new_security:
            return changes
        
        old_users = old_security.get('users', {})
        new_users = new_security.get('users', {})
        old_groups = old_security.get('groups', {})
        new_groups = new_security.get('groups', {})
        
        # New users
        new_user_names = set(new_users.keys()) - set(old_users.keys())
        for username in new_user_names:
            user_info = new_users[username]
            
            changes.append(SystemChange(
                category="security",
                change_type=ChangeType.ADDED,
                entity_id=f"user_{username}",
                old_value=None,
                new_value=username,
                significance=0.9,  # User creation is highly significant
                metadata={
                    'event_type': 'user_created',
                    'username': username,
                    'uid': user_info['uid'],
                    'home_directory': user_info['home'],
                    'shell': user_info['shell']
                },
                timestamp=timestamp
            ))
        
        # Removed users
        removed_users = set(old_users.keys()) - set(new_users.keys())
        for username in removed_users:
            changes.append(SystemChange(
                category="security",
                change_type=ChangeType.REMOVED,
                entity_id=f"user_{username}",
                old_value=username,
                new_value=None,
                significance=1.0,  # User removal is critical
                metadata={
                    'event_type': 'user_removed',
                    'username': username
                },
                timestamp=timestamp
            ))
        
        # Modified users
        common_users = set(old_users.keys()) & set(new_users.keys())
        for username in common_users:
            old_user = old_users[username]
            new_user = new_users[username]
            
            # Check for significant changes
            if old_user['shell'] != new_user['shell']:
                changes.append(SystemChange(
                    category="security",
                    change_type=ChangeType.MODIFIED,
                    entity_id=f"user_{username}_shell",
                    old_value=old_user['shell'],
                    new_value=new_user['shell'],
                    significance=0.7,
                    metadata={
                        'event_type': 'user_shell_changed',
                        'username': username
                    },
                    timestamp=timestamp
                ))
            
            if old_user['home'] != new_user['home']:
                changes.append(SystemChange(
                    category="security",
                    change_type=ChangeType.MODIFIED,
                    entity_id=f"user_{username}_home",
                    old_value=old_user['home'],
                    new_value=new_user['home'],
                    significance=0.6,
                    metadata={
                        'event_type': 'user_home_changed',
                        'username': username
                    },
                    timestamp=timestamp
                ))
        
        # New groups
        new_group_names = set(new_groups.keys()) - set(old_groups.keys())
        for groupname in new_group_names:
            group_info = new_groups[groupname]
            
            changes.append(SystemChange(
                category="security",
                change_type=ChangeType.ADDED,
                entity_id=f"group_{groupname}",
                old_value=None,
                new_value=groupname,
                significance=0.7,
                metadata={
                    'event_type': 'group_created',
                    'groupname': groupname,
                    'gid': group_info['gid'],
                    'members': group_info['members']
                },
                timestamp=timestamp
            ))
        
        # Group membership changes
        common_groups = set(old_groups.keys()) & set(new_groups.keys())
        for groupname in common_groups:
            old_members = set(old_groups[groupname]['members'])
            new_members = set(new_groups[groupname]['members'])
            
            added_members = new_members - old_members
            removed_members = old_members - new_members
            
            for member in added_members:
                changes.append(SystemChange(
                    category="security",
                    change_type=ChangeType.ADDED,
                    entity_id=f"group_{groupname}_member_{member}",
                    old_value=None,
                    new_value=member,
                    significance=0.8,
                    metadata={
                        'event_type': 'group_member_added',
                        'groupname': groupname,
                        'member': member
                    },
                    timestamp=timestamp
                ))
            
            for member in removed_members:
                changes.append(SystemChange(
                    category="security",
                    change_type=ChangeType.REMOVED,
                    entity_id=f"group_{groupname}_member_{member}",
                    old_value=member,
                    new_value=None,
                    significance=0.8,
                    metadata={
                        'event_type': 'group_member_removed',
                        'groupname': groupname,
                        'member': member
                    },
                    timestamp=timestamp
                ))
        
        return changes
    
    def _detect_authentication_events(self, timestamp: datetime) -> List[SystemChange]:
        """Detect authentication events from system logs."""
        changes = []
        
        try:
            # Check authentication logs for recent events
            cutoff_time = self._last_auth_check
            self._last_auth_check = timestamp
            
            for log_path in self._auth_log_paths:
                if os.path.exists(log_path):
                    changes.extend(self._parse_auth_log(log_path, cutoff_time, timestamp))
                    break  # Use first available log
            
        except Exception as e:
            logger.error(f"Error detecting authentication events: {e}")
        
        return changes
    
    def _parse_auth_log(self, log_path: str, cutoff_time: datetime, timestamp: datetime) -> List[SystemChange]:
        """Parse authentication events from log file."""
        changes = []
        
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
            
            # Check recent lines (last 1000 to avoid reading huge files)
            recent_lines = lines[-1000:]
            
            for line in recent_lines:
                line = line.strip()
                
                # Failed login attempts
                if any(pattern in line.lower() for pattern in self._suspicious_patterns['failed_login']):
                    # Extract username if possible
                    username = self._extract_username_from_log(line)
                    
                    changes.append(SystemChange(
                        category="security",
                        change_type=ChangeType.MODIFIED,
                        entity_id=f"failed_login_{username or 'unknown'}",
                        old_value=False,
                        new_value=True,
                        significance=0.8,
                        metadata={
                            'event_type': 'failed_authentication',
                            'username': username,
                            'log_entry': line,
                            'log_source': log_path
                        },
                        timestamp=timestamp
                    ))
                
                # Privilege escalation
                if any(pattern in line.lower() for pattern in self._suspicious_patterns['privilege_escalation']):
                    username = self._extract_username_from_log(line)
                    
                    changes.append(SystemChange(
                        category="security",
                        change_type=ChangeType.MODIFIED,
                        entity_id=f"privilege_escalation_{username or 'unknown'}",
                        old_value=False,
                        new_value=True,
                        significance=0.9,
                        metadata={
                            'event_type': 'privilege_escalation',
                            'username': username,
                            'log_entry': line,
                            'log_source': log_path
                        },
                        timestamp=timestamp
                    ))
                
                # Account creation/modification
                if any(pattern in line.lower() for pattern in self._suspicious_patterns['account_creation']):
                    username = self._extract_username_from_log(line)
                    
                    changes.append(SystemChange(
                        category="security",
                        change_type=ChangeType.ADDED,
                        entity_id=f"account_activity_{username or 'unknown'}",
                        old_value=None,
                        new_value=True,
                        significance=0.7,
                        metadata={
                            'event_type': 'account_management',
                            'username': username,
                            'log_entry': line,
                            'log_source': log_path
                        },
                        timestamp=timestamp
                    ))
                        
        except Exception as e:
            logger.error(f"Error parsing auth log {log_path}: {e}")
        
        return changes
    
    def _detect_permission_changes(self, old_security: Dict[str, Any], 
                                  new_security: Dict[str, Any], timestamp: datetime) -> List[SystemChange]:
        """Detect file permission and ownership changes."""
        changes = []
        
        old_files = old_security.get('security_files', {})
        new_files = new_security.get('security_files', {})
        
        # Check for changes in monitored security files
        for file_path in set(old_files.keys()) | set(new_files.keys()):
            old_file = old_files.get(file_path, {})
            new_file = new_files.get(file_path, {})
            
            # File permission changes
            if old_file.get('mode') and new_file.get('mode'):
                if old_file['mode'] != new_file['mode']:
                    changes.append(SystemChange(
                        category="security",
                        change_type=ChangeType.MODIFIED,
                        entity_id=f"file_permissions_{file_path.replace('/', '_')}",
                        old_value=oct(old_file['mode']),
                        new_value=oct(new_file['mode']),
                        significance=0.9,  # Permission changes are highly significant
                        metadata={
                            'event_type': 'permission_change',
                            'file_path': file_path,
                            'file_type': self._monitored_security_files.get(file_path, 'unknown')
                        },
                        timestamp=timestamp
                    ))
            
            # Ownership changes
            if (old_file.get('uid') and new_file.get('uid') and 
                old_file['uid'] != new_file['uid']):
                
                changes.append(SystemChange(
                    category="security",
                    change_type=ChangeType.MODIFIED,
                    entity_id=f"file_ownership_{file_path.replace('/', '_')}",
                    old_value=old_file['uid'],
                    new_value=new_file['uid'],
                    significance=0.9,
                    metadata={
                        'event_type': 'ownership_change',
                        'file_path': file_path,
                        'change_type': 'uid'
                    },
                    timestamp=timestamp
                ))
            
            if (old_file.get('gid') and new_file.get('gid') and 
                old_file['gid'] != new_file['gid']):
                
                changes.append(SystemChange(
                    category="security",
                    change_type=ChangeType.MODIFIED,
                    entity_id=f"file_group_{file_path.replace('/', '_')}",
                    old_value=old_file['gid'],
                    new_value=new_file['gid'],
                    significance=0.9,
                    metadata={
                        'event_type': 'ownership_change',
                        'file_path': file_path,
                        'change_type': 'gid'
                    },
                    timestamp=timestamp
                ))
        
        return changes
    
    def _detect_security_config_changes(self, old_security: Dict[str, Any], 
                                       new_security: Dict[str, Any], timestamp: datetime) -> List[SystemChange]:
        """Detect security configuration changes."""
        changes = []
        
        # File modification time changes for security files
        old_files = old_security.get('security_files', {})
        new_files = new_security.get('security_files', {})
        
        for file_path in set(old_files.keys()) & set(new_files.keys()):
            old_file = old_files[file_path]
            new_file = new_files[file_path]
            
            old_mtime = old_file.get('mtime', 0)
            new_mtime = new_file.get('mtime', 0)
            
            if old_mtime != new_mtime and new_mtime > old_mtime:
                changes.append(SystemChange(
                    category="security",
                    change_type=ChangeType.MODIFIED,
                    entity_id=f"security_config_{file_path.replace('/', '_')}",
                    old_value=old_mtime,
                    new_value=new_mtime,
                    significance=1.0,  # Security config changes are critical
                    metadata={
                        'event_type': 'security_config_modified',
                        'file_path': file_path,
                        'config_type': self._monitored_security_files.get(file_path, 'unknown'),
                        'modification_time': datetime.fromtimestamp(new_mtime).isoformat()
                    },
                    timestamp=timestamp
                ))
        
        return changes
    
    def _detect_privilege_changes(self, old_security: Dict[str, Any], 
                                 new_security: Dict[str, Any], timestamp: datetime) -> List[SystemChange]:
        """Detect process privilege escalation or changes."""
        changes = []
        
        old_processes = {p['pid']: p for p in old_security.get('processes', [])}
        new_processes = {p['pid']: p for p in new_security.get('processes', [])}
        
        # Check for UID/GID changes in existing processes
        common_pids = set(old_processes.keys()) & set(new_processes.keys())
        
        for pid in common_pids:
            old_proc = old_processes[pid]
            new_proc = new_processes[pid]
            
            # Effective UID change (privilege escalation/de-escalation)
            if old_proc['effective_uid'] != new_proc['effective_uid']:
                escalation = new_proc['effective_uid'] < old_proc['effective_uid']  # Lower UID = more privileges
                
                changes.append(SystemChange(
                    category="security",
                    change_type=ChangeType.MODIFIED,
                    entity_id=f"process_{pid}_privileges",
                    old_value=old_proc['effective_uid'],
                    new_value=new_proc['effective_uid'],
                    significance=0.9 if escalation else 0.7,
                    metadata={
                        'event_type': 'privilege_escalation' if escalation else 'privilege_de_escalation',
                        'pid': pid,
                        'process_name': new_proc['name'],
                        'old_euid': old_proc['effective_uid'],
                        'new_euid': new_proc['effective_uid']
                    },
                    timestamp=timestamp
                ))
        
        return changes
    
    def _detect_suspicious_activities(self, timestamp: datetime) -> List[SystemChange]:
        """Detect suspicious activities from various sources."""
        changes = []
        
        try:
            # Check for suspicious command patterns in bash history (if accessible)
            changes.extend(self._check_suspicious_commands(timestamp))
            
            # Check for unusual network connections (if we had network data)
            # This would require integration with network monitoring
            
            # Check for unusual file access patterns
            # This would require file access monitoring (inotify, etc.)
            
        except Exception as e:
            logger.error(f"Error detecting suspicious activities: {e}")
        
        return changes
    
    def _check_suspicious_commands(self, timestamp: datetime) -> List[SystemChange]:
        """Check for suspicious commands in system logs or process list."""
        changes = []
        
        # This is a simplified implementation
        # In practice, you'd check current processes for suspicious patterns
        
        return changes
    
    def _extract_username_from_log(self, log_line: str) -> Optional[str]:
        """Extract username from log line."""
        # Simple username extraction - would need more sophisticated parsing
        parts = log_line.split()
        for i, part in enumerate(parts):
            if part.lower() in ['user', 'for'] and i + 1 < len(parts):
                return parts[i + 1].strip(':')
        return None
    
    def _count_recent_auth_events(self, event_type: str) -> int:
        """Count recent authentication events of specific type."""
        # Simplified - would check actual logs
        return 0
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security status summary."""
        try:
            current_state = self.get_current_security_state()
            
            return {
                'total_users': len(current_state.get('users', [])),
                'total_groups': len(current_state.get('groups', [])),
                'failed_logins_recent': current_state.get('failed_logins_recent', 0),
                'sudo_activities_recent': current_state.get('sudo_activities_recent', 0),
                'security_files_monitored': len([f for f in current_state.get('security_files', {}).values() 
                                               if f.get('exists', False)]),
                'security_recommendation': self._get_security_recommendation(current_state)
            }
            
        except Exception as e:
            logger.error(f"Error getting security summary: {e}")
            return {}
    
    def _get_security_recommendation(self, state: Dict[str, Any]) -> str:
        """Get security recommendation based on current state."""
        failed_logins = state.get('failed_logins_recent', 0)
        
        if failed_logins > 10:
            return "HIGH: Multiple failed login attempts detected - review access controls"
        elif failed_logins > 5:
            return "MEDIUM: Several failed login attempts - monitor authentication logs"
        else:
            return "LOW: No immediate security concerns detected"
    
    def get_detector_stats(self) -> Dict[str, Any]:
        """Get detector statistics and current state."""
        current_state = self.get_current_security_state()
        security_summary = self.get_security_summary()
        
        return {
            'detector_type': 'security',
            'current_security_state': current_state,
            'security_summary': security_summary,
            'monitoring': {
                'monitored_security_files': list(self._monitored_security_files.keys()),
                'auth_log_paths': self._auth_log_paths,
                'suspicious_patterns': list(self._suspicious_patterns.keys())
            },
            'last_auth_check': self._last_auth_check.isoformat()
        }