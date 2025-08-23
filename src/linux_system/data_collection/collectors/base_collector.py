"""
Base Collector
=============

Abstract base class for all Ubuntu system data collectors.
Provides common functionality for safe, structured data collection.
"""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import subprocess
import json
import hashlib


class BaseCollector(ABC):
    """Base class for all data collectors with safety and structure."""
    
    def __init__(self, name: str, collection_interval: int = 60):
        """
        Initialize base collector.
        
        Args:
            name: Collector name for logging
            collection_interval: Default collection interval in seconds
        """
        self.name = name
        self.collection_interval = collection_interval
        self.logger = logging.getLogger(f"collectors.{name}")
        self.last_collection_time = None
        self.collection_count = 0
        
        # Safety settings
        self.max_file_size = 50 * 1024 * 1024  # 50MB max file read
        self.max_command_timeout = 30  # 30 second command timeout
        self.allowed_commands = self._get_allowed_commands()
        
    @abstractmethod
    def collect(self) -> Dict[str, Any]:
        """
        Collect data for this collector.
        
        Returns:
            Dictionary with structured data and metadata
        """
        pass
    
    @abstractmethod
    def _get_allowed_commands(self) -> List[str]:
        """Return list of allowed read-only commands for this collector."""
        pass
    
    def safe_read_file(self, file_path: str, max_lines: Optional[int] = None) -> Optional[str]:
        """
        Safely read a file with size and permission checks.
        
        Args:
            file_path: Path to file to read
            max_lines: Maximum lines to read (None for all)
            
        Returns:
            File content or None if failed
        """
        try:
            path = Path(file_path)
            
            # Check if file exists and is readable
            if not path.exists():
                self.logger.warning(f"File does not exist: {file_path}")
                return None
                
            if not path.is_file():
                self.logger.warning(f"Path is not a file: {file_path}")
                return None
                
            # Check file size
            if path.stat().st_size > self.max_file_size:
                self.logger.warning(f"File too large: {file_path} ({path.stat().st_size} bytes)")
                return None
            
            # Read file content
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                if max_lines:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        lines.append(line)
                    return ''.join(lines)
                else:
                    return f.read()
                    
        except PermissionError:
            self.logger.warning(f"Permission denied reading: {file_path}")
            return None
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {e}")
            return None
    
    def safe_execute_command(self, command: List[str]) -> Optional[Dict[str, Any]]:
        """
        Safely execute a read-only command with validation.
        
        Args:
            command: Command and arguments as list
            
        Returns:
            Dictionary with stdout, stderr, return_code, or None if failed
        """
        try:
            # Validate command is allowed
            if not command or command[0] not in self.allowed_commands:
                self.logger.warning(f"Command not allowed: {' '.join(command)}")
                return None
            
            # Execute with timeout
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.max_command_timeout,
                encoding='utf-8',
                errors='ignore'
            )
            
            return {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode,
                'command': ' '.join(command),
                'timestamp': datetime.now().isoformat()
            }
            
        except subprocess.TimeoutExpired:
            self.logger.warning(f"Command timed out: {' '.join(command)}")
            return None
        except FileNotFoundError:
            self.logger.warning(f"Command not found: {command[0]}")
            return None
        except Exception as e:
            self.logger.error(f"Error executing {' '.join(command)}: {e}")
            return None
    
    def create_data_snapshot(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a standardized data snapshot with metadata.
        
        Args:
            data: Raw collected data
            
        Returns:
            Structured data snapshot with metadata
        """
        timestamp = datetime.now()
        
        # Create data hash for change detection
        data_str = json.dumps(data, sort_keys=True, default=str)
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()[:16]
        
        snapshot = {
            'metadata': {
                'collector': self.name,
                'timestamp': timestamp.isoformat(),
                'collection_count': self.collection_count,
                'data_hash': data_hash,
                'collection_duration_ms': None  # Set by caller if needed
            },
            'data': data
        }
        
        self.last_collection_time = timestamp
        self.collection_count += 1
        
        return snapshot
    
    def should_collect(self) -> bool:
        """
        Check if enough time has passed for next collection.
        
        Returns:
            True if collection should run
        """
        if self.last_collection_time is None:
            return True
            
        elapsed = (datetime.now() - self.last_collection_time).total_seconds()
        return elapsed >= self.collection_interval
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get collector status information.
        
        Returns:
            Status dictionary
        """
        return {
            'name': self.name,
            'collection_interval': self.collection_interval,
            'last_collection_time': self.last_collection_time.isoformat() if self.last_collection_time else None,
            'collection_count': self.collection_count,
            'status': 'active'
        }