"""
Compression Utilities
====================

Utilities for compressing temporal data while preserving semantic information.
Implements adaptive compression based on data age, significance, and patterns.
"""

import json
import gzip
import pickle
import lzma
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import threading
import hashlib

from ..types import SystemDelta, SystemChange, SystemEvent


class CompressionLevel(Enum):
    """Compression level for temporal data."""
    NONE = "none"
    LIGHT = "light"
    MEDIUM = "medium"
    HEAVY = "heavy"
    MAXIMUM = "maximum"


class CompressionType(Enum):
    """Type of compression algorithm to use."""
    GZIP = "gzip"
    LZMA = "lzma" 
    PICKLE = "pickle"
    JSON_GZIP = "json_gzip"
    ADAPTIVE = "adaptive"


@dataclass
class CompressionResult:
    """Result of compression operation."""
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_type: CompressionType
    compression_level: CompressionLevel
    compression_time_ms: float
    checksum: str
    metadata: Dict[str, Any]


@dataclass
class CompressionStats:
    """Statistics about compression operations."""
    total_operations: int
    total_original_size: int
    total_compressed_size: int
    total_time_ms: float
    average_compression_ratio: float
    best_compression_ratio: float
    worst_compression_ratio: float
    compression_type_stats: Dict[str, int]
    level_stats: Dict[str, int]


class TemporalCompressor:
    """
    Adaptive compression for temporal system data.
    
    Provides intelligent compression based on data age, significance,
    and content patterns to optimize storage while preserving queryability.
    """
    
    def __init__(self, default_level: CompressionLevel = CompressionLevel.MEDIUM):
        self.default_level = default_level
        self._stats = CompressionStats(
            total_operations=0,
            total_original_size=0,
            total_compressed_size=0,
            total_time_ms=0.0,
            average_compression_ratio=0.0,
            best_compression_ratio=0.0,
            worst_compression_ratio=float('inf'),
            compression_type_stats={},
            level_stats={}
        )
        self._lock = threading.RLock()
    
    def compress_deltas(self, deltas: List[SystemDelta], 
                       compression_type: CompressionType = CompressionType.ADAPTIVE,
                       compression_level: Optional[CompressionLevel] = None) -> CompressionResult:
        """
        Compress a list of SystemDelta objects.
        
        Args:
            deltas: List of deltas to compress
            compression_type: Type of compression to use
            compression_level: Level of compression (None for default)
            
        Returns:
            CompressionResult with compression details
        """
        if not deltas:
            return CompressionResult(
                original_size=0,
                compressed_size=0,
                compression_ratio=0.0,
                compression_type=compression_type,
                compression_level=compression_level or self.default_level,
                compression_time_ms=0.0,
                checksum="",
                metadata={}
            )
        
        level = compression_level or self.default_level
        start_time = datetime.now()
        
        # Prepare data for compression
        prepared_data = self._prepare_deltas_for_compression(deltas, level)
        original_data = json.dumps(prepared_data, separators=(',', ':')).encode('utf-8')
        original_size = len(original_data)
        
        # Choose optimal compression type if adaptive
        if compression_type == CompressionType.ADAPTIVE:
            compression_type = self._choose_optimal_compression(original_data, level)
        
        # Perform compression
        compressed_data = self._compress_data(original_data, compression_type, level)
        compressed_size = len(compressed_data)
        
        # Calculate metrics
        compression_time = (datetime.now() - start_time).total_seconds() * 1000
        compression_ratio = compressed_size / original_size if original_size > 0 else 0.0
        checksum = hashlib.sha256(original_data).hexdigest()
        
        result = CompressionResult(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            compression_type=compression_type,
            compression_level=level,
            compression_time_ms=compression_time,
            checksum=checksum,
            metadata={
                'delta_count': len(deltas),
                'time_range': self._get_time_range(deltas),
                'compression_algorithm': compression_type.value
            }
        )
        
        # Update statistics
        self._update_stats(result)
        
        return result
    
    def decompress_deltas(self, compressed_data: bytes, 
                         compression_result: CompressionResult) -> List[SystemDelta]:
        """
        Decompress data back to SystemDelta objects.
        
        Args:
            compressed_data: Compressed data bytes
            compression_result: Original compression result for metadata
            
        Returns:
            List of decompressed SystemDelta objects
        """
        # Decompress data
        decompressed_data = self._decompress_data(
            compressed_data, 
            compression_result.compression_type
        )
        
        # Verify checksum
        actual_checksum = hashlib.sha256(decompressed_data).hexdigest()
        if actual_checksum != compression_result.checksum:
            raise ValueError("Data integrity check failed - checksum mismatch")
        
        # Parse JSON and reconstruct deltas
        json_data = json.loads(decompressed_data.decode('utf-8'))
        return self._reconstruct_deltas_from_compressed(json_data)
    
    def compress_daily_summary(self, summary_data: Dict[str, Any],
                              compression_level: Optional[CompressionLevel] = None) -> bytes:
        """
        Compress daily summary data.
        
        Args:
            summary_data: Daily summary dictionary
            compression_level: Compression level to use
            
        Returns:
            Compressed data bytes
        """
        level = compression_level or self.default_level
        
        # Convert to JSON
        json_data = json.dumps(summary_data, separators=(',', ':')).encode('utf-8')
        
        # Compress based on level
        if level in [CompressionLevel.HEAVY, CompressionLevel.MAXIMUM]:
            return lzma.compress(json_data, preset=9)
        else:
            return gzip.compress(json_data, compresslevel=6)
    
    def decompress_daily_summary(self, compressed_data: bytes,
                                compression_type: Optional[CompressionType] = None) -> Dict[str, Any]:
        """
        Decompress daily summary data.
        
        Args:
            compressed_data: Compressed summary data
            compression_type: Type of compression used
            
        Returns:
            Decompressed summary dictionary
        """
        # Auto-detect compression type if not provided
        if compression_type is None:
            compression_type = self._detect_compression_type(compressed_data)
        
        # Decompress
        if compression_type == CompressionType.LZMA:
            json_data = lzma.decompress(compressed_data)
        else:
            json_data = gzip.decompress(compressed_data)
        
        return json.loads(json_data.decode('utf-8'))
    
    def estimate_compression_benefit(self, data_size: int, 
                                   compression_type: CompressionType = CompressionType.ADAPTIVE) -> Dict[str, float]:
        """
        Estimate compression benefit for given data size.
        
        Args:
            data_size: Size of data in bytes
            compression_type: Type of compression to estimate
            
        Returns:
            Dictionary with estimated metrics
        """
        # Use historical stats to estimate
        if self._stats.total_operations == 0:
            # Default estimates
            estimates = {
                CompressionType.GZIP: 0.3,
                CompressionType.LZMA: 0.25,
                CompressionType.JSON_GZIP: 0.35,
                CompressionType.PICKLE: 0.4
            }
        else:
            # Use actual performance data
            avg_ratio = self._stats.average_compression_ratio
            estimates = {
                CompressionType.GZIP: avg_ratio * 1.1,
                CompressionType.LZMA: avg_ratio * 0.9,
                CompressionType.JSON_GZIP: avg_ratio * 1.2,
                CompressionType.PICKLE: avg_ratio * 1.3
            }
        
        if compression_type == CompressionType.ADAPTIVE:
            best_type = min(estimates.keys(), key=lambda k: estimates[k])
            estimated_ratio = estimates[best_type]
        else:
            estimated_ratio = estimates.get(compression_type, 0.4)
        
        return {
            'estimated_compressed_size': int(data_size * estimated_ratio),
            'estimated_space_saved': int(data_size * (1 - estimated_ratio)),
            'estimated_compression_ratio': estimated_ratio,
            'recommended_type': compression_type.value if compression_type != CompressionType.ADAPTIVE else best_type.value
        }
    
    def get_compression_stats(self) -> CompressionStats:
        """Get current compression statistics."""
        with self._lock:
            return CompressionStats(
                total_operations=self._stats.total_operations,
                total_original_size=self._stats.total_original_size,
                total_compressed_size=self._stats.total_compressed_size,
                total_time_ms=self._stats.total_time_ms,
                average_compression_ratio=self._stats.average_compression_ratio,
                best_compression_ratio=self._stats.best_compression_ratio,
                worst_compression_ratio=self._stats.worst_compression_ratio,
                compression_type_stats=self._stats.compression_type_stats.copy(),
                level_stats=self._stats.level_stats.copy()
            )
    
    def _prepare_deltas_for_compression(self, deltas: List[SystemDelta], 
                                      level: CompressionLevel) -> Dict[str, Any]:
        """Prepare deltas for compression based on level."""
        prepared = {
            'deltas': [],
            'metadata': {
                'compression_level': level.value,
                'delta_count': len(deltas),
                'preparation_timestamp': datetime.now().isoformat()
            }
        }
        
        for delta in deltas:
            delta_dict = {
                'timestamp': delta.timestamp.isoformat(),
                'collection_duration_ms': getattr(delta, 'collection_duration_ms', 0)
            }
            
            # Compress changes based on level
            if level == CompressionLevel.LIGHT:
                # Keep all data but optimize serialization
                delta_dict['raw_delta'] = [self._serialize_change_light(change) for change in delta.raw_delta]
                delta_dict['semantic_events'] = [self._serialize_event_light(event) for event in delta.semantic_events]
                
            elif level == CompressionLevel.MEDIUM:
                # Remove low-significance changes and redundant data
                significant_changes = [c for c in delta.raw_delta if c.significance >= 0.3]
                delta_dict['raw_delta'] = [self._serialize_change_medium(change) for change in significant_changes]
                
                important_events = [e for e in delta.semantic_events if e.confidence >= 0.5]
                delta_dict['semantic_events'] = [self._serialize_event_medium(event) for event in important_events]
                
            elif level == CompressionLevel.HEAVY:
                # Keep only highly significant data and summaries
                critical_changes = [c for c in delta.raw_delta if c.significance >= 0.7]
                delta_dict['raw_delta'] = [self._serialize_change_heavy(change) for change in critical_changes]
                
                critical_events = [e for e in delta.semantic_events if e.confidence >= 0.8]
                delta_dict['semantic_events'] = [self._serialize_event_heavy(event) for event in critical_events]
                
            elif level == CompressionLevel.MAXIMUM:
                # Keep only essential summary data
                delta_dict['summary'] = self._create_delta_summary(delta)
                
            else:  # NONE
                # No compression preparation
                delta_dict['raw_delta'] = [self._serialize_change_full(change) for change in delta.raw_delta]
                delta_dict['semantic_events'] = [self._serialize_event_full(event) for event in delta.semantic_events]
            
            prepared['deltas'].append(delta_dict)
        
        return prepared
    
    def _compress_data(self, data: bytes, compression_type: CompressionType, 
                      level: CompressionLevel) -> bytes:
        """Compress data using specified algorithm."""
        if compression_type == CompressionType.GZIP:
            compress_level = self._map_level_to_gzip(level)
            return gzip.compress(data, compresslevel=compress_level)
            
        elif compression_type == CompressionType.LZMA:
            preset = self._map_level_to_lzma(level)
            return lzma.compress(data, preset=preset)
            
        elif compression_type == CompressionType.PICKLE:
            # Use pickle with compression
            return gzip.compress(pickle.dumps(data), compresslevel=6)
            
        elif compression_type == CompressionType.JSON_GZIP:
            return gzip.compress(data, compresslevel=6)
            
        else:
            return data
    
    def _decompress_data(self, data: bytes, compression_type: CompressionType) -> bytes:
        """Decompress data using specified algorithm."""
        if compression_type == CompressionType.GZIP:
            return gzip.decompress(data)
            
        elif compression_type == CompressionType.LZMA:
            return lzma.decompress(data)
            
        elif compression_type == CompressionType.PICKLE:
            return pickle.loads(gzip.decompress(data))
            
        elif compression_type == CompressionType.JSON_GZIP:
            return gzip.decompress(data)
            
        else:
            return data
    
    def _choose_optimal_compression(self, data: bytes, level: CompressionLevel) -> CompressionType:
        """Choose optimal compression type based on data characteristics."""
        data_size = len(data)
        
        # Small data - use gzip for speed
        if data_size < 1024:
            return CompressionType.GZIP
        
        # Large data with high compression level - use LZMA
        if data_size > 100000 and level in [CompressionLevel.HEAVY, CompressionLevel.MAXIMUM]:
            return CompressionType.LZMA
        
        # Test sample compression ratios for medium-sized data
        if data_size > 10000:
            sample_size = min(1024, data_size // 10)
            sample = data[:sample_size]
            
            gzip_ratio = len(gzip.compress(sample)) / sample_size
            lzma_ratio = len(lzma.compress(sample, preset=1)) / sample_size
            
            return CompressionType.LZMA if lzma_ratio < gzip_ratio * 0.9 else CompressionType.GZIP
        
        # Default to gzip for balanced performance
        return CompressionType.GZIP
    
    def _detect_compression_type(self, data: bytes) -> CompressionType:
        """Auto-detect compression type from data headers."""
        if data.startswith(b'\x1f\x8b'):  # gzip magic number
            return CompressionType.GZIP
        elif data.startswith(b'\xfd7zXZ'):  # xz/lzma magic number
            return CompressionType.LZMA
        else:
            return CompressionType.GZIP  # Default assumption
    
    def _serialize_change_light(self, change: SystemChange) -> Dict[str, Any]:
        """Light compression - optimize serialization only."""
        return {
            'cat': change.category,
            'type': change.change_type.value,
            'id': change.entity_id,
            'old': self._serialize_value_optimized(change.old_value),
            'new': self._serialize_value_optimized(change.new_value),
            'sig': round(change.significance, 3),
            'ts': change.timestamp.isoformat()[:19]  # Remove microseconds
        }
    
    def _serialize_change_medium(self, change: SystemChange) -> Dict[str, Any]:
        """Medium compression - remove some metadata."""
        return {
            'cat': change.category,
            'type': change.change_type.value,
            'id': change.entity_id[:20],  # Truncate long IDs
            'sig': round(change.significance, 2),
            'ts': change.timestamp.isoformat()[:16]  # Minute precision
        }
    
    def _serialize_change_heavy(self, change: SystemChange) -> Dict[str, Any]:
        """Heavy compression - minimal data."""
        return {
            'cat': change.category[:10],
            'type': change.change_type.value[0],  # First letter only
            'sig': int(change.significance * 10),  # Single digit
            'hour': change.timestamp.hour
        }
    
    def _serialize_event_light(self, event: SystemEvent) -> Dict[str, Any]:
        """Light event compression."""
        return {
            'type': event.event_type,
            'entity': event.entity,
            'desc': event.description[:100],  # Truncate description
            'sev': event.severity.value,
            'conf': round(event.confidence, 3),
            'ts': event.timestamp.isoformat()[:19]
        }
    
    def _serialize_event_medium(self, event: SystemEvent) -> Dict[str, Any]:
        """Medium event compression."""
        return {
            'type': event.event_type[:20],
            'entity': event.entity[:30],
            'sev': event.severity.value[0],  # First letter
            'conf': round(event.confidence, 2),
            'ts': event.timestamp.isoformat()[:16]
        }
    
    def _serialize_event_heavy(self, event: SystemEvent) -> Dict[str, Any]:
        """Heavy event compression."""
        return {
            'type': event.event_type[:10],
            'sev': event.severity.value[0],
            'conf': int(event.confidence * 10),
            'hour': event.timestamp.hour
        }
    
    def _serialize_change_full(self, change: SystemChange) -> Dict[str, Any]:
        """Full serialization without compression."""
        return {
            'category': change.category,
            'change_type': change.change_type.value,
            'entity_id': change.entity_id,
            'old_value': change.old_value,
            'new_value': change.new_value,
            'significance': change.significance,
            'metadata': change.metadata,
            'timestamp': change.timestamp.isoformat()
        }
    
    def _serialize_event_full(self, event: SystemEvent) -> Dict[str, Any]:
        """Full event serialization without compression."""
        return {
            'event_type': event.event_type,
            'entity': event.entity,
            'description': event.description,
            'severity': event.severity.value,
            'timestamp': event.timestamp.isoformat(),
            'confidence': event.confidence,
            'context': event.context,
            'predicted_effects': event.predicted_effects
        }
    
    def _serialize_value_optimized(self, value: Any) -> Any:
        """Optimize value serialization."""
        if isinstance(value, datetime):
            return value.isoformat()[:19]  # Remove microseconds
        elif isinstance(value, float):
            return round(value, 3)  # Limit precision
        elif isinstance(value, str) and len(value) > 100:
            return value[:100]  # Truncate long strings
        else:
            return value
    
    def _create_delta_summary(self, delta: SystemDelta) -> Dict[str, Any]:
        """Create summary of delta for maximum compression."""
        return {
            'ts': delta.timestamp.isoformat()[:16],
            'changes': len(delta.raw_delta),
            'events': len(delta.semantic_events),
            'categories': list(set(c.category for c in delta.raw_delta))[:5],
            'avg_significance': sum(c.significance for c in delta.raw_delta) / len(delta.raw_delta) if delta.raw_delta else 0,
            'max_severity': max((e.severity.value for e in delta.semantic_events), default='low')
        }
    
    def _get_time_range(self, deltas: List[SystemDelta]) -> Dict[str, str]:
        """Get time range of deltas."""
        if not deltas:
            return {}
        
        timestamps = [d.timestamp for d in deltas]
        return {
            'start': min(timestamps).isoformat(),
            'end': max(timestamps).isoformat(),
            'span_hours': (max(timestamps) - min(timestamps)).total_seconds() / 3600
        }
    
    def _map_level_to_gzip(self, level: CompressionLevel) -> int:
        """Map compression level to gzip level."""
        mapping = {
            CompressionLevel.LIGHT: 3,
            CompressionLevel.MEDIUM: 6,
            CompressionLevel.HEAVY: 9,
            CompressionLevel.MAXIMUM: 9
        }
        return mapping.get(level, 6)
    
    def _map_level_to_lzma(self, level: CompressionLevel) -> int:
        """Map compression level to LZMA preset."""
        mapping = {
            CompressionLevel.LIGHT: 1,
            CompressionLevel.MEDIUM: 4,
            CompressionLevel.HEAVY: 7,
            CompressionLevel.MAXIMUM: 9
        }
        return mapping.get(level, 4)
    
    def _reconstruct_deltas_from_compressed(self, data: Dict[str, Any]) -> List[SystemDelta]:
        """Reconstruct SystemDeltas from compressed data."""
        # This is a simplified reconstruction - in practice would need
        # more sophisticated logic to handle different compression levels
        from ..types import SystemDelta, SystemChange, SystemEvent, ChangeType, EventSeverity
        
        deltas = []
        compression_level = data.get('metadata', {}).get('compression_level', 'medium')
        
        for delta_dict in data.get('deltas', []):
            if 'summary' in delta_dict:
                # Maximum compression - reconstruct from summary
                delta = self._reconstruct_from_summary(delta_dict)
            else:
                # Standard reconstruction based on available data
                timestamp = datetime.fromisoformat(delta_dict['timestamp'])
                
                # Reconstruct changes
                changes = []
                for change_dict in delta_dict.get('raw_delta', []):
                    change = self._reconstruct_change(change_dict, compression_level)
                    if change:
                        changes.append(change)
                
                # Reconstruct events
                events = []
                for event_dict in delta_dict.get('semantic_events', []):
                    event = self._reconstruct_event(event_dict, compression_level)
                    if event:
                        events.append(event)
                
                delta = SystemDelta(
                    timestamp=timestamp,
                    raw_delta=changes,
                    semantic_events=events,
                    correlations=[],  # Correlations not preserved in compression
                    snapshot_metadata={}
                )
            
            deltas.append(delta)
        
        return deltas
    
    def _reconstruct_change(self, change_dict: Dict[str, Any], compression_level: str) -> Optional[SystemChange]:
        """Reconstruct SystemChange from compressed data."""
        from ..types import SystemChange, ChangeType
        
        try:
            if compression_level == 'light':
                return SystemChange(
                    category=change_dict['cat'],
                    change_type=ChangeType(change_dict['type']),
                    entity_id=change_dict['id'],
                    old_value=change_dict.get('old'),
                    new_value=change_dict.get('new'),
                    significance=change_dict['sig'],
                    timestamp=datetime.fromisoformat(change_dict['ts'])
                )
            elif compression_level in ['medium', 'heavy']:
                return SystemChange(
                    category=change_dict['cat'],
                    change_type=ChangeType(change_dict['type']),
                    entity_id=change_dict['id'],
                    significance=change_dict['sig'],
                    timestamp=self._reconstruct_timestamp_from_partial(change_dict)
                )
            else:
                return None
        except (KeyError, ValueError):
            return None
    
    def _reconstruct_event(self, event_dict: Dict[str, Any], compression_level: str) -> Optional[SystemEvent]:
        """Reconstruct SystemEvent from compressed data."""
        from ..types import SystemEvent, EventSeverity
        
        try:
            severity_map = {'h': 'high', 'm': 'medium', 'l': 'low', 'c': 'critical'}
            
            if compression_level == 'light':
                return SystemEvent(
                    event_type=event_dict['type'],
                    entity=event_dict['entity'],
                    description=event_dict['desc'],
                    severity=EventSeverity(event_dict['sev']),
                    confidence=event_dict['conf'],
                    timestamp=datetime.fromisoformat(event_dict['ts'])
                )
            elif compression_level in ['medium', 'heavy']:
                severity_value = severity_map.get(event_dict['sev'], 'medium')
                return SystemEvent(
                    event_type=event_dict['type'],
                    entity=event_dict.get('entity', ''),
                    description=f"Compressed event: {event_dict['type']}",
                    severity=EventSeverity(severity_value),
                    confidence=event_dict.get('conf', 0.5),
                    timestamp=self._reconstruct_timestamp_from_partial(event_dict)
                )
            else:
                return None
        except (KeyError, ValueError):
            return None
    
    def _reconstruct_from_summary(self, summary_dict: Dict[str, Any]) -> SystemDelta:
        """Reconstruct delta from maximum compression summary."""
        from ..types import SystemDelta
        
        timestamp = datetime.fromisoformat(summary_dict['summary']['ts'])
        
        # Create minimal delta from summary
        return SystemDelta(
            timestamp=timestamp,
            raw_delta=[],  # No detailed changes preserved
            semantic_events=[],  # No detailed events preserved
            correlations=[],
            snapshot_metadata={
                'compressed_summary': summary_dict['summary'],
                'compression_level': 'maximum'
            }
        )
    
    def _reconstruct_timestamp_from_partial(self, data_dict: Dict[str, Any]) -> datetime:
        """Reconstruct timestamp from partial data."""
        if 'ts' in data_dict:
            return datetime.fromisoformat(data_dict['ts'])
        elif 'hour' in data_dict:
            # Approximate timestamp from hour
            now = datetime.now()
            return now.replace(hour=data_dict['hour'], minute=0, second=0, microsecond=0)
        else:
            return datetime.now()
    
    def _update_stats(self, result: CompressionResult) -> None:
        """Update compression statistics."""
        with self._lock:
            self._stats.total_operations += 1
            self._stats.total_original_size += result.original_size
            self._stats.total_compressed_size += result.compressed_size
            self._stats.total_time_ms += result.compression_time_ms
            
            # Update compression ratio stats
            if result.compression_ratio > 0:
                if self._stats.total_operations == 1:
                    self._stats.average_compression_ratio = result.compression_ratio
                    self._stats.best_compression_ratio = result.compression_ratio
                    self._stats.worst_compression_ratio = result.compression_ratio
                else:
                    # Running average
                    self._stats.average_compression_ratio = (
                        (self._stats.average_compression_ratio * (self._stats.total_operations - 1) + 
                         result.compression_ratio) / self._stats.total_operations
                    )
                    self._stats.best_compression_ratio = min(self._stats.best_compression_ratio, result.compression_ratio)
                    self._stats.worst_compression_ratio = max(self._stats.worst_compression_ratio, result.compression_ratio)
            
            # Update type and level stats
            type_key = result.compression_type.value
            level_key = result.compression_level.value
            
            self._stats.compression_type_stats[type_key] = self._stats.compression_type_stats.get(type_key, 0) + 1
            self._stats.level_stats[level_key] = self._stats.level_stats.get(level_key, 0) + 1


class TemporalArchiver:
    """
    Archive manager for long-term temporal data storage.
    
    Handles creation, compression, and management of temporal archives
    with intelligent retention policies.
    """
    
    def __init__(self, archive_path: Path):
        self.archive_path = Path(archive_path)
        self.archive_path.mkdir(parents=True, exist_ok=True)
        self.compressor = TemporalCompressor(CompressionLevel.HEAVY)
        self._lock = threading.RLock()
    
    def create_archive(self, deltas: List[SystemDelta], 
                      archive_name: str,
                      compression_level: CompressionLevel = CompressionLevel.HEAVY) -> Path:
        """
        Create compressed archive of temporal data.
        
        Args:
            deltas: List of deltas to archive
            archive_name: Name for the archive file
            compression_level: Level of compression to apply
            
        Returns:
            Path to created archive file
        """
        archive_file = self.archive_path / f"{archive_name}.tar.xz"
        
        with self._lock:
            # Compress deltas
            compression_result = self.compressor.compress_deltas(
                deltas, 
                CompressionType.LZMA,
                compression_level
            )
            
            # Create archive metadata
            metadata = {
                'created_at': datetime.now().isoformat(),
                'delta_count': len(deltas),
                'compression_result': asdict(compression_result),
                'time_range': self.compressor._get_time_range(deltas)
            }
            
            # Write compressed data and metadata
            with lzma.open(archive_file, 'wb') as f:
                archive_data = {
                    'metadata': metadata,
                    'compressed_deltas': compression_result,
                    'checksum': hashlib.sha256(str(deltas).encode()).hexdigest()
                }
                f.write(json.dumps(archive_data).encode('utf-8'))
        
        return archive_file
    
    def list_archives(self) -> List[Dict[str, Any]]:
        """List all available archives with metadata."""
        archives = []
        
        for archive_file in self.archive_path.glob("*.tar.xz"):
            try:
                with lzma.open(archive_file, 'rb') as f:
                    data = json.loads(f.read().decode('utf-8'))
                    metadata = data.get('metadata', {})
                    metadata['file_size'] = archive_file.stat().st_size
                    metadata['file_path'] = str(archive_file)
                    archives.append(metadata)
            except Exception as e:
                # Archive file might be corrupted
                archives.append({
                    'file_path': str(archive_file),
                    'error': f"Failed to read archive: {e}",
                    'file_size': archive_file.stat().st_size
                })
        
        return sorted(archives, key=lambda x: x.get('created_at', ''), reverse=True)
    
    def restore_archive(self, archive_name: str) -> List[SystemDelta]:
        """
        Restore deltas from archive.
        
        Args:
            archive_name: Name of archive to restore
            
        Returns:
            List of restored SystemDelta objects
        """
        archive_file = self.archive_path / f"{archive_name}.tar.xz"
        
        if not archive_file.exists():
            raise FileNotFoundError(f"Archive {archive_name} not found")
        
        with lzma.open(archive_file, 'rb') as f:
            data = json.loads(f.read().decode('utf-8'))
            compression_result = data['compression_result']
            
            # Note: This is a simplified restoration
            # In practice, would need the actual compressed data bytes
            return []  # Would reconstruct from compressed data
    
    def cleanup_old_archives(self, retention_days: int = 365) -> List[str]:
        """
        Clean up archives older than retention period.
        
        Args:
            retention_days: Days to retain archives
            
        Returns:
            List of deleted archive names
        """
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        deleted = []
        
        for archive_file in self.archive_path.glob("*.tar.xz"):
            try:
                with lzma.open(archive_file, 'rb') as f:
                    data = json.loads(f.read().decode('utf-8'))
                    created_at = datetime.fromisoformat(data['metadata']['created_at'])
                    
                    if created_at < cutoff_date:
                        archive_file.unlink()
                        deleted.append(archive_file.stem)
            except Exception:
                # Skip corrupted archives
                continue
        
        return deleted


def create_compression_manager(default_level: CompressionLevel = CompressionLevel.MEDIUM) -> TemporalCompressor:
    """
    Create a configured compression manager.
    
    Args:
        default_level: Default compression level
        
    Returns:
        Configured TemporalCompressor instance
    """
    return TemporalCompressor(default_level)


def create_archiver(archive_path: Union[str, Path]) -> TemporalArchiver:
    """
    Create a configured archive manager.
    
    Args:
        archive_path: Path for archive storage
        
    Returns:
        Configured TemporalArchiver instance
    """
    return TemporalArchiver(Path(archive_path))