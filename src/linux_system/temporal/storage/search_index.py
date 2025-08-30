"""
Temporal Search Index
====================

Fast indexing and search capabilities for temporal data.
Provides efficient querying across time ranges, categories, and content.
"""

import json
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass

from ..types import SystemDelta, SystemChange, SystemEvent


@dataclass
class IndexEntry:
    """Single index entry."""
    delta_id: str
    timestamp: datetime
    categories: Set[str]
    event_types: Set[str]
    max_significance: float
    content_hash: str
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """Search result with scoring."""
    delta_id: str
    score: float
    matched_fields: List[str]
    timestamp: datetime
    summary: str


class TemporalSearchIndex:
    """
    Efficient search index for temporal data.
    
    Provides fast lookup and filtering capabilities across
    time ranges, categories, and content patterns.
    """
    
    def __init__(self, indexed_categories: List[str] = None):
        self.indexed_categories = set(indexed_categories) if indexed_categories else set()
        self._entries: Dict[str, IndexEntry] = {}
        self._time_index: Dict[datetime, List[str]] = defaultdict(list)  # timestamp -> delta_ids
        self._category_index: Dict[str, Set[str]] = defaultdict(set)    # category -> delta_ids
        self._event_type_index: Dict[str, Set[str]] = defaultdict(set) # event_type -> delta_ids
        self._significance_index: List[Tuple[float, str]] = []          # (significance, delta_id) sorted by significance
        self._content_index: Dict[str, Set[str]] = defaultdict(set)    # content_term -> delta_ids
        
        self._lock = threading.RLock()
        self._next_delta_id = 0
        
        # Index statistics
        self._total_deltas_indexed = 0
        self._last_update = None
    
    def update(self, system_delta: SystemDelta) -> str:
        """
        Update index with new system delta.
        
        Args:
            system_delta: System delta to index
            
        Returns:
            Delta ID assigned to this delta
        """
        with self._lock:
            # Generate unique delta ID
            delta_id = f"delta_{self._next_delta_id:08d}"
            self._next_delta_id += 1
            
            # Extract indexable information
            categories = set(change.category for change in system_delta.raw_delta)
            event_types = set(event.event_type for event in system_delta.semantic_events)
            
            # Calculate maximum significance
            max_significance = 0.0
            if system_delta.raw_delta:
                max_significance = max(change.significance for change in system_delta.raw_delta)
            
            # Generate content hash for deduplication
            content_hash = self._generate_content_hash(system_delta)
            
            # Extract searchable content terms
            content_terms = self._extract_content_terms(system_delta)
            
            # Create index entry
            entry = IndexEntry(
                delta_id=delta_id,
                timestamp=system_delta.timestamp,
                categories=categories,
                event_types=event_types,
                max_significance=max_significance,
                content_hash=content_hash,
                metadata={
                    'change_count': len(system_delta.raw_delta),
                    'event_count': len(system_delta.semantic_events),
                    'correlation_count': len(system_delta.correlations)
                }
            )
            
            # Store entry
            self._entries[delta_id] = entry
            
            # Update indices
            self._update_time_index(delta_id, system_delta.timestamp)
            self._update_category_index(delta_id, categories)
            self._update_event_type_index(delta_id, event_types)
            self._update_significance_index(delta_id, max_significance)
            self._update_content_index(delta_id, content_terms)
            
            # Update statistics
            self._total_deltas_indexed += 1
            self._last_update = datetime.now()
            
            return delta_id
    
    def search(self, 
               start_time: Optional[datetime] = None,
               end_time: Optional[datetime] = None,
               categories: Optional[List[str]] = None,
               event_types: Optional[List[str]] = None,
               min_significance: Optional[float] = None,
               content_terms: Optional[List[str]] = None,
               limit: int = 100) -> List[SearchResult]:
        """
        Search index with multiple criteria.
        
        Args:
            start_time: Search start time
            end_time: Search end time
            categories: Required categories
            event_types: Required event types
            min_significance: Minimum significance threshold
            content_terms: Content search terms
            limit: Maximum results to return
            
        Returns:
            List of search results sorted by relevance
        """
        with self._lock:
            # Start with all deltas if no time range specified
            if start_time is None and end_time is None:
                candidate_ids = set(self._entries.keys())
            else:
                candidate_ids = self._search_by_time_range(start_time, end_time)
            
            # Filter by categories
            if categories:
                category_ids = self._search_by_categories(categories)
                candidate_ids &= category_ids
            
            # Filter by event types
            if event_types:
                event_type_ids = self._search_by_event_types(event_types)
                candidate_ids &= event_type_ids
            
            # Filter by significance
            if min_significance is not None:
                significance_ids = self._search_by_significance(min_significance)
                candidate_ids &= significance_ids
            
            # Filter by content terms
            if content_terms:
                content_ids = self._search_by_content(content_terms)
                candidate_ids &= content_ids
            
            # Score and rank results
            results = []
            for delta_id in candidate_ids:
                if delta_id in self._entries:
                    entry = self._entries[delta_id]
                    score = self._calculate_result_score(entry, categories, event_types, min_significance, content_terms)
                    
                    result = SearchResult(
                        delta_id=delta_id,
                        score=score,
                        matched_fields=self._get_matched_fields(entry, categories, event_types, content_terms),
                        timestamp=entry.timestamp,
                        summary=self._generate_result_summary(entry)
                    )
                    results.append(result)
            
            # Sort by score (descending) and timestamp (descending for ties)
            results.sort(key=lambda r: (-r.score, -r.timestamp.timestamp()))
            
            return results[:limit]
    
    def _search_by_time_range(self, start_time: Optional[datetime], end_time: Optional[datetime]) -> Set[str]:
        """Search by time range."""
        if start_time is None and end_time is None:
            return set(self._entries.keys())
        
        candidate_ids = set()
        
        for timestamp, delta_ids in self._time_index.items():
            include = True
            
            if start_time and timestamp < start_time:
                include = False
            if end_time and timestamp > end_time:
                include = False
            
            if include:
                candidate_ids.update(delta_ids)
        
        return candidate_ids
    
    def _search_by_categories(self, categories: List[str]) -> Set[str]:
        """Search by categories (AND logic)."""
        if not categories:
            return set(self._entries.keys())
        
        result_ids = None
        
        for category in categories:
            category_ids = self._category_index.get(category, set())
            
            if result_ids is None:
                result_ids = category_ids.copy()
            else:
                result_ids &= category_ids
        
        return result_ids or set()
    
    def _search_by_event_types(self, event_types: List[str]) -> Set[str]:
        """Search by event types (OR logic)."""
        if not event_types:
            return set(self._entries.keys())
        
        result_ids = set()
        
        for event_type in event_types:
            event_type_ids = self._event_type_index.get(event_type, set())
            result_ids.update(event_type_ids)
        
        return result_ids
    
    def _search_by_significance(self, min_significance: float) -> Set[str]:
        """Search by minimum significance."""
        result_ids = set()
        
        # Binary search for efficiency (significance index is sorted)
        for significance, delta_id in self._significance_index:
            if significance >= min_significance:
                result_ids.add(delta_id)
        
        return result_ids
    
    def _search_by_content(self, content_terms: List[str]) -> Set[str]:
        """Search by content terms (OR logic)."""
        if not content_terms:
            return set(self._entries.keys())
        
        result_ids = set()
        
        for term in content_terms:
            term_lower = term.lower()
            for indexed_term, delta_ids in self._content_index.items():
                if term_lower in indexed_term:
                    result_ids.update(delta_ids)
        
        return result_ids
    
    def _calculate_result_score(self, entry: IndexEntry, categories: Optional[List[str]], 
                              event_types: Optional[List[str]], min_significance: Optional[float],
                              content_terms: Optional[List[str]]) -> float:
        """Calculate relevance score for search result."""
        score = 0.0
        
        # Base score from significance
        score += entry.max_significance * 0.3
        
        # Category match bonus
        if categories:
            matched_categories = len(set(categories) & entry.categories)
            category_ratio = matched_categories / len(categories)
            score += category_ratio * 0.2
        
        # Event type match bonus
        if event_types:
            matched_event_types = len(set(event_types) & entry.event_types)
            event_type_ratio = matched_event_types / len(event_types)
            score += event_type_ratio * 0.2
        
        # Significance threshold bonus
        if min_significance and entry.max_significance >= min_significance:
            excess_significance = entry.max_significance - min_significance
            score += excess_significance * 0.1
        
        # Content match bonus (simplified)
        if content_terms:
            # This is a simplified content scoring - could be enhanced
            score += 0.2
        
        # Recency bonus (more recent = slightly higher score)
        days_ago = (datetime.now() - entry.timestamp).days
        recency_bonus = max(0, 0.1 - days_ago * 0.001)  # Decays over time
        score += recency_bonus
        
        return score
    
    def _get_matched_fields(self, entry: IndexEntry, categories: Optional[List[str]],
                          event_types: Optional[List[str]], content_terms: Optional[List[str]]) -> List[str]:
        """Get list of fields that matched the search criteria."""
        matched = []
        
        if categories:
            matched_categories = set(categories) & entry.categories
            if matched_categories:
                matched.extend([f"category:{cat}" for cat in matched_categories])
        
        if event_types:
            matched_event_types = set(event_types) & entry.event_types
            if matched_event_types:
                matched.extend([f"event_type:{et}" for et in matched_event_types])
        
        if content_terms:
            # Simplified - could be enhanced to show actual matched terms
            matched.append("content_match")
        
        matched.append(f"significance:{entry.max_significance:.2f}")
        
        return matched
    
    def _generate_result_summary(self, entry: IndexEntry) -> str:
        """Generate human-readable summary of search result."""
        timestamp_str = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        category_str = ", ".join(sorted(entry.categories)[:3])  # Top 3 categories
        if len(entry.categories) > 3:
            category_str += "..."
        
        event_str = ""
        if entry.event_types:
            event_types = sorted(entry.event_types)[:2]  # Top 2 event types
            event_str = f" - Events: {', '.join(event_types)}"
            if len(entry.event_types) > 2:
                event_str += "..."
        
        return f"{timestamp_str} | {category_str}{event_str} | Significance: {entry.max_significance:.2f}"
    
    def _update_time_index(self, delta_id: str, timestamp: datetime) -> None:
        """Update time-based index."""
        # Round timestamp to minute for efficient indexing
        rounded_timestamp = timestamp.replace(second=0, microsecond=0)
        self._time_index[rounded_timestamp].append(delta_id)
    
    def _update_category_index(self, delta_id: str, categories: Set[str]) -> None:
        """Update category-based index."""
        for category in categories:
            self._category_index[category].add(delta_id)
    
    def _update_event_type_index(self, delta_id: str, event_types: Set[str]) -> None:
        """Update event type index."""
        for event_type in event_types:
            self._event_type_index[event_type].add(delta_id)
    
    def _update_significance_index(self, delta_id: str, significance: float) -> None:
        """Update significance index."""
        self._significance_index.append((significance, delta_id))
        # Keep significance index sorted for binary search
        self._significance_index.sort(key=lambda x: x[0], reverse=True)
    
    def _update_content_index(self, delta_id: str, content_terms: Set[str]) -> None:
        """Update content-based index."""
        for term in content_terms:
            self._content_index[term.lower()].add(delta_id)
    
    def _generate_content_hash(self, system_delta: SystemDelta) -> str:
        """Generate content hash for delta."""
        import hashlib
        
        # Create content signature
        content_parts = []
        
        # Add change signatures
        for change in system_delta.raw_delta:
            content_parts.append(f"{change.category}:{change.change_type.value}:{change.entity_id}")
        
        # Add event signatures
        for event in system_delta.semantic_events:
            content_parts.append(f"event:{event.event_type}:{event.entity}")
        
        content_str = "|".join(sorted(content_parts))
        return hashlib.md5(content_str.encode()).hexdigest()[:12]
    
    def _extract_content_terms(self, system_delta: SystemDelta) -> Set[str]:
        """Extract searchable content terms from delta."""
        terms = set()
        
        # Extract from change entity IDs
        for change in system_delta.raw_delta:
            # Split entity ID into searchable parts
            entity_parts = change.entity_id.replace(':', ' ').replace('_', ' ').split()
            terms.update(part.lower() for part in entity_parts if len(part) > 2)
            
            # Add category
            terms.add(change.category.lower())
            
            # Extract from metadata
            if change.metadata:
                for key, value in change.metadata.items():
                    if isinstance(value, str):
                        terms.add(key.lower())
                        if len(value) < 50:  # Avoid long values
                            terms.add(value.lower())
        
        # Extract from events
        for event in system_delta.semantic_events:
            # Event type terms
            event_parts = event.event_type.replace('_', ' ').split()
            terms.update(part.lower() for part in event_parts if len(part) > 2)
            
            # Entity terms
            entity_parts = event.entity.replace(':', ' ').replace('_', ' ').split()
            terms.update(part.lower() for part in entity_parts if len(part) > 2)
            
            # Description terms (first 10 words)
            desc_words = event.description.lower().split()[:10]
            terms.update(word.strip('.,!?') for word in desc_words if len(word) > 3)
        
        return terms
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        with self._lock:
            return {
                'total_deltas_indexed': self._total_deltas_indexed,
                'unique_categories': len(self._category_index),
                'unique_event_types': len(self._event_type_index),
                'unique_content_terms': len(self._content_index),
                'time_range': self._get_time_range(),
                'last_update': self._last_update.isoformat() if self._last_update else None,
                'index_size_estimate': self._estimate_index_size()
            }
    
    def _get_time_range(self) -> Optional[Tuple[datetime, datetime]]:
        """Get time range of indexed data."""
        if not self._time_index:
            return None
        
        timestamps = list(self._time_index.keys())
        return min(timestamps), max(timestamps)
    
    def _estimate_index_size(self) -> Dict[str, int]:
        """Estimate index memory usage."""
        return {
            'entries': len(self._entries),
            'time_index_buckets': len(self._time_index),
            'category_mappings': sum(len(delta_ids) for delta_ids in self._category_index.values()),
            'event_type_mappings': sum(len(delta_ids) for delta_ids in self._event_type_index.values()),
            'content_term_mappings': sum(len(delta_ids) for delta_ids in self._content_index.values())
        }
    
    def get_popular_categories(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most popular categories by delta count."""
        with self._lock:
            category_counts = [(cat, len(delta_ids)) for cat, delta_ids in self._category_index.items()]
            category_counts.sort(key=lambda x: x[1], reverse=True)
            return category_counts[:limit]
    
    def get_popular_event_types(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most popular event types by delta count."""
        with self._lock:
            event_type_counts = [(et, len(delta_ids)) for et, delta_ids in self._event_type_index.items()]
            event_type_counts.sort(key=lambda x: x[1], reverse=True)
            return event_type_counts[:limit]
    
    def get_recent_activity(self, hours: int = 24) -> Dict[str, Any]:
        """Get recent activity statistics."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_deltas = []
            
            for timestamp, delta_ids in self._time_index.items():
                if timestamp >= cutoff_time:
                    recent_deltas.extend(delta_ids)
            
            # Analyze recent activity
            recent_categories = defaultdict(int)
            recent_event_types = defaultdict(int)
            total_significance = 0.0
            
            for delta_id in recent_deltas:
                entry = self._entries.get(delta_id)
                if entry:
                    for category in entry.categories:
                        recent_categories[category] += 1
                    for event_type in entry.event_types:
                        recent_event_types[event_type] += 1
                    total_significance += entry.max_significance
            
            return {
                'delta_count': len(recent_deltas),
                'avg_significance': total_significance / len(recent_deltas) if recent_deltas else 0.0,
                'active_categories': dict(recent_categories),
                'active_event_types': dict(recent_event_types),
                'time_range_hours': hours
            }
    
    def rebuild(self) -> Dict[str, Any]:
        """Rebuild index (optimization operation)."""
        with self._lock:
            start_time = datetime.now()
            
            # Clear derived indices (keep entries)
            old_entries = self._entries.copy()
            self._time_index.clear()
            self._category_index.clear()
            self._event_type_index.clear()
            self._significance_index.clear()
            self._content_index.clear()
            
            # Rebuild indices from entries
            for entry in old_entries.values():
                self._update_time_index(entry.delta_id, entry.timestamp)
                self._update_category_index(entry.delta_id, entry.categories)
                self._update_event_type_index(entry.delta_id, entry.event_types)
                self._update_significance_index(entry.delta_id, entry.max_significance)
                # Content index would need original delta data to rebuild
            
            rebuild_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'rebuild_time_seconds': rebuild_time,
                'entries_processed': len(old_entries),
                'indices_rebuilt': ['time', 'category', 'event_type', 'significance']
            }
    
    def clear(self) -> None:
        """Clear all index data."""
        with self._lock:
            self._entries.clear()
            self._time_index.clear()
            self._category_index.clear()
            self._event_type_index.clear()
            self._significance_index.clear()
            self._content_index.clear()
            self._next_delta_id = 0
            self._total_deltas_indexed = 0
            self._last_update = None
    
    def persist(self) -> None:
        """Persist index to disk (placeholder)."""
        # Implementation would save index to disk
        pass
    
    def backup_to(self, backup_path: Path) -> None:
        """Backup index to specified path."""
        backup_path.mkdir(parents=True, exist_ok=True)
        
        with self._lock:
            # Backup main entries
            entries_file = backup_path / 'entries.json'
            with open(entries_file, 'w') as f:
                entries_data = {}
                for delta_id, entry in self._entries.items():
                    entries_data[delta_id] = {
                        'delta_id': entry.delta_id,
                        'timestamp': entry.timestamp.isoformat(),
                        'categories': list(entry.categories),
                        'event_types': list(entry.event_types),
                        'max_significance': entry.max_significance,
                        'content_hash': entry.content_hash,
                        'metadata': entry.metadata
                    }
                json.dump(entries_data, f, indent=2)
            
            # Backup statistics
            stats_file = backup_path / 'statistics.json'
            with open(stats_file, 'w') as f:
                stats = self.get_statistics()
                json.dump(stats, f, indent=2)
    
    def restore_from(self, backup_path: Path) -> None:
        """Restore index from backup path."""
        if not backup_path.exists():
            return
        
        entries_file = backup_path / 'entries.json'
        if not entries_file.exists():
            return
        
        with self._lock:
            self.clear()
            
            try:
                with open(entries_file, 'r') as f:
                    entries_data = json.load(f)
                
                for delta_id, entry_dict in entries_data.items():
                    entry = IndexEntry(
                        delta_id=entry_dict['delta_id'],
                        timestamp=datetime.fromisoformat(entry_dict['timestamp']),
                        categories=set(entry_dict['categories']),
                        event_types=set(entry_dict['event_types']),
                        max_significance=entry_dict['max_significance'],
                        content_hash=entry_dict['content_hash'],
                        metadata=entry_dict['metadata']
                    )
                    
                    self._entries[delta_id] = entry
                    
                    # Rebuild derived indices
                    self._update_time_index(entry.delta_id, entry.timestamp)
                    self._update_category_index(entry.delta_id, entry.categories)
                    self._update_event_type_index(entry.delta_id, entry.event_types)
                    self._update_significance_index(entry.delta_id, entry.max_significance)
                    
                    # Update counters
                    self._total_deltas_indexed += 1
                
                # Update next ID counter
                if self._entries:
                    max_id = max(int(delta_id.split('_')[1]) for delta_id in self._entries.keys())
                    self._next_delta_id = max_id + 1
                
                self._last_update = datetime.now()
                
            except Exception as e:
                print(f"Error restoring index from backup: {e}")
                self.clear()  # Clear partially restored state