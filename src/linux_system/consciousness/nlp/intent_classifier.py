"""
Intent Classifier - Natural Language Understanding for System Queries
=====================================================================

Classifies user intents from natural language queries and extracts relevant
system entities, time contexts, and query parameters for system consciousness.

Key capabilities:
- Intent classification (monitoring, analysis, troubleshooting, optimization)
- Entity extraction (GPU, containers, processes, time ranges)
- Temporal context understanding (yesterday, past 4 hours, last week)
- System component identification and parameter extraction
"""

import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Types of user intents for system queries."""
    MONITORING = "monitoring"           # "Show me GPU temperature"
    ANALYSIS = "analysis"               # "Analyze memory usage patterns"
    TROUBLESHOOTING = "troubleshooting" # "Why did my GPU throttle?"
    OPTIMIZATION = "optimization"       # "How can I optimize container performance?"
    EXPLORATION = "exploration"         # "What happened yesterday?"
    PREDICTION = "prediction"           # "Will I run out of memory?"
    COMPARISON = "comparison"           # "Compare GPU vs CPU usage"
    UNKNOWN = "unknown"                 # Unclear intent


class SystemEntity(Enum):
    """System entities that can be referenced in queries."""
    GPU = "gpu"
    CPU = "cpu" 
    MEMORY = "memory"
    STORAGE = "storage"
    CONTAINERS = "containers"
    PROCESSES = "processes"
    NETWORK = "network"
    THERMAL = "thermal"
    SYSTEM = "system"
    
    # Specific container names
    LLAMA_GPU = "llama-gpu"
    LLAMA_CPU = "llama-cpu"
    VLLM_GPU = "vllm-gpu"
    OPEN_WEBUI = "open-webui"
    
    # Specific metrics
    TEMPERATURE = "temperature"
    UTILIZATION = "utilization"
    USAGE = "usage"
    PERFORMANCE = "performance"
    HEALTH = "health"


@dataclass
class TimeContext:
    """Temporal context extracted from natural language."""
    relative_time: Optional[str] = None      # "yesterday", "last hour"
    specific_time: Optional[datetime] = None # Parsed specific time
    time_range: Optional[Tuple[datetime, datetime]] = None
    granularity: Optional[str] = None        # "minute", "hour", "day"
    
    def to_query_parameters(self) -> Dict[str, Any]:
        """Convert to query parameters for system queries."""
        params = {}
        
        if self.time_range:
            params["start_time"] = self.time_range[0]
            params["end_time"] = self.time_range[1]
        elif self.specific_time:
            # Create 1-hour window around specific time
            params["start_time"] = self.specific_time - timedelta(minutes=30)
            params["end_time"] = self.specific_time + timedelta(minutes=30)
        elif self.relative_time:
            params["time_range"] = self.relative_time
            
        if self.granularity:
            params["granularity"] = self.granularity
            
        return params


@dataclass
class ExtractedQuery:
    """Structured representation of extracted query information."""
    intent: QueryIntent
    entities: List[SystemEntity]
    time_context: Optional[TimeContext]
    query_parameters: Dict[str, Any]
    confidence: float
    original_query: str
    processing_notes: List[str]


class IntentClassifier:
    """
    Natural language understanding for system consciousness queries.
    
    Analyzes user queries to understand intent, extract system entities,
    and prepare structured queries for the consciousness system.
    """
    
    def __init__(self):
        """Initialize intent classifier with patterns and mappings."""
        self.intent_patterns = self._build_intent_patterns()
        self.entity_patterns = self._build_entity_patterns() 
        self.time_patterns = self._build_time_patterns()
        
        # Common words to filter out
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", 
            "for", "of", "with", "by", "from", "is", "are", "was", "were",
            "will", "would", "could", "should", "can", "may", "might",
            "my", "your", "his", "her", "its", "our", "their"
        }
        
    def classify_query(self, query: str) -> ExtractedQuery:
        """
        Classify natural language query and extract structured information.
        
        Main entry point for natural language understanding.
        """
        try:
            # Normalize query
            normalized_query = self._normalize_query(query)
            processing_notes = []
            
            # Extract intent
            intent, intent_confidence = self._classify_intent(normalized_query)
            processing_notes.append(f"Intent: {intent.value} (confidence: {intent_confidence:.2f})")
            
            # Extract entities
            entities = self._extract_entities(normalized_query)
            processing_notes.append(f"Entities: {[e.value for e in entities]}")
            
            # Extract temporal context
            time_context = self._extract_time_context(normalized_query)
            if time_context and (time_context.relative_time or time_context.specific_time):
                processing_notes.append(f"Time context: {time_context.relative_time or time_context.specific_time}")
            
            # Extract additional query parameters
            query_params = self._extract_query_parameters(normalized_query, intent, entities)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                intent_confidence, entities, time_context, query_params
            )
            
            return ExtractedQuery(
                intent=intent,
                entities=entities,
                time_context=time_context,
                query_parameters=query_params,
                confidence=overall_confidence,
                original_query=query,
                processing_notes=processing_notes
            )
            
        except Exception as e:
            logger.error(f"Error classifying query '{query}': {e}")
            return ExtractedQuery(
                intent=QueryIntent.UNKNOWN,
                entities=[],
                time_context=None,
                query_parameters={},
                confidence=0.0,
                original_query=query,
                processing_notes=[f"Classification error: {str(e)}"]
            )
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for processing."""
        # Convert to lowercase and strip whitespace
        normalized = query.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
    def _build_intent_patterns(self) -> Dict[QueryIntent, List[str]]:
        """Build patterns for intent classification."""
        return {
            QueryIntent.MONITORING: [
                r'\b(show|display|get|current|now|status|what)\b.*\b(gpu|cpu|memory|container|process|temperature|usage|utilization)\b',
                r'\bhow\s+(is|are)\b.*\b(running|performing|doing)\b',
                r'\b(current|present|now)\b.*\b(state|status|condition)\b',
                r'\bwhat.*\b(temperature|usage|utilization|status)\b'
            ],
            
            QueryIntent.ANALYSIS: [
                r'\b(analyze|analysis|examine|study|investigate)\b',
                r'\b(pattern|trend|behavior|correlation)\b',
                r'\bhow\s+(often|much|many)\b',
                r'\b(compare|comparison|versus|vs|against)\b.*\b(usage|performance|efficiency)\b',
                r'\b(statistics|stats|metrics|data)\b.*\b(over|during|for)\b'
            ],
            
            QueryIntent.TROUBLESHOOTING: [
                r'\b(why|what\s+caused|what\s+happened|problem|issue|error|fail|crash)\b',
                r'\b(throttl|overheat|slow|lag|stuck|freeze)\b',
                r'\b(not\s+working|broken|malfunction|unstable)\b',
                r'\b(diagnose|troubleshoot|debug|fix)\b',
                r'\bwhat\s+went\s+wrong\b'
            ],
            
            QueryIntent.OPTIMIZATION: [
                r'\b(optimize|optimization|improve|enhance|better|faster|efficient)\b',
                r'\bhow\s+(can|could|should)\s+i\b.*\b(improve|optimize|enhance|speed)\b',
                r'\b(recommend|suggestion|advice|best\s+practice)\b',
                r'\b(reduce|minimize|maximize|increase)\b.*\b(usage|performance|efficiency)\b',
                r'\btune|tuning|configuration|config\b'
            ],
            
            QueryIntent.EXPLORATION: [
                r'\b(what\s+happened|what\s+was|show\s+me)\b.*\b(yesterday|last|during|when)\b',
                r'\b(history|historical|past|previous|earlier)\b',
                r'\b(timeline|over\s+time|progression)\b',
                r'\bwhen\s+(did|was|were)\b',
                r'\b(browse|explore|look\s+at)\b.*\b(data|logs|events)\b'
            ],
            
            QueryIntent.PREDICTION: [
                r'\b(will|going\s+to|predict|forecast|expect)\b',
                r'\b(when\s+will|how\s+long|estimate)\b',
                r'\b(future|upcoming|next|soon)\b',
                r'\b(run\s+out|exceed|reach|hit)\b.*\b(limit|capacity|threshold)\b',
                r'\b(likely|probably|chance|risk)\b.*\b(fail|crash|throttle)\b'
            ],
            
            QueryIntent.COMPARISON: [
                r'\b(compare|comparison|versus|vs|against|between)\b',
                r'\b(difference|differ|similar|same|better|worse)\b',
                r'\b(gpu\s+vs\s+cpu|cpu\s+vs\s+gpu)\b',
                r'\b(container.*container|process.*process)\b',
                r'\bwhich\s+(is|has|uses)\s+(more|less|better|faster)\b'
            ]
        }
    
    def _build_entity_patterns(self) -> Dict[SystemEntity, List[str]]:
        """Build patterns for entity extraction."""
        return {
            SystemEntity.GPU: [
                r'\b(gpu|graphics|rtx|5090|nvidia|cuda|vram|video\s+memory)\b',
                r'\b(graphics\s+card|video\s+card)\b'
            ],
            
            SystemEntity.CPU: [
                r'\b(cpu|processor|amd|ryzen|9950x|cores|threads)\b',
                r'\b(central\s+processing|processing\s+unit)\b'
            ],
            
            SystemEntity.MEMORY: [
                r'\b(memory|ram|ddr5|128gb|system\s+memory)\b',
                r'\b(swap|virtual\s+memory)\b'
            ],
            
            SystemEntity.STORAGE: [
                r'\b(storage|disk|drive|ssd|nvme|filesystem|mount)\b',
                r'\b(990\s+pro|990\s+evo|samsung)\b'
            ],
            
            SystemEntity.CONTAINERS: [
                r'\b(container|docker|llama|vllm|webui)\b',
                r'\b(containerized|orchestration)\b'
            ],
            
            SystemEntity.PROCESSES: [
                r'\b(process|processes|pid|service|daemon|application|app)\b',
                r'\b(running|active|background)\b.*\b(task|job|program)\b'
            ],
            
            SystemEntity.NETWORK: [
                r'\b(network|networking|connection|bandwidth|interface)\b',
                r'\b(ethernet|wifi|tcp|udp|port|traffic)\b'
            ],
            
            SystemEntity.THERMAL: [
                r'\b(temperature|temp|thermal|heat|cooling|fan|throttle|throttling)\b',
                r'\b(overheat|hot|cold)\b'
            ],
            
            # Specific containers
            SystemEntity.LLAMA_GPU: [
                r'\bllama-gpu\b'
            ],
            
            SystemEntity.LLAMA_CPU: [
                r'\bllama-cpu\b'
            ],
            
            SystemEntity.VLLM_GPU: [
                r'\bvllm-gpu\b'
            ],
            
            SystemEntity.OPEN_WEBUI: [
                r'\b(open-webui|webui|web\s+interface)\b'
            ],
            
            # Specific metrics
            SystemEntity.TEMPERATURE: [
                r'\b(temperature|temp|degrees|celsius|°c)\b'
            ],
            
            SystemEntity.UTILIZATION: [
                r'\b(utilization|util|usage|load|busy)\b'
            ],
            
            SystemEntity.PERFORMANCE: [
                r'\b(performance|perf|speed|latency|throughput|fps)\b'
            ],
            
            SystemEntity.HEALTH: [
                r'\b(health|healthy|status|condition|state)\b'
            ]
        }
    
    def _build_time_patterns(self) -> Dict[str, str]:
        """Build patterns for temporal context extraction."""
        return {
            # Relative time expressions
            "yesterday": r'\byesterday\b',
            "today": r'\btoday\b',
            "now": r'\b(now|currently|right\s+now|at\s+this\s+moment)\b',
            
            # Time ranges
            "last_hour": r'\b(last|past|previous)\s+(hour|hr)\b',
            "last_4_hours": r'\b(last|past|previous)\s+(4|four)\s+(hours|hrs|h)\b',
            "last_day": r'\b(last|past|previous)\s+(24\s+hours|day)\b',
            "last_week": r'\b(last|past|previous)\s+(week|7\s+days)\b',
            "last_month": r'\b(last|past|previous)\s+(month|30\s+days)\b',
            
            # Specific times
            "time_with_am_pm": r'\b(\d{1,2})(:\d{2})?\s*(am|pm|a\.m\.|p\.m\.)\b',
            "time_24h": r'\b([01]?\d|2[0-3]):([0-5]\d)\b',
            
            # Time periods
            "during_period": r'\bduring\s+(\w+)\b',
            "between_times": r'\bbetween\s+(.+?)\s+and\s+(.+?)\b',
            "since_time": r'\bsince\s+(.+?)\b',
            
            # Duration expressions  
            "for_duration": r'\bfor\s+(\d+)\s+(minute|hour|day|week)s?\b',
            "over_period": r'\bover\s+(.+?)\b'
        }
    
    def _classify_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Classify user intent from query."""
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            matches = 0
            
            for pattern in patterns:
                if re.search(pattern, query):
                    matches += 1
                    # Weight earlier matches higher
                    score += (len(patterns) - patterns.index(pattern)) / len(patterns)
            
            if matches > 0:
                # Normalize score by pattern count and boost for multiple matches
                intent_scores[intent] = (score / len(patterns)) * (1 + matches * 0.1)
        
        if not intent_scores:
            return QueryIntent.UNKNOWN, 0.0
        
        # Return intent with highest score
        best_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k])
        confidence = min(intent_scores[best_intent], 1.0)
        
        return best_intent, confidence
    
    def _extract_entities(self, query: str) -> List[SystemEntity]:
        """Extract system entities from query."""
        found_entities = set()
        
        for entity, patterns in self.entity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    found_entities.add(entity)
                    break  # Found this entity, move to next
        
        # Convert to list and sort for consistency
        return sorted(list(found_entities), key=lambda x: x.value)
    
    def _extract_time_context(self, query: str) -> Optional[TimeContext]:
        """Extract temporal context from query."""
        time_context = TimeContext()
        now = datetime.now()
        
        # Check for relative time expressions
        for time_key, pattern in self.time_patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if time_key == "yesterday":
                    start_time = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
                    end_time = start_time + timedelta(days=1)
                    time_context.time_range = (start_time, end_time)
                    time_context.relative_time = "1d"
                    time_context.granularity = "hour"
                    
                elif time_key == "today":
                    start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
                    time_context.time_range = (start_time, now)
                    time_context.relative_time = "1d"
                    time_context.granularity = "hour"
                    
                elif time_key == "last_hour":
                    time_context.time_range = (now - timedelta(hours=1), now)
                    time_context.relative_time = "1h"
                    time_context.granularity = "minute"
                    
                elif time_key == "last_4_hours":
                    time_context.time_range = (now - timedelta(hours=4), now)
                    time_context.relative_time = "4h"
                    time_context.granularity = "minute"
                    
                elif time_key == "last_day":
                    time_context.time_range = (now - timedelta(days=1), now)
                    time_context.relative_time = "1d"
                    time_context.granularity = "hour"
                    
                elif time_key == "last_week":
                    time_context.time_range = (now - timedelta(weeks=1), now)
                    time_context.relative_time = "1w"
                    time_context.granularity = "hour"
                    
                elif time_key == "time_with_am_pm":
                    # Parse specific time (simplified - would need more robust parsing)
                    time_text = match.group(0)
                    try:
                        # This is a simplified parser - in production would use more robust datetime parsing
                        time_context.specific_time = self._parse_time_string(time_text, now)
                    except:
                        time_context.relative_time = "1h"  # Fallback
                
                break  # Use first match
        
        # Return context only if we found something meaningful
        if time_context.relative_time or time_context.specific_time or time_context.time_range:
            return time_context
        
        return None
    
    def _parse_time_string(self, time_text: str, reference_date: datetime) -> datetime:
        """Parse time string to datetime (simplified implementation)."""
        # This is a simplified parser - in production would use dateutil.parser or similar
        
        # Extract hour and AM/PM
        match = re.match(r'(\d{1,2})(:\d{2})?\s*(am|pm|a\.m\.|p\.m\.)', time_text.lower())
        if match:
            hour = int(match.group(1))
            minutes = int(match.group(2)[1:]) if match.group(2) else 0
            am_pm = match.group(3)
            
            # Convert to 24-hour format
            if 'pm' in am_pm or 'p.m.' in am_pm:
                if hour != 12:
                    hour += 12
            elif 'am' in am_pm or 'a.m.' in am_pm:
                if hour == 12:
                    hour = 0
            
            # Use today's date with parsed time
            return reference_date.replace(hour=hour, minute=minutes, second=0, microsecond=0)
        
        # If parsing fails, return current time
        return reference_date
    
    def _extract_query_parameters(self, query: str, intent: QueryIntent, entities: List[SystemEntity]) -> Dict[str, Any]:
        """Extract additional query parameters."""
        params = {}
        
        # Extract aggregation preferences
        if re.search(r'\b(average|avg|mean)\b', query):
            params['aggregation'] = 'avg'
        elif re.search(r'\b(maximum|max|peak|highest)\b', query):
            params['aggregation'] = 'max'
        elif re.search(r'\b(minimum|min|lowest)\b', query):
            params['aggregation'] = 'min'
        elif re.search(r'\b(sum|total)\b', query):
            params['aggregation'] = 'sum'
        
        # Extract granularity preferences
        if re.search(r'\b(minute|min|per\s+minute)\b', query):
            params['granularity'] = 'minute'
        elif re.search(r'\b(hour|hourly|per\s+hour)\b', query):
            params['granularity'] = 'hour'
        elif re.search(r'\b(day|daily|per\s+day)\b', query):
            params['granularity'] = 'day'
        
        # Extract filtering preferences
        if re.search(r'\b(high|above|over|greater\s+than)\b', query):
            params['filter_type'] = 'above_threshold'
        elif re.search(r'\b(low|below|under|less\s+than)\b', query):
            params['filter_type'] = 'below_threshold'
        
        # Extract significance preferences
        if intent == QueryIntent.TROUBLESHOOTING:
            params['significance_threshold'] = 0.7  # Higher threshold for troubleshooting
        elif intent == QueryIntent.MONITORING:
            params['significance_threshold'] = 0.3  # Lower threshold for monitoring
        else:
            params['significance_threshold'] = 0.5  # Default threshold
        
        # Extract sorting preferences
        if re.search(r'\b(latest|recent|newest|last)\b', query):
            params['sort_order'] = 'desc'
        elif re.search(r'\b(earliest|oldest|first)\b', query):
            params['sort_order'] = 'asc'
        
        # Set default metric types based on entities
        metric_types = []
        for entity in entities:
            if entity == SystemEntity.GPU:
                metric_types.append('gpu')
            elif entity == SystemEntity.CPU:
                metric_types.append('cpu')
            elif entity == SystemEntity.MEMORY:
                metric_types.append('memory')
            elif entity == SystemEntity.CONTAINERS:
                metric_types.append('containers')
            elif entity == SystemEntity.PROCESSES:
                metric_types.append('processes')
            elif entity == SystemEntity.STORAGE:
                metric_types.append('storage')
            elif entity == SystemEntity.NETWORK:
                metric_types.append('network')
            elif entity == SystemEntity.THERMAL:
                metric_types.append('thermal')
        
        if metric_types:
            params['metric_types'] = metric_types
        elif intent == QueryIntent.MONITORING:
            params['metric_types'] = ['system']  # Default to system overview
        
        return params
    
    def _calculate_overall_confidence(self, intent_confidence: float, entities: List[SystemEntity], 
                                    time_context: Optional[TimeContext], params: Dict[str, Any]) -> float:
        """Calculate overall confidence in query understanding."""
        confidence = intent_confidence
        
        # Boost confidence for entity matches
        if entities:
            confidence += len(entities) * 0.1
        
        # Boost confidence for time context
        if time_context and (time_context.relative_time or time_context.specific_time):
            confidence += 0.1
        
        # Boost confidence for extracted parameters
        if params:
            confidence += len(params) * 0.05
        
        # Cap at 1.0
        return min(confidence, 1.0)
    
    def get_query_suggestions(self, partial_query: str, max_suggestions: int = 5) -> List[str]:
        """Generate query suggestions based on partial input."""
        suggestions = []
        
        # Common query templates
        templates = [
            "Show me GPU temperature",
            "What happened yesterday?", 
            "Why did my GPU throttle?",
            "Compare CPU vs GPU usage",
            "Analyze memory usage patterns",
            "How is the llama-gpu container performing?",
            "What's the current system health?",
            "Show me thermal management status",
            "When did containers restart last?",
            "Optimize my GPU utilization"
        ]
        
        # Filter templates based on partial query
        query_lower = partial_query.lower()
        for template in templates:
            if query_lower in template.lower() or any(word in template.lower() for word in query_lower.split()):
                suggestions.append(template)
                if len(suggestions) >= max_suggestions:
                    break
        
        return suggestions