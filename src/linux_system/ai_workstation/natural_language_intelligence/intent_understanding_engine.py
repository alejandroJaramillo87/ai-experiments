"""
Intent Understanding Engine

Advanced ML-based intent classification and entity extraction for natural language
queries about the AI workstation consciousness system. Uses sophisticated pattern
matching and contextual understanding to route queries to appropriate intelligence
systems.

Leverages the existing ML infrastructure and temporal intelligence for robust
natural language understanding that goes beyond simple template matching.
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Try to import advanced NLP libraries with graceful degradation
try:
    import spacy
    SPACY_AVAILABLE = True
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None
        SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Advanced intent classification for AI workstation queries"""
    # Diagnostic intents - understanding what happened
    CAUSAL_ANALYSIS = "causal_analysis"      # "Why did X happen?"
    ROOT_CAUSE = "root_cause"                # "What caused the throttling?"
    FAILURE_ANALYSIS = "failure_analysis"    # "Why did the service crash?"
    
    # Status and monitoring intents
    SYSTEM_STATUS = "system_status"          # "How is the system running?"
    COMPONENT_STATUS = "component_status"    # "How is the GPU performing?"
    HEALTH_CHECK = "health_check"            # "Is everything healthy?"
    
    # Pattern and intelligence intents  
    PATTERN_DISCOVERY = "pattern_discovery"  # "What patterns do you see?"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis" # "How does usage change over time?"
    CORRELATION_ANALYSIS = "correlation_analysis" # "What's correlated with X?"
    
    # Predictive and optimization intents
    PREDICTIVE_QUERY = "predictive_query"    # "Will X happen if Y?"
    OPTIMIZATION_REQUEST = "optimization_request" # "How can I optimize?"
    RESOURCE_PLANNING = "resource_planning"  # "Should I start training now?"
    
    # Historical and temporal intents
    TEMPORAL_QUERY = "temporal_query"        # "What happened yesterday?"
    TREND_ANALYSIS = "trend_analysis"        # "What's the trend in performance?"
    COMPARATIVE_ANALYSIS = "comparative_analysis" # "How does X compare to Y?"
    
    # Intelligence-specific intents
    CONSCIOUSNESS_QUERY = "consciousness_query" # "What does the system think?"
    LEARNING_STATUS = "learning_status"      # "What has the system learned?"
    
    # Fallback
    GENERAL_INQUIRY = "general_inquiry"      # Unclear or general questions


class SystemComponent(Enum):
    """AI workstation consciousness components that can be queried"""
    # Hardware components
    RTX5090_GPU = "rtx5090_gpu"
    AMD_ZEN5_CPU = "amd_zen5_cpu" 
    THERMAL_SYSTEM = "thermal_system"
    MEMORY_SYSTEM = "memory_system"
    STORAGE_SYSTEM = "storage_system"
    
    # Intelligence systems
    CONTAINER_CONSCIOUSNESS = "container_consciousness"
    HARDWARE_SPECIALIZATION = "hardware_specialization"
    MULTI_MODEL_ORACLE = "multi_model_oracle" 
    TEMPORAL_INTELLIGENCE = "temporal_intelligence"
    
    # AI Services
    AI_SERVICES = "ai_services"
    INFERENCE_ENGINES = "inference_engines"
    MODEL_MANAGEMENT = "model_management"
    
    # System-wide
    SYSTEM_CONSCIOUSNESS = "system_consciousness"
    ALL_SYSTEMS = "all_systems"


@dataclass
class QueryContext:
    """Rich contextual information extracted from natural language queries"""
    original_query: str
    intent: QueryIntent
    confidence: float
    components: List[SystemComponent]
    
    # Temporal context
    time_range: Optional[Tuple[datetime, datetime]] = None
    temporal_keywords: List[str] = None
    
    # Entity extraction
    entities: Dict[str, List[str]] = None
    technical_terms: List[str] = None
    
    # Intelligence routing hints
    requires_causal_analysis: bool = False
    requires_prediction: bool = False
    requires_optimization: bool = False
    requires_historical_data: bool = False
    
    # Context enrichment
    related_queries: List[str] = None
    suggested_followups: List[str] = None


class IntentUnderstandingEngine:
    """Advanced intent understanding for AI workstation consciousness queries"""
    
    def __init__(self):
        self.intent_patterns = self._build_advanced_intent_patterns()
        self.component_patterns = self._build_component_patterns()  
        self.temporal_extractors = self._build_temporal_extractors()
        self.technical_vocabulary = self._build_technical_vocabulary()
        
        # Initialize ML-based features if available
        self.ml_features_available = SPACY_AVAILABLE and nlp is not None
        
        logger.info(f"Intent Understanding Engine initialized. ML features: {self.ml_features_available}")
    
    def understand_query(self, query: str, session_context: Optional[Dict[str, Any]] = None) -> QueryContext:
        """
        Perform deep understanding of a natural language query about the AI workstation.
        
        Args:
            query: Natural language query from user
            session_context: Optional session context for continuity
            
        Returns:
            Rich QueryContext with intent, components, and routing information
        """
        query_normalized = query.lower().strip()
        
        # Extract primary intent with confidence scoring
        intent, intent_confidence = self._classify_intent(query_normalized)
        
        # Extract target system components
        components = self._extract_components(query_normalized)
        
        # Extract temporal context 
        time_range, temporal_keywords = self._extract_temporal_context(query_normalized)
        
        # Perform entity extraction if ML available
        entities = self._extract_entities(query) if self.ml_features_available else {}
        
        # Extract technical terms
        technical_terms = self._extract_technical_terms(query_normalized)
        
        # Determine intelligence routing requirements
        routing_reqs = self._analyze_intelligence_requirements(intent, components, query_normalized)
        
        # Generate context enrichment
        related_queries, followups = self._generate_contextual_suggestions(intent, components)
        
        return QueryContext(
            original_query=query,
            intent=intent,
            confidence=intent_confidence,
            components=components,
            time_range=time_range,
            temporal_keywords=temporal_keywords or [],
            entities=entities or {},
            technical_terms=technical_terms,
            requires_causal_analysis=routing_reqs['causal'],
            requires_prediction=routing_reqs['prediction'],
            requires_optimization=routing_reqs['optimization'],
            requires_historical_data=routing_reqs['historical'],
            related_queries=related_queries,
            suggested_followups=followups
        )
    
    def _build_advanced_intent_patterns(self) -> Dict[QueryIntent, Dict[str, Any]]:
        """Build sophisticated intent classification patterns"""
        return {
            QueryIntent.CAUSAL_ANALYSIS: {
                'patterns': [
                    r'\bwhy\s+did\b.*\b(happen|occur|fail|crash|throttle|restart)\b',
                    r'\bwhat\s+caused\b.*\b(the|this|that)\b',
                    r'\broot\s+cause\b.*\b(of|for|behind)\b',
                    r'\b(because|reason|cause)\b.*\bwhy\b'
                ],
                'weight': 1.0,
                'requires': ['causal', 'historical']
            },
            QueryIntent.PREDICTIVE_QUERY: {
                'patterns': [
                    r'\bwill\b.*\bif\b.*\b(start|run|execute|train)\b',
                    r'\bwhat\s+happens?\s+if\b',
                    r'\bshould\s+i\b.*\b(start|begin|run|execute|train|optimize)\b',
                    r'\b(predict|forecast|expect|anticipate)\b'
                ],
                'weight': 1.0,
                'requires': ['prediction', 'optimization']
            },
            QueryIntent.SYSTEM_STATUS: {
                'patterns': [
                    r'\bhow\s+is\b.*\b(performing|running|working|doing)\b',
                    r'\bstatus\s+of\b',
                    r'\bcurrent\s+state\b',
                    r'\bhealth\b.*\b(check|status|condition)\b'
                ],
                'weight': 0.9,
                'requires': ['current']
            },
            QueryIntent.PATTERN_DISCOVERY: {
                'patterns': [
                    r'\bwhat\s+patterns?\b.*\b(do\s+you\s+see|exist|emerge)\b',
                    r'\b(trends?|behaviors?|patterns?)\b.*\b(over\s+time|historical)\b',
                    r'\brepeat\b.*\b(pattern|behavior|cycle)\b',
                    r'\b(correlation|relationship)\b.*\bbetween\b'
                ],
                'weight': 1.0,
                'requires': ['historical', 'causal']
            },
            QueryIntent.OPTIMIZATION_REQUEST: {
                'patterns': [
                    r'\bhow\s+can\s+i\b.*\b(optimize|improve|enhance|boost)\b',
                    r'\b(optimize|optimization|improve|enhancement)\b.*\b(strategy|approach|plan)\b',
                    r'\bbest\s+way\s+to\b.*\b(improve|optimize|enhance)\b',
                    r'\b(recommend|suggest|advise)\b.*\b(optimization|improvement)\b'
                ],
                'weight': 1.0,
                'requires': ['optimization', 'prediction']
            },
            QueryIntent.CONSCIOUSNESS_QUERY: {
                'patterns': [
                    r'\bwhat\s+does\s+the\s+system\b.*\b(think|know|understand|learn)\b',
                    r'\bsystem\s+(consciousness|intelligence|awareness)\b',
                    r'\bai\s+workstation\b.*\b(thinks?|knows?|learned?)\b',
                    r'\bmachine\s+(consciousness|intelligence|learning)\b'
                ],
                'weight': 1.0,
                'requires': ['consciousness']
            }
        }
    
    def _build_component_patterns(self) -> Dict[SystemComponent, List[str]]:
        """Build patterns for sophisticated component identification"""
        return {
            SystemComponent.RTX5090_GPU: [
                r'\b(gpu|graphics|rtx\s*5090|blackwall)\b',
                r'\b(vram|tensor\s+cores?|cuda|nvlink)\b',
                r'\bgraphics?\s+(card|unit|processor)\b'
            ],
            SystemComponent.AMD_ZEN5_CPU: [
                r'\b(cpu|processor|amd|zen\s*5|9950x)\b',
                r'\b(cores?|threads?|aocl|ccx)\b',
                r'\bprocessor\s+(cores?|performance)\b'
            ],
            SystemComponent.THERMAL_SYSTEM: [
                r'\b(thermal|temperature|cooling|heat)\b',
                r'\b(fans?|airflow|thermodynamic)\b',
                r'\b(15[\-\s]fan|cooling\s+system)\b',
                r'\b(throttl|thermal\s+management)\b'
            ],
            SystemComponent.CONTAINER_CONSCIOUSNESS: [
                r'\b(container|docker|service|orchestr)\b',
                r'\b(llama|vllm|inference\s+engine)\b',
                r'\b(microservice|service\s+mesh)\b'
            ],
            SystemComponent.MULTI_MODEL_ORACLE: [
                r'\b(oracle|prediction|optimization)\b',
                r'\b(ml\s+model|machine\s+learning|ai\s+model)\b',
                r'\b(strategy|planning|decision)\b'
            ],
            SystemComponent.TEMPORAL_INTELLIGENCE: [
                r'\b(temporal|time\s+series|historical)\b',
                r'\b(pattern|trend|correlation)\b',
                r'\b(causality|causal|temporal\s+analysis)\b'
            ],
            SystemComponent.SYSTEM_CONSCIOUSNESS: [
                r'\b(system|workstation|consciousness)\b',
                r'\b(ai\s+workstation|digital\s+twin)\b',
                r'\b(overall|general|system[\-\s]wide)\b'
            ]
        }
    
    def _build_temporal_extractors(self) -> List[Tuple[str, callable]]:
        """Build advanced temporal context extractors"""
        return [
            (r'\b(yesterday|last\s+night)\b', lambda: self._get_yesterday()),
            (r'\blast\s+(hour|day|week|month|year)\b', lambda m: self._get_last_period(m.group(1))),
            (r'\b(\d+)\s+(minutes?|hours?|days?|weeks?)\s+ago\b', 
             lambda m: self._get_time_ago(int(m.group(1)), m.group(2))),
            (r'\b(this\s+(morning|afternoon|evening|night))\b',
             lambda m: self._get_time_of_day_range(m.group(2))),
            (r'\bbetween\s+(\d+):?(\d*)\s*(?:am|pm)?\s+and\s+(\d+):?(\d*)\s*(?:am|pm)?\b',
             lambda m: self._parse_time_range(m.groups())),
            (r'\bduring\s+(\w+\s+\w+|\w+)\b', lambda m: self._parse_duration_context(m.group(1)))
        ]
    
    def _build_technical_vocabulary(self) -> Dict[str, str]:
        """Build comprehensive technical vocabulary mapping"""
        return {
            # Performance metrics
            'utilization': 'resource_usage',
            'throughput': 'performance_metric',
            'latency': 'response_time',
            'bandwidth': 'data_transfer_rate',
            
            # System events  
            'throttling': 'thermal_limiting',
            'scaling': 'resource_adjustment',
            'balancing': 'load_distribution',
            
            # Intelligence terms
            'inference': 'model_execution',
            'training': 'model_learning',
            'fine-tuning': 'model_adaptation',
            'optimization': 'performance_improvement'
        }
    
    def _classify_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Advanced intent classification with confidence scoring"""
        max_confidence = 0.0
        best_intent = QueryIntent.GENERAL_INQUIRY
        
        for intent, config in self.intent_patterns.items():
            confidence = 0.0
            
            for pattern in config['patterns']:
                if re.search(pattern, query, re.IGNORECASE):
                    # Weight patterns based on specificity
                    pattern_confidence = config['weight'] / len(config['patterns'])
                    confidence += pattern_confidence
            
            # Apply intelligence requirement bonuses
            if confidence > 0 and 'requires' in config:
                for req in config['requires']:
                    if self._has_requirement_indicators(query, req):
                        confidence += 0.1
            
            if confidence > max_confidence:
                max_confidence = confidence
                best_intent = intent
        
        return best_intent, min(max_confidence, 1.0)
    
    def _extract_components(self, query: str) -> List[SystemComponent]:
        """Extract target system components with sophisticated matching"""
        components = []
        component_scores = {}
        
        for component, patterns in self.component_patterns.items():
            score = 0.0
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                score += len(matches) * (1.0 / len(patterns))
            
            if score > 0:
                component_scores[component] = score
        
        # Sort by relevance and return top matches
        sorted_components = sorted(component_scores.items(), key=lambda x: x[1], reverse=True)
        components = [comp for comp, score in sorted_components if score > 0.3]
        
        # Default to system consciousness if no specific components identified
        if not components:
            components = [SystemComponent.SYSTEM_CONSCIOUSNESS]
        
        return components
    
    def _extract_temporal_context(self, query: str) -> Tuple[Optional[Tuple[datetime, datetime]], List[str]]:
        """Extract sophisticated temporal context"""
        temporal_keywords = []
        
        # Look for temporal expressions
        for pattern, parser_func in self.temporal_extractors:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    time_range = parser_func(match)
                    temporal_keywords.append(match.group(0))
                    return time_range, temporal_keywords
                except Exception as e:
                    logger.warning(f"Failed to parse temporal expression: {e}")
        
        return None, temporal_keywords
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Advanced entity extraction using spaCy"""
        if not self.ml_features_available:
            return {}
        
        doc = nlp(query)
        entities = {}
        
        for ent in doc.ents:
            entity_type = ent.label_.lower()
            if entity_type in ['time', 'date', 'cardinal', 'percent', 'quantity']:
                if entity_type not in entities:
                    entities[entity_type] = []
                entities[entity_type].append(ent.text)
        
        return entities
    
    def _extract_technical_terms(self, query: str) -> List[str]:
        """Extract technical terms relevant to AI workstation"""
        technical_terms = []
        
        for term, category in self.technical_vocabulary.items():
            if term in query:
                technical_terms.append(term)
        
        # Also extract numerical values that might be thresholds
        numerical_matches = re.findall(r'\b\d+(?:\.\d+)?(?:%|gb|mb|°c|watts?|rpm|hz)\b', query, re.IGNORECASE)
        technical_terms.extend(numerical_matches)
        
        return technical_terms
    
    def _analyze_intelligence_requirements(self, intent: QueryIntent, components: List[SystemComponent], 
                                         query: str) -> Dict[str, bool]:
        """Determine what types of intelligence analysis are needed"""
        requirements = {
            'causal': False,
            'prediction': False, 
            'optimization': False,
            'historical': False,
            'consciousness': False
        }
        
        # Intent-based requirements
        if intent in [QueryIntent.CAUSAL_ANALYSIS, QueryIntent.ROOT_CAUSE, QueryIntent.FAILURE_ANALYSIS]:
            requirements['causal'] = True
            requirements['historical'] = True
            
        if intent in [QueryIntent.PREDICTIVE_QUERY, QueryIntent.RESOURCE_PLANNING]:
            requirements['prediction'] = True
            
        if intent == QueryIntent.OPTIMIZATION_REQUEST:
            requirements['optimization'] = True
            requirements['prediction'] = True
            
        if intent in [QueryIntent.TEMPORAL_QUERY, QueryIntent.TREND_ANALYSIS, QueryIntent.PATTERN_DISCOVERY]:
            requirements['historical'] = True
            
        if intent == QueryIntent.CONSCIOUSNESS_QUERY:
            requirements['consciousness'] = True
        
        # Query text-based indicators
        requirements['causal'] |= bool(re.search(r'\b(why|cause|reason|because)\b', query))
        requirements['prediction'] |= bool(re.search(r'\b(will|predict|forecast|future)\b', query))
        requirements['optimization'] |= bool(re.search(r'\b(optimize|improve|better|enhance)\b', query))
        requirements['historical'] |= bool(re.search(r'\b(yesterday|last|ago|history|trend|pattern)\b', query))
        
        return requirements
    
    def _generate_contextual_suggestions(self, intent: QueryIntent, components: List[SystemComponent]) -> Tuple[List[str], List[str]]:
        """Generate contextually relevant related queries and follow-ups"""
        related_queries = []
        followups = []
        
        # Intent-based suggestions
        if intent == QueryIntent.CAUSAL_ANALYSIS:
            followups = [
                "How can I prevent this from happening again?",
                "What other components were affected?",
                "Show me the performance timeline for that period"
            ]
        elif intent == QueryIntent.SYSTEM_STATUS:
            followups = [
                "What optimization opportunities exist?",
                "How does this compare to yesterday?",
                "What patterns do you see in recent performance?"
            ]
        elif intent == QueryIntent.PREDICTIVE_QUERY:
            followups = [
                "What factors influence this prediction?",
                "How can I improve the expected outcome?",
                "Show me similar historical patterns"
            ]
        
        # Component-based related queries
        for component in components:
            if component == SystemComponent.RTX5090_GPU:
                related_queries.extend([
                    "What's the GPU thermal pattern?",
                    "How is VRAM utilization trending?",
                    "Are there any tensor core efficiency issues?"
                ])
            elif component == SystemComponent.CONTAINER_CONSCIOUSNESS:
                related_queries.extend([
                    "How are AI services interacting?",
                    "What's the container resource distribution?",
                    "Any service orchestration patterns?"
                ])
        
        return related_queries[:3], followups[:3]  # Limit suggestions
    
    # Helper methods for temporal parsing
    def _has_requirement_indicators(self, query: str, requirement: str) -> bool:
        """Check if query has indicators for specific intelligence requirements"""
        indicators = {
            'causal': [r'\bwhy\b', r'\bcause\b', r'\breason\b'],
            'prediction': [r'\bwill\b', r'\bpredict\b', r'\bfuture\b'],
            'optimization': [r'\boptimize\b', r'\bimprove\b', r'\bbetter\b'],
            'historical': [r'\byesterday\b', r'\blast\b', r'\bago\b', r'\bhistory\b'],
            'consciousness': [r'\bsystem.*think\b', r'\bai.*know\b', r'\bconsciousness\b']
        }
        
        if requirement not in indicators:
            return False
            
        for pattern in indicators[requirement]:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _get_yesterday(self) -> Tuple[datetime, datetime]:
        """Get yesterday's time range"""
        yesterday = datetime.now() - timedelta(days=1)
        start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        return start, end
    
    def _get_last_period(self, period: str) -> Tuple[datetime, datetime]:
        """Get time range for 'last X' expressions"""
        now = datetime.now()
        period_map = {
            'hour': timedelta(hours=1),
            'day': timedelta(days=1),
            'week': timedelta(weeks=1),
            'month': timedelta(days=30),
            'year': timedelta(days=365)
        }
        
        delta = period_map.get(period, timedelta(hours=1))
        return now - delta, now
    
    def _get_time_ago(self, amount: int, unit: str) -> Tuple[datetime, datetime]:
        """Get time range for 'X units ago' expressions"""
        now = datetime.now()
        
        if 'minute' in unit:
            delta = timedelta(minutes=amount)
        elif 'hour' in unit:
            delta = timedelta(hours=amount)
        elif 'day' in unit:
            delta = timedelta(days=amount)
        elif 'week' in unit:
            delta = timedelta(weeks=amount)
        else:
            delta = timedelta(hours=amount)
        
        target_time = now - delta
        return target_time - timedelta(minutes=30), target_time + timedelta(minutes=30)
    
    def _get_time_of_day_range(self, period: str) -> Tuple[datetime, datetime]:
        """Get time range for 'this morning/afternoon/etc' expressions"""
        today = datetime.now().date()
        
        ranges = {
            'morning': (6, 12),
            'afternoon': (12, 18),
            'evening': (18, 22),
            'night': (22, 6)  # Special case handled below
        }
        
        if period == 'night':
            # Night spans midnight
            start = datetime.combine(today, datetime.min.time().replace(hour=22))
            end = datetime.combine(today + timedelta(days=1), datetime.min.time().replace(hour=6))
        else:
            start_hour, end_hour = ranges.get(period, (0, 24))
            start = datetime.combine(today, datetime.min.time().replace(hour=start_hour))
            end = datetime.combine(today, datetime.min.time().replace(hour=end_hour))
        
        return start, end
    
    def _parse_time_range(self, groups: Tuple[str, ...]) -> Tuple[datetime, datetime]:
        """Parse 'between X and Y' time expressions"""
        start_hour = int(groups[0])
        start_min = int(groups[1]) if groups[1] else 0
        end_hour = int(groups[2])
        end_min = int(groups[3]) if groups[3] else 0
        
        today = datetime.now().date()
        start = datetime.combine(today, datetime.min.time().replace(hour=start_hour, minute=start_min))
        end = datetime.combine(today, datetime.min.time().replace(hour=end_hour, minute=end_min))
        
        # If times are in the future, assume yesterday
        now = datetime.now()
        if start > now:
            start -= timedelta(days=1)
            end -= timedelta(days=1)
        
        return start, end
    
    def _parse_duration_context(self, duration: str) -> Tuple[datetime, datetime]:
        """Parse contextual duration expressions like 'heavy usage' or 'training session'"""
        # This could be enhanced with more sophisticated duration context understanding
        now = datetime.now()
        return now - timedelta(hours=2), now  # Default 2-hour window