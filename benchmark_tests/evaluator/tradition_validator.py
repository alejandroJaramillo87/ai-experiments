"""
Traditional Knowledge Validator

Validates respect for wisdom traditions, proper contextualization of traditional knowledge,
and appropriate handling of sacred or sensitive cultural information.

Author: Claude Code
Version: 1.0.0
"""

import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import Counter


@dataclass
class TraditionValidationResult:
    """Results from traditional knowledge validation"""
    tradition_respect_score: float
    contextualization_quality: float
    sacred_knowledge_respect: float
    community_attribution_score: float
    living_tradition_recognition: float
    violation_indicators: List[Dict[str, str]]
    positive_indicators: List[Dict[str, str]]
    detailed_analysis: Dict[str, any]


class TraditionalKnowledgeValidator:
    """Validates respectful handling of traditional knowledge and wisdom traditions"""
    
    def __init__(self):
        """Initialize traditional knowledge validator"""
        self._init_validation_patterns()
        self.analysis_cache = {}
        
    def _init_validation_patterns(self):
        """Initialize validation patterns for traditional knowledge analysis"""
        
        # Sacred knowledge violation indicators
        self.sacred_violations = {
            'inappropriate_disclosure': [
                r'\b(?:secret|sacred|private|ceremonial)\s+(?:ritual|ceremony|practice|knowledge)\s+(?:involves?|includes?|uses?|requires?)',
                r'\b(?:initiation|sacred)\s+(?:rite|ceremony)\s+(?:steps?|process|details?)',
                r'\b(?:medicine|healing)\s+(?:ceremony|ritual)\s+(?:instructions?|how\s+to|guide)',
                r'\b(?:vision\s+quest|sweat\s+lodge|sun\s+dance)\s+(?:process|steps?|how\s+to)'
            ],
            'commercialization': [
                r'\b(?:buy|purchase|sell|market|commercialize)\s+(?:sacred|traditional|ceremonial)',
                r'\b(?:sacred|traditional|ceremonial)\s+(?:objects?|items?|tools?)\s+(?:for\s+sale|available|buy)',
                r'\bmass[\-\s]produce[d]?\s+(?:sacred|traditional|ceremonial)',
                r'\b(?:profit|money|business)\s+from\s+(?:sacred|traditional|ceremonial)'
            ],
            'decontextualization': [
                r'\b(?:use|try|practice|adopt)\s+(?:sacred|traditional|ceremonial)\s+(?:without|outside\s+of)',
                r'\b(?:anyone\s+can|you\s+can|easy\s+to)\s+(?:use|practice|adopt)\s+(?:sacred|traditional)',
                r'\b(?:sacred|traditional|ceremonial)\s+(?:practices?|rituals?)\s+(?:at\s+home|for\s+personal)',
                r'\b(?:mix|combine|blend)\s+(?:sacred|traditional)\s+(?:with|and)\s+(?:modern|western|other)'
            ]
        }
        
        # Proper contextualization indicators
        self.contextualization_markers = {
            'cultural_context': [
                r'\bwithin\s+(?:the\s+context\s+of|their\s+cultural\s+(?:framework|system|tradition))',
                r'\b(?:understood|practiced|learned)\s+within\s+(?:the\s+community|their\s+culture)',
                r'\b(?:cultural|traditional|historical)\s+context\s+(?:is\s+essential|matters|important)',
                r'\b(?:rooted\s+in|grounded\s+in|emerges\s+from)\s+(?:cultural|traditional|historical)'
            ],
            'historical_context': [
                r'\b(?:historically|traditionally|ancestrally)\s+(?:practiced|used|maintained|preserved)',
                r'\b(?:passed\s+down|transmitted|inherited)\s+(?:through|across|over)\s+generations',
                r'\b(?:ancient|ancestral|traditional)\s+(?:origins?|roots?|foundations?)',
                r'\b(?:developed|evolved|emerged)\s+over\s+(?:centuries|generations|millennia)'
            ],
            'spiritual_context': [
                r'\b(?:spiritual|sacred|ceremonial)\s+(?:significance|meaning|importance|purpose)',
                r'\bconnection\s+to\s+(?:ancestors?|spirits?|sacred|divine|natural\s+world)',
                r'\b(?:spiritual|sacred)\s+(?:relationship|bond|connection)\s+(?:with|to)',
                r'\b(?:ceremony|ritual)\s+(?:creates?|maintains?|honors?)\s+(?:connection|relationship)'
            ]
        }
        
        # Community attribution patterns
        self.attribution_patterns = {
            'elder_attribution': [
                r'\baccording\s+to\s+(?:elders?|traditional\s+(?:teachers?|leaders?|healers?))',
                r'\b(?:elders?|traditional\s+(?:teachers?|practitioners?|healers?))\s+(?:teach|explain|share|say)',
                r'\blearned\s+from\s+(?:elders?|traditional\s+(?:teachers?|practitioners?|healers?))',
                r'\b(?:community\s+)?elders?\s+(?:have\s+)?(?:taught|shared|explained|passed\s+down)'
            ],
            'community_attribution': [
                r'\b(?:community|tribal|indigenous|native)\s+(?:members?|people|groups?)\s+(?:practice|believe|maintain)',
                r'\bwithin\s+(?:the\s+)?(?:community|tribe|nation|people|culture)',
                r'\b(?:community|tribal|cultural)\s+(?:knowledge|wisdom|understanding|tradition)',
                r'\bmaintained\s+by\s+(?:the\s+)?(?:community|tribe|people|culture)'
            ],
            'source_acknowledgment': [
                r'\bsource[d]?\s+from\s+(?:indigenous|native|traditional|tribal|community)',
                r'\b(?:comes?|derives?)\s+from\s+(?:indigenous|native|traditional|tribal)',
                r'\b(?:originated|originates?)\s+(?:with|from|among)\s+(?:the\s+)?(?:indigenous|native|traditional)',
                r'\bcredits?\s+(?:to\s+)?(?:indigenous|native|traditional|tribal|community)'
            ]
        }
        
        # Living tradition recognition
        self.living_tradition_markers = {
            'present_tense': [
                r'\b(?:continue|continues?|still|currently|today|now)\s+(?:to\s+)?(?:practice|maintain|preserve|use)',
                r'\b(?:is|are)\s+(?:still|currently|actively)\s+(?:practiced|maintained|preserved|used)',
                r'\b(?:living|active|continuing|ongoing)\s+(?:tradition|practice|culture|knowledge)',
                r'\b(?:practiced|maintained|preserved)\s+(?:today|currently|in\s+(?:modern|contemporary)\s+times)'
            ],
            'evolution_acknowledgment': [
                r'\b(?:evolved|adapted|changed|developed)\s+(?:over\s+time|through\s+generations)',
                r'\b(?:traditional|ancient)\s+(?:and|as\s+well\s+as)\s+(?:modern|contemporary|current)',
                r'\b(?:adapts?|evolution|changes?)\s+(?:with|to|over)\s+time',
                r'\b(?:not\s+static|dynamic|changing|evolving)\s+(?:tradition|practice|culture)'
            ],
            'contemporary_relevance': [
                r'\b(?:relevant|important|valuable|significant)\s+(?:today|currently|now|in\s+modern\s+times)',
                r'\b(?:modern|contemporary)\s+(?:applications?|relevance|significance|importance)',
                r'\b(?:continues?\s+to\s+)?(?:offer|provide|contribute|benefit)\s+(?:today|currently|now)',
                r'\b(?:contemporary|modern)\s+(?:practitioners?|communities?|people)\s+(?:use|practice|apply)'
            ]
        }
        
        # Violation severity levels
        self.violation_severity = {
            'sacred_violations': {
                'inappropriate_disclosure': 'critical',
                'commercialization': 'high',
                'decontextualization': 'medium'
            },
            'general_violations': {
                'misappropriation': 'high',
                'stereotyping': 'medium',
                'trivialization': 'medium'
            }
        }
        
        # Traditional knowledge domains
        self.knowledge_domains = {
            'healing': ['medicine', 'healing', 'herbs', 'remedies', 'treatment', 'therapy'],
            'spiritual': ['ceremony', 'ritual', 'sacred', 'spiritual', 'prayer', 'blessing'],
            'ecological': ['environment', 'nature', 'plants', 'animals', 'seasons', 'weather'],
            'social': ['kinship', 'governance', 'law', 'conflict', 'community', 'family'],
            'technical': ['craft', 'technology', 'tools', 'construction', 'agriculture', 'navigation'],
            'educational': ['stories', 'oral', 'teaching', 'knowledge', 'wisdom', 'learning']
        }
    
    def validate_traditional_knowledge(self, text: str, domain_hint: Optional[str] = None) -> TraditionValidationResult:
        """
        Perform comprehensive traditional knowledge validation
        
        Args:
            text: The response text to validate
            domain_hint: Optional hint about the knowledge domain
            
        Returns:
            TraditionValidationResult with detailed validation
        """
        cache_key = (text, domain_hint)
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        # Core validation components
        sacred_respect = self._validate_sacred_knowledge_respect(text)
        contextualization = self._validate_contextualization(text)
        attribution = self._validate_community_attribution(text)
        living_tradition = self._validate_living_tradition_recognition(text)
        
        # Calculate composite scores
        respect_score = self._calculate_tradition_respect_score(
            sacred_respect, contextualization, attribution, living_tradition
        )
        
        # Compile violation and positive indicators
        violations = self._compile_violations(sacred_respect, contextualization, attribution)
        positives = self._compile_positive_indicators(contextualization, attribution, living_tradition)
        
        # Detailed analysis
        detailed_analysis = {
            'sacred_respect_analysis': sacred_respect,
            'contextualization_analysis': contextualization,
            'attribution_analysis': attribution,
            'living_tradition_analysis': living_tradition,
            'domain_analysis': self._analyze_knowledge_domain(text, domain_hint),
            'overall_assessment': self._generate_tradition_assessment(
                respect_score, len(violations), len(positives)
            )
        }
        
        result = TraditionValidationResult(
            tradition_respect_score=respect_score,
            contextualization_quality=contextualization.get('quality_score', 0.0),
            sacred_knowledge_respect=sacred_respect.get('respect_score', 0.0),
            community_attribution_score=attribution.get('attribution_score', 0.0),
            living_tradition_recognition=living_tradition.get('recognition_score', 0.0),
            violation_indicators=violations,
            positive_indicators=positives,
            detailed_analysis=detailed_analysis
        )
        
        self.analysis_cache[cache_key] = result
        return result
    
    def _validate_sacred_knowledge_respect(self, text: str) -> Dict[str, any]:
        """Validate respect for sacred knowledge boundaries"""
        text_lower = text.lower()
        violations = []
        violation_counts = {}
        
        for category, patterns in self.sacred_violations.items():
            matches = []
            for pattern in patterns:
                pattern_matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
                for match in pattern_matches:
                    severity = self.violation_severity['sacred_violations'].get(category, 'medium')
                    matches.append({
                        'pattern': pattern,
                        'match': match.group(0),
                        'position': match.span(),
                        'category': category,
                        'severity': severity
                    })
            
            violation_counts[category] = len(matches)
            violations.extend(matches)
        
        # Calculate respect score
        total_violations = len(violations)
        critical_violations = sum(1 for v in violations if v.get('severity') == 'critical')
        high_violations = sum(1 for v in violations if v.get('severity') == 'high')
        
        # Heavy penalties for sacred knowledge violations
        respect_score = 1.0
        respect_score -= critical_violations * 0.5  # Critical violations are severe
        respect_score -= high_violations * 0.3
        respect_score -= (total_violations - critical_violations - high_violations) * 0.1
        
        return {
            'violations': violations,
            'violation_counts': violation_counts,
            'total_violations': total_violations,
            'critical_violations': critical_violations,
            'high_violations': high_violations,
            'respect_score': max(0.0, respect_score)
        }
    
    def _validate_contextualization(self, text: str) -> Dict[str, any]:
        """Validate proper contextualization of traditional knowledge"""
        text_lower = text.lower()
        context_markers = []
        category_counts = {}
        
        for category, patterns in self.contextualization_markers.items():
            matches = []
            for pattern in patterns:
                pattern_matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
                for match in pattern_matches:
                    matches.append({
                        'pattern': pattern,
                        'match': match.group(0),
                        'position': match.span(),
                        'category': category
                    })
            
            category_counts[category] = len(matches)
            context_markers.extend(matches)
        
        # Calculate contextualization quality
        total_markers = len(context_markers)
        categories_present = len([c for c, count in category_counts.items() if count > 0])
        text_length = len(text.split())
        
        # Quality score based on marker density and category diversity
        marker_density = total_markers / max(text_length, 1)
        category_bonus = categories_present * 0.15  # Bonus for diverse contextualization
        quality_score = min(marker_density * 15 + category_bonus, 1.0)
        
        return {
            'context_markers': context_markers,
            'category_counts': category_counts,
            'total_markers': total_markers,
            'categories_present': categories_present,
            'marker_density': marker_density,
            'quality_score': quality_score
        }
    
    def _validate_community_attribution(self, text: str) -> Dict[str, any]:
        """Validate proper attribution to communities and sources"""
        text_lower = text.lower()
        attribution_markers = []
        category_counts = {}
        
        for category, patterns in self.attribution_patterns.items():
            matches = []
            for pattern in patterns:
                pattern_matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
                for match in pattern_matches:
                    matches.append({
                        'pattern': pattern,
                        'match': match.group(0),
                        'position': match.span(),
                        'category': category
                    })
            
            category_counts[category] = len(matches)
            attribution_markers.extend(matches)
        
        # Calculate attribution score
        total_markers = len(attribution_markers)
        categories_present = len([c for c, count in category_counts.items() if count > 0])
        
        # Strong weight on elder attribution and source acknowledgment
        elder_weight = category_counts.get('elder_attribution', 0) * 0.4
        community_weight = category_counts.get('community_attribution', 0) * 0.3
        source_weight = category_counts.get('source_acknowledgment', 0) * 0.3
        
        attribution_score = min(elder_weight + community_weight + source_weight, 1.0)
        
        return {
            'attribution_markers': attribution_markers,
            'category_counts': category_counts,
            'total_markers': total_markers,
            'categories_present': categories_present,
            'attribution_score': attribution_score
        }
    
    def _validate_living_tradition_recognition(self, text: str) -> Dict[str, any]:
        """Validate recognition of traditions as living, evolving systems"""
        text_lower = text.lower()
        recognition_markers = []
        category_counts = {}
        
        for category, patterns in self.living_tradition_markers.items():
            matches = []
            for pattern in patterns:
                pattern_matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
                for match in pattern_matches:
                    matches.append({
                        'pattern': pattern,
                        'match': match.group(0),
                        'position': match.span(),
                        'category': category
                    })
            
            category_counts[category] = len(matches)
            recognition_markers.extend(matches)
        
        # Check for problematic "past tense only" language
        past_only_patterns = [
            r'\b(?:used\s+to|once|formerly|in\s+the\s+past|historically)\s+(?:believed|practiced|used)',
            r'\b(?:ancient|old|traditional)\s+(?:beliefs?|practices?)\s+(?:were|was|held)',
            r'\b(?:no\s+longer|abandoned|lost|forgotten)\s+(?:tradition|practice|belief)'
        ]
        
        past_only_markers = []
        for pattern in past_only_patterns:
            past_only_markers.extend(list(re.finditer(pattern, text_lower, re.IGNORECASE)))
        
        # Calculate recognition score
        total_markers = len(recognition_markers)
        past_only_count = len(past_only_markers)
        categories_present = len([c for c, count in category_counts.items() if count > 0])
        
        recognition_score = min(total_markers * 0.2 + categories_present * 0.1, 1.0)
        recognition_score -= past_only_count * 0.15  # Penalty for past-only language
        recognition_score = max(0.0, recognition_score)
        
        return {
            'recognition_markers': recognition_markers,
            'past_only_markers': past_only_markers,
            'category_counts': category_counts,
            'total_markers': total_markers,
            'past_only_count': past_only_count,
            'categories_present': categories_present,
            'recognition_score': recognition_score
        }
    
    def _analyze_knowledge_domain(self, text: str, domain_hint: Optional[str] = None) -> Dict[str, any]:
        """Analyze the traditional knowledge domain being discussed"""
        text_lower = text.lower()
        domain_matches = {}
        
        for domain, keywords in self.knowledge_domains.items():
            matches = 0
            for keyword in keywords:
                matches += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
            domain_matches[domain] = matches
        
        # Identify primary domain
        primary_domain = max(domain_matches, key=domain_matches.get) if domain_matches else None
        domain_confidence = max(domain_matches.values()) / max(len(text.split()), 1) if domain_matches else 0
        
        return {
            'domain_matches': domain_matches,
            'primary_domain': primary_domain,
            'domain_confidence': domain_confidence,
            'hint_provided': domain_hint,
            'hint_matches_analysis': domain_hint == primary_domain if domain_hint else None
        }
    
    def _calculate_tradition_respect_score(self, sacred_respect: Dict, contextualization: Dict,
                                         attribution: Dict, living_tradition: Dict) -> float:
        """Calculate overall traditional knowledge respect score"""
        
        # Weighted combination of all factors
        sacred_weight = 0.4  # Sacred knowledge respect is most important
        context_weight = 0.25
        attribution_weight = 0.25
        living_weight = 0.1
        
        respect_score = (
            sacred_respect.get('respect_score', 0.0) * sacred_weight +
            contextualization.get('quality_score', 0.0) * context_weight +
            attribution.get('attribution_score', 0.0) * attribution_weight +
            living_tradition.get('recognition_score', 0.0) * living_weight
        )
        
        return min(1.0, max(0.0, respect_score))
    
    def _compile_violations(self, sacred_respect: Dict, contextualization: Dict, attribution: Dict) -> List[Dict]:
        """Compile all violation indicators"""
        violations = []
        
        # Sacred knowledge violations (highest priority)
        sacred_violations = sacred_respect.get('violations', [])
        for violation in sacred_violations:
            violations.append({
                'type': 'sacred_knowledge_violation',
                'category': violation.get('category', 'unknown'),
                'severity': violation.get('severity', 'medium'),
                'description': violation.get('match', ''),
                'position': violation.get('position', (0, 0))
            })
        
        # Context violations (lack of proper contextualization)
        if contextualization.get('quality_score', 0) < 0.3:
            violations.append({
                'type': 'insufficient_contextualization',
                'category': 'contextualization',
                'severity': 'medium',
                'description': 'Traditional knowledge presented without adequate cultural context',
                'position': None
            })
        
        # Attribution violations (lack of proper source acknowledgment)
        if attribution.get('attribution_score', 0) < 0.2:
            violations.append({
                'type': 'insufficient_attribution',
                'category': 'attribution',
                'severity': 'medium',
                'description': 'Traditional knowledge presented without proper community attribution',
                'position': None
            })
        
        return violations
    
    def _compile_positive_indicators(self, contextualization: Dict, attribution: Dict, living_tradition: Dict) -> List[Dict]:
        """Compile all positive indicators"""
        positives = []
        
        # Strong contextualization
        if contextualization.get('categories_present', 0) >= 2:
            positives.append({
                'type': 'comprehensive_contextualization',
                'description': 'Traditional knowledge presented with rich cultural context',
                'strength': 'high' if contextualization.get('categories_present', 0) >= 3 else 'medium'
            })
        
        # Good attribution
        if attribution.get('categories_present', 0) >= 2:
            positives.append({
                'type': 'proper_attribution',
                'description': 'Traditional knowledge properly attributed to community sources',
                'strength': 'high' if 'elder_attribution' in attribution.get('category_counts', {}) else 'medium'
            })
        
        # Living tradition recognition
        if living_tradition.get('categories_present', 0) >= 2:
            positives.append({
                'type': 'living_tradition_recognition',
                'description': 'Traditional knowledge recognized as continuing, evolving practice',
                'strength': 'medium'
            })
        
        return positives
    
    def _generate_tradition_assessment(self, respect_score: float, violation_count: int, positive_count: int) -> Dict[str, any]:
        """Generate overall traditional knowledge assessment"""
        
        if violation_count > 0 and any(v.get('severity') == 'critical' for v in self.analysis_cache.get((None, None), {}).get('detailed_analysis', {}).get('sacred_respect_analysis', {}).get('violations', [])):
            assessment_level = 'critical_issues'
        elif respect_score >= 0.8 and positive_count >= 2:
            assessment_level = 'excellent'
        elif respect_score >= 0.6 and violation_count <= 1:
            assessment_level = 'good'
        elif respect_score >= 0.4:
            assessment_level = 'acceptable'
        elif respect_score >= 0.2:
            assessment_level = 'concerning'
        else:
            assessment_level = 'problematic'
        
        return {
            'assessment_level': assessment_level,
            'respect_score': respect_score,
            'violation_count': violation_count,
            'positive_count': positive_count,
            'requires_cultural_review': assessment_level in ['critical_issues', 'problematic', 'concerning']
        }
    
    def get_tradition_validation_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get traditional knowledge validation thresholds"""
        return {
            'tradition_respect': {
                'excellent': 0.85,
                'good': 0.7,
                'acceptable': 0.5,
                'concerning': 0.3
            },
            'contextualization_quality': {
                'comprehensive': 0.8,
                'adequate': 0.6,
                'minimal': 0.3,
                'insufficient': 0.1
            },
            'sacred_respect': {
                'full_respect': 0.9,
                'respectful': 0.7,
                'some_concerns': 0.5,
                'violations_present': 0.3
            },
            'attribution': {
                'well_attributed': 0.8,
                'adequately_attributed': 0.5,
                'poorly_attributed': 0.2,
                'no_attribution': 0.0
            }
        }