"""
Cross-Cultural Coherence Checker

Ensures knowledge systems are presented on their own terms without inappropriate
framework imposition, and validates respectful cross-cultural understanding.

Author: Claude Code
Version: 1.0.0
"""

import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import Counter


@dataclass
class CrossCulturalCoherenceResult:
    """Results from cross-cultural coherence analysis"""
    coherence_score: float
    framework_imposition_score: float
    knowledge_system_integrity: float
    translation_quality: float
    comparative_appropriateness: float
    imposition_indicators: List[Dict[str, str]]
    integrity_indicators: List[Dict[str, str]]
    translation_issues: List[Dict[str, str]]
    detailed_analysis: Dict[str, any]


class CrossCulturalCoherenceChecker:
    """Validates coherent and respectful cross-cultural knowledge presentation"""
    
    def __init__(self):
        """Initialize cross-cultural coherence checker"""
        self._init_coherence_patterns()
        self.analysis_cache = {}
        
    def _init_coherence_patterns(self):
        """Initialize patterns for cross-cultural coherence analysis"""
        
        # Framework imposition indicators
        self.framework_imposition = {
            'scientific_reductionism': [
                r'\b(?:scientifically|objectively|empirically)\s+(?:speaking|proven|validated|confirmed)',
                r'\b(?:traditional|indigenous|folk)\s+(?:beliefs?|practices?|knowledge)\s+(?:can\s+be\s+explained|are\s+actually|is\s+really)',
                r'\bfrom\s+a\s+scientific\s+(?:perspective|viewpoint|standpoint)',
                r'\b(?:science|research|studies?)\s+(?:shows?|proves?|demonstrates?)\s+(?:that\s+)?(?:traditional|indigenous|folk)'
            ],
            'western_psychology': [
                r'\b(?:psychological|cognitive|behavioral)\s+(?:explanation|basis|foundation)\s+(?:for|of|behind)',
                r'\b(?:traditional|indigenous|folk)\s+(?:healing|practices?|rituals?)\s+(?:work|function)\s+(?:through|via|by)\s+(?:psychology|placebo)',
                r'\bpsychological\s+(?:mechanisms?|processes?|effects?)\s+(?:underlying|behind|in)',
                r'\b(?:belief|faith|ritual)\s+(?:activates?|triggers?|creates?)\s+(?:psychological|neurological|biochemical)'
            ],
            'economic_framing': [
                r'\b(?:traditional|indigenous|folk)\s+(?:practices?|knowledge|systems?)\s+(?:are\s+)?(?:economically|financially)\s+(?:viable|valuable|beneficial)',
                r'\b(?:cost[\-\s]effective|profitable|marketable)\s+(?:traditional|indigenous|folk)',
                r'\b(?:traditional|indigenous|folk)\s+(?:knowledge|practices?)\s+(?:can\s+be\s+)?(?:commercialized|monetized|scaled)',
                r'\beconomic\s+(?:value|benefit|potential)\s+of\s+(?:traditional|indigenous|folk)'
            ],
            'technological_determinism': [
                r'\b(?:traditional|indigenous|folk)\s+(?:practices?|knowledge|methods?)\s+(?:enhanced|improved|optimized)\s+(?:by|with|through)\s+technology',
                r'\b(?:digital|technological|modern)\s+(?:solutions?|tools?|methods?)\s+(?:for|to\s+improve|to\s+enhance)\s+(?:traditional|indigenous)',
                r'\b(?:apps?|software|digital\s+platforms?)\s+(?:for|to\s+preserve|to\s+document)\s+(?:traditional|indigenous|folk)',
                r'\btechnology\s+(?:can\s+)?(?:preserve|enhance|improve|modernize)\s+(?:traditional|indigenous|folk)'
            ]
        }
        
        # Knowledge system integrity indicators
        self.integrity_markers = {
            'holistic_understanding': [
                r'\b(?:holistic|interconnected|integrated|unified)\s+(?:approach|understanding|system|worldview)',
                r'\b(?:cannot\s+be\s+separated|inseparable|intertwined|connected)\s+(?:from|with)',
                r'\b(?:whole|complete|entire)\s+(?:system|worldview|framework|context)\s+(?:must\s+be|is\s+important)',
                r'\b(?:relationships?|connections?|interdependence)\s+(?:between|among|within)'
            ],
            'indigenous_frameworks': [
                r'\b(?:within|according\s+to|from\s+the\s+perspective\s+of)\s+(?:indigenous|native|traditional|tribal)\s+(?:understanding|worldview|framework)',
                r'\b(?:indigenous|native|traditional|tribal)\s+(?:knowledge|epistemology|ways?\s+of\s+knowing|understanding)',
                r'\bon\s+(?:its|their)\s+own\s+terms',
                r'\brespecting\s+(?:the|their|indigenous|native|traditional)\s+(?:framework|worldview|understanding|epistemology)'
            ],
            'cultural_autonomy': [
                r'\b(?:self[\-\s]determination|autonomy|sovereignty)\s+(?:of|over|in)\s+(?:cultural|traditional|indigenous)',
                r'\b(?:cultural|traditional|indigenous)\s+(?:self[\-\s]determination|autonomy|sovereignty)',
                r'\b(?:communities?|peoples?)\s+(?:define|determine|control)\s+(?:their\s+own|how\s+their)',
                r'\b(?:not\s+for\s+outsiders|external\s+validation\s+not\s+needed|internally\s+valid)'
            ]
        }
        
        # Translation quality indicators
        self.translation_quality = {
            'concept_explanation': [
                r'\b(?:concept|term|word)\s+(?:that\s+)?(?:roughly|approximately|somewhat)\s+(?:translates?|means?|corresponds?)',
                r'\b(?:no\s+direct|difficult\s+to|hard\s+to)\s+(?:translate|render\s+in|express\s+in)\s+(?:english|western)',
                r'\b(?:best\s+understood\s+as|closest\s+equivalent|similar\s+to\s+but\s+not\s+exactly)',
                r'\b(?:encompasses|includes|covers)\s+(?:more\s+than|broader\s+than|additional\s+meanings?)'
            ],
            'cultural_nuance': [
                r'\b(?:cultural|contextual|traditional)\s+(?:nuances?|subtleties|complexities)',
                r'\b(?:culturally\s+specific|culture[\-\s]specific|context[\-\s]dependent)',
                r'\b(?:meaning|significance|understanding)\s+(?:varies|differs|depends\s+on)\s+(?:cultural|contextual)',
                r'\b(?:deep|rich|complex|nuanced)\s+(?:cultural|traditional|indigenous)\s+(?:meaning|significance|understanding)'
            ],
            'respectful_approximation': [
                r'\b(?:approximate|rough|limited)\s+(?:translation|understanding|explanation)',
                r'\b(?:this\s+is\s+a\s+)?(?:simplified|basic|general)\s+(?:explanation|understanding|description)',
                r'\bmay\s+not\s+fully\s+(?:capture|convey|represent|express)\s+the\s+(?:depth|richness|complexity)',
                r'\b(?:fuller|deeper|complete)\s+understanding\s+(?:requires|needs|depends\s+on)\s+(?:cultural|traditional|indigenous)'
            ]
        }
        
        # Inappropriate comparison patterns
        self.inappropriate_comparisons = {
            'hierarchical_comparisons': [
                r'\b(?:more|less)\s+(?:advanced|sophisticated|developed|evolved|primitive|backward)',
                r'\b(?:superior|inferior)\s+(?:to|than|compared\s+to)',
                r'\b(?:higher|lower)\s+(?:level|form|stage)\s+of\s+(?:development|evolution|civilization)',
                r'\b(?:progressed|evolved|advanced)\s+(?:beyond|from|past)\s+(?:traditional|indigenous|primitive)'
            ],
            'false_equivalencies': [
                r'\b(?:same\s+as|equivalent\s+to|just\s+like|identical\s+to)\s+(?:western|modern|scientific)',
                r'\b(?:western|modern|scientific)\s+(?:version|equivalent|counterpart)\s+of',
                r'\b(?:traditional|indigenous|folk)\s+(?:science|medicine|psychology|philosophy)',
                r'\ballthe\s+(?:traditional|indigenous|folk)\s+(?:cultures?|societies?|peoples?)\s+(?:believe|practice|share)'
            ],
            'reductive_analogies': [
                r'\b(?:like|similar\s+to|comparable\s+to)\s+(?:western|modern|scientific)\s+(?:but|except|only)',
                r'\bbasically\s+(?:the\s+same|equivalent|like)\s+(?:as|to)\s+(?:western|modern|scientific)',
                r'\b(?:traditional|indigenous|folk)\s+(?:version|form|type)\s+of\s+(?:western|modern|scientific)',
                r'\bthink\s+of\s+it\s+(?:as|like)\s+(?:western|modern|scientific)\s+(?:but|except|with)'
            ]
        }
        
        # Cultural knowledge systems
        self.knowledge_systems = {
            'indigenous': [
                'traditional ecological knowledge', 'tek', 'indigenous science',
                'native wisdom', 'ancestral knowledge', 'oral tradition',
                'indigenous worldview', 'native epistemology'
            ],
            'eastern': [
                'ayurveda', 'traditional chinese medicine', 'tcm', 'yoga philosophy',
                'buddhist philosophy', 'confucian thought', 'taoist principles',
                'vedic knowledge', 'zen philosophy'
            ],
            'african': [
                'ubuntu philosophy', 'african traditional medicine', 'ancestral wisdom',
                'african cosmology', 'traditional african healing', 'african philosophy'
            ],
            'other_traditional': [
                'traditional medicine', 'folk healing', 'ancestral practices',
                'traditional knowledge systems', 'indigenous medicine',
                'cultural practices', 'ceremonial practices'
            ]
        }
    
    def check_cross_cultural_coherence(self, text: str, cultural_context: Optional[str] = None) -> CrossCulturalCoherenceResult:
        """
        Check cross-cultural coherence and respectful presentation
        
        Args:
            text: The response text to check
            cultural_context: Optional context about the cultural domain
            
        Returns:
            CrossCulturalCoherenceResult with detailed analysis
        """
        cache_key = (text, cultural_context)
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        # Core analysis components
        imposition_analysis = self._analyze_framework_imposition(text)
        integrity_analysis = self._analyze_knowledge_system_integrity(text)
        translation_analysis = self._analyze_translation_quality(text)
        comparison_analysis = self._analyze_comparative_appropriateness(text)
        
        # Calculate composite scores
        coherence_score = self._calculate_coherence_score(
            imposition_analysis, integrity_analysis, translation_analysis, comparison_analysis
        )
        
        # Compile indicators and issues
        imposition_indicators = imposition_analysis.get('indicators', [])
        integrity_indicators = integrity_analysis.get('indicators', [])
        translation_issues = translation_analysis.get('issues', [])
        
        # Detailed analysis
        detailed_analysis = {
            'framework_imposition_analysis': imposition_analysis,
            'integrity_analysis': integrity_analysis,
            'translation_analysis': translation_analysis,
            'comparison_analysis': comparison_analysis,
            'knowledge_system_analysis': self._analyze_knowledge_systems_present(text),
            'overall_assessment': self._generate_coherence_assessment(
                coherence_score, len(imposition_indicators), len(integrity_indicators)
            )
        }
        
        result = CrossCulturalCoherenceResult(
            coherence_score=coherence_score,
            framework_imposition_score=imposition_analysis.get('imposition_score', 0.0),
            knowledge_system_integrity=integrity_analysis.get('integrity_score', 0.0),
            translation_quality=translation_analysis.get('quality_score', 0.0),
            comparative_appropriateness=comparison_analysis.get('appropriateness_score', 0.0),
            imposition_indicators=imposition_indicators,
            integrity_indicators=integrity_indicators,
            translation_issues=translation_issues,
            detailed_analysis=detailed_analysis
        )
        
        self.analysis_cache[cache_key] = result
        return result
    
    def _analyze_framework_imposition(self, text: str) -> Dict[str, any]:
        """Analyze inappropriate framework imposition"""
        text_lower = text.lower()
        imposition_indicators = []
        category_counts = {}
        
        for category, patterns in self.framework_imposition.items():
            matches = []
            for pattern in patterns:
                pattern_matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
                for match in pattern_matches:
                    severity = self._assess_imposition_severity(match.group(0), category)
                    matches.append({
                        'pattern': pattern,
                        'match': match.group(0),
                        'position': match.span(),
                        'category': category,
                        'severity': severity
                    })
            
            category_counts[category] = len(matches)
            imposition_indicators.extend(matches)
        
        # Calculate imposition score (lower is better)
        total_indicators = len(imposition_indicators)
        high_severity_count = sum(1 for i in imposition_indicators if i.get('severity') == 'high')
        text_length = len(text.split())
        
        imposition_density = total_indicators / max(text_length, 1)
        severity_weight = (high_severity_count * 2 + total_indicators) / max(text_length, 1)
        
        # Imposition score: 0 = heavy imposition, 1 = no imposition
        imposition_score = max(0.0, 1.0 - min(severity_weight * 10, 1.0))
        
        return {
            'indicators': imposition_indicators,
            'category_counts': category_counts,
            'total_indicators': total_indicators,
            'high_severity_count': high_severity_count,
            'imposition_density': imposition_density,
            'imposition_score': imposition_score
        }
    
    def _analyze_knowledge_system_integrity(self, text: str) -> Dict[str, any]:
        """Analyze respect for knowledge system integrity"""
        text_lower = text.lower()
        integrity_indicators = []
        category_counts = {}
        
        for category, patterns in self.integrity_markers.items():
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
            integrity_indicators.extend(matches)
        
        # Calculate integrity score (higher is better)
        total_indicators = len(integrity_indicators)
        categories_present = len([c for c, count in category_counts.items() if count > 0])
        text_length = len(text.split())
        
        integrity_density = total_indicators / max(text_length, 1)
        category_bonus = categories_present * 0.2  # Bonus for diverse integrity markers
        
        integrity_score = min(integrity_density * 15 + category_bonus, 1.0)
        
        return {
            'indicators': integrity_indicators,
            'category_counts': category_counts,
            'total_indicators': total_indicators,
            'categories_present': categories_present,
            'integrity_density': integrity_density,
            'integrity_score': integrity_score
        }
    
    def _analyze_translation_quality(self, text: str) -> Dict[str, any]:
        """Analyze cultural concept translation quality"""
        text_lower = text.lower()
        quality_indicators = []
        translation_issues = []
        category_counts = {}
        
        for category, patterns in self.translation_quality.items():
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
            quality_indicators.extend(matches)
        
        # Check for problematic direct translations
        problematic_patterns = [
            r'\b(?:exactly|directly|literally)\s+(?:translates?|means?|is)',
            r'\b(?:simply|just|merely)\s+(?:means?|is|translates?\s+to)',
            r'\bthe\s+(?:english|western)\s+(?:word|term|equivalent)\s+(?:for|of)\s+this\s+is'
        ]
        
        for pattern in problematic_patterns:
            matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
            for match in matches:
                translation_issues.append({
                    'type': 'oversimplified_translation',
                    'match': match.group(0),
                    'position': match.span(),
                    'issue': 'Overly direct translation without cultural nuance'
                })
        
        # Calculate translation quality score
        positive_indicators = len(quality_indicators)
        translation_problems = len(translation_issues)
        categories_present = len([c for c, count in category_counts.items() if count > 0])
        
        quality_score = min(positive_indicators * 0.2 + categories_present * 0.15, 1.0)
        quality_score -= translation_problems * 0.2  # Penalty for problematic translations
        quality_score = max(0.0, quality_score)
        
        return {
            'quality_indicators': quality_indicators,
            'issues': translation_issues,
            'category_counts': category_counts,
            'positive_indicators': positive_indicators,
            'translation_problems': translation_problems,
            'categories_present': categories_present,
            'quality_score': quality_score
        }
    
    def _analyze_comparative_appropriateness(self, text: str) -> Dict[str, any]:
        """Analyze appropriateness of cross-cultural comparisons"""
        text_lower = text.lower()
        inappropriate_comparisons = []
        category_counts = {}
        
        for category, patterns in self.inappropriate_comparisons.items():
            matches = []
            for pattern in patterns:
                pattern_matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
                for match in pattern_matches:
                    matches.append({
                        'pattern': pattern,
                        'match': match.group(0),
                        'position': match.span(),
                        'category': category,
                        'severity': self._assess_comparison_severity(match.group(0), category)
                    })
            
            category_counts[category] = len(matches)
            inappropriate_comparisons.extend(matches)
        
        # Calculate appropriateness score (higher is better)
        total_inappropriate = len(inappropriate_comparisons)
        high_severity_count = sum(1 for c in inappropriate_comparisons if c.get('severity') == 'high')
        text_length = len(text.split())
        
        inappropriateness_density = total_inappropriate / max(text_length, 1)
        severity_penalty = (high_severity_count * 2 + total_inappropriate) / max(text_length, 1)
        
        appropriateness_score = max(0.0, 1.0 - min(severity_penalty * 8, 1.0))
        
        return {
            'inappropriate_comparisons': inappropriate_comparisons,
            'category_counts': category_counts,
            'total_inappropriate': total_inappropriate,
            'high_severity_count': high_severity_count,
            'inappropriateness_density': inappropriateness_density,
            'appropriateness_score': appropriateness_score
        }
    
    def _analyze_knowledge_systems_present(self, text: str) -> Dict[str, any]:
        """Analyze which knowledge systems are referenced"""
        text_lower = text.lower()
        systems_mentioned = {}
        
        for system_type, keywords in self.knowledge_systems.items():
            mentions = []
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                matches = list(re.finditer(pattern, text_lower))
                mentions.extend([match.group(0) for match in matches])
            
            if mentions:
                systems_mentioned[system_type] = {
                    'count': len(mentions),
                    'terms': list(set(mentions))
                }
        
        # Identify primary knowledge system
        primary_system = None
        if systems_mentioned:
            primary_system = max(systems_mentioned, key=lambda x: systems_mentioned[x]['count'])
        
        return {
            'systems_mentioned': systems_mentioned,
            'primary_system': primary_system,
            'total_systems': len(systems_mentioned),
            'cross_cultural_discussion': len(systems_mentioned) > 1
        }
    
    def _assess_imposition_severity(self, match_text: str, category: str) -> str:
        """Assess severity of framework imposition"""
        severity_mapping = {
            'scientific_reductionism': 'high',
            'western_psychology': 'medium',
            'economic_framing': 'medium',
            'technological_determinism': 'low'
        }
        
        # Upgrade severity for certain keywords
        if any(word in match_text.lower() for word in ['objectively', 'proven', 'actually', 'really']):
            return 'high'
        
        return severity_mapping.get(category, 'medium')
    
    def _assess_comparison_severity(self, match_text: str, category: str) -> str:
        """Assess severity of inappropriate comparison"""
        severity_mapping = {
            'hierarchical_comparisons': 'high',
            'false_equivalencies': 'medium',
            'reductive_analogies': 'medium'
        }
        
        # Upgrade severity for particularly problematic terms
        if any(word in match_text.lower() for word in ['superior', 'inferior', 'primitive', 'backward', 'advanced']):
            return 'high'
        
        return severity_mapping.get(category, 'medium')
    
    def _calculate_coherence_score(self, imposition_analysis: Dict, integrity_analysis: Dict,
                                 translation_analysis: Dict, comparison_analysis: Dict) -> float:
        """Calculate overall cross-cultural coherence score"""
        
        # Weighted combination emphasizing framework imposition avoidance
        imposition_weight = 0.4  # Most important: avoiding framework imposition
        integrity_weight = 0.3   # Respecting knowledge system integrity
        translation_weight = 0.2  # Quality cultural translation
        comparison_weight = 0.1   # Appropriate comparisons
        
        coherence_score = (
            imposition_analysis.get('imposition_score', 0.0) * imposition_weight +
            integrity_analysis.get('integrity_score', 0.0) * integrity_weight +
            translation_analysis.get('quality_score', 0.0) * translation_weight +
            comparison_analysis.get('appropriateness_score', 0.0) * comparison_weight
        )
        
        return min(1.0, max(0.0, coherence_score))
    
    def _generate_coherence_assessment(self, coherence_score: float, imposition_count: int, integrity_count: int) -> Dict[str, any]:
        """Generate overall coherence assessment"""
        
        if imposition_count > 3 or coherence_score < 0.3:
            assessment_level = 'problematic'
        elif imposition_count > 1 or coherence_score < 0.5:
            assessment_level = 'concerning'
        elif coherence_score >= 0.8 and integrity_count >= 2:
            assessment_level = 'excellent'
        elif coherence_score >= 0.6:
            assessment_level = 'good'
        else:
            assessment_level = 'acceptable'
        
        return {
            'assessment_level': assessment_level,
            'coherence_score': coherence_score,
            'framework_imposition_count': imposition_count,
            'integrity_indicator_count': integrity_count,
            'requires_cultural_review': assessment_level in ['problematic', 'concerning']
        }
    
    def get_cross_cultural_coherence_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Get cross-cultural coherence evaluation thresholds"""
        return {
            'coherence_score': {
                'excellent': 0.85,
                'good': 0.7,
                'acceptable': 0.5,
                'concerning': 0.3
            },
            'framework_imposition': {
                'minimal': 0.9,
                'low': 0.7,
                'moderate': 0.5,
                'high': 0.3
            },
            'knowledge_integrity': {
                'high': 0.8,
                'adequate': 0.6,
                'limited': 0.3,
                'poor': 0.1
            },
            'translation_quality': {
                'excellent': 0.8,
                'good': 0.6,
                'adequate': 0.4,
                'poor': 0.2
            }
        }