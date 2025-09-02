"""
Social Hierarchy Navigation Assessor

Specialized evaluator for assessing competence in navigating social hierarchies,
including power distance awareness, authority recognition, status sensitivity,
deference patterns, hierarchy adaptation, and cross-cultural hierarchy navigation.

"""

from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import re
from collections import defaultdict, Counter

from ..core.domain_evaluator_base import (
    MultiDimensionalEvaluator, 
    EvaluationDimension, 
    DomainEvaluationResult,
    CulturalContext
)


class HierarchyNavigationType(Enum):
    """Types of hierarchy navigation assessment."""
    POWER_DISTANCE_AWARENESS = "power_distance_awareness"
    AUTHORITY_RECOGNITION = "authority_recognition"
    STATUS_SENSITIVITY = "status_sensitivity"
    DEFERENCE_PATTERNS = "deference_patterns"
    HIERARCHY_ADAPTATION = "hierarchy_adaptation"
    VERTICAL_COMMUNICATION = "vertical_communication"
    CROSS_CULTURAL_HIERARCHY = "cross_cultural_hierarchy"


class SocialHierarchyNavigationAssessor(MultiDimensionalEvaluator):
    """Evaluates competence in navigating social hierarchies across cultural contexts."""
    
    VERSION = "1.0.0"
    
    def _initialize_evaluator(self):
        """Initialize social hierarchy navigation evaluation components."""
        
        # Power distance awareness patterns
        self.power_distance_patterns = {
            "high_power_distance": [
                "respect for authority", "hierarchy is important", "clear chain of command",
                "formal protocols", "senior leadership", "proper channels",
                "established procedures", "rank structure", "authority figures",
                "vertical organization", "top-down approach", "command structure"
            ],
            "low_power_distance": [
                "flat organization", "accessible leadership", "open door policy",
                "democratic decision making", "equal participation", "informal communication",
                "questioning authority", "collaborative approach", "team-based",
                "horizontal structure", "peer relationships", "participative management"
            ],
            "power_distance_concepts": [
                "power distance", "hofstede", "cultural dimension", "authority acceptance",
                "inequality expectation", "subordinate behavior", "superior-subordinate",
                "power distribution", "status differential", "hierarchical distance"
            ]
        }
        
        # Authority recognition markers
        self.authority_recognition_patterns = {
            "formal_authority": [
                "official position", "title", "rank", "designation", "appointment",
                "elected official", "institutional authority", "organizational hierarchy",
                "formal leadership", "administrative position", "executive role"
            ],
            "traditional_authority": [
                "elder", "patriarch", "matriarch", "traditional leader", "ceremonial role",
                "ancestral authority", "customary leadership", "tribal chief",
                "family head", "community elder", "respected elder", "wisdom keeper"
            ],
            "charismatic_authority": [
                "natural leader", "inspiring figure", "influential person", "thought leader",
                "visionary", "charismatic leadership", "personal magnetism",
                "inspirational authority", "transformational leader"
            ],
            "expert_authority": [
                "subject matter expert", "technical authority", "professional expertise",
                "specialist knowledge", "domain expert", "consultant authority",
                "advisory role", "knowledge leader", "technical leadership"
            ]
        }
        
        # Status sensitivity indicators
        self.status_sensitivity_patterns = {
            "status_markers": [
                "social status", "prestige", "reputation", "standing", "position",
                "social rank", "class position", "status symbol", "social capital",
                "cultural capital", "symbolic capital", "status hierarchy"
            ],
            "status_behavior": [
                "status-conscious", "face-saving", "dignity preservation", "honor",
                "respect for status", "status maintenance", "status enhancement",
                "status anxiety", "status competition", "status display"
            ],
            "status_communication": [
                "appropriate deference", "respectful address", "formal titles",
                "status-appropriate behavior", "protocol awareness", "etiquette",
                "ceremonial respect", "status acknowledgment", "proper recognition"
            ]
        }
        
        # Deference patterns
        self.deference_patterns = {
            "linguistic_deference": [
                "honorifics", "respectful language", "formal address", "polite forms",
                "humble language", "deferential speech", "respectful tone",
                "appropriate titles", "sir", "madam", "your honor", "your excellence"
            ],
            "behavioral_deference": [
                "bowing", "standing when appropriate", "waiting to be seated",
                "letting others go first", "seeking permission", "asking approval",
                "deferential posture", "respectful gesture", "appropriate distance"
            ],
            "cultural_deference": [
                "age respect", "gender deference", "professional courtesy",
                "academic respect", "religious deference", "cultural protocol",
                "traditional respect", "ceremonial deference", "ritual respect"
            ]
        }
        
        # Hierarchy adaptation skills
        self.hierarchy_adaptation_patterns = {
            "context_sensitivity": [
                "reading the room", "situational awareness", "context adaptation",
                "environmental scanning", "social cues", "cultural context",
                "situational adjustment", "adaptive behavior", "context-appropriate"
            ],
            "role_flexibility": [
                "switching roles", "role adaptation", "position adjustment",
                "behavioral flexibility", "role-appropriate", "situational roles",
                "context-dependent behavior", "adaptive positioning"
            ],
            "communication_adaptation": [
                "adjusting communication style", "tone modification", "register switching",
                "formal vs informal", "adapting approach", "style flexibility",
                "communication adjustment", "appropriate register"
            ]
        }
        
        # Vertical communication patterns
        self.vertical_communication_patterns = {
            "upward_communication": [
                "reporting to superiors", "briefing leadership", "escalation",
                "seeking guidance", "formal reporting", "status updates",
                "requesting approval", "upward feedback", "management communication"
            ],
            "downward_communication": [
                "delegating authority", "providing direction", "team leadership",
                "instructional communication", "performance guidance",
                "supervisory communication", "subordinate interaction"
            ],
            "communication_protocols": [
                "chain of command", "proper channels", "official procedures",
                "formal protocols", "communication hierarchy", "reporting structure",
                "procedural communication", "protocol adherence"
            ]
        }
        
        # Cross-cultural hierarchy navigation
        self.cross_cultural_hierarchy_patterns = {
            "cultural_variation_awareness": [
                "different hierarchy styles", "cultural differences in authority",
                "varying power structures", "cultural hierarchy norms",
                "cross-cultural sensitivity", "hierarchy variation",
                "cultural adaptation", "context-specific hierarchy"
            ],
            "adaptation_strategies": [
                "observing before acting", "following local customs",
                "asking about protocols", "cultural accommodation",
                "respectful inquiry", "learning local norms",
                "adaptive strategy", "cultural adjustment"
            ],
            "conflict_navigation": [
                "hierarchy conflicts", "authority disputes", "power struggles",
                "status conflicts", "jurisdictional issues", "role conflicts",
                "boundary disputes", "authority challenges"
            ]
        }
        
        # Cultural hierarchy systems
        self.cultural_hierarchy_systems = {
            "east_asian": [
                "confucian hierarchy", "age respect", "filial piety", "seniority system",
                "face concept", "guanxi", "social harmony", "collective hierarchy",
                "group orientation", "consensus building", "indirect communication"
            ],
            "western_corporate": [
                "organizational chart", "corporate hierarchy", "executive levels",
                "management structure", "professional hierarchy", "merit-based",
                "performance hierarchy", "functional authority", "matrix organization"
            ],
            "traditional_african": [
                "elder system", "tribal hierarchy", "community leadership",
                "age-based authority", "consensus decision-making", "ubuntu philosophy",
                "collective responsibility", "traditional council", "wisdom hierarchy"
            ],
            "latin_american": [
                "familismo", "personalismo", "patron-client relationships",
                "informal hierarchy", "relationship-based authority", "compadrazgo",
                "family hierarchy", "personal loyalty", "informal networks"
            ],
            "middle_eastern": [
                "tribal hierarchy", "family honor", "hospitality protocols",
                "gender roles", "religious authority", "traditional respect",
                "family patriarch", "community elder", "Islamic hierarchy"
            ]
        }
        
        # Hierarchy violations and mistakes
        self.hierarchy_violations = {
            "protocol_violations": [
                "bypassing authority", "inappropriate directness", "protocol breach",
                "chain of command violation", "disrespectful behavior", "status insensitivity",
                "hierarchy disruption", "authority challenge", "improper approach"
            ],
            "cultural_insensitivity": [
                "cultural misunderstanding", "hierarchy ignorance", "inappropriate behavior",
                "cultural offense", "status blindness", "protocol ignorance",
                "cultural violation", "hierarchy misreading", "context insensitivity"
            ]
        }
    
    def get_domain_name(self) -> str:
        """Return the domain name this evaluator handles."""
        return "social_hierarchy_navigation"
    
    def get_supported_evaluation_types(self) -> List[str]:
        """Return list of evaluation types this evaluator supports."""
        return [evaluation_type.value for evaluation_type in HierarchyNavigationType]
    
    def get_evaluation_dimensions(self) -> List[str]:
        """Return list of dimensions this evaluator assesses."""
        return [
            "power_distance_awareness",
            "authority_recognition",
            "status_sensitivity",
            "deference_competence",
            "hierarchy_adaptation",
            "vertical_communication",
            "cross_cultural_navigation",
            "protocol_adherence"
        ]
    
    def evaluate_dimension(self, 
                          dimension: str,
                          response_text: str, 
                          test_metadata: Dict[str, Any], 
                          cultural_context: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate a specific dimension."""
        cultural_ctx = self._create_cultural_context(cultural_context)
        
        if dimension == "power_distance_awareness":
            return self._evaluate_power_distance_awareness(response_text, cultural_ctx, test_metadata)
        elif dimension == "authority_recognition":
            return self._evaluate_authority_recognition(response_text, cultural_ctx, test_metadata)
        elif dimension == "status_sensitivity":
            return self._evaluate_status_sensitivity(response_text, cultural_ctx, test_metadata)
        elif dimension == "deference_competence":
            return self._evaluate_deference_competence(response_text, cultural_ctx, test_metadata)
        elif dimension == "hierarchy_adaptation":
            return self._evaluate_hierarchy_adaptation(response_text, cultural_ctx, test_metadata)
        elif dimension == "vertical_communication":
            return self._evaluate_vertical_communication(response_text, cultural_ctx, test_metadata)
        elif dimension == "cross_cultural_navigation":
            return self._evaluate_cross_cultural_navigation(response_text, cultural_ctx, test_metadata)
        elif dimension == "protocol_adherence":
            return self._evaluate_protocol_adherence(response_text, cultural_ctx, test_metadata)
        else:
            return EvaluationDimension(
                name=dimension,
                score=0.0,
                confidence=0.0,
                cultural_relevance=0.0,
                evidence=[f"Unknown dimension: {dimension}"],
                cultural_markers=[]
            )
    
    def _evaluate_power_distance_awareness(self, response: str, cultural_context: CulturalContext,
                                         test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate awareness of power distance concepts and cultural variation."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count power distance indicators
        high_pd_count = sum(1 for pattern in self.power_distance_patterns["high_power_distance"]
                           if pattern in response_lower)
        low_pd_count = sum(1 for pattern in self.power_distance_patterns["low_power_distance"]
                          if pattern in response_lower)
        concept_count = sum(1 for concept in self.power_distance_patterns["power_distance_concepts"]
                           if concept in response_lower)
        
        if high_pd_count > 0:
            evidence.append(f"High power distance awareness: {high_pd_count} instances")
            cultural_markers.append("high_power_distance_competence")
        
        if low_pd_count > 0:
            evidence.append(f"Low power distance awareness: {low_pd_count} instances")
            cultural_markers.append("low_power_distance_competence")
        
        if concept_count > 0:
            evidence.append(f"Power distance concepts: {concept_count} instances")
            cultural_markers.append("theoretical_awareness")
        
        # Cultural context bonus
        cultural_bonus = 0.0
        if any(group in ["east_asian", "middle_eastern", "african_traditional"] 
               for group in cultural_context.cultural_groups):
            if high_pd_count > 0:
                cultural_bonus = 0.15
                cultural_markers.append("high_pd_cultural_alignment")
        elif any(group in ["scandinavian", "australian", "dutch"] 
                for group in cultural_context.cultural_groups):
            if low_pd_count > 0:
                cultural_bonus = 0.15
                cultural_markers.append("low_pd_cultural_alignment")
        
        total_score = (high_pd_count * 0.35 + low_pd_count * 0.35 + concept_count * 0.3)
        
        score = min(1.0, (total_score * 0.12) + cultural_bonus)
        confidence = min(1.0, (high_pd_count + low_pd_count + concept_count) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="power_distance_awareness",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_authority_recognition(self, response: str, cultural_context: CulturalContext,
                                      test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate recognition of different types of authority."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count authority type recognition
        formal_count = sum(1 for auth in self.authority_recognition_patterns["formal_authority"]
                          if auth in response_lower)
        traditional_count = sum(1 for auth in self.authority_recognition_patterns["traditional_authority"]
                               if auth in response_lower)
        charismatic_count = sum(1 for auth in self.authority_recognition_patterns["charismatic_authority"]
                               if auth in response_lower)
        expert_count = sum(1 for auth in self.authority_recognition_patterns["expert_authority"]
                          if auth in response_lower)
        
        if formal_count > 0:
            evidence.append(f"Formal authority recognition: {formal_count} instances")
            cultural_markers.append("formal_authority_competence")
        
        if traditional_count > 0:
            evidence.append(f"Traditional authority recognition: {traditional_count} instances")
            cultural_markers.append("traditional_authority_competence")
        
        if charismatic_count > 0:
            evidence.append(f"Charismatic authority recognition: {charismatic_count} instances")
            cultural_markers.append("charismatic_authority_competence")
        
        if expert_count > 0:
            evidence.append(f"Expert authority recognition: {expert_count} instances")
            cultural_markers.append("expert_authority_competence")
        
        # Bonus for traditional contexts
        traditional_bonus = 0.0
        if "traditional" in cultural_context.traditions and traditional_count > 0:
            traditional_bonus = 0.1
            cultural_markers.append("traditional_context_alignment")
        
        total_score = (formal_count * 0.25 + traditional_count * 0.25 + 
                      charismatic_count * 0.25 + expert_count * 0.25)
        
        score = min(1.0, (total_score * 0.15) + traditional_bonus)
        confidence = min(1.0, (formal_count + traditional_count + charismatic_count + expert_count) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="authority_recognition",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_status_sensitivity(self, response: str, cultural_context: CulturalContext,
                                   test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate sensitivity to status differences and social ranking."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count status sensitivity indicators
        status_markers = sum(1 for marker in self.status_sensitivity_patterns["status_markers"]
                            if marker in response_lower)
        status_behavior = sum(1 for behavior in self.status_sensitivity_patterns["status_behavior"]
                             if behavior in response_lower)
        status_communication = sum(1 for comm in self.status_sensitivity_patterns["status_communication"]
                                  if comm in response_lower)
        
        if status_markers > 0:
            evidence.append(f"Status markers awareness: {status_markers} instances")
            cultural_markers.append("status_marker_competence")
        
        if status_behavior > 0:
            evidence.append(f"Status behavior awareness: {status_behavior} instances")
            cultural_markers.append("status_behavior_competence")
        
        if status_communication > 0:
            evidence.append(f"Status communication awareness: {status_communication} instances")
            cultural_markers.append("status_communication_competence")
        
        total_score = (status_markers * 0.4 + status_behavior * 0.35 + status_communication * 0.25)
        
        score = min(1.0, total_score * 0.15)
        confidence = min(1.0, (status_markers + status_behavior + status_communication) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="status_sensitivity",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_deference_competence(self, response: str, cultural_context: CulturalContext,
                                     test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate competence in showing appropriate deference."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count deference indicators
        linguistic_def = sum(1 for def_pattern in self.deference_patterns["linguistic_deference"]
                            if def_pattern in response_lower)
        behavioral_def = sum(1 for def_pattern in self.deference_patterns["behavioral_deference"]
                            if def_pattern in response_lower)
        cultural_def = sum(1 for def_pattern in self.deference_patterns["cultural_deference"]
                          if def_pattern in response_lower)
        
        if linguistic_def > 0:
            evidence.append(f"Linguistic deference: {linguistic_def} instances")
            cultural_markers.append("linguistic_deference_competence")
        
        if behavioral_def > 0:
            evidence.append(f"Behavioral deference: {behavioral_def} instances")
            cultural_markers.append("behavioral_deference_competence")
        
        if cultural_def > 0:
            evidence.append(f"Cultural deference: {cultural_def} instances")
            cultural_markers.append("cultural_deference_competence")
        
        total_score = (linguistic_def * 0.4 + behavioral_def * 0.35 + cultural_def * 0.25)
        
        score = min(1.0, total_score * 0.15)
        confidence = min(1.0, (linguistic_def + behavioral_def + cultural_def) * 0.12)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="deference_competence",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_hierarchy_adaptation(self, response: str, cultural_context: CulturalContext,
                                     test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate ability to adapt to different hierarchical contexts."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count adaptation indicators
        context_sens = sum(1 for sens in self.hierarchy_adaptation_patterns["context_sensitivity"]
                          if sens in response_lower)
        role_flex = sum(1 for flex in self.hierarchy_adaptation_patterns["role_flexibility"]
                       if flex in response_lower)
        comm_adapt = sum(1 for adapt in self.hierarchy_adaptation_patterns["communication_adaptation"]
                        if adapt in response_lower)
        
        if context_sens > 0:
            evidence.append(f"Context sensitivity: {context_sens} instances")
            cultural_markers.append("contextual_adaptation")
        
        if role_flex > 0:
            evidence.append(f"Role flexibility: {role_flex} instances")
            cultural_markers.append("role_adaptation")
        
        if comm_adapt > 0:
            evidence.append(f"Communication adaptation: {comm_adapt} instances")
            cultural_markers.append("communication_flexibility")
        
        total_score = (context_sens * 0.4 + role_flex * 0.35 + comm_adapt * 0.25)
        
        score = min(1.0, total_score * 0.15)
        confidence = min(1.0, (context_sens + role_flex + comm_adapt) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="hierarchy_adaptation",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_vertical_communication(self, response: str, cultural_context: CulturalContext,
                                       test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate competence in vertical communication patterns."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count vertical communication indicators
        upward_comm = sum(1 for comm in self.vertical_communication_patterns["upward_communication"]
                         if comm in response_lower)
        downward_comm = sum(1 for comm in self.vertical_communication_patterns["downward_communication"]
                           if comm in response_lower)
        protocols = sum(1 for protocol in self.vertical_communication_patterns["communication_protocols"]
                       if protocol in response_lower)
        
        if upward_comm > 0:
            evidence.append(f"Upward communication: {upward_comm} instances")
            cultural_markers.append("upward_communication_competence")
        
        if downward_comm > 0:
            evidence.append(f"Downward communication: {downward_comm} instances")
            cultural_markers.append("downward_communication_competence")
        
        if protocols > 0:
            evidence.append(f"Communication protocols: {protocols} instances")
            cultural_markers.append("protocol_competence")
        
        total_score = (upward_comm * 0.4 + downward_comm * 0.35 + protocols * 0.25)
        
        score = min(1.0, total_score * 0.15)
        confidence = min(1.0, (upward_comm + downward_comm + protocols) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="vertical_communication",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_cross_cultural_navigation(self, response: str, cultural_context: CulturalContext,
                                          test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate cross-cultural hierarchy navigation skills."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count cross-cultural navigation indicators
        variation_aware = sum(1 for var in self.cross_cultural_hierarchy_patterns["cultural_variation_awareness"]
                             if var in response_lower)
        adaptation_strategies = sum(1 for strategy in self.cross_cultural_hierarchy_patterns["adaptation_strategies"]
                                   if strategy in response_lower)
        conflict_nav = sum(1 for conflict in self.cross_cultural_hierarchy_patterns["conflict_navigation"]
                          if conflict in response_lower)
        
        if variation_aware > 0:
            evidence.append(f"Cultural variation awareness: {variation_aware} instances")
            cultural_markers.append("cultural_hierarchy_awareness")
        
        if adaptation_strategies > 0:
            evidence.append(f"Adaptation strategies: {adaptation_strategies} instances")
            cultural_markers.append("cross_cultural_adaptation")
        
        if conflict_nav > 0:
            evidence.append(f"Conflict navigation: {conflict_nav} instances")
            cultural_markers.append("hierarchy_conflict_competence")
        
        # Multicultural bonus
        multicultural_bonus = 0.0
        if len(cultural_context.cultural_groups) > 1:
            multicultural_bonus = 0.2
            cultural_markers.append("multicultural_hierarchy_competence")
        
        total_score = (variation_aware * 0.4 + adaptation_strategies * 0.35 + conflict_nav * 0.25)
        
        score = min(1.0, (total_score * 0.12) + multicultural_bonus)
        confidence = min(1.0, (variation_aware + adaptation_strategies + conflict_nav) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="cross_cultural_navigation",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_protocol_adherence(self, response: str, cultural_context: CulturalContext,
                                   test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate adherence to hierarchical protocols and avoidance of violations."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count protocol adherence indicators (positive)
        protocol_awareness = sum(1 for protocol in self.vertical_communication_patterns["communication_protocols"]
                               if protocol in response_lower)
        
        # Count violations (negative indicators)
        protocol_violations = sum(1 for violation in self.hierarchy_violations["protocol_violations"]
                                 if violation in response_lower)
        cultural_insensitivity = sum(1 for insens in self.hierarchy_violations["cultural_insensitivity"]
                                    if insens in response_lower)
        
        if protocol_awareness > 0:
            evidence.append(f"Protocol awareness: {protocol_awareness} instances")
            cultural_markers.append("protocol_competence")
        
        if protocol_violations > 0:
            evidence.append(f"Protocol violations identified: {protocol_violations} instances")
            cultural_markers.append("violation_awareness")
        
        if cultural_insensitivity > 0:
            evidence.append(f"Cultural insensitivity identified: {cultural_insensitivity} instances")
            cultural_markers.append("insensitivity_awareness")
        
        # Calculate score with penalties for violations
        base_score = protocol_awareness * 0.3
        violation_penalty = (protocol_violations + cultural_insensitivity) * 0.1
        
        score = max(0.0, min(1.0, base_score - violation_penalty))
        confidence = min(1.0, (protocol_awareness + protocol_violations + cultural_insensitivity) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="protocol_adherence",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _create_cultural_context(self, cultural_context: Dict[str, Any]) -> CulturalContext:
        """Create CulturalContext object from dictionary input."""
        if isinstance(cultural_context, CulturalContext):
            return cultural_context
        
        return CulturalContext(
            traditions=cultural_context.get('traditions', []),
            knowledge_systems=cultural_context.get('knowledge_systems', []),
            performance_aspects=cultural_context.get('performance_aspects', []),
            cultural_groups=cultural_context.get('cultural_groups', []),
            linguistic_varieties=cultural_context.get('linguistic_varieties', [])
        )
    
    def _calculate_cultural_relevance(self, cultural_markers: List[str], 
                                    cultural_context: CulturalContext) -> float:
        """Calculate cultural relevance score based on detected markers and context."""
        if not cultural_markers:
            return 0.5  # Default relevance when no markers detected
        
        total_relevance = 0.0
        marker_count = 0
        
        # Check alignment with cultural context
        for marker in cultural_markers:
            marker_type = marker.split('_')[0] if '_' in marker else marker
            relevance_score = 0.5  # Default relevance
            
            # Higher relevance for markers aligned with hierarchical cultures
            if marker_type in ["high", "formal", "traditional"] and any(
                group in ["east_asian", "middle_eastern", "african_traditional", "latin_american"] 
                for group in cultural_context.cultural_groups
            ):
                relevance_score = 0.95
            elif marker_type in ["low", "informal", "egalitarian"] and any(
                group in ["scandinavian", "dutch", "australian"] 
                for group in cultural_context.cultural_groups
            ):
                relevance_score = 0.9
            elif marker_type in ["multicultural", "cross"] and len(cultural_context.cultural_groups) > 1:
                relevance_score = 0.9
            elif marker_type in ["power", "authority", "status", "hierarchy"]:
                relevance_score = 0.85
            elif marker_type in ["deference", "protocol", "communication"]:
                relevance_score = 0.8
            elif marker_type in ["adaptation", "flexibility", "contextual"]:
                relevance_score = 0.8
            elif marker_type in ["violation", "insensitivity"] and len(cultural_context.cultural_groups) > 0:
                relevance_score = 0.9  # High relevance for awareness of violations
            
            total_relevance += relevance_score
            marker_count += 1
        
        return min(1.0, total_relevance / marker_count) if marker_count > 0 else 0.5