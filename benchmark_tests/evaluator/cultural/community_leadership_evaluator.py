"""
Community Leadership Evaluator

Specialized evaluator for assessing community leadership competence across cultural contexts
including visionary leadership, collective mobilization, consensus building, cultural preservation,
community empowerment, collaborative governance, and sustainable development.

Author: Claude Code
Version: 1.0.0
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


class CommunityLeadershipType(Enum):
    """Types of community leadership evaluation."""
    VISIONARY_LEADERSHIP = "visionary_leadership"
    COLLECTIVE_MOBILIZATION = "collective_mobilization"
    CONSENSUS_BUILDING = "consensus_building"
    CULTURAL_PRESERVATION = "cultural_preservation"
    COMMUNITY_EMPOWERMENT = "community_empowerment"
    COLLABORATIVE_GOVERNANCE = "collaborative_governance"
    SUSTAINABLE_DEVELOPMENT = "sustainable_development"


class CommunityLeadershipEvaluator(MultiDimensionalEvaluator):
    """Evaluates community leadership competence across cultural contexts."""
    
    VERSION = "1.0.0"
    
    def _initialize_evaluator(self):
        """Initialize community leadership evaluation components."""
        
        # Visionary leadership patterns
        self.visionary_leadership_patterns = {
            "vision_articulation": [
                "shared vision", "collective future", "common purpose", "inspiring vision",
                "community goals", "long-term vision", "collective aspiration",
                "shared dreams", "community destiny", "future direction"
            ],
            "transformational_thinking": [
                "transformative change", "paradigm shift", "revolutionary approach",
                "systemic transformation", "fundamental change", "breakthrough",
                "innovation", "creative solutions", "new possibilities"
            ],
            "inspirational_motivation": [
                "inspire community", "motivate others", "rally support", "energize people",
                "build enthusiasm", "create excitement", "generate passion",
                "foster commitment", "kindle hope", "ignite action"
            ],
            "future_orientation": [
                "next generation", "future generations", "legacy building",
                "sustainable future", "long-term thinking", "forward-looking",
                "anticipatory", "proactive planning", "future-focused"
            ]
        }
        
        # Collective mobilization patterns
        self.collective_mobilization_patterns = {
            "community_organizing": [
                "grassroots organizing", "community mobilization", "collective action",
                "community engagement", "mass participation", "popular movement",
                "social movement", "community organizing", "citizen participation"
            ],
            "resource_mobilization": [
                "resource pooling", "collective resources", "community assets",
                "shared resources", "resource coordination", "resource leveraging",
                "asset mapping", "community wealth", "collective investment"
            ],
            "network_building": [
                "building coalitions", "creating alliances", "partnership development",
                "network expansion", "relationship building", "collaboration",
                "interconnection", "community bonds", "social capital"
            ],
            "collective_efficacy": [
                "collective strength", "community power", "unified action",
                "shared capability", "collective competence", "group effectiveness",
                "community capacity", "collective agency", "united effort"
            ]
        }
        
        # Consensus building patterns
        self.consensus_building_patterns = {
            "participatory_processes": [
                "inclusive participation", "democratic process", "stakeholder involvement",
                "community input", "participatory decision making", "collective dialogue",
                "open forum", "town hall", "community consultation"
            ],
            "conflict_resolution": [
                "mediation", "conflict transformation", "peaceful resolution",
                "dialogue facilitation", "reconciliation", "bridge building",
                "healing divisions", "finding common ground", "harmony"
            ],
            "facilitation_skills": [
                "meeting facilitation", "group facilitation", "process management",
                "discussion leading", "consensus facilitation", "group dynamics",
                "collaborative process", "structured dialogue", "guided discussion"
            ],
            "decision_making": [
                "consensus decision", "collective choice", "shared decision",
                "democratic choice", "community decision", "participatory decision",
                "collaborative decision", "inclusive decision making"
            ]
        }
        
        # Cultural preservation patterns
        self.cultural_preservation_patterns = {
            "tradition_maintaining": [
                "preserving traditions", "cultural heritage", "ancestral wisdom",
                "traditional practices", "cultural continuity", "heritage protection",
                "tradition passing", "cultural transmission", "legacy preservation"
            ],
            "knowledge_preservation": [
                "traditional knowledge", "indigenous wisdom", "cultural knowledge",
                "ancestral teachings", "oral history", "cultural memory",
                "community knowledge", "collective wisdom", "cultural learning"
            ],
            "identity_strengthening": [
                "cultural identity", "community identity", "ethnic identity",
                "cultural pride", "identity formation", "cultural affirmation",
                "identity preservation", "cultural strength", "community pride"
            ],
            "intergenerational_transfer": [
                "elder wisdom", "youth engagement", "intergenerational dialogue",
                "knowledge transfer", "generational bridge", "elder-youth connection",
                "cultural mentoring", "wisdom sharing", "generational continuity"
            ]
        }
        
        # Community empowerment patterns
        self.community_empowerment_patterns = {
            "capacity_building": [
                "skill development", "capacity building", "competence enhancement",
                "ability strengthening", "capability development", "training provision",
                "education programs", "skill sharing", "knowledge building"
            ],
            "self_determination": [
                "community autonomy", "self-governance", "self-determination",
                "community control", "local sovereignty", "autonomous decision",
                "community ownership", "self-directed", "independent action"
            ],
            "agency_development": [
                "empowering individuals", "building confidence", "fostering agency",
                "enabling participation", "encouraging initiative", "promoting leadership",
                "developing potential", "unlocking capabilities", "personal growth"
            ],
            "collective_ownership": [
                "community ownership", "shared ownership", "collective responsibility",
                "community assets", "shared control", "participatory ownership",
                "democratic ownership", "community-controlled", "locally owned"
            ]
        }
        
        # Collaborative governance patterns
        self.collaborative_governance_patterns = {
            "stakeholder_engagement": [
                "multi-stakeholder", "stakeholder participation", "inclusive governance",
                "stakeholder involvement", "participatory governance", "collaborative governance",
                "partnership approach", "shared governance", "co-governance"
            ],
            "transparency_accountability": [
                "transparent process", "accountable leadership", "open governance",
                "public accountability", "transparent decision making", "responsible leadership",
                "ethical governance", "integrity", "trustworthy leadership"
            ],
            "adaptive_management": [
                "adaptive leadership", "flexible approach", "responsive governance",
                "adaptive management", "learning organization", "continuous improvement",
                "iterative process", "feedback integration", "adaptive capacity"
            ],
            "power_sharing": [
                "power distribution", "shared authority", "distributed leadership",
                "decentralized power", "power sharing", "collaborative authority",
                "joint leadership", "shared control", "democratic power"
            ]
        }
        
        # Sustainable development patterns
        self.sustainable_development_patterns = {
            "environmental_stewardship": [
                "environmental protection", "ecological stewardship", "sustainability",
                "environmental responsibility", "ecological preservation", "green development",
                "environmental care", "sustainable practices", "ecological mindfulness"
            ],
            "economic_sustainability": [
                "economic viability", "sustainable economy", "local economy",
                "economic resilience", "sustainable livelihoods", "economic development",
                "community economics", "cooperative economy", "inclusive economy"
            ],
            "social_sustainability": [
                "social cohesion", "community well-being", "social equity",
                "inclusive development", "social justice", "community resilience",
                "social capital", "community health", "social sustainability"
            ],
            "holistic_development": [
                "integrated development", "holistic approach", "comprehensive development",
                "multidimensional progress", "balanced development", "whole systems",
                "interconnected development", "systemic approach", "integrated solutions"
            ]
        }
        
        # Cultural leadership styles
        self.cultural_leadership_styles = {
            "indigenous_leadership": [
                "consensus leadership", "circular leadership", "elder guidance",
                "ceremonial leadership", "spiritual leadership", "traditional governance",
                "indigenous governance", "tribal leadership", "ancestral guidance"
            ],
            "african_ubuntu": [
                "ubuntu philosophy", "communal leadership", "collective wisdom",
                "shared humanity", "interconnectedness", "community solidarity",
                "mutual support", "collective responsibility", "ubuntu principles"
            ],
            "asian_collective": [
                "collective leadership", "harmony emphasis", "consensus building",
                "face-saving", "relationship priority", "group harmony",
                "collective decision", "social harmony", "group-first"
            ],
            "latin_community": [
                "personalismo", "familismo", "community bonds", "personal relationships",
                "extended family", "compadrazgo", "community solidarity",
                "relationship-based leadership", "personal connections"
            ]
        }
        
        # Leadership challenges and competencies
        self.leadership_competencies = {
            "emotional_intelligence": [
                "emotional awareness", "empathy", "social intelligence",
                "relationship management", "emotional regulation", "social skills",
                "interpersonal competence", "emotional competence", "people skills"
            ],
            "cultural_competence": [
                "cultural sensitivity", "cross-cultural competence", "cultural awareness",
                "multicultural leadership", "cultural adaptation", "cultural intelligence",
                "intercultural skills", "cultural responsiveness", "diversity leadership"
            ],
            "systems_thinking": [
                "systems perspective", "holistic thinking", "interconnected systems",
                "systemic solutions", "complexity thinking", "systems leadership",
                "whole systems approach", "integrated thinking", "systems analysis"
            ],
            "resilience_building": [
                "community resilience", "adaptive capacity", "crisis management",
                "recovery leadership", "resilience development", "crisis response",
                "emergency leadership", "disaster recovery", "community healing"
            ]
        }
    
    def get_domain_name(self) -> str:
        """Return the domain name this evaluator handles."""
        return "community_leadership"
    
    def get_supported_evaluation_types(self) -> List[str]:
        """Return list of evaluation types this evaluator supports."""
        return [evaluation_type.value for evaluation_type in CommunityLeadershipType]
    
    def get_evaluation_dimensions(self) -> List[str]:
        """Return list of dimensions this evaluator assesses."""
        return [
            "visionary_leadership",
            "collective_mobilization",
            "consensus_building",
            "cultural_preservation",
            "community_empowerment",
            "collaborative_governance",
            "sustainable_development",
            "leadership_competencies"
        ]
    
    def evaluate_dimension(self, 
                          dimension: str,
                          response_text: str, 
                          test_metadata: Dict[str, Any], 
                          cultural_context: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate a specific dimension."""
        cultural_ctx = self._create_cultural_context(cultural_context)
        
        if dimension == "visionary_leadership":
            return self._evaluate_visionary_leadership(response_text, cultural_ctx, test_metadata)
        elif dimension == "collective_mobilization":
            return self._evaluate_collective_mobilization(response_text, cultural_ctx, test_metadata)
        elif dimension == "consensus_building":
            return self._evaluate_consensus_building(response_text, cultural_ctx, test_metadata)
        elif dimension == "cultural_preservation":
            return self._evaluate_cultural_preservation(response_text, cultural_ctx, test_metadata)
        elif dimension == "community_empowerment":
            return self._evaluate_community_empowerment(response_text, cultural_ctx, test_metadata)
        elif dimension == "collaborative_governance":
            return self._evaluate_collaborative_governance(response_text, cultural_ctx, test_metadata)
        elif dimension == "sustainable_development":
            return self._evaluate_sustainable_development(response_text, cultural_ctx, test_metadata)
        elif dimension == "leadership_competencies":
            return self._evaluate_leadership_competencies(response_text, cultural_ctx, test_metadata)
        else:
            return EvaluationDimension(
                name=dimension,
                score=0.0,
                confidence=0.0,
                cultural_relevance=0.0,
                evidence=[f"Unknown dimension: {dimension}"],
                cultural_markers=[]
            )
    
    def _evaluate_visionary_leadership(self, response: str, cultural_context: CulturalContext,
                                     test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate visionary leadership capabilities."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count visionary leadership indicators
        vision_articulation = sum(1 for vision in self.visionary_leadership_patterns["vision_articulation"]
                                 if vision in response_lower)
        transformational = sum(1 for trans in self.visionary_leadership_patterns["transformational_thinking"]
                              if trans in response_lower)
        inspirational = sum(1 for insp in self.visionary_leadership_patterns["inspirational_motivation"]
                           if insp in response_lower)
        future_oriented = sum(1 for future in self.visionary_leadership_patterns["future_orientation"]
                             if future in response_lower)
        
        if vision_articulation > 0:
            evidence.append(f"Vision articulation: {vision_articulation} instances")
            cultural_markers.append("vision_articulation_competence")
        
        if transformational > 0:
            evidence.append(f"Transformational thinking: {transformational} instances")
            cultural_markers.append("transformational_leadership")
        
        if inspirational > 0:
            evidence.append(f"Inspirational motivation: {inspirational} instances")
            cultural_markers.append("inspirational_competence")
        
        if future_oriented > 0:
            evidence.append(f"Future orientation: {future_oriented} instances")
            cultural_markers.append("future_oriented_thinking")
        
        total_score = (vision_articulation * 0.3 + transformational * 0.25 + 
                      inspirational * 0.25 + future_oriented * 0.2)
        
        score = min(1.0, total_score * 0.15)
        confidence = min(1.0, (vision_articulation + transformational + inspirational + future_oriented) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="visionary_leadership",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_collective_mobilization(self, response: str, cultural_context: CulturalContext,
                                        test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate collective mobilization skills."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count mobilization indicators
        organizing = sum(1 for org in self.collective_mobilization_patterns["community_organizing"]
                        if org in response_lower)
        resources = sum(1 for res in self.collective_mobilization_patterns["resource_mobilization"]
                       if res in response_lower)
        networking = sum(1 for net in self.collective_mobilization_patterns["network_building"]
                        if net in response_lower)
        efficacy = sum(1 for eff in self.collective_mobilization_patterns["collective_efficacy"]
                      if eff in response_lower)
        
        if organizing > 0:
            evidence.append(f"Community organizing: {organizing} instances")
            cultural_markers.append("community_organizing_competence")
        
        if resources > 0:
            evidence.append(f"Resource mobilization: {resources} instances")
            cultural_markers.append("resource_mobilization_competence")
        
        if networking > 0:
            evidence.append(f"Network building: {networking} instances")
            cultural_markers.append("network_building_competence")
        
        if efficacy > 0:
            evidence.append(f"Collective efficacy: {efficacy} instances")
            cultural_markers.append("collective_efficacy_competence")
        
        total_score = (organizing * 0.3 + resources * 0.25 + networking * 0.25 + efficacy * 0.2)
        
        score = min(1.0, total_score * 0.15)
        confidence = min(1.0, (organizing + resources + networking + efficacy) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="collective_mobilization",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_consensus_building(self, response: str, cultural_context: CulturalContext,
                                   test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate consensus building and facilitation skills."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count consensus building indicators
        participatory = sum(1 for part in self.consensus_building_patterns["participatory_processes"]
                           if part in response_lower)
        conflict_resolution = sum(1 for conf in self.consensus_building_patterns["conflict_resolution"]
                                 if conf in response_lower)
        facilitation = sum(1 for fac in self.consensus_building_patterns["facilitation_skills"]
                          if fac in response_lower)
        decision_making = sum(1 for dec in self.consensus_building_patterns["decision_making"]
                             if dec in response_lower)
        
        if participatory > 0:
            evidence.append(f"Participatory processes: {participatory} instances")
            cultural_markers.append("participatory_competence")
        
        if conflict_resolution > 0:
            evidence.append(f"Conflict resolution: {conflict_resolution} instances")
            cultural_markers.append("conflict_resolution_competence")
        
        if facilitation > 0:
            evidence.append(f"Facilitation skills: {facilitation} instances")
            cultural_markers.append("facilitation_competence")
        
        if decision_making > 0:
            evidence.append(f"Consensus decision making: {decision_making} instances")
            cultural_markers.append("consensus_decision_competence")
        
        # Cultural style bonus
        cultural_bonus = 0.0
        for style, patterns in self.cultural_leadership_styles.items():
            style_count = sum(1 for pattern in patterns if pattern in response_lower)
            if style_count > 0:
                if style == "indigenous_leadership" and "indigenous" in cultural_context.cultural_groups:
                    cultural_bonus = 0.15
                elif style == "african_ubuntu" and "african" in cultural_context.cultural_groups:
                    cultural_bonus = 0.15
                elif style == "asian_collective" and any(group in ["east_asian", "asian"] for group in cultural_context.cultural_groups):
                    cultural_bonus = 0.15
                cultural_markers.append(f"{style}_alignment")
                break
        
        total_score = (participatory * 0.3 + conflict_resolution * 0.25 + 
                      facilitation * 0.25 + decision_making * 0.2)
        
        score = min(1.0, (total_score * 0.12) + cultural_bonus)
        confidence = min(1.0, (participatory + conflict_resolution + facilitation + decision_making) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="consensus_building",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_cultural_preservation(self, response: str, cultural_context: CulturalContext,
                                      test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate cultural preservation and heritage leadership."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count cultural preservation indicators
        tradition_maintaining = sum(1 for trad in self.cultural_preservation_patterns["tradition_maintaining"]
                                   if trad in response_lower)
        knowledge_preservation = sum(1 for know in self.cultural_preservation_patterns["knowledge_preservation"]
                                    if know in response_lower)
        identity_strengthening = sum(1 for ident in self.cultural_preservation_patterns["identity_strengthening"]
                                    if ident in response_lower)
        intergenerational = sum(1 for inter in self.cultural_preservation_patterns["intergenerational_transfer"]
                               if inter in response_lower)
        
        if tradition_maintaining > 0:
            evidence.append(f"Tradition maintaining: {tradition_maintaining} instances")
            cultural_markers.append("tradition_preservation_competence")
        
        if knowledge_preservation > 0:
            evidence.append(f"Knowledge preservation: {knowledge_preservation} instances")
            cultural_markers.append("knowledge_preservation_competence")
        
        if identity_strengthening > 0:
            evidence.append(f"Identity strengthening: {identity_strengthening} instances")
            cultural_markers.append("identity_strengthening_competence")
        
        if intergenerational > 0:
            evidence.append(f"Intergenerational transfer: {intergenerational} instances")
            cultural_markers.append("intergenerational_competence")
        
        # High bonus for cultural preservation contexts
        preservation_bonus = 0.0
        if "cultural_preservation" in cultural_context.traditions or "heritage" in cultural_context.knowledge_systems:
            preservation_bonus = 0.2
            cultural_markers.append("cultural_preservation_context")
        
        total_score = (tradition_maintaining * 0.25 + knowledge_preservation * 0.25 + 
                      identity_strengthening * 0.25 + intergenerational * 0.25)
        
        score = min(1.0, (total_score * 0.15) + preservation_bonus)
        confidence = min(1.0, (tradition_maintaining + knowledge_preservation + 
                              identity_strengthening + intergenerational) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="cultural_preservation",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_community_empowerment(self, response: str, cultural_context: CulturalContext,
                                       test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate community empowerment and capacity building."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count empowerment indicators
        capacity_building = sum(1 for cap in self.community_empowerment_patterns["capacity_building"]
                               if cap in response_lower)
        self_determination = sum(1 for self_det in self.community_empowerment_patterns["self_determination"]
                                if self_det in response_lower)
        agency_development = sum(1 for agency in self.community_empowerment_patterns["agency_development"]
                                if agency in response_lower)
        collective_ownership = sum(1 for own in self.community_empowerment_patterns["collective_ownership"]
                                  if own in response_lower)
        
        if capacity_building > 0:
            evidence.append(f"Capacity building: {capacity_building} instances")
            cultural_markers.append("capacity_building_competence")
        
        if self_determination > 0:
            evidence.append(f"Self-determination promotion: {self_determination} instances")
            cultural_markers.append("self_determination_competence")
        
        if agency_development > 0:
            evidence.append(f"Agency development: {agency_development} instances")
            cultural_markers.append("agency_development_competence")
        
        if collective_ownership > 0:
            evidence.append(f"Collective ownership: {collective_ownership} instances")
            cultural_markers.append("collective_ownership_competence")
        
        total_score = (capacity_building * 0.3 + self_determination * 0.25 + 
                      agency_development * 0.25 + collective_ownership * 0.2)
        
        score = min(1.0, total_score * 0.15)
        confidence = min(1.0, (capacity_building + self_determination + 
                              agency_development + collective_ownership) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="community_empowerment",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_collaborative_governance(self, response: str, cultural_context: CulturalContext,
                                         test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate collaborative governance and shared leadership."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count collaborative governance indicators
        stakeholder = sum(1 for stake in self.collaborative_governance_patterns["stakeholder_engagement"]
                         if stake in response_lower)
        transparency = sum(1 for trans in self.collaborative_governance_patterns["transparency_accountability"]
                          if trans in response_lower)
        adaptive = sum(1 for adapt in self.collaborative_governance_patterns["adaptive_management"]
                      if adapt in response_lower)
        power_sharing = sum(1 for power in self.collaborative_governance_patterns["power_sharing"]
                           if power in response_lower)
        
        if stakeholder > 0:
            evidence.append(f"Stakeholder engagement: {stakeholder} instances")
            cultural_markers.append("stakeholder_engagement_competence")
        
        if transparency > 0:
            evidence.append(f"Transparency & accountability: {transparency} instances")
            cultural_markers.append("transparency_competence")
        
        if adaptive > 0:
            evidence.append(f"Adaptive management: {adaptive} instances")
            cultural_markers.append("adaptive_management_competence")
        
        if power_sharing > 0:
            evidence.append(f"Power sharing: {power_sharing} instances")
            cultural_markers.append("power_sharing_competence")
        
        total_score = (stakeholder * 0.3 + transparency * 0.25 + adaptive * 0.25 + power_sharing * 0.2)
        
        score = min(1.0, total_score * 0.15)
        confidence = min(1.0, (stakeholder + transparency + adaptive + power_sharing) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="collaborative_governance",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_sustainable_development(self, response: str, cultural_context: CulturalContext,
                                        test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate sustainable development leadership."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count sustainable development indicators
        environmental = sum(1 for env in self.sustainable_development_patterns["environmental_stewardship"]
                           if env in response_lower)
        economic = sum(1 for econ in self.sustainable_development_patterns["economic_sustainability"]
                      if econ in response_lower)
        social = sum(1 for soc in self.sustainable_development_patterns["social_sustainability"]
                    if soc in response_lower)
        holistic = sum(1 for hol in self.sustainable_development_patterns["holistic_development"]
                      if hol in response_lower)
        
        if environmental > 0:
            evidence.append(f"Environmental stewardship: {environmental} instances")
            cultural_markers.append("environmental_stewardship_competence")
        
        if economic > 0:
            evidence.append(f"Economic sustainability: {economic} instances")
            cultural_markers.append("economic_sustainability_competence")
        
        if social > 0:
            evidence.append(f"Social sustainability: {social} instances")
            cultural_markers.append("social_sustainability_competence")
        
        if holistic > 0:
            evidence.append(f"Holistic development: {holistic} instances")
            cultural_markers.append("holistic_development_competence")
        
        total_score = (environmental * 0.25 + economic * 0.25 + social * 0.25 + holistic * 0.25)
        
        score = min(1.0, total_score * 0.15)
        confidence = min(1.0, (environmental + economic + social + holistic) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="sustainable_development",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_leadership_competencies(self, response: str, cultural_context: CulturalContext,
                                        test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate core leadership competencies."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count leadership competencies
        emotional_intelligence = sum(1 for ei in self.leadership_competencies["emotional_intelligence"]
                                    if ei in response_lower)
        cultural_competence = sum(1 for cc in self.leadership_competencies["cultural_competence"]
                                 if cc in response_lower)
        systems_thinking = sum(1 for st in self.leadership_competencies["systems_thinking"]
                              if st in response_lower)
        resilience = sum(1 for res in self.leadership_competencies["resilience_building"]
                        if res in response_lower)
        
        if emotional_intelligence > 0:
            evidence.append(f"Emotional intelligence: {emotional_intelligence} instances")
            cultural_markers.append("emotional_intelligence_competence")
        
        if cultural_competence > 0:
            evidence.append(f"Cultural competence: {cultural_competence} instances")
            cultural_markers.append("cultural_competence")
        
        if systems_thinking > 0:
            evidence.append(f"Systems thinking: {systems_thinking} instances")
            cultural_markers.append("systems_thinking_competence")
        
        if resilience > 0:
            evidence.append(f"Resilience building: {resilience} instances")
            cultural_markers.append("resilience_building_competence")
        
        # Bonus for multicultural leadership contexts
        multicultural_bonus = 0.0
        if len(cultural_context.cultural_groups) > 1 and cultural_competence > 0:
            multicultural_bonus = 0.1
            cultural_markers.append("multicultural_leadership_competence")
        
        total_score = (emotional_intelligence * 0.25 + cultural_competence * 0.25 + 
                      systems_thinking * 0.25 + resilience * 0.25)
        
        score = min(1.0, (total_score * 0.15) + multicultural_bonus)
        confidence = min(1.0, (emotional_intelligence + cultural_competence + 
                              systems_thinking + resilience) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="leadership_competencies",
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
            
            # Higher relevance for markers aligned with community and cultural contexts
            if marker_type in ["vision", "transformational", "inspirational"] and "visionary_leadership" in cultural_context.performance_aspects:
                relevance_score = 0.95
            elif marker_type in ["community", "collective", "organizing"] and "community_organizing" in cultural_context.traditions:
                relevance_score = 0.95
            elif marker_type in ["consensus", "participatory", "collaborative"] and any(
                group in ["indigenous", "african", "asian"] 
                for group in cultural_context.cultural_groups
            ):
                relevance_score = 0.9
            elif marker_type in ["cultural", "tradition", "heritage"] and "cultural_preservation" in cultural_context.traditions:
                relevance_score = 0.95
            elif marker_type in ["empowerment", "capacity", "self"] and "community_empowerment" in cultural_context.knowledge_systems:
                relevance_score = 0.9
            elif marker_type in ["governance", "stakeholder", "transparency"]:
                relevance_score = 0.85
            elif marker_type in ["sustainable", "environmental", "economic", "social"]:
                relevance_score = 0.85
            elif marker_type in ["emotional", "systems", "resilience"] and len(cultural_context.cultural_groups) > 0:
                relevance_score = 0.8
            elif marker_type in ["multicultural", "intercultural"] and len(cultural_context.cultural_groups) > 1:
                relevance_score = 0.95
            elif marker_type in ["indigenous", "african", "asian", "latin"] and any(
                group in marker_type for group in cultural_context.cultural_groups
            ):
                relevance_score = 0.95
            
            total_relevance += relevance_score
            marker_count += 1
        
        return min(1.0, total_relevance / marker_count) if marker_count > 0 else 0.5