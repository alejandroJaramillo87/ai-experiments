"""
Conflict Resolution and Consensus Evaluator

Specialized evaluator for assessing conflict resolution skills, mediation competence,
consensus-building abilities, and de-escalation techniques across cultural contexts.

Author: Claude Code
Version: 1.0.0
"""

from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import re
from collections import defaultdict, Counter

from evaluator.domain_evaluator_base import (
    MultiDimensionalEvaluator, 
    EvaluationDimension, 
    DomainEvaluationResult,
    CulturalContext
)


class ConflictResolutionType(Enum):
    """Types of conflict resolution evaluation."""
    MEDIATION = "mediation"
    NEGOTIATION = "negotiation"
    DE_ESCALATION = "de_escalation"
    CONSENSUS_BUILDING = "consensus_building"
    RESTORATIVE_JUSTICE = "restorative_justice"
    CULTURAL_CONFLICT = "cultural_conflict"
    INTERPERSONAL_CONFLICT = "interpersonal_conflict"


class ConflictResolutionEvaluator(MultiDimensionalEvaluator):
    """Evaluates conflict resolution and consensus-building competence."""
    
    VERSION = "1.0.0"
    
    def _initialize_evaluator(self):
        """Initialize conflict resolution evaluation components."""
        
        # De-escalation techniques
        self.deescalation_patterns = {
            "calming_language": [
                "let's take a step back", "I understand you're frustrated",
                "let's find common ground", "I hear what you're saying",
                "let's approach this calmly", "I appreciate your perspective",
                "let's work together", "I want to understand better"
            ],
            "reframing": [
                "another way to look at this", "what if we considered",
                "perhaps we could frame it as", "from a different angle",
                "let's redefine the problem", "the real issue might be"
            ],
            "validation": [
                "your feelings are valid", "I can see why you'd feel",
                "that must be difficult", "I understand your concern",
                "your point is well-taken", "I acknowledge that"
            ],
            "cooling_techniques": [
                "let's pause for a moment", "take a deep breath",
                "let's slow down", "give it some time", "step back",
                "cool off period", "break to reflect"
            ]
        }
        
        # Mediation strategies
        self.mediation_patterns = {
            "neutral_positioning": [
                "I'm here to help both parties", "as a neutral facilitator",
                "I don't take sides", "my role is to help you both",
                "I'm not here to judge", "impartially", "objectively"
            ],
            "active_listening": [
                "what I'm hearing is", "let me reflect back",
                "so you're saying", "if I understand correctly",
                "let me paraphrase", "correct me if I'm wrong"
            ],
            "process_management": [
                "let's establish ground rules", "one person at a time",
                "let's keep this respectful", "focus on the issue",
                "stay on topic", "productive discussion"
            ],
            "solution_generation": [
                "what would work for both of you", "creative solutions",
                "win-win outcome", "mutually beneficial", "compromise",
                "meet in the middle", "alternative approaches"
            ]
        }
        
        # Consensus building techniques
        self.consensus_patterns = {
            "inclusive_participation": [
                "let's hear from everyone", "all voices matter",
                "what does everyone think", "include all perspectives",
                "democratic process", "collective decision",
                "group input", "shared ownership"
            ],
            "synthesis_skills": [
                "combining these ideas", "building on that",
                "merging different approaches", "finding the synthesis",
                "common themes", "integrated solution",
                "bridging different views", "unified approach"
            ],
            "facilitation": [
                "let's move to the next point", "summarizing our discussion",
                "where do we have agreement", "outstanding issues",
                "action items", "next steps", "follow-up plan"
            ],
            "decision_processes": [
                "consensus check", "do we all agree", "any objections",
                "final decision", "voting", "unanimous agreement",
                "majority support", "modified consensus"
            ]
        }
        
        # Negotiation strategies
        self.negotiation_patterns = {
            "interest_exploration": [
                "what's most important to you", "underlying interests",
                "core needs", "what do you really want", "priorities",
                "fundamental concerns", "bottom line", "deal breakers"
            ],
            "option_generation": [
                "brainstorming solutions", "multiple options",
                "creative alternatives", "what if we tried",
                "different possibilities", "range of choices",
                "innovative approaches", "out-of-the-box thinking"
            ],
            "value_creation": [
                "expand the pie", "mutual gains", "win-win scenario",
                "added value", "joint benefits", "shared prosperity",
                "positive-sum outcome", "collaborative advantage"
            ],
            "strategic_communication": [
                "strategic concession", "conditional offer",
                "package deal", "trade-offs", "contingent agreement",
                "phased implementation", "trial period"
            ]
        }
        
        # Cultural conflict patterns
        self.cultural_conflict_patterns = {
            "cultural_sensitivity": [
                "cultural differences", "different traditions",
                "cultural perspective", "respect for customs",
                "cultural background", "traditional values",
                "cross-cultural understanding", "cultural competence"
            ],
            "bridge_building": [
                "cultural bridge", "finding common ground",
                "shared humanity", "universal values",
                "transcend differences", "cultural exchange",
                "mutual respect", "cultural learning"
            ],
            "adaptation": [
                "adapt to cultural context", "culturally appropriate",
                "modified approach", "cultural accommodation",
                "flexible strategies", "context-sensitive",
                "culturally informed", "local customs"
            ]
        }
        
        # Restorative justice principles
        self.restorative_patterns = {
            "accountability": [
                "taking responsibility", "acknowledging harm",
                "owning the mistake", "accepting consequences",
                "making amends", "repair the damage",
                "restore relationships", "healing process"
            ],
            "empathy_building": [
                "impact on others", "understanding the pain",
                "seeing from their perspective", "emotional impact",
                "human cost", "ripple effects", "affected parties"
            ],
            "community_involvement": [
                "community healing", "collective response",
                "community support", "social fabric",
                "rebuilding trust", "community values",
                "shared responsibility", "collective healing"
            ]
        }
        
        # Emotional intelligence markers
        self.emotional_intelligence = {
            "self_awareness": [
                "I recognize my bias", "I need to check myself",
                "my emotional response", "I'm feeling frustrated",
                "I need to be mindful", "self-reflection"
            ],
            "emotional_regulation": [
                "staying calm", "managing my emotions",
                "keeping my cool", "emotional control",
                "composed response", "measured reaction"
            ],
            "empathy": [
                "I understand how you feel", "putting myself in your shoes",
                "seeing your perspective", "emotional intelligence",
                "empathetic response", "feeling with you"
            ],
            "social_skills": [
                "building rapport", "establishing trust",
                "connecting with people", "relationship building",
                "communication skills", "social awareness"
            ]
        }
    
    def get_domain_name(self) -> str:
        """Return the domain name this evaluator handles."""
        return "conflict_resolution"
    
    def get_supported_evaluation_types(self) -> List[str]:
        """Return list of evaluation types this evaluator supports."""
        return [evaluation_type.value for evaluation_type in ConflictResolutionType]
    
    def get_evaluation_dimensions(self) -> List[str]:
        """Return list of dimensions this evaluator assesses."""
        return [
            "de_escalation_skills",
            "mediation_competence",
            "consensus_building",
            "negotiation_strategy",
            "cultural_conflict_navigation",
            "restorative_justice_principles",
            "emotional_intelligence"
        ]
    
    def evaluate_dimension(self, 
                          dimension: str,
                          response_text: str, 
                          test_metadata: Dict[str, Any], 
                          cultural_context: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate a specific dimension."""
        cultural_ctx = self._create_cultural_context(cultural_context)
        
        if dimension == "de_escalation_skills":
            return self._evaluate_de_escalation_skills(response_text, cultural_ctx, test_metadata)
        elif dimension == "mediation_competence":
            return self._evaluate_mediation_competence(response_text, cultural_ctx, test_metadata)
        elif dimension == "consensus_building":
            return self._evaluate_consensus_building(response_text, cultural_ctx, test_metadata)
        elif dimension == "negotiation_strategy":
            return self._evaluate_negotiation_strategy(response_text, cultural_ctx, test_metadata)
        elif dimension == "cultural_conflict_navigation":
            return self._evaluate_cultural_conflict_navigation(response_text, cultural_ctx, test_metadata)
        elif dimension == "restorative_justice_principles":
            return self._evaluate_restorative_justice_principles(response_text, cultural_ctx, test_metadata)
        elif dimension == "emotional_intelligence":
            return self._evaluate_emotional_intelligence(response_text, cultural_ctx, test_metadata)
        else:
            return EvaluationDimension(
                name=dimension,
                score=0.0,
                confidence=0.0,
                cultural_relevance=0.0,
                evidence=[f"Unknown dimension: {dimension}"],
                cultural_markers=[]
            )
    
    def _evaluate_de_escalation_skills(self, response: str, cultural_context: CulturalContext,
                                     test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate de-escalation techniques and calming strategies."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count de-escalation techniques
        calming_count = sum(1 for phrase in self.deescalation_patterns["calming_language"]
                           if phrase in response_lower)
        reframing_count = sum(1 for phrase in self.deescalation_patterns["reframing"]
                             if phrase in response_lower)
        validation_count = sum(1 for phrase in self.deescalation_patterns["validation"]
                              if phrase in response_lower)
        cooling_count = sum(1 for phrase in self.deescalation_patterns["cooling_techniques"]
                           if phrase in response_lower)
        
        if calming_count > 0:
            evidence.append(f"Calming language: {calming_count} instances")
            cultural_markers.append("calming_communication")
        
        if reframing_count > 0:
            evidence.append(f"Reframing techniques: {reframing_count} instances")
            cultural_markers.append("cognitive_reframing")
        
        if validation_count > 0:
            evidence.append(f"Validation statements: {validation_count} instances")
            cultural_markers.append("emotional_validation")
        
        if cooling_count > 0:
            evidence.append(f"Cooling techniques: {cooling_count} instances")
            cultural_markers.append("tension_reduction")
        
        # Calculate weighted score
        total_score = (calming_count * 0.3 + reframing_count * 0.25 + 
                      validation_count * 0.25 + cooling_count * 0.2)
        
        score = min(1.0, total_score * 0.15)
        confidence = min(1.0, (calming_count + reframing_count + validation_count + cooling_count) * 0.12)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="de_escalation_skills",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_mediation_competence(self, response: str, cultural_context: CulturalContext,
                                     test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate mediation skills and neutral facilitation."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count mediation techniques
        neutral_count = sum(1 for phrase in self.mediation_patterns["neutral_positioning"]
                           if phrase in response_lower)
        listening_count = sum(1 for phrase in self.mediation_patterns["active_listening"]
                             if phrase in response_lower)
        process_count = sum(1 for phrase in self.mediation_patterns["process_management"]
                           if phrase in response_lower)
        solution_count = sum(1 for phrase in self.mediation_patterns["solution_generation"]
                            if phrase in response_lower)
        
        if neutral_count > 0:
            evidence.append(f"Neutral positioning: {neutral_count} instances")
            cultural_markers.append("neutral_facilitation")
        
        if listening_count > 0:
            evidence.append(f"Active listening: {listening_count} instances")
            cultural_markers.append("reflective_listening")
        
        if process_count > 0:
            evidence.append(f"Process management: {process_count} instances")
            cultural_markers.append("structured_facilitation")
        
        if solution_count > 0:
            evidence.append(f"Solution generation: {solution_count} instances")
            cultural_markers.append("collaborative_problem_solving")
        
        total_score = (neutral_count * 0.25 + listening_count * 0.25 + 
                      process_count * 0.25 + solution_count * 0.25)
        
        score = min(1.0, total_score * 0.15)
        confidence = min(1.0, (neutral_count + listening_count + process_count + solution_count) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="mediation_competence",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_consensus_building(self, response: str, cultural_context: CulturalContext,
                                   test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate consensus building and group facilitation skills."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count consensus building techniques
        inclusion_count = sum(1 for phrase in self.consensus_patterns["inclusive_participation"]
                             if phrase in response_lower)
        synthesis_count = sum(1 for phrase in self.consensus_patterns["synthesis_skills"]
                             if phrase in response_lower)
        facilitation_count = sum(1 for phrase in self.consensus_patterns["facilitation"]
                                if phrase in response_lower)
        decision_count = sum(1 for phrase in self.consensus_patterns["decision_processes"]
                            if phrase in response_lower)
        
        if inclusion_count > 0:
            evidence.append(f"Inclusive participation: {inclusion_count} instances")
            cultural_markers.append("inclusive_facilitation")
        
        if synthesis_count > 0:
            evidence.append(f"Synthesis skills: {synthesis_count} instances")
            cultural_markers.append("idea_integration")
        
        if facilitation_count > 0:
            evidence.append(f"Group facilitation: {facilitation_count} instances")
            cultural_markers.append("group_process_management")
        
        if decision_count > 0:
            evidence.append(f"Decision processes: {decision_count} instances")
            cultural_markers.append("democratic_decision_making")
        
        # Cultural context bonus for collectivist cultures
        cultural_bonus = 0.0
        if any(group in ["east_asian", "african_traditional", "indigenous"] 
               for group in cultural_context.cultural_groups):
            cultural_bonus = 0.1
            cultural_markers.append("collectivist_consensus")
        
        total_score = (inclusion_count * 0.3 + synthesis_count * 0.25 + 
                      facilitation_count * 0.25 + decision_count * 0.2)
        
        score = min(1.0, (total_score * 0.12) + cultural_bonus)
        confidence = min(1.0, (inclusion_count + synthesis_count + facilitation_count + decision_count) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="consensus_building",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_negotiation_strategy(self, response: str, cultural_context: CulturalContext,
                                     test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate negotiation skills and strategic thinking."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count negotiation techniques
        interest_count = sum(1 for phrase in self.negotiation_patterns["interest_exploration"]
                            if phrase in response_lower)
        option_count = sum(1 for phrase in self.negotiation_patterns["option_generation"]
                          if phrase in response_lower)
        value_count = sum(1 for phrase in self.negotiation_patterns["value_creation"]
                         if phrase in response_lower)
        strategic_count = sum(1 for phrase in self.negotiation_patterns["strategic_communication"]
                             if phrase in response_lower)
        
        if interest_count > 0:
            evidence.append(f"Interest exploration: {interest_count} instances")
            cultural_markers.append("principled_negotiation")
        
        if option_count > 0:
            evidence.append(f"Option generation: {option_count} instances")
            cultural_markers.append("creative_problem_solving")
        
        if value_count > 0:
            evidence.append(f"Value creation: {value_count} instances")
            cultural_markers.append("mutual_gain_orientation")
        
        if strategic_count > 0:
            evidence.append(f"Strategic communication: {strategic_count} instances")
            cultural_markers.append("tactical_competence")
        
        total_score = (interest_count * 0.3 + option_count * 0.25 + 
                      value_count * 0.25 + strategic_count * 0.2)
        
        score = min(1.0, total_score * 0.15)
        confidence = min(1.0, (interest_count + option_count + value_count + strategic_count) * 0.12)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="negotiation_strategy",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_cultural_conflict_navigation(self, response: str, cultural_context: CulturalContext,
                                             test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate skills in navigating cultural conflicts."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count cultural conflict techniques
        sensitivity_count = sum(1 for phrase in self.cultural_conflict_patterns["cultural_sensitivity"]
                               if phrase in response_lower)
        bridge_count = sum(1 for phrase in self.cultural_conflict_patterns["bridge_building"]
                          if phrase in response_lower)
        adaptation_count = sum(1 for phrase in self.cultural_conflict_patterns["adaptation"]
                              if phrase in response_lower)
        
        if sensitivity_count > 0:
            evidence.append(f"Cultural sensitivity: {sensitivity_count} instances")
            cultural_markers.append("cultural_awareness")
        
        if bridge_count > 0:
            evidence.append(f"Cultural bridge building: {bridge_count} instances")
            cultural_markers.append("intercultural_mediation")
        
        if adaptation_count > 0:
            evidence.append(f"Cultural adaptation: {adaptation_count} instances")
            cultural_markers.append("contextual_flexibility")
        
        # High relevance for multicultural contexts
        multicultural_bonus = 0.0
        if len(cultural_context.cultural_groups) > 1:
            multicultural_bonus = 0.2
            cultural_markers.append("multicultural_competence")
        
        total_score = (sensitivity_count * 0.4 + bridge_count * 0.35 + adaptation_count * 0.25)
        
        score = min(1.0, (total_score * 0.15) + multicultural_bonus)
        confidence = min(1.0, (sensitivity_count + bridge_count + adaptation_count) * 0.15)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="cultural_conflict_navigation",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_restorative_justice_principles(self, response: str, cultural_context: CulturalContext,
                                               test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate understanding of restorative justice principles."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count restorative justice elements
        accountability_count = sum(1 for phrase in self.restorative_patterns["accountability"]
                                  if phrase in response_lower)
        empathy_count = sum(1 for phrase in self.restorative_patterns["empathy_building"]
                           if phrase in response_lower)
        community_count = sum(1 for phrase in self.restorative_patterns["community_involvement"]
                             if phrase in response_lower)
        
        if accountability_count > 0:
            evidence.append(f"Accountability focus: {accountability_count} instances")
            cultural_markers.append("responsibility_orientation")
        
        if empathy_count > 0:
            evidence.append(f"Empathy building: {empathy_count} instances")
            cultural_markers.append("victim_impact_awareness")
        
        if community_count > 0:
            evidence.append(f"Community involvement: {community_count} instances")
            cultural_markers.append("collective_healing")
        
        # Cultural context bonus for communal cultures
        communal_bonus = 0.0
        if any(tradition in ["restorative_practices", "traditional_justice"] 
               for tradition in cultural_context.traditions):
            communal_bonus = 0.15
            cultural_markers.append("traditional_justice_alignment")
        
        total_score = (accountability_count * 0.4 + empathy_count * 0.35 + community_count * 0.25)
        
        score = min(1.0, (total_score * 0.2) + communal_bonus)
        confidence = min(1.0, (accountability_count + empathy_count + community_count) * 0.15)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="restorative_justice_principles",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_emotional_intelligence(self, response: str, cultural_context: CulturalContext,
                                       test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate emotional intelligence in conflict resolution."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Count emotional intelligence indicators
        self_awareness_count = sum(1 for phrase in self.emotional_intelligence["self_awareness"]
                                  if phrase in response_lower)
        regulation_count = sum(1 for phrase in self.emotional_intelligence["emotional_regulation"]
                              if phrase in response_lower)
        empathy_count = sum(1 for phrase in self.emotional_intelligence["empathy"]
                           if phrase in response_lower)
        social_count = sum(1 for phrase in self.emotional_intelligence["social_skills"]
                          if phrase in response_lower)
        
        if self_awareness_count > 0:
            evidence.append(f"Self-awareness: {self_awareness_count} instances")
            cultural_markers.append("emotional_self_awareness")
        
        if regulation_count > 0:
            evidence.append(f"Emotional regulation: {regulation_count} instances")
            cultural_markers.append("emotional_control")
        
        if empathy_count > 0:
            evidence.append(f"Empathy: {empathy_count} instances")
            cultural_markers.append("empathetic_responding")
        
        if social_count > 0:
            evidence.append(f"Social skills: {social_count} instances")
            cultural_markers.append("interpersonal_competence")
        
        total_score = (self_awareness_count * 0.25 + regulation_count * 0.25 + 
                      empathy_count * 0.25 + social_count * 0.25)
        
        score = min(1.0, total_score * 0.2)
        confidence = min(1.0, (self_awareness_count + regulation_count + empathy_count + social_count) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="emotional_intelligence",
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
            
            # Higher relevance for markers aligned with cultural context
            if marker_type in ["collective", "consensus", "community"] and any(
                group in ["east_asian", "african_traditional", "indigenous"] 
                for group in cultural_context.cultural_groups
            ):
                relevance_score = 0.9
            elif marker_type in ["restorative", "traditional"] and "traditional_justice" in cultural_context.traditions:
                relevance_score = 0.9
            elif marker_type in ["multicultural", "intercultural"] and len(cultural_context.cultural_groups) > 1:
                relevance_score = 0.9
            elif marker_type in ["mediation", "facilitation", "negotiation"]:
                relevance_score = 0.8
            elif marker_type in ["emotional", "empathy", "social"]:
                relevance_score = 0.7
            
            total_relevance += relevance_score
            marker_count += 1
        
        return min(1.0, total_relevance / marker_count) if marker_count > 0 else 0.5