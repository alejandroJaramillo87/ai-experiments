"""
Social Domain Evaluator

Evaluates social competence across 7 comprehensive categories:
1. Social Appropriateness - Context-sensitive social behavior
2. Hierarchy Navigation - Understanding and respecting social hierarchies
3. Relationship Maintenance - Building and sustaining relationships
4. Community Dynamics - Understanding group interactions and collective behavior
5. Cultural Etiquette - Following cultural norms and protocols
6. Conflict Resolution - Managing disputes and tensions constructively
7. Intercultural Competence - Navigating cross-cultural social situations

Author: Claude Code
Version: 1.0.0
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

from ..core.domain_evaluator_base import (
    MultiDimensionalEvaluator, 
    DomainEvaluationResult,
    EvaluationDimension,
    CulturalContext
)


class SocialEvaluationType(Enum):
    """Types of social evaluation."""
    SOCIAL_CONTEXT = "social_context"
    HIERARCHY_NAVIGATION = "hierarchy_navigation"
    RELATIONSHIP_BUILDING = "relationship_building"
    COMMUNITY_ENGAGEMENT = "community_engagement"
    CULTURAL_ETIQUETTE = "cultural_etiquette"
    CONFLICT_RESOLUTION = "conflict_resolution"
    INTERCULTURAL_COMPETENCE = "intercultural_competence"


@dataclass
class SocialIndicator:
    """Represents a social competence indicator."""
    indicator_name: str
    score: float  # 0.0 to 1.0
    evidence: List[str]
    cultural_context: str
    appropriateness_level: str  # "excellent", "good", "acceptable", "concerning", "problematic"


class SocialEvaluator(MultiDimensionalEvaluator):
    """Evaluates social competence across multiple dimensions."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self._initialize_evaluator()
    
    def _initialize_evaluator(self):
        """Initialize domain-specific components and resources."""
        self.social_patterns = self._initialize_social_patterns()
        self.hierarchy_markers = self._initialize_hierarchy_markers()
        self.relationship_indicators = self._initialize_relationship_indicators()
        self.community_patterns = self._initialize_community_patterns()
        self.etiquette_systems = self._initialize_etiquette_systems()
        self.conflict_resolution_patterns = self._initialize_conflict_patterns()
        self.intercultural_competence_markers = self._initialize_intercultural_markers()
    
    def get_domain_name(self) -> str:
        """Return the domain name this evaluator handles."""
        return "social"
    
    def get_evaluation_dimensions(self) -> List[str]:
        """Return list of dimensions this evaluator assesses."""
        return [
            "social_appropriateness",
            "hierarchy_navigation", 
            "relationship_maintenance",
            "community_dynamics",
            "cultural_etiquette",
            "conflict_resolution",
            "intercultural_competence"
        ]
    
    def get_supported_evaluation_types(self) -> List[str]:
        """Return list of evaluation types this evaluator supports."""
        return [eval_type.value for eval_type in SocialEvaluationType]
    
    def evaluate_dimension(self, dimension: str, response_text: str, 
                         test_metadata: Dict[str, Any], 
                         cultural_context: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate a specific dimension."""
        cultural_ctx = self._create_cultural_context(cultural_context)
        
        if dimension == "social_appropriateness":
            return self._evaluate_social_appropriateness(response_text, cultural_ctx, test_metadata)
        elif dimension == "hierarchy_navigation":
            return self._evaluate_hierarchy_navigation(response_text, cultural_ctx, test_metadata)
        elif dimension == "relationship_maintenance":
            return self._evaluate_relationship_maintenance(response_text, cultural_ctx, test_metadata)
        elif dimension == "community_dynamics":
            return self._evaluate_community_dynamics(response_text, cultural_ctx, test_metadata)
        elif dimension == "cultural_etiquette":
            return self._evaluate_cultural_etiquette(response_text, cultural_ctx, test_metadata)
        elif dimension == "conflict_resolution":
            return self._evaluate_conflict_resolution(response_text, cultural_ctx, test_metadata)
        elif dimension == "intercultural_competence":
            return self._evaluate_intercultural_competence(response_text, cultural_ctx, test_metadata)
        else:
            # Default dimension with minimal score
            return EvaluationDimension(
                name=dimension,
                score=0.0,
                confidence=0.0,
                cultural_relevance=0.0,
                evidence=[f"Unknown dimension: {dimension}"],
                cultural_markers=[]
            )
    
    def _initialize_social_patterns(self) -> Dict[str, List[str]]:
        """Initialize social appropriateness patterns."""
        return {
            "polite_language": [
                "please", "thank you", "excuse me", "pardon", "may I", "would you mind",
                "I appreciate", "if you don't mind", "with respect", "kindly"
            ],
            "respectful_address": [
                "sir", "madam", "mr", "ms", "dr", "professor", "elder", "respected",
                "honorable", "your excellency", "your honor"
            ],
            "social_awareness": [
                "I understand this might", "I recognize that", "considering others",
                "sensitive to", "aware of the impact", "mindful of", "respectful of"
            ],
            "inappropriate_markers": [
                "shut up", "whatever", "don't care", "who cares", "stupid", "idiot",
                "waste of time", "pointless", "obviously wrong"
            ],
            "boundary_respect": [
                "personal space", "privacy", "consent", "permission", "boundaries",
                "comfortable with", "respect your decision", "understand if you prefer"
            ]
        }
    
    def _initialize_hierarchy_markers(self) -> Dict[str, List[str]]:
        """Initialize hierarchy navigation patterns."""
        return {
            "authority_recognition": [
                "following protocol", "chain of command", "proper channels",
                "appropriate authority", "seek permission", "defer to",
                "senior leadership", "organizational structure"
            ],
            "respectful_disagreement": [
                "I respectfully disagree", "with due respect", "I see your point, however",
                "may I suggest", "another perspective", "humbly submit", "if I may"
            ],
            "cultural_hierarchy": [
                "elder wisdom", "traditional authority", "cultural leader",
                "community elder", "ancestral guidance", "traditional protocol",
                "generational respect", "cultural precedence"
            ],
            "power_dynamics": [
                "power imbalance", "positional authority", "influence", "decision maker",
                "stakeholder", "vested interest", "institutional power", "social capital"
            ],
            "hierarchy_violations": [
                "going over heads", "bypassing authority", "undermining leadership",
                "disregarding protocol", "ignoring chain of command", "insubordinate"
            ]
        }
    
    def _initialize_relationship_indicators(self) -> Dict[str, List[str]]:
        """Initialize relationship maintenance patterns."""
        return {
            "trust_building": [
                "reliability", "consistency", "trustworthy", "dependable",
                "keep promises", "follow through", "honor commitments", "integrity"
            ],
            "empathy_expression": [
                "I understand how you feel", "that must be difficult", "I can imagine",
                "putting myself in your shoes", "your perspective", "emotional support",
                "validate your feelings", "acknowledge your experience"
            ],
            "relationship_investment": [
                "spend time together", "get to know", "shared experiences",
                "mutual interest", "common ground", "building connection",
                "strengthen our relationship", "invest in friendship"
            ],
            "conflict_avoidance": [
                "let's agree to disagree", "find common ground", "bridge differences",
                "compromise", "meet halfway", "mutual understanding", "peaceful resolution"
            ],
            "relationship_damage": [
                "broken trust", "betrayal", "disappointment", "let down",
                "unreliable", "inconsistent", "hurt feelings", "damaged relationship"
            ]
        }
    
    def _initialize_community_patterns(self) -> Dict[str, List[str]]:
        """Initialize community dynamics patterns."""
        return {
            "collective_thinking": [
                "we as a community", "our shared", "collective responsibility",
                "community welfare", "common good", "group harmony", "unity",
                "together we", "community spirit", "solidarity"
            ],
            "participation_encouragement": [
                "everyone's voice matters", "inclusive decision-making", "participate",
                "contribute", "involvement", "engagement", "collaboration",
                "shared leadership", "democratic process"
            ],
            "community_support": [
                "mutual aid", "helping neighbors", "community assistance",
                "support network", "look out for each other", "community care",
                "collective support", "standing together"
            ],
            "social_cohesion": [
                "community bonds", "social fabric", "connectedness",
                "belonging", "community identity", "shared values",
                "cultural continuity", "social integration"
            ],
            "individualistic_focus": [
                "every person for themselves", "individual success", "personal gain",
                "competitive advantage", "self-interest", "independent action"
            ]
        }
    
    def _initialize_etiquette_systems(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize cultural etiquette systems."""
        return {
            "western_formal": {
                "greetings": ["good morning", "good afternoon", "pleased to meet you"],
                "courtesy": ["please", "thank you", "you're welcome", "excuse me"],
                "table_manners": ["proper utensils", "chew quietly", "napkin on lap"],
                "conversation": ["active listening", "turn-taking", "eye contact"]
            },
            "east_asian": {
                "respect_hierarchy": ["bow", "honorifics", "age deference", "seniority"],
                "face_saving": ["indirect disagreement", "avoid embarrassment", "dignity"],
                "group_harmony": ["consensus", "collective decision", "group before self"],
                "gift_giving": ["both hands", "reciprocity", "appropriate wrapping"]
            },
            "african_traditional": {
                "elder_respect": ["greet elders first", "stand when elder enters", "defer"],
                "community_focus": ["collective welfare", "ubuntu", "interconnectedness"],
                "oral_tradition": ["story telling", "proverb sharing", "wisdom passing"],
                "ceremonial": ["ritual respect", "traditional protocol", "sacred boundaries"]
            },
            "middle_eastern": {
                "hospitality": ["guest honor", "generous hosting", "refuse politely"],
                "family_respect": ["family honor", "generational respect", "lineage pride"],
                "religious_consideration": ["prayer times", "dietary laws", "modesty"],
                "business_relations": ["relationship first", "trust building", "patience"]
            },
            "indigenous": {
                "land_connection": ["territorial acknowledgment", "environmental respect"],
                "spiritual_awareness": ["sacred spaces", "ritual boundaries", "ancestors"],
                "circle_communication": ["everyone speaks", "listening circle", "consensus"],
                "traditional_knowledge": ["elder teachings", "cultural protocols", "ceremonies"]
            }
        }
    
    def _initialize_conflict_patterns(self) -> Dict[str, List[str]]:
        """Initialize conflict resolution patterns."""
        return {
            "de_escalation": [
                "let's take a step back", "calm down", "lower tensions",
                "peaceful discussion", "find common ground", "reduce hostility",
                "cool off period", "neutral space"
            ],
            "active_listening": [
                "I hear you saying", "let me understand", "your perspective is",
                "help me understand", "listening carefully", "acknowledge your point",
                "validate your concerns", "reflect back"
            ],
            "mediation_skills": [
                "neutral third party", "facilitate discussion", "mediate",
                "impartial perspective", "help both sides", "bridge differences",
                "fair process", "structured dialogue"
            ],
            "solution_focus": [
                "find solutions", "work together", "collaborative approach",
                "win-win situation", "mutual benefit", "creative solutions",
                "problem-solving", "constructive outcome"
            ],
            "escalation_markers": [
                "you always", "you never", "blame", "fault", "wrong",
                "stupid", "ridiculous", "impossible", "threat", "ultimatum"
            ]
        }
    
    def _initialize_intercultural_markers(self) -> Dict[str, List[str]]:
        """Initialize intercultural competence markers."""
        return {
            "cultural_curiosity": [
                "tell me about your culture", "I'd like to learn", "cultural differences",
                "interesting customs", "help me understand", "cultural background",
                "traditional ways", "cultural practices"
            ],
            "cultural_humility": [
                "I don't know much about", "please correct me", "still learning",
                "appreciate your patience", "open to feedback", "willing to learn",
                "recognize my limitations", "cultural sensitivity"
            ],
            "adaptation_skills": [
                "adjust my approach", "when in Rome", "respect local customs",
                "follow your lead", "adapt to", "cultural flexibility",
                "modify behavior", "accommodate differences"
            ],
            "stereotype_awareness": [
                "avoid generalizations", "individual differences", "not all",
                "diverse within culture", "unique person", "avoid assumptions",
                "cultural diversity", "personal experience"
            ],
            "cultural_bridge_building": [
                "find common ground", "shared human experience", "universal values",
                "bridge cultures", "cultural exchange", "mutual understanding",
                "cross-cultural friendship", "cultural ambassador"
            ]
        }
    
    def evaluate(self, response: str, test_metadata: Dict[str, Any], 
                cultural_context: Dict[str, Any]) -> DomainEvaluationResult:
        """
        Evaluate social competence across all 7 dimensions.
        
        Args:
            response: The text response to evaluate
            test_metadata: Test metadata and context
            cultural_context: Cultural context for evaluation
            
        Returns:
            DomainEvaluationResult with social competence analysis
        """
        # Convert cultural context to CulturalContext object
        cultural_ctx = self._create_cultural_context(cultural_context)
        
        # Evaluate each dimension
        dimensions = []
        processing_notes = []
        
        # 1. Social Appropriateness
        social_appropriateness = self._evaluate_social_appropriateness(response, cultural_ctx, test_metadata)
        dimensions.append(social_appropriateness)
        
        # 2. Hierarchy Navigation
        hierarchy_navigation = self._evaluate_hierarchy_navigation(response, cultural_ctx, test_metadata)
        dimensions.append(hierarchy_navigation)
        
        # 3. Relationship Maintenance
        relationship_maintenance = self._evaluate_relationship_maintenance(response, cultural_ctx, test_metadata)
        dimensions.append(relationship_maintenance)
        
        # 4. Community Dynamics
        community_dynamics = self._evaluate_community_dynamics(response, cultural_ctx, test_metadata)
        dimensions.append(community_dynamics)
        
        # 5. Cultural Etiquette
        cultural_etiquette = self._evaluate_cultural_etiquette(response, cultural_ctx, test_metadata)
        dimensions.append(cultural_etiquette)
        
        # 6. Conflict Resolution
        conflict_resolution = self._evaluate_conflict_resolution(response, cultural_ctx, test_metadata)
        dimensions.append(conflict_resolution)
        
        # 7. Intercultural Competence
        intercultural_competence = self._evaluate_intercultural_competence(response, cultural_ctx, test_metadata)
        dimensions.append(intercultural_competence)
        
        # Calculate overall score
        overall_score = self._calculate_overall_social_score(dimensions, cultural_ctx)
        
        processing_notes.append(f"Evaluated {len(dimensions)} social competence dimensions")
        processing_notes.append(f"Cultural context: {', '.join(cultural_ctx.cultural_groups)}")
        
        return DomainEvaluationResult(
            domain="social",
            evaluation_type=self._determine_social_evaluation_type(test_metadata),
            overall_score=overall_score,
            dimensions=dimensions,
            cultural_context=cultural_ctx,
            metadata={
                "social_categories": len(dimensions),
                "primary_cultural_groups": cultural_ctx.cultural_groups,
                "evaluation_focus": test_metadata.get("category", "general_social")
            },
            processing_notes=processing_notes
        )
    
    def _evaluate_social_appropriateness(self, response: str, cultural_context: CulturalContext,
                                       test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate social appropriateness and context sensitivity."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Check for polite language
        polite_count = sum(1 for phrase in self.social_patterns["polite_language"] 
                          if phrase in response_lower)
        if polite_count > 0:
            evidence.append(f"Uses polite language ({polite_count} instances)")
            cultural_markers.append("politeness")
        
        # Check for respectful address
        respectful_count = sum(1 for term in self.social_patterns["respectful_address"]
                              if term in response_lower)
        if respectful_count > 0:
            evidence.append(f"Uses respectful forms of address ({respectful_count} instances)")
            cultural_markers.append("respect")
        
        # Check for social awareness
        awareness_count = sum(1 for phrase in self.social_patterns["social_awareness"]
                             if phrase in response_lower)
        if awareness_count > 0:
            evidence.append(f"Demonstrates social awareness ({awareness_count} instances)")
            cultural_markers.append("social_awareness")
        
        # Check for inappropriate language (penalty)
        inappropriate_count = sum(1 for phrase in self.social_patterns["inappropriate_markers"]
                                 if phrase in response_lower)
        if inappropriate_count > 0:
            evidence.append(f"Contains inappropriate language ({inappropriate_count} instances)")
            cultural_markers.append("inappropriate_language")
        
        # Check for boundary respect
        boundary_count = sum(1 for phrase in self.social_patterns["boundary_respect"]
                            if phrase in response_lower)
        if boundary_count > 0:
            evidence.append(f"Respects boundaries ({boundary_count} instances)")
            cultural_markers.append("boundary_respect")
        
        # Calculate score
        positive_score = (polite_count * 0.2 + respectful_count * 0.25 + 
                         awareness_count * 0.3 + boundary_count * 0.25)
        penalty = inappropriate_count * 0.3
        
        score = max(0.0, min(1.0, positive_score - penalty))
        confidence = min(1.0, (polite_count + respectful_count + awareness_count + boundary_count) * 0.1)
        
        # Cultural relevance based on context
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="social_appropriateness",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_hierarchy_navigation(self, response: str, cultural_context: CulturalContext,
                                     test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate understanding and navigation of social hierarchies."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Check for authority recognition
        authority_count = sum(1 for phrase in self.hierarchy_markers["authority_recognition"]
                             if phrase in response_lower)
        if authority_count > 0:
            evidence.append(f"Recognizes authority structures ({authority_count} instances)")
            cultural_markers.append("authority_recognition")
        
        # Check for respectful disagreement
        disagreement_count = sum(1 for phrase in self.hierarchy_markers["respectful_disagreement"]
                                if phrase in response_lower)
        if disagreement_count > 0:
            evidence.append(f"Uses respectful disagreement ({disagreement_count} instances)")
            cultural_markers.append("respectful_disagreement")
        
        # Check for cultural hierarchy awareness
        cultural_hierarchy_count = sum(1 for phrase in self.hierarchy_markers["cultural_hierarchy"]
                                      if phrase in response_lower)
        if cultural_hierarchy_count > 0:
            evidence.append(f"Understands cultural hierarchy ({cultural_hierarchy_count} instances)")
            cultural_markers.append("cultural_hierarchy")
        
        # Check for power dynamics awareness
        power_count = sum(1 for phrase in self.hierarchy_markers["power_dynamics"]
                         if phrase in response_lower)
        if power_count > 0:
            evidence.append(f"Recognizes power dynamics ({power_count} instances)")
            cultural_markers.append("power_awareness")
        
        # Check for hierarchy violations (penalty)
        violation_count = sum(1 for phrase in self.hierarchy_markers["hierarchy_violations"]
                             if phrase in response_lower)
        if violation_count > 0:
            evidence.append(f"Contains hierarchy violations ({violation_count} instances)")
            cultural_markers.append("hierarchy_violations")
        
        # Calculate score with cultural context weighting
        base_score = (authority_count * 0.25 + disagreement_count * 0.25 + 
                     cultural_hierarchy_count * 0.3 + power_count * 0.2)
        penalty = violation_count * 0.4
        
        # Adjust for cultural context
        cultural_weight = 1.0
        if any(group in ["east_asian", "african_traditional", "indigenous"] 
               for group in cultural_context.cultural_groups):
            cultural_weight = 1.2  # Higher importance in hierarchy-conscious cultures
        
        score = max(0.0, min(1.0, (base_score * cultural_weight) - penalty))
        confidence = min(1.0, (authority_count + disagreement_count + cultural_hierarchy_count) * 0.15)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="hierarchy_navigation",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_relationship_maintenance(self, response: str, cultural_context: CulturalContext,
                                         test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate relationship building and maintenance skills."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Check for trust building
        trust_count = sum(1 for phrase in self.relationship_indicators["trust_building"]
                         if phrase in response_lower)
        if trust_count > 0:
            evidence.append(f"Demonstrates trust building ({trust_count} instances)")
            cultural_markers.append("trust_building")
        
        # Check for empathy expression
        empathy_count = sum(1 for phrase in self.relationship_indicators["empathy_expression"]
                           if phrase in response_lower)
        if empathy_count > 0:
            evidence.append(f"Shows empathy and understanding ({empathy_count} instances)")
            cultural_markers.append("empathy")
        
        # Check for relationship investment
        investment_count = sum(1 for phrase in self.relationship_indicators["relationship_investment"]
                              if phrase in response_lower)
        if investment_count > 0:
            evidence.append(f"Invests in relationships ({investment_count} instances)")
            cultural_markers.append("relationship_investment")
        
        # Check for conflict avoidance/resolution in relationships
        avoidance_count = sum(1 for phrase in self.relationship_indicators["conflict_avoidance"]
                             if phrase in response_lower)
        if avoidance_count > 0:
            evidence.append(f"Seeks relationship harmony ({avoidance_count} instances)")
            cultural_markers.append("conflict_avoidance")
        
        # Check for relationship damage markers (penalty)
        damage_count = sum(1 for phrase in self.relationship_indicators["relationship_damage"]
                          if phrase in response_lower)
        if damage_count > 0:
            evidence.append(f"Contains relationship-damaging elements ({damage_count} instances)")
            cultural_markers.append("relationship_damage")
        
        # Calculate score
        positive_score = (trust_count * 0.3 + empathy_count * 0.3 + 
                         investment_count * 0.25 + avoidance_count * 0.15)
        penalty = damage_count * 0.5
        
        score = max(0.0, min(1.0, positive_score - penalty))
        confidence = min(1.0, (trust_count + empathy_count + investment_count) * 0.12)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="relationship_maintenance",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_community_dynamics(self, response: str, cultural_context: CulturalContext,
                                   test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate understanding of community dynamics and collective behavior."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Check for collective thinking
        collective_count = sum(1 for phrase in self.community_patterns["collective_thinking"]
                              if phrase in response_lower)
        if collective_count > 0:
            evidence.append(f"Demonstrates collective thinking ({collective_count} instances)")
            cultural_markers.append("collective_thinking")
        
        # Check for participation encouragement
        participation_count = sum(1 for phrase in self.community_patterns["participation_encouragement"]
                                 if phrase in response_lower)
        if participation_count > 0:
            evidence.append(f"Encourages participation ({participation_count} instances)")
            cultural_markers.append("participation_encouragement")
        
        # Check for community support
        support_count = sum(1 for phrase in self.community_patterns["community_support"]
                           if phrase in response_lower)
        if support_count > 0:
            evidence.append(f"Supports community welfare ({support_count} instances)")
            cultural_markers.append("community_support")
        
        # Check for social cohesion awareness
        cohesion_count = sum(1 for phrase in self.community_patterns["social_cohesion"]
                            if phrase in response_lower)
        if cohesion_count > 0:
            evidence.append(f"Understands social cohesion ({cohesion_count} instances)")
            cultural_markers.append("social_cohesion")
        
        # Check for individualistic focus (penalty in community contexts)
        individualistic_count = sum(1 for phrase in self.community_patterns["individualistic_focus"]
                                   if phrase in response_lower)
        if individualistic_count > 0:
            evidence.append(f"Shows individualistic focus ({individualistic_count} instances)")
            cultural_markers.append("individualistic_focus")
        
        # Calculate score with cultural context consideration
        base_score = (collective_count * 0.3 + participation_count * 0.25 + 
                     support_count * 0.25 + cohesion_count * 0.2)
        
        # Individualistic penalty varies by cultural context
        penalty_weight = 0.2
        if any(group in ["african_traditional", "indigenous", "east_asian"] 
               for group in cultural_context.cultural_groups):
            penalty_weight = 0.4  # Higher penalty in collectivist cultures
        
        penalty = individualistic_count * penalty_weight
        score = max(0.0, min(1.0, base_score - penalty))
        confidence = min(1.0, (collective_count + participation_count + support_count + cohesion_count) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="community_dynamics",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_cultural_etiquette(self, response: str, cultural_context: CulturalContext,
                                   test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate adherence to cultural etiquette systems."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        etiquette_score = 0.0
        etiquette_matches = 0
        
        # Check relevant etiquette systems based on cultural context
        relevant_systems = self._identify_relevant_etiquette_systems(cultural_context)
        
        for system_name, system_patterns in relevant_systems.items():
            for category, patterns in system_patterns.items():
                matches = sum(1 for pattern in patterns if pattern in response_lower)
                if matches > 0:
                    etiquette_matches += matches
                    etiquette_score += matches * 0.1
                    evidence.append(f"Follows {system_name} {category} ({matches} instances)")
                    cultural_markers.append(f"{system_name}_{category}")
        
        # General etiquette markers
        if etiquette_matches == 0:
            # Check for universal etiquette markers
            universal_etiquette = ["respectful", "courteous", "polite", "considerate", "appropriate"]
            universal_matches = sum(1 for marker in universal_etiquette if marker in response_lower)
            if universal_matches > 0:
                etiquette_score = universal_matches * 0.05
                evidence.append(f"Shows general etiquette awareness ({universal_matches} instances)")
                cultural_markers.append("general_etiquette")
        
        score = min(1.0, etiquette_score)
        confidence = min(1.0, etiquette_matches * 0.08)
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="cultural_etiquette",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_conflict_resolution(self, response: str, cultural_context: CulturalContext,
                                    test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate conflict resolution and mediation skills."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Check for de-escalation techniques
        deescalation_count = sum(1 for phrase in self.conflict_resolution_patterns["de_escalation"]
                                if phrase in response_lower)
        if deescalation_count > 0:
            evidence.append(f"Uses de-escalation techniques ({deescalation_count} instances)")
            cultural_markers.append("de_escalation")
        
        # Check for active listening
        listening_count = sum(1 for phrase in self.conflict_resolution_patterns["active_listening"]
                             if phrase in response_lower)
        if listening_count > 0:
            evidence.append(f"Demonstrates active listening ({listening_count} instances)")
            cultural_markers.append("active_listening")
        
        # Check for mediation skills
        mediation_count = sum(1 for phrase in self.conflict_resolution_patterns["mediation_skills"]
                             if phrase in response_lower)
        if mediation_count > 0:
            evidence.append(f"Shows mediation skills ({mediation_count} instances)")
            cultural_markers.append("mediation")
        
        # Check for solution focus
        solution_count = sum(1 for phrase in self.conflict_resolution_patterns["solution_focus"]
                            if phrase in response_lower)
        if solution_count > 0:
            evidence.append(f"Focuses on solutions ({solution_count} instances)")
            cultural_markers.append("solution_focus")
        
        # Check for escalation markers (penalty)
        escalation_count = sum(1 for phrase in self.conflict_resolution_patterns["escalation_markers"]
                              if phrase in response_lower)
        if escalation_count > 0:
            evidence.append(f"Contains escalating language ({escalation_count} instances)")
            cultural_markers.append("escalation")
        
        # Calculate score
        positive_score = (deescalation_count * 0.3 + listening_count * 0.25 + 
                         mediation_count * 0.25 + solution_count * 0.2)
        penalty = escalation_count * 0.4
        
        score = max(0.0, min(1.0, positive_score - penalty))
        confidence = min(1.0, (deescalation_count + listening_count + mediation_count + solution_count) * 0.1)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="conflict_resolution",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _evaluate_intercultural_competence(self, response: str, cultural_context: CulturalContext,
                                         test_metadata: Dict[str, Any]) -> EvaluationDimension:
        """Evaluate intercultural competence and cross-cultural navigation skills."""
        evidence = []
        cultural_markers = []
        
        response_lower = response.lower()
        
        # Check for cultural curiosity
        curiosity_count = sum(1 for phrase in self.intercultural_competence_markers["cultural_curiosity"]
                             if phrase in response_lower)
        if curiosity_count > 0:
            evidence.append(f"Shows cultural curiosity ({curiosity_count} instances)")
            cultural_markers.append("cultural_curiosity")
        
        # Check for cultural humility
        humility_count = sum(1 for phrase in self.intercultural_competence_markers["cultural_humility"]
                            if phrase in response_lower)
        if humility_count > 0:
            evidence.append(f"Demonstrates cultural humility ({humility_count} instances)")
            cultural_markers.append("cultural_humility")
        
        # Check for adaptation skills
        adaptation_count = sum(1 for phrase in self.intercultural_competence_markers["adaptation_skills"]
                              if phrase in response_lower)
        if adaptation_count > 0:
            evidence.append(f"Shows adaptation skills ({adaptation_count} instances)")
            cultural_markers.append("adaptation")
        
        # Check for stereotype awareness
        stereotype_count = sum(1 for phrase in self.intercultural_competence_markers["stereotype_awareness"]
                              if phrase in response_lower)
        if stereotype_count > 0:
            evidence.append(f"Demonstrates stereotype awareness ({stereotype_count} instances)")
            cultural_markers.append("stereotype_awareness")
        
        # Check for cultural bridge building
        bridge_count = sum(1 for phrase in self.intercultural_competence_markers["cultural_bridge_building"]
                          if phrase in response_lower)
        if bridge_count > 0:
            evidence.append(f"Builds cultural bridges ({bridge_count} instances)")
            cultural_markers.append("cultural_bridge_building")
        
        # Calculate score
        score = min(1.0, (curiosity_count * 0.2 + humility_count * 0.25 + adaptation_count * 0.25 + 
                         stereotype_count * 0.15 + bridge_count * 0.15))
        
        confidence = min(1.0, (curiosity_count + humility_count + adaptation_count + 
                              stereotype_count + bridge_count) * 0.08)
        
        cultural_relevance = self._calculate_cultural_relevance(cultural_markers, cultural_context)
        
        return EvaluationDimension(
            name="intercultural_competence",
            score=score,
            confidence=confidence,
            cultural_relevance=cultural_relevance,
            evidence=evidence,
            cultural_markers=cultural_markers
        )
    
    def _identify_relevant_etiquette_systems(self, cultural_context: CulturalContext) -> Dict[str, Dict[str, List[str]]]:
        """Identify relevant etiquette systems based on cultural context."""
        relevant_systems = {}
        
        # Map cultural groups to etiquette systems
        cultural_mappings = {
            "western": "western_formal",
            "european": "western_formal",
            "north_american": "western_formal",
            "east_asian": "east_asian",
            "chinese": "east_asian",
            "japanese": "east_asian",
            "korean": "east_asian",
            "west_african": "african_traditional",
            "african": "african_traditional",
            "middle_eastern": "middle_eastern",
            "arab": "middle_eastern",
            "persian": "middle_eastern",
            "indigenous": "indigenous",
            "aboriginal": "indigenous",
            "native_american": "indigenous"
        }
        
        for group in cultural_context.cultural_groups:
            if group in cultural_mappings:
                system_name = cultural_mappings[group]
                if system_name in self.etiquette_systems:
                    relevant_systems[system_name] = self.etiquette_systems[system_name]
        
        # If no specific systems found, include general western formal
        if not relevant_systems:
            relevant_systems["western_formal"] = self.etiquette_systems["western_formal"]
        
        return relevant_systems
    
    def _calculate_overall_social_score(self, dimensions: List[EvaluationDimension], 
                                      cultural_context: CulturalContext) -> float:
        """Calculate overall social competence score."""
        if not dimensions:
            return 0.0
        
        # Base weights for each dimension
        dimension_weights = {
            "social_appropriateness": 0.20,
            "hierarchy_navigation": 0.15,
            "relationship_maintenance": 0.15,
            "community_dynamics": 0.15,
            "cultural_etiquette": 0.15,
            "conflict_resolution": 0.10,
            "intercultural_competence": 0.10
        }
        
        # Adjust weights based on cultural context
        if any(group in ["east_asian", "african_traditional", "indigenous"] 
               for group in cultural_context.cultural_groups):
            # Increase importance of hierarchy and community in hierarchical cultures
            dimension_weights["hierarchy_navigation"] = 0.20
            dimension_weights["community_dynamics"] = 0.20
            dimension_weights["social_appropriateness"] = 0.15
            dimension_weights["cultural_etiquette"] = 0.20
            dimension_weights["relationship_maintenance"] = 0.10
            dimension_weights["conflict_resolution"] = 0.10
            dimension_weights["intercultural_competence"] = 0.05
        
        # Calculate weighted score
        total_score = 0.0
        total_weight = 0.0
        
        for dimension in dimensions:
            weight = dimension_weights.get(dimension.name, 0.0)
            if weight > 0:
                # Factor in cultural relevance and confidence
                adjusted_score = dimension.score * dimension.cultural_relevance * dimension.confidence
                total_score += adjusted_score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_social_evaluation_type(self, test_metadata: Dict[str, Any]) -> str:
        """Determine the specific social evaluation type."""
        category = test_metadata.get("category", "").lower()
        
        if "hierarchy" in category:
            return SocialEvaluationType.HIERARCHY_NAVIGATION.value
        elif "relationship" in category:
            return SocialEvaluationType.RELATIONSHIP_BUILDING.value
        elif "community" in category:
            return SocialEvaluationType.COMMUNITY_ENGAGEMENT.value
        elif "etiquette" in category:
            return SocialEvaluationType.CULTURAL_ETIQUETTE.value
        elif "conflict" in category:
            return SocialEvaluationType.CONFLICT_RESOLUTION.value
        elif "intercultural" in category or "cross-cultural" in category:
            return SocialEvaluationType.INTERCULTURAL_COMPETENCE.value
        else:
            return SocialEvaluationType.SOCIAL_CONTEXT.value
    
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
        
        # Check alignment with cultural traditions
        for marker in cultural_markers:
            marker_type = marker.split(':')[0] if ':' in marker else marker
            relevance_score = 0.5  # Default relevance
            
            # Higher relevance for markers aligned with cultural context
            if marker_type in ["hierarchy", "authority"] and any(
                group in ["east_asian", "african_traditional", "middle_eastern"] 
                for group in cultural_context.cultural_groups
            ):
                relevance_score = 0.9
            elif marker_type in ["community", "collective"] and any(
                group in ["indigenous", "african_traditional", "east_asian"] 
                for group in cultural_context.cultural_groups
            ):
                relevance_score = 0.9
            elif marker_type in ["individual", "independence"] and any(
                group in ["western", "north_american", "european"] 
                for group in cultural_context.cultural_groups
            ):
                relevance_score = 0.8
            elif marker_type in ["etiquette", "formal", "respectful"]:
                relevance_score = 0.8
            elif marker_type in ["cultural_bridge_building", "adaptation"]:
                relevance_score = 0.9
            
            total_relevance += relevance_score
            marker_count += 1
        
        return min(1.0, total_relevance / marker_count) if marker_count > 0 else 0.5