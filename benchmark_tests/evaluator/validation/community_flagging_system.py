from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import json
import csv
import time
import uuid
from pathlib import Path
from datetime import datetime

from ..core.domain_evaluator_base import DomainEvaluationResult, CulturalContext
from ..core.evaluation_aggregator import ValidationFlag


class FlagSeverity(Enum):
    """Severity levels for community flags."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FlagCategory(Enum):
    """Categories of community flags."""
    CULTURAL_INACCURACY = "cultural_inaccuracy"
    BIAS_DETECTED = "bias_detected"
    MISSING_CONTEXT = "missing_context"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    EVALUATION_ERROR = "evaluation_error"
    TECHNICAL_ISSUE = "technical_issue"
    COMMUNITY_CONCERN = "community_concern"
    SUGGESTION = "suggestion"


class ReviewStatus(Enum):
    """Status of community flag reviews."""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    RESOLVED = "resolved"


@dataclass
class CommunityFlag:
    """A flag raised by the community for review."""
    flag_id: str
    timestamp: float
    category: FlagCategory
    severity: FlagSeverity
    description: str
    cultural_context: Dict[str, Any]
    affected_content: str
    evaluation_id: Optional[str]
    flagged_dimensions: List[str]
    cultural_groups_affected: List[str]
    evidence: List[str]
    recommended_action: str
    submitter_info: Dict[str, Any]  # Anonymous or minimal info
    review_status: ReviewStatus
    reviewer_notes: List[str]
    resolution_notes: str
    metadata: Dict[str, Any]


@dataclass
class CommunityFeedback:
    """Community feedback on evaluation results."""
    feedback_id: str
    timestamp: float
    evaluation_id: str
    overall_rating: float  # 1.0 to 5.0
    dimension_ratings: Dict[str, float]  # dimension -> rating
    cultural_accuracy_rating: float
    comments: str
    cultural_background: Optional[str]
    expertise_level: str  # "community_member", "cultural_expert", "academic"
    suggested_improvements: List[str]
    metadata: Dict[str, Any]


@dataclass
class FlagAnalytics:
    """Analytics for community flagging system."""
    total_flags: int
    flags_by_category: Dict[str, int]
    flags_by_severity: Dict[str, int]
    flags_by_status: Dict[str, int]
    resolution_time_stats: Dict[str, float]
    most_flagged_dimensions: List[str]
    most_affected_cultural_groups: List[str]
    flag_trends: Dict[str, List[int]]  # category -> counts over time
    community_engagement_metrics: Dict[str, Any]


class CommunityFlaggingSystem:
    """System for community reporting and flagging of evaluation issues."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.data_dir = Path(self.config.get('data_dir', './data/community'))
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize data files
        self.flags_file = self.data_dir / 'community_flags.json'
        self.feedback_file = self.data_dir / 'community_feedback.json'
        self.export_dir = self.data_dir / 'exports'
        self.export_dir.mkdir(exist_ok=True)
        
        # In-memory storage (would be database in production)
        self.flags: Dict[str, CommunityFlag] = self._load_flags()
        self.feedback: Dict[str, CommunityFeedback] = self._load_feedback()
        
        # Auto-flagging rules
        self.auto_flag_rules = self._initialize_auto_flag_rules()
    
    def _load_flags(self) -> Dict[str, CommunityFlag]:
        """Load community flags from storage."""
        if self.flags_file.exists():
            try:
                with open(self.flags_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    flags = {}
                    for flag_id, flag_data in data.items():
                        # Convert dict back to dataclass
                        flag_data['category'] = FlagCategory(flag_data['category'])
                        flag_data['severity'] = FlagSeverity(flag_data['severity'])
                        flag_data['review_status'] = ReviewStatus(flag_data['review_status'])
                        flags[flag_id] = CommunityFlag(**flag_data)
                    return flags
            except Exception as e:
                print(f"Error loading community flags: {e}")
        
        return {}
    
    def _load_feedback(self) -> Dict[str, CommunityFeedback]:
        """Load community feedback from storage."""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    feedback = {}
                    for feedback_id, feedback_data in data.items():
                        feedback[feedback_id] = CommunityFeedback(**feedback_data)
                    return feedback
            except Exception as e:
                print(f"Error loading community feedback: {e}")
        
        return {}
    
    def _save_flags(self):
        """Save community flags to storage."""
        try:
            data = {}
            for flag_id, flag in self.flags.items():
                flag_dict = asdict(flag)
                # Convert enums to strings for JSON serialization
                flag_dict['category'] = flag_dict['category'].value
                flag_dict['severity'] = flag_dict['severity'].value
                flag_dict['review_status'] = flag_dict['review_status'].value
                data[flag_id] = flag_dict
            
            with open(self.flags_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving community flags: {e}")
    
    def _save_feedback(self):
        """Save community feedback to storage."""
        try:
            data = {feedback_id: asdict(feedback) 
                   for feedback_id, feedback in self.feedback.items()}
            
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving community feedback: {e}")
    
    def _initialize_auto_flag_rules(self) -> Dict[str, Any]:
        """Initialize automatic flagging rules."""
        return {
            'low_confidence_threshold': 0.3,
            'high_disagreement_threshold': 0.7,
            'bias_score_threshold': 0.4,
            'cultural_authenticity_threshold': 0.2,
            'missing_cultural_context_threshold': 0.1
        }
    
    def submit_community_flag(self, 
                            category: FlagCategory,
                            severity: FlagSeverity,
                            description: str,
                            evaluation_result: DomainEvaluationResult,
                            evidence: List[str],
                            recommended_action: str,
                            submitter_info: Dict[str, Any] = None) -> str:
        """
        Submit a community flag for review.
        
        Args:
            category: Category of the flag
            severity: Severity level
            description: Description of the issue
            evaluation_result: The evaluation result being flagged
            evidence: Supporting evidence
            recommended_action: Suggested action to take
            submitter_info: Information about submitter (kept minimal/anonymous)
            
        Returns:
            Flag ID
        """
        flag_id = f"flag_{uuid.uuid4().hex[:12]}"
        
        flag = CommunityFlag(
            flag_id=flag_id,
            timestamp=time.time(),
            category=category,
            severity=severity,
            description=description,
            cultural_context=self._extract_cultural_context_dict(evaluation_result.cultural_context),
            affected_content=evaluation_result.metadata.get('content_summary', '')[:500],
            evaluation_id=evaluation_result.metadata.get('evaluation_id'),
            flagged_dimensions=[dim.name for dim in evaluation_result.dimensions],
            cultural_groups_affected=evaluation_result.cultural_context.cultural_groups,
            evidence=evidence,
            recommended_action=recommended_action,
            submitter_info=submitter_info or {'type': 'anonymous'},
            review_status=ReviewStatus.PENDING,
            reviewer_notes=[],
            resolution_notes='',
            metadata={
                'domain': evaluation_result.domain,
                'evaluation_type': evaluation_result.evaluation_type,
                'overall_score': evaluation_result.overall_score
            }
        )
        
        self.flags[flag_id] = flag
        self._save_flags()
        
        return flag_id
    
    def submit_community_feedback(self,
                                evaluation_id: str,
                                overall_rating: float,
                                dimension_ratings: Dict[str, float],
                                cultural_accuracy_rating: float,
                                comments: str,
                                cultural_background: Optional[str] = None,
                                expertise_level: str = "community_member",
                                suggested_improvements: List[str] = None) -> str:
        """
        Submit community feedback on evaluation results.
        
        Args:
            evaluation_id: ID of evaluation being reviewed
            overall_rating: Overall rating (1.0 to 5.0)
            dimension_ratings: Ratings for specific dimensions
            cultural_accuracy_rating: Rating for cultural accuracy
            comments: Text comments
            cultural_background: Background of reviewer (optional)
            expertise_level: Level of expertise
            suggested_improvements: List of improvement suggestions
            
        Returns:
            Feedback ID
        """
        feedback_id = f"feedback_{int(time.time() * 1000)}"
        
        feedback = CommunityFeedback(
            feedback_id=feedback_id,
            timestamp=time.time(),
            evaluation_id=evaluation_id,
            overall_rating=max(1.0, min(5.0, overall_rating)),
            dimension_ratings=dimension_ratings,
            cultural_accuracy_rating=max(1.0, min(5.0, cultural_accuracy_rating)),
            comments=comments,
            cultural_background=cultural_background,
            expertise_level=expertise_level,
            suggested_improvements=suggested_improvements or [],
            metadata={
                'submission_time': datetime.now().isoformat(),
                'ratings_count': len(dimension_ratings)
            }
        )
        
        self.feedback[feedback_id] = feedback
        self._save_feedback()
        
        return feedback_id
    
    def auto_flag_evaluation(self, evaluation_result: DomainEvaluationResult, 
                           validation_flags: List[ValidationFlag]) -> List[str]:
        """
        Automatically flag evaluations based on predefined rules.
        
        Args:
            evaluation_result: Evaluation result to check
            validation_flags: Existing validation flags
            
        Returns:
            List of auto-generated flag IDs
        """
        auto_flags = []
        
        # Check for low confidence
        overall_confidence = evaluation_result.metadata.get('evaluation_confidence', 1.0)
        if overall_confidence < self.auto_flag_rules['low_confidence_threshold']:
            flag_id = self.submit_community_flag(
                category=FlagCategory.EVALUATION_ERROR,
                severity=FlagSeverity.MEDIUM,
                description=f"Automatic flag: Low evaluation confidence ({overall_confidence:.2f})",
                evaluation_result=evaluation_result,
                evidence=[f"Confidence score: {overall_confidence:.2f}"],
                recommended_action="Manual review recommended due to low confidence",
                submitter_info={'type': 'auto_system', 'rule': 'low_confidence'}
            )
            auto_flags.append(flag_id)
        
        # Check for high disagreement
        disagreement_level = evaluation_result.metadata.get('disagreement_level', 0.0)
        if disagreement_level > self.auto_flag_rules['high_disagreement_threshold']:
            flag_id = self.submit_community_flag(
                category=FlagCategory.EVALUATION_ERROR,
                severity=FlagSeverity.HIGH,
                description=f"Automatic flag: High evaluator disagreement ({disagreement_level:.2f})",
                evaluation_result=evaluation_result,
                evidence=[f"Disagreement level: {disagreement_level:.2f}"],
                recommended_action="Review evaluation criteria and methodology",
                submitter_info={'type': 'auto_system', 'rule': 'high_disagreement'}
            )
            auto_flags.append(flag_id)
        
        # Check for cultural authenticity issues
        cultural_competence = evaluation_result.calculate_cultural_competence()
        if cultural_competence < self.auto_flag_rules['cultural_authenticity_threshold']:
            flag_id = self.submit_community_flag(
                category=FlagCategory.CULTURAL_INACCURACY,
                severity=FlagSeverity.HIGH,
                description=f"Automatic flag: Low cultural authenticity score ({cultural_competence:.2f})",
                evaluation_result=evaluation_result,
                evidence=[f"Cultural competence: {cultural_competence:.2f}"],
                recommended_action="Community review for cultural accuracy",
                submitter_info={'type': 'auto_system', 'rule': 'cultural_authenticity'}
            )
            auto_flags.append(flag_id)
        
        # Check existing validation flags for auto-flagging
        for val_flag in validation_flags:
            if val_flag.severity == 'high' and val_flag.flag_type == 'bias':
                flag_id = self.submit_community_flag(
                    category=FlagCategory.BIAS_DETECTED,
                    severity=FlagSeverity.HIGH,
                    description=f"Automatic flag: Bias detection - {val_flag.description}",
                    evaluation_result=evaluation_result,
                    evidence=[val_flag.description],
                    recommended_action=val_flag.recommendation,
                    submitter_info={'type': 'auto_system', 'rule': 'bias_detection'}
                )
                auto_flags.append(flag_id)
        
        return auto_flags
    
    def review_flag(self, flag_id: str, 
                   reviewer_notes: str,
                   new_status: ReviewStatus,
                   resolution_notes: str = '') -> bool:
        """
        Review and update a community flag.
        
        Args:
            flag_id: ID of flag to review
            reviewer_notes: Notes from reviewer
            new_status: New status for the flag
            resolution_notes: Resolution notes if resolving
            
        Returns:
            True if successful, False otherwise
        """
        if flag_id not in self.flags:
            return False
        
        flag = self.flags[flag_id]
        flag.reviewer_notes.append(f"{datetime.now().isoformat()}: {reviewer_notes}")
        flag.review_status = new_status
        
        if resolution_notes:
            flag.resolution_notes = resolution_notes
        
        self._save_flags()
        return True
    
    def get_pending_flags(self, category: Optional[FlagCategory] = None,
                         severity: Optional[FlagSeverity] = None) -> List[CommunityFlag]:
        """Get pending community flags for review."""
        flags = [flag for flag in self.flags.values() 
                if flag.review_status == ReviewStatus.PENDING]
        
        if category:
            flags = [flag for flag in flags if flag.category == category]
        
        if severity:
            flags = [flag for flag in flags if flag.severity == severity]
        
        # Sort by severity and timestamp
        severity_order = {FlagSeverity.CRITICAL: 0, FlagSeverity.HIGH: 1, 
                         FlagSeverity.MEDIUM: 2, FlagSeverity.LOW: 3}
        
        flags.sort(key=lambda f: (severity_order[f.severity], f.timestamp))
        
        return flags
    
    def get_cultural_group_flags(self, cultural_groups: List[str]) -> List[CommunityFlag]:
        """Get flags affecting specific cultural groups."""
        flags = []
        for flag in self.flags.values():
            if any(group in flag.cultural_groups_affected for group in cultural_groups):
                flags.append(flag)
        
        return flags
    
    def export_flagged_items_csv(self, filename: Optional[str] = None) -> str:
        """
        Export flagged items to CSV for review.
        
        Args:
            filename: Optional filename, defaults to timestamp-based name
            
        Returns:
            Path to exported CSV file
        """
        if filename is None:
            filename = f"flagged_items_{int(time.time())}.csv"
        
        export_path = self.export_dir / filename
        
        # Prepare data for CSV export
        fieldnames = [
            'flag_id', 'timestamp', 'category', 'severity', 'description',
            'cultural_groups_affected', 'flagged_dimensions', 'review_status',
            'recommended_action', 'evaluation_id', 'overall_score',
            'cultural_context_summary'
        ]
        
        with open(export_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for flag in self.flags.values():
                row = {
                    'flag_id': flag.flag_id,
                    'timestamp': datetime.fromtimestamp(flag.timestamp).isoformat(),
                    'category': flag.category.value,
                    'severity': flag.severity.value,
                    'description': flag.description,
                    'cultural_groups_affected': ';'.join(flag.cultural_groups_affected),
                    'flagged_dimensions': ';'.join(str(dim) for dim in flag.flagged_dimensions),
                    'review_status': flag.review_status.value,
                    'recommended_action': flag.recommended_action,
                    'evaluation_id': flag.evaluation_id or '',
                    'overall_score': flag.metadata.get('overall_score', ''),
                    'cultural_context_summary': self._summarize_cultural_context(flag.cultural_context)
                }
                writer.writerow(row)
        
        return str(export_path)
    
    def export_feedback_json(self, filename: Optional[str] = None) -> str:
        """
        Export community feedback to JSON for analysis.
        
        Args:
            filename: Optional filename, defaults to timestamp-based name
            
        Returns:
            Path to exported JSON file
        """
        if filename is None:
            filename = f"community_feedback_{int(time.time())}.json"
        
        export_path = self.export_dir / filename
        
        # Prepare feedback data for export
        export_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'total_feedback_entries': len(self.feedback),
                'average_overall_rating': self._calculate_average_rating()
            },
            'feedback': [asdict(feedback) for feedback in self.feedback.values()]
        }
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return str(export_path)
    
    def generate_flag_analytics(self) -> FlagAnalytics:
        """Generate analytics for community flagging system."""
        flags = list(self.flags.values())
        
        # Basic counts
        total_flags = len(flags)
        flags_by_category = {}
        flags_by_severity = {}
        flags_by_status = {}
        
        for flag in flags:
            # Category counts
            cat = flag.category.value.lower()
            flags_by_category[cat] = flags_by_category.get(cat, 0) + 1
            
            # Severity counts
            sev = flag.severity.value.lower()
            flags_by_severity[sev] = flags_by_severity.get(sev, 0) + 1
            
            # Status counts
            status = flag.review_status.value.lower()
            flags_by_status[status] = flags_by_status.get(status, 0) + 1
        
        # Resolution time statistics (for resolved flags)
        resolved_flags = [f for f in flags if f.review_status == ReviewStatus.RESOLVED]
        resolution_times = []
        
        for flag in resolved_flags:
            # Calculate resolution time (simplified - would track actual resolution timestamp)
            resolution_time = time.time() - flag.timestamp
            resolution_times.append(resolution_time)
        
        resolution_time_stats = {}
        if resolution_times:
            resolution_time_stats = {
                'mean_hours': sum(resolution_times) / len(resolution_times) / 3600,
                'min_hours': min(resolution_times) / 3600,
                'max_hours': max(resolution_times) / 3600,
                'count': len(resolution_times)
            }
        
        # Most flagged dimensions
        dimension_counts = {}
        for flag in flags:
            for dimension in flag.flagged_dimensions:
                dimension_counts[dimension] = dimension_counts.get(dimension, 0) + 1
        
        most_flagged_dimensions = sorted(dimension_counts.keys(), 
                                       key=dimension_counts.get, reverse=True)[:10]
        
        # Most affected cultural groups
        group_counts = {}
        for flag in flags:
            for group in flag.cultural_groups_affected:
                group_counts[group] = group_counts.get(group, 0) + 1
        
        most_affected_groups = sorted(group_counts.keys(),
                                    key=group_counts.get, reverse=True)[:10]
        
        # Community engagement metrics
        engagement_metrics = {
            'total_community_submissions': len([f for f in flags 
                                             if f.submitter_info.get('type') != 'auto_system']),
            'auto_generated_flags': len([f for f in flags 
                                       if f.submitter_info.get('type') == 'auto_system']),
            'total_feedback_entries': len(self.feedback),
            'average_feedback_rating': self._calculate_average_rating(),
            'unique_cultural_groups_represented': len(set().union(
                *[flag.cultural_groups_affected for flag in flags]
            ))
        }
        
        return FlagAnalytics(
            total_flags=total_flags,
            flags_by_category=flags_by_category,
            flags_by_severity=flags_by_severity,
            flags_by_status=flags_by_status,
            resolution_time_stats=resolution_time_stats,
            most_flagged_dimensions=most_flagged_dimensions,
            most_affected_cultural_groups=most_affected_groups,
            flag_trends={},  # Would implement trend analysis with time series data
            community_engagement_metrics=engagement_metrics
        )
    
    def _extract_cultural_context_dict(self, cultural_context: CulturalContext) -> Dict[str, Any]:
        """Extract cultural context as dictionary."""
        return {
            'traditions': cultural_context.traditions,
            'knowledge_systems': cultural_context.knowledge_systems,
            'performance_aspects': cultural_context.performance_aspects,
            'cultural_groups': cultural_context.cultural_groups,
            'linguistic_varieties': cultural_context.linguistic_varieties
        }
    
    def _summarize_cultural_context(self, cultural_context: Dict[str, Any]) -> str:
        """Create summary of cultural context for export."""
        summary_parts = []
        
        if cultural_context.get('cultural_groups'):
            summary_parts.append(f"Groups: {', '.join(cultural_context['cultural_groups'][:3])}")
        
        if cultural_context.get('traditions'):
            summary_parts.append(f"Traditions: {', '.join(cultural_context['traditions'][:2])}")
        
        return '; '.join(summary_parts)
    
    def _calculate_average_rating(self) -> float:
        """Calculate average rating from community feedback."""
        if not self.feedback:
            return 0.0
        
        ratings = [feedback.overall_rating for feedback in self.feedback.values()]
        return sum(ratings) / len(ratings)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of community flagging system."""
        analytics = self.generate_flag_analytics()
        
        return {
            'system_info': {
                'total_flags': analytics.total_flags,
                'pending_flags': analytics.flags_by_status.get('pending', 0),
                'total_feedback': len(self.feedback),
                'data_directory': str(self.data_dir),
                'export_directory': str(self.export_dir)
            },
            'recent_activity': {
                'flags_last_24h': len([f for f in self.flags.values() 
                                     if time.time() - f.timestamp < 86400]),
                'feedback_last_24h': len([f for f in self.feedback.values() 
                                        if time.time() - f.timestamp < 86400])
            },
            'priority_items': {
                'critical_flags': len([f for f in self.flags.values() 
                                     if f.severity == FlagSeverity.CRITICAL and 
                                     f.review_status == ReviewStatus.PENDING]),
                'high_severity_pending': len([f for f in self.flags.values() 
                                            if f.severity == FlagSeverity.HIGH and 
                                            f.review_status == ReviewStatus.PENDING])
            },
            'auto_flagging_stats': {
                'auto_flags_generated': len([f for f in self.flags.values() 
                                           if f.submitter_info.get('type') == 'auto_system']),
                'auto_flag_rules_active': len(self.auto_flag_rules)
            }
        }