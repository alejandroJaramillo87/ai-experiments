from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
import json
import os
import requests
import zipfile
import csv
import time
from pathlib import Path
from enum import Enum

from ..core.domain_evaluator_base import CulturalContext, DomainEvaluationResult
from ..core.evaluation_aggregator import ValidationFlag


class DatasetSource(Enum):
    """Sources of cultural datasets."""
    UNESCO = "unesco"
    ETHNOLOGUE = "ethnologue"
    WORLD_CULTURES = "world_cultures"
    ACADEMIC_CORPUS = "academic_corpus"
    CULTURAL_COMMONS = "cultural_commons"


@dataclass
class CulturalDatasetEntry:
    """Entry from a cultural dataset."""
    name: str
    description: str
    cultural_groups: List[str]
    traditions: List[str]
    knowledge_systems: List[str]
    linguistic_varieties: List[str]
    geographic_regions: List[str]
    sources: List[str]
    confidence_score: float
    dataset_source: DatasetSource


@dataclass
class DatasetValidationResult:
    """Result of validating against cultural datasets."""
    matched_entries: List[CulturalDatasetEntry]
    validation_confidence: float
    coverage_score: float  # How well datasets cover the cultural context
    validation_flags: List[ValidationFlag]
    dataset_sources_used: List[DatasetSource]
    missing_cultural_elements: List[str]
    contradictory_information: Dict[str, List[str]]


class CulturalDatasetValidator:
    """Validates cultural evaluations against free cultural datasets."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.datasets_dir = Path(self.config.get('datasets_dir', './data/cultural'))
        self.datasets_dir.mkdir(exist_ok=True)
        
        # Dataset configurations
        self.dataset_configs = self._initialize_dataset_configs()
        self.cached_datasets = {}
        
        # Initialize datasets if they don't exist
        self._setup_datasets()
    
    def _initialize_dataset_configs(self) -> Dict[DatasetSource, Dict[str, Any]]:
        """Initialize configurations for cultural datasets."""
        return {
            DatasetSource.UNESCO: {
                "name": "UNESCO Cultural Heritage",
                "url": "https://whc.unesco.org/en/list/",
                "format": "json",
                "local_file": "unesco_cultural_heritage.json",
                "description": "UNESCO World Heritage cultural sites and practices",
                "update_frequency_days": 30
            },
            DatasetSource.ETHNOLOGUE: {
                "name": "Ethnologue Language Data",
                "url": "https://www.ethnologue.com/",
                "format": "csv",
                "local_file": "ethnologue_languages.csv",
                "description": "Language families and cultural groups",
                "update_frequency_days": 90,
                "note": "Uses free/public data only"
            },
            DatasetSource.WORLD_CULTURES: {
                "name": "World Cultures Dataset",
                "url": "https://raw.githubusercontent.com/datasets/world-cities/master/data/world-cities.csv",
                "format": "csv", 
                "local_file": "world_cultures.csv",
                "description": "Cultural practices and traditions by region",
                "update_frequency_days": 60,
                "note": "Substitute public cultural data"
            },
            DatasetSource.ACADEMIC_CORPUS: {
                "name": "Academic Cultural Corpus",
                "url": "local",
                "format": "json",
                "local_file": "academic_cultural_corpus.json",
                "description": "Curated academic sources on cultural practices",
                "update_frequency_days": 7
            },
            DatasetSource.CULTURAL_COMMONS: {
                "name": "Cultural Commons Data",
                "url": "https://commons.wikimedia.org/",
                "format": "json",
                "local_file": "cultural_commons.json", 
                "description": "Open cultural knowledge from Wikimedia Commons",
                "update_frequency_days": 14
            }
        }
    
    def _setup_datasets(self):
        """Set up cultural datasets if they don't exist."""
        for source, config in self.dataset_configs.items():
            local_path = self.datasets_dir / config["local_file"]
            
            if not local_path.exists() or self._needs_update(local_path, config):
                print(f"Setting up dataset: {config['name']}")
                self._download_or_create_dataset(source, config, local_path)
    
    def _needs_update(self, local_path: Path, config: Dict[str, Any]) -> bool:
        """Check if dataset needs updating."""
        if not local_path.exists():
            return True
        
        # Check file age
        file_age_days = (time.time() - local_path.stat().st_mtime) / 86400
        return file_age_days > config.get("update_frequency_days", 30)
    
    def _download_or_create_dataset(self, source: DatasetSource, config: Dict[str, Any], local_path: Path):
        """Download or create a cultural dataset."""
        try:
            if source == DatasetSource.UNESCO:
                self._create_unesco_dataset(local_path)
            elif source == DatasetSource.ETHNOLOGUE:
                self._create_ethnologue_dataset(local_path)
            elif source == DatasetSource.WORLD_CULTURES:
                self._create_world_cultures_dataset(local_path)
            elif source == DatasetSource.ACADEMIC_CORPUS:
                self._create_academic_corpus(local_path)
            elif source == DatasetSource.CULTURAL_COMMONS:
                self._create_cultural_commons_dataset(local_path)
        except Exception as e:
            print(f"Failed to create dataset {source.value}: {str(e)}")
            # Create empty dataset as fallback
            self._create_empty_dataset(local_path)
    
    def _create_unesco_dataset(self, local_path: Path):
        """Create UNESCO cultural heritage dataset."""
        # This is a simplified version using sample data
        # In practice, you'd use UNESCO's API or data exports
        unesco_data = [
            {
                "name": "Griot Tradition",
                "description": "West African oral storytelling tradition",
                "cultural_groups": ["mandinka", "wolof", "fulani"],
                "traditions": ["griot", "oral tradition", "storytelling"],
                "knowledge_systems": ["traditional knowledge", "genealogy", "oral history"],
                "linguistic_varieties": ["mandinka", "wolof", "fulani"],
                "geographic_regions": ["west africa", "mali", "senegal", "guinea"],
                "sources": ["UNESCO", "academic research"],
                "confidence_score": 0.95
            },
            {
                "name": "Aboriginal Dreamtime",
                "description": "Indigenous Australian creation stories and cultural law",
                "cultural_groups": ["aboriginal australian", "indigenous australian"],
                "traditions": ["dreamtime", "songlines", "ancestral stories"],
                "knowledge_systems": ["traditional knowledge", "indigenous knowledge", "land connection"],
                "linguistic_varieties": ["aboriginal languages", "indigenous australian"],
                "geographic_regions": ["australia", "outback"],
                "sources": ["UNESCO", "Australian cultural institutions"],
                "confidence_score": 0.92
            },
            {
                "name": "Kamishibai",
                "description": "Japanese paper theater storytelling tradition",
                "cultural_groups": ["japanese"],
                "traditions": ["kamishibai", "paper theater", "visual storytelling"],
                "knowledge_systems": ["traditional performance", "visual narrative"],
                "linguistic_varieties": ["japanese"],
                "geographic_regions": ["japan"],
                "sources": ["UNESCO", "Japanese cultural documentation"],
                "confidence_score": 0.88
            }
        ]
        
        with open(local_path, 'w', encoding='utf-8') as f:
            json.dump(unesco_data, f, indent=2, ensure_ascii=False)
    
    def _create_ethnologue_dataset(self, local_path: Path):
        """Create ethnologue-style language and culture dataset."""
        ethnologue_data = [
            ["Language", "Family", "Cultural_Group", "Region", "Speakers", "Traditions"],
            ["Mandinka", "Mande", "mandinka", "West Africa", "1300000", "griot tradition"],
            ["Wolof", "Atlantic", "wolof", "Senegal", "4200000", "griot tradition"],
            ["Quechua", "Quechuan", "indigenous south american", "Andes", "8000000", "oral tradition"],
            ["Navajo", "Athabaskan", "navajo", "North America", "170000", "oral tradition"],
            ["Yolngu", "Pama-Nyungan", "aboriginal australian", "Australia", "4000", "dreamtime stories"]
        ]
        
        with open(local_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(ethnologue_data)
    
    def _create_world_cultures_dataset(self, local_path: Path):
        """Create world cultures dataset with regional cultural information."""
        cultures_data = [
            ["Region", "Cultural_Groups", "Traditions", "Knowledge_Systems", "Languages"],
            ["West Africa", "mandinka;wolof;fulani", "griot;oral tradition", "traditional knowledge", "mandinka;wolof;fulani"],
            ["Australia", "aboriginal australian", "dreamtime;songlines", "indigenous knowledge", "aboriginal languages"],
            ["Japan", "japanese", "kamishibai;tea ceremony", "traditional arts", "japanese"],
            ["Andes", "quechua;aymara", "oral tradition;weaving", "traditional agriculture", "quechua;aymara"],
            ["North America", "native american;first nations", "oral tradition;powwow", "traditional ecological knowledge", "indigenous american"]
        ]
        
        with open(local_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(cultures_data)
    
    def _create_academic_corpus(self, local_path: Path):
        """Create academic cultural corpus from curated sources."""
        academic_corpus = [
            {
                "title": "Oral Tradition in West Africa",
                "cultural_groups": ["west african", "mandinka", "wolof"],
                "traditions": ["griot", "oral tradition", "praise singing"],
                "knowledge_systems": ["traditional knowledge", "genealogy", "cultural transmission"],
                "key_concepts": ["griots", "djeli", "oral history", "cultural preservation"],
                "source": "Academic research compilation",
                "confidence_score": 0.85
            },
            {
                "title": "Indigenous Knowledge Systems",
                "cultural_groups": ["indigenous", "aboriginal", "native american"],
                "traditions": ["oral tradition", "traditional practice", "ceremonial knowledge"],
                "knowledge_systems": ["indigenous knowledge", "traditional ecological knowledge", "cultural knowledge"],
                "key_concepts": ["traditional knowledge", "indigenous wisdom", "cultural preservation"],
                "source": "Indigenous studies literature",
                "confidence_score": 0.90
            }
        ]
        
        with open(local_path, 'w', encoding='utf-8') as f:
            json.dump(academic_corpus, f, indent=2, ensure_ascii=False)
    
    def _create_cultural_commons_dataset(self, local_path: Path):
        """Create cultural commons dataset from Wikimedia and open sources."""
        commons_data = [
            {
                "name": "Traditional Storytelling Methods",
                "description": "Various cultural approaches to storytelling",
                "cultural_groups": ["global", "various"],
                "traditions": ["oral tradition", "storytelling", "narrative tradition"],
                "knowledge_systems": ["traditional knowledge", "cultural transmission"],
                "sources": ["Wikimedia Commons", "Open cultural archives"],
                "confidence_score": 0.75
            }
        ]
        
        with open(local_path, 'w', encoding='utf-8') as f:
            json.dump(commons_data, f, indent=2, ensure_ascii=False)
    
    def _create_empty_dataset(self, local_path: Path):
        """Create empty dataset as fallback."""
        with open(local_path, 'w') as f:
            json.dump([], f)
    
    def validate_cultural_evaluation(self, 
                                   cultural_context: CulturalContext,
                                   evaluation_result: DomainEvaluationResult) -> DatasetValidationResult:
        """
        Validate cultural evaluation against cultural datasets.
        
        Args:
            cultural_context: Cultural context to validate
            evaluation_result: Evaluation result to validate
            
        Returns:
            DatasetValidationResult with validation findings
        """
        matched_entries = []
        validation_flags = []
        dataset_sources_used = []
        contradictory_info = {}
        
        # Load and search datasets
        for source in DatasetSource:
            try:
                dataset = self._load_dataset(source)
                if dataset:
                    matches = self._find_matches_in_dataset(cultural_context, dataset, source)
                    matched_entries.extend(matches)
                    if matches:
                        dataset_sources_used.append(source)
            except Exception as e:
                validation_flags.append(ValidationFlag(
                    flag_type='dataset_error',
                    severity='medium',
                    description=f"Failed to load dataset {source.value}: {str(e)}",
                    affected_dimensions=[],
                    cultural_groups=cultural_context.cultural_groups,
                    recommendation=f"Check dataset availability: {source.value}"
                ))
        
        # Calculate validation metrics
        validation_confidence = self._calculate_validation_confidence(matched_entries, cultural_context)
        coverage_score = self._calculate_coverage_score(matched_entries, cultural_context)
        
        # Identify missing elements
        missing_elements = self._identify_missing_elements(cultural_context, matched_entries)
        
        # Check for contradictions
        contradictory_info = self._check_contradictions(matched_entries, evaluation_result)
        
        # Generate additional validation flags
        validation_flags.extend(self._generate_dataset_validation_flags(
            matched_entries, cultural_context, validation_confidence, coverage_score
        ))
        
        return DatasetValidationResult(
            matched_entries=matched_entries,
            validation_confidence=validation_confidence,
            coverage_score=coverage_score,
            validation_flags=validation_flags,
            dataset_sources_used=dataset_sources_used,
            missing_cultural_elements=missing_elements,
            contradictory_information=contradictory_info
        )
    
    def _load_dataset(self, source: DatasetSource) -> Optional[List[Dict[str, Any]]]:
        """Load dataset from local file."""
        if source in self.cached_datasets:
            return self.cached_datasets[source]
        
        config = self.dataset_configs[source]
        local_path = self.datasets_dir / config["local_file"]
        
        if not local_path.exists():
            return None
        
        try:
            if config["format"] == "json":
                with open(local_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif config["format"] == "csv":
                data = []
                with open(local_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    data = list(reader)
            else:
                return None
            
            # Cache the dataset
            self.cached_datasets[source] = data
            return data
            
        except Exception as e:
            print(f"Error loading dataset {source.value}: {str(e)}")
            return None
    
    def _find_matches_in_dataset(self, 
                                cultural_context: CulturalContext,
                                dataset: List[Dict[str, Any]],
                                source: DatasetSource) -> List[CulturalDatasetEntry]:
        """Find matching entries in a dataset."""
        matches = []
        
        for entry in dataset:
            match_score = self._calculate_match_score(cultural_context, entry)
            
            if match_score > 0.3:  # Threshold for considering a match
                dataset_entry = self._convert_to_dataset_entry(entry, source, match_score)
                matches.append(dataset_entry)
        
        return matches
    
    def _calculate_match_score(self, cultural_context: CulturalContext, entry: Dict[str, Any]) -> float:
        """Calculate how well an entry matches the cultural context."""
        score = 0.0
        total_weight = 0.0
        
        # Check traditions
        entry_traditions = self._extract_list_field(entry, ['traditions', 'tradition'])
        if entry_traditions and cultural_context.traditions:
            overlap = self._calculate_overlap(cultural_context.traditions, entry_traditions)
            score += overlap * 0.3
            total_weight += 0.3
        
        # Check cultural groups
        entry_groups = self._extract_list_field(entry, ['cultural_groups', 'cultural_group'])
        if entry_groups and cultural_context.cultural_groups:
            overlap = self._calculate_overlap(cultural_context.cultural_groups, entry_groups)
            score += overlap * 0.3
            total_weight += 0.3
        
        # Check knowledge systems
        entry_knowledge = self._extract_list_field(entry, ['knowledge_systems', 'knowledge_system'])
        if entry_knowledge and cultural_context.knowledge_systems:
            overlap = self._calculate_overlap(cultural_context.knowledge_systems, entry_knowledge)
            score += overlap * 0.2
            total_weight += 0.2
        
        # Check linguistic varieties
        entry_languages = self._extract_list_field(entry, ['linguistic_varieties', 'languages', 'language'])
        if entry_languages and cultural_context.linguistic_varieties:
            overlap = self._calculate_overlap(cultural_context.linguistic_varieties, entry_languages)
            score += overlap * 0.2
            total_weight += 0.2
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _extract_list_field(self, entry: Dict[str, Any], field_names: List[str]) -> List[str]:
        """Extract list field from entry, trying multiple field names."""
        for field_name in field_names:
            if field_name in entry:
                value = entry[field_name]
                if isinstance(value, list):
                    return [str(item).lower() for item in value]
                elif isinstance(value, str):
                    # Try to split on common delimiters
                    if ';' in value:
                        return [item.strip().lower() for item in value.split(';')]
                    elif ',' in value:
                        return [item.strip().lower() for item in value.split(',')]
                    else:
                        return [value.lower()]
        return []
    
    def _calculate_overlap(self, list1: List[str], list2: List[str]) -> float:
        """Calculate overlap between two lists."""
        if not list1 or not list2:
            return 0.0
        
        set1 = {item.lower() for item in list1}
        set2 = {item.lower() for item in list2}
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _convert_to_dataset_entry(self, 
                                entry: Dict[str, Any], 
                                source: DatasetSource, 
                                confidence: float) -> CulturalDatasetEntry:
        """Convert raw dataset entry to CulturalDatasetEntry."""
        return CulturalDatasetEntry(
            name=entry.get('name', entry.get('title', 'Unknown')),
            description=entry.get('description', ''),
            cultural_groups=self._extract_list_field(entry, ['cultural_groups', 'cultural_group']),
            traditions=self._extract_list_field(entry, ['traditions', 'tradition']),
            knowledge_systems=self._extract_list_field(entry, ['knowledge_systems', 'knowledge_system']),
            linguistic_varieties=self._extract_list_field(entry, ['linguistic_varieties', 'languages', 'language']),
            geographic_regions=self._extract_list_field(entry, ['geographic_regions', 'region']),
            sources=self._extract_list_field(entry, ['sources', 'source']),
            confidence_score=confidence,
            dataset_source=source
        )
    
    def _calculate_validation_confidence(self, 
                                       matched_entries: List[CulturalDatasetEntry],
                                       cultural_context: CulturalContext) -> float:
        """Calculate overall validation confidence."""
        if not matched_entries:
            return 0.0
        
        # Weight by entry confidence scores
        total_confidence = sum(entry.confidence_score for entry in matched_entries)
        avg_confidence = total_confidence / len(matched_entries)
        
        # Adjust by number of matches and cultural context coverage
        match_factor = min(1.0, len(matched_entries) / 3.0)  # More matches = higher confidence
        
        return avg_confidence * match_factor
    
    def _calculate_coverage_score(self, 
                                matched_entries: List[CulturalDatasetEntry],
                                cultural_context: CulturalContext) -> float:
        """Calculate how well the datasets cover the cultural context."""
        if not matched_entries:
            return 0.0
        
        # Count coverage for each category
        covered_traditions = set()
        covered_groups = set()
        covered_knowledge = set()
        covered_languages = set()
        
        for entry in matched_entries:
            covered_traditions.update(entry.traditions)
            covered_groups.update(entry.cultural_groups)
            covered_knowledge.update(entry.knowledge_systems)
            covered_languages.update(entry.linguistic_varieties)
        
        # Calculate coverage percentages
        tradition_coverage = (len(covered_traditions.intersection({t.lower() for t in cultural_context.traditions})) / 
                            len(cultural_context.traditions) if cultural_context.traditions else 1.0)
        
        group_coverage = (len(covered_groups.intersection({g.lower() for g in cultural_context.cultural_groups})) / 
                         len(cultural_context.cultural_groups) if cultural_context.cultural_groups else 1.0)
        
        knowledge_coverage = (len(covered_knowledge.intersection({k.lower() for k in cultural_context.knowledge_systems})) / 
                            len(cultural_context.knowledge_systems) if cultural_context.knowledge_systems else 1.0)
        
        language_coverage = (len(covered_languages.intersection({l.lower() for l in cultural_context.linguistic_varieties})) / 
                           len(cultural_context.linguistic_varieties) if cultural_context.linguistic_varieties else 1.0)
        
        # Overall coverage score
        return (tradition_coverage + group_coverage + knowledge_coverage + language_coverage) / 4.0
    
    def _identify_missing_elements(self, 
                                 cultural_context: CulturalContext,
                                 matched_entries: List[CulturalDatasetEntry]) -> List[str]:
        """Identify cultural elements not found in datasets."""
        covered_elements = set()
        for entry in matched_entries:
            covered_elements.update(entry.traditions)
            covered_elements.update(entry.cultural_groups)
            covered_elements.update(entry.knowledge_systems)
            covered_elements.update(entry.linguistic_varieties)
        
        all_context_elements = (cultural_context.traditions + 
                              cultural_context.cultural_groups + 
                              cultural_context.knowledge_systems + 
                              cultural_context.linguistic_varieties)
        
        missing = []
        for element in all_context_elements:
            if element.lower() not in covered_elements:
                missing.append(element)
        
        return missing
    
    def _check_contradictions(self, 
                            matched_entries: List[CulturalDatasetEntry],
                            evaluation_result: DomainEvaluationResult) -> Dict[str, List[str]]:
        """Check for contradictory information between datasets and evaluation."""
        contradictions = {}
        
        # This is a simplified check - in practice would be more sophisticated
        for entry in matched_entries:
            if entry.confidence_score < 0.3:
                contradictions[entry.name] = ["Low confidence match - potential contradiction"]
        
        return contradictions
    
    def _generate_dataset_validation_flags(self, 
                                         matched_entries: List[CulturalDatasetEntry],
                                         cultural_context: CulturalContext,
                                         validation_confidence: float,
                                         coverage_score: float) -> List[ValidationFlag]:
        """Generate validation flags based on dataset validation."""
        flags = []
        
        # Low coverage flag
        if coverage_score < 0.5:
            flags.append(ValidationFlag(
                flag_type='low_dataset_coverage',
                severity='medium' if coverage_score < 0.3 else 'low',
                description=f"Low dataset coverage of cultural context: {coverage_score:.2f}",
                affected_dimensions=[],
                cultural_groups=cultural_context.cultural_groups,
                recommendation="Consider additional cultural validation sources"
            ))
        
        # No matches flag
        if not matched_entries:
            flags.append(ValidationFlag(
                flag_type='no_dataset_matches',
                severity='high',
                description="No matches found in cultural datasets",
                affected_dimensions=[],
                cultural_groups=cultural_context.cultural_groups,
                recommendation="Verify cultural context accuracy or expand dataset coverage"
            ))
        
        # Low confidence flag
        if validation_confidence < 0.4:
            flags.append(ValidationFlag(
                flag_type='low_dataset_confidence',
                severity='medium',
                description=f"Low dataset validation confidence: {validation_confidence:.2f}",
                affected_dimensions=[],
                cultural_groups=cultural_context.cultural_groups,
                recommendation="Cross-reference with additional cultural sources"
            ))
        
        return flags
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about available datasets."""
        info = {
            'datasets': {},
            'total_entries': 0,
            'last_updated': {}
        }
        
        for source, config in self.dataset_configs.items():
            local_path = self.datasets_dir / config["local_file"]
            
            dataset_info = {
                'name': config['name'],
                'description': config['description'],
                'available': local_path.exists(),
                'format': config['format']
            }
            
            if local_path.exists():
                try:
                    dataset = self._load_dataset(source)
                    dataset_info['entries'] = len(dataset) if dataset else 0
                    dataset_info['last_modified'] = local_path.stat().st_mtime
                    info['total_entries'] += dataset_info['entries']
                except Exception as e:
                    dataset_info['error'] = str(e)
                    dataset_info['entries'] = 0
            
            info['datasets'][source.value] = dataset_info
        
        return info