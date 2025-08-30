from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import requests
import time
import json
from enum import Enum
from abc import ABC, abstractmethod

from .domain_evaluator_base import CulturalContext
from .evaluation_aggregator import ValidationFlag


class CulturalAPIProvider(Enum):
    """Available cultural API providers."""
    WIKIMEDIA_COMMONS = "wikimedia_commons"
    CULTURAL_HERITAGE_API = "cultural_heritage"
    ETHNOLOGUE_API = "ethnologue"
    UNESCO_API = "unesco"
    OPEN_CULTURE_API = "open_culture"
    DBPEDIA = "dbpedia"
    WIKIDATA = "wikidata"


@dataclass
class CulturalAPIConfig:
    """Configuration for cultural API."""
    provider: CulturalAPIProvider
    base_url: str
    api_key: Optional[str]
    rate_limit: int  # requests per minute
    timeout: int
    free_tier: bool
    description: str


@dataclass
class CulturalAPIResponse:
    """Response from cultural API."""
    provider: CulturalAPIProvider
    query: str
    results: List[Dict[str, Any]]
    confidence: float
    response_time: float
    success: bool
    error_message: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class CulturalValidationResult:
    """Result of cultural validation using APIs."""
    validated_elements: List[str]
    api_responses: List[CulturalAPIResponse]
    overall_confidence: float
    validation_flags: List[ValidationFlag]
    cross_reference_matches: Dict[str, List[str]]  # element -> supporting APIs
    contradictions: Dict[str, List[str]]  # element -> conflicting information
    coverage_score: float


class CulturalAPIClient(ABC):
    """Abstract base for cultural API clients."""
    
    def __init__(self, config: CulturalAPIConfig):
        self.config = config
        self.rate_limiter = self._initialize_rate_limiter()
    
    def _initialize_rate_limiter(self) -> Dict[str, float]:
        """Initialize rate limiting."""
        return {
            'last_request': 0.0,
            'requests_this_minute': 0,
            'minute_start': 0.0
        }
    
    def _check_rate_limit(self) -> bool:
        """Check if request is within rate limit."""
        current_time = time.time()
        
        # Reset minute counter if needed
        if current_time - self.rate_limiter['minute_start'] > 60:
            self.rate_limiter['minute_start'] = current_time
            self.rate_limiter['requests_this_minute'] = 0
        
        # Check rate limit
        if self.rate_limiter['requests_this_minute'] >= self.config.rate_limit:
            return False
        
        return True
    
    def _update_rate_limiter(self):
        """Update rate limiter after request."""
        self.rate_limiter['last_request'] = time.time()
        self.rate_limiter['requests_this_minute'] += 1
    
    @abstractmethod
    async def search_cultural_element(self, element: str, context: CulturalContext) -> CulturalAPIResponse:
        """Search for cultural element using this API."""
        pass
    
    @abstractmethod
    def validate_cultural_claim(self, claim: str, context: CulturalContext) -> CulturalAPIResponse:
        """Validate a cultural claim using this API."""
        pass


class WikimediaCommonsClient(CulturalAPIClient):
    """Client for Wikimedia Commons cultural data."""
    
    async def search_cultural_element(self, element: str, context: CulturalContext) -> CulturalAPIResponse:
        """Search Wikimedia Commons for cultural element."""
        if not self._check_rate_limit():
            return self._create_rate_limited_response(element)
        
        start_time = time.time()
        
        try:
            # Search Wikimedia Commons API
            search_url = f"{self.config.base_url}/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': f"{element} cultural heritage tradition",
                'srlimit': 5,
                'srnamespace': '6|0'  # File and Main namespaces
            }
            
            response = requests.get(search_url, params=params, timeout=self.config.timeout)
            self._update_rate_limiter()
            
            if response.status_code == 200:
                data = response.json()
                results = self._process_wikimedia_results(data, element, context)
                
                return CulturalAPIResponse(
                    provider=self.config.provider,
                    query=element,
                    results=results,
                    confidence=self._calculate_wikimedia_confidence(results, element),
                    response_time=time.time() - start_time,
                    success=True,
                    error_message=None,
                    metadata={'search_params': params}
                )
            else:
                return self._create_error_response(element, f"HTTP {response.status_code}")
                
        except Exception as e:
            return self._create_error_response(element, str(e))
    
    def validate_cultural_claim(self, claim: str, context: CulturalContext) -> CulturalAPIResponse:
        """Validate cultural claim against Wikimedia Commons."""
        # For now, delegate to search - could be enhanced with specific validation logic
        import asyncio
        return asyncio.run(self.search_cultural_element(claim, context))
    
    def _process_wikimedia_results(self, data: Dict[str, Any], element: str, context: CulturalContext) -> List[Dict[str, Any]]:
        """Process Wikimedia API results."""
        results = []
        search_results = data.get('query', {}).get('search', [])
        
        for result in search_results:
            processed_result = {
                'title': result.get('title', ''),
                'snippet': result.get('snippet', ''),
                'size': result.get('size', 0),
                'wordcount': result.get('wordcount', 0),
                'timestamp': result.get('timestamp', ''),
                'relevance_score': self._calculate_relevance(result, element, context),
                'url': f"https://commons.wikimedia.org/wiki/{result.get('title', '').replace(' ', '_')}"
            }
            results.append(processed_result)
        
        return results
    
    def _calculate_relevance(self, result: Dict[str, Any], element: str, context: CulturalContext) -> float:
        """Calculate relevance score for Wikimedia result."""
        relevance = 0.0
        
        title = result.get('title', '').lower()
        snippet = result.get('snippet', '').lower()
        element_lower = element.lower()
        
        # Title relevance
        if element_lower in title:
            relevance += 0.5
        
        # Snippet relevance
        if element_lower in snippet:
            relevance += 0.3
        
        # Cultural context relevance
        context_terms = (context.traditions + context.cultural_groups + 
                        context.knowledge_systems + context.linguistic_varieties)
        for term in context_terms:
            if term.lower() in title or term.lower() in snippet:
                relevance += 0.1
        
        return min(1.0, relevance)
    
    def _calculate_wikimedia_confidence(self, results: List[Dict[str, Any]], element: str) -> float:
        """Calculate confidence in Wikimedia results."""
        if not results:
            return 0.0
        
        # Base confidence on result quality and relevance
        total_confidence = 0.0
        for result in results:
            result_confidence = result.get('relevance_score', 0.0)
            
            # Boost confidence for larger articles
            size_boost = min(0.2, result.get('wordcount', 0) / 1000.0)
            result_confidence += size_boost
            
            total_confidence += result_confidence
        
        return min(1.0, total_confidence / len(results))
    
    def _create_rate_limited_response(self, element: str) -> CulturalAPIResponse:
        """Create response for rate limited request."""
        return CulturalAPIResponse(
            provider=self.config.provider,
            query=element,
            results=[],
            confidence=0.0,
            response_time=0.0,
            success=False,
            error_message="Rate limit exceeded",
            metadata={}
        )
    
    def _create_error_response(self, element: str, error: str) -> CulturalAPIResponse:
        """Create error response."""
        return CulturalAPIResponse(
            provider=self.config.provider,
            query=element,
            results=[],
            confidence=0.0,
            response_time=0.0,
            success=False,
            error_message=error,
            metadata={}
        )


class WikidataClient(CulturalAPIClient):
    """Client for Wikidata cultural information."""
    
    async def search_cultural_element(self, element: str, context: CulturalContext) -> CulturalAPIResponse:
        """Search Wikidata for cultural element."""
        if not self._check_rate_limit():
            return self._create_rate_limited_response(element)
        
        start_time = time.time()
        
        try:
            # Search Wikidata using SPARQL
            sparql_query = self._build_wikidata_query(element, context)
            
            sparql_url = f"{self.config.base_url}/sparql"
            headers = {
                'Accept': 'application/json',
                'User-Agent': 'BenchmarkTests/1.0 (Cultural Validation; educational use)'
            }
            
            response = requests.get(
                sparql_url, 
                params={'query': sparql_query, 'format': 'json'},
                headers=headers,
                timeout=self.config.timeout
            )
            self._update_rate_limiter()
            
            if response.status_code == 200:
                data = response.json()
                results = self._process_wikidata_results(data, element, context)
                
                return CulturalAPIResponse(
                    provider=self.config.provider,
                    query=element,
                    results=results,
                    confidence=self._calculate_wikidata_confidence(results, element),
                    response_time=time.time() - start_time,
                    success=True,
                    error_message=None,
                    metadata={'sparql_query': sparql_query}
                )
            else:
                return self._create_error_response(element, f"HTTP {response.status_code}")
                
        except Exception as e:
            return self._create_error_response(element, str(e))
    
    def validate_cultural_claim(self, claim: str, context: CulturalContext) -> CulturalAPIResponse:
        """Validate cultural claim against Wikidata."""
        import asyncio
        return asyncio.run(self.search_cultural_element(claim, context))
    
    def _build_wikidata_query(self, element: str, context: CulturalContext) -> str:
        """Build SPARQL query for Wikidata."""
        # Simple SPARQL query to find cultural items
        query = f"""
        SELECT ?item ?itemLabel ?description ?culturalGroup ?tradition WHERE {{
          ?item rdfs:label|skos:altLabel|schema:name ?itemLabel .
          FILTER(CONTAINS(LCASE(?itemLabel), LCASE("{element}")))
          
          OPTIONAL {{ ?item wdt:P31/wdt:P279* wd:Q11042 }}  # Cultural heritage
          OPTIONAL {{ ?item wdt:P31/wdt:P279* wd:Q1792379 }}  # Tradition
          OPTIONAL {{ ?item wdt:P31/wdt:P279* wd:Q309481 }}  # Cultural practice
          
          OPTIONAL {{ ?item schema:description ?description }}
          OPTIONAL {{ ?item wdt:P172 ?culturalGroup }}
          OPTIONAL {{ ?item wdt:P361 ?tradition }}
          
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" }}
        }}
        LIMIT 10
        """
        return query
    
    def _process_wikidata_results(self, data: Dict[str, Any], element: str, context: CulturalContext) -> List[Dict[str, Any]]:
        """Process Wikidata SPARQL results."""
        results = []
        bindings = data.get('results', {}).get('bindings', [])
        
        for binding in bindings:
            result = {
                'item_id': binding.get('item', {}).get('value', '').split('/')[-1],
                'label': binding.get('itemLabel', {}).get('value', ''),
                'description': binding.get('description', {}).get('value', ''),
                'cultural_group': binding.get('culturalGroup', {}).get('value', ''),
                'tradition': binding.get('tradition', {}).get('value', ''),
                'relevance_score': self._calculate_wikidata_relevance(binding, element, context),
                'url': binding.get('item', {}).get('value', '')
            }
            results.append(result)
        
        return results
    
    def _calculate_wikidata_relevance(self, binding: Dict[str, Any], element: str, context: CulturalContext) -> float:
        """Calculate relevance for Wikidata result."""
        relevance = 0.0
        element_lower = element.lower()
        
        # Label match
        label = binding.get('itemLabel', {}).get('value', '').lower()
        if element_lower in label:
            relevance += 0.6
        
        # Description match
        description = binding.get('description', {}).get('value', '').lower()
        if element_lower in description:
            relevance += 0.3
        
        # Cultural context match
        cultural_group = binding.get('culturalGroup', {}).get('value', '').lower()
        tradition = binding.get('tradition', {}).get('value', '').lower()
        
        context_terms = [term.lower() for term in (context.traditions + context.cultural_groups)]
        for term in context_terms:
            if term in cultural_group or term in tradition:
                relevance += 0.2
        
        return min(1.0, relevance)
    
    def _calculate_wikidata_confidence(self, results: List[Dict[str, Any]], element: str) -> float:
        """Calculate confidence in Wikidata results."""
        if not results:
            return 0.0
        
        # Higher confidence for Wikidata due to structured data
        relevance_scores = [result.get('relevance_score', 0.0) for result in results]
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        
        # Boost for having multiple results
        result_count_boost = min(0.3, len(results) * 0.1)
        
        return min(1.0, avg_relevance + result_count_boost)
    
    def _create_rate_limited_response(self, element: str) -> CulturalAPIResponse:
        """Create response for rate limited request."""
        return CulturalAPIResponse(
            provider=self.config.provider,
            query=element,
            results=[],
            confidence=0.0,
            response_time=0.0,
            success=False,
            error_message="Rate limit exceeded",
            metadata={}
        )
    
    def _create_error_response(self, element: str, error: str) -> CulturalAPIResponse:
        """Create error response."""
        return CulturalAPIResponse(
            provider=self.config.provider,
            query=element,
            results=[],
            confidence=0.0,
            response_time=0.0,
            success=False,
            error_message=error,
            metadata={}
        )


class DBpediaClient(CulturalAPIClient):
    """Client for DBpedia cultural information."""
    
    async def search_cultural_element(self, element: str, context: CulturalContext) -> CulturalAPIResponse:
        """Search DBpedia for cultural element."""
        if not self._check_rate_limit():
            return self._create_rate_limited_response(element)
        
        start_time = time.time()
        
        try:
            # Use DBpedia lookup API
            lookup_url = f"{self.config.base_url}/lookup-application/api/search"
            params = {
                'query': element,
                'format': 'json',
                'maxResults': 10
            }
            
            response = requests.get(lookup_url, params=params, timeout=self.config.timeout)
            self._update_rate_limiter()
            
            if response.status_code == 200:
                data = response.json()
                results = self._process_dbpedia_results(data, element, context)
                
                return CulturalAPIResponse(
                    provider=self.config.provider,
                    query=element,
                    results=results,
                    confidence=self._calculate_dbpedia_confidence(results, element),
                    response_time=time.time() - start_time,
                    success=True,
                    error_message=None,
                    metadata={'search_params': params}
                )
            else:
                return self._create_error_response(element, f"HTTP {response.status_code}")
                
        except Exception as e:
            return self._create_error_response(element, str(e))
    
    def validate_cultural_claim(self, claim: str, context: CulturalContext) -> CulturalAPIResponse:
        """Validate cultural claim against DBpedia."""
        import asyncio
        return asyncio.run(self.search_cultural_element(claim, context))
    
    def _process_dbpedia_results(self, data: Dict[str, Any], element: str, context: CulturalContext) -> List[Dict[str, Any]]:
        """Process DBpedia lookup results."""
        results = []
        docs = data.get('docs', [])
        
        for doc in docs:
            result = {
                'label': doc.get('label', [''])[0] if doc.get('label') else '',
                'uri': doc.get('resource', [''])[0] if doc.get('resource') else '',
                'description': doc.get('description', [''])[0] if doc.get('description') else '',
                'categories': doc.get('category', []),
                'types': doc.get('type', []),
                'relevance_score': self._calculate_dbpedia_relevance(doc, element, context)
            }
            results.append(result)
        
        return results
    
    def _calculate_dbpedia_relevance(self, doc: Dict[str, Any], element: str, context: CulturalContext) -> float:
        """Calculate relevance for DBpedia result."""
        relevance = 0.0
        element_lower = element.lower()
        
        # Label match
        labels = doc.get('label', [])
        for label in labels:
            if element_lower in label.lower():
                relevance += 0.5
                break
        
        # Description match
        descriptions = doc.get('description', [])
        for desc in descriptions:
            if element_lower in desc.lower():
                relevance += 0.3
                break
        
        # Category/Type match for cultural items
        categories = doc.get('category', []) + doc.get('type', [])
        cultural_terms = ['culture', 'tradition', 'heritage', 'folk', 'ethnic', 'indigenous']
        
        for category in categories:
            category_lower = category.lower()
            if any(term in category_lower for term in cultural_terms):
                relevance += 0.2
                break
        
        return min(1.0, relevance)
    
    def _calculate_dbpedia_confidence(self, results: List[Dict[str, Any]], element: str) -> float:
        """Calculate confidence in DBpedia results."""
        if not results:
            return 0.0
        
        relevance_scores = [result.get('relevance_score', 0.0) for result in results]
        return min(1.0, sum(relevance_scores) / len(relevance_scores))
    
    def _create_rate_limited_response(self, element: str) -> CulturalAPIResponse:
        """Create response for rate limited request."""
        return CulturalAPIResponse(
            provider=self.config.provider,
            query=element,
            results=[],
            confidence=0.0,
            response_time=0.0,
            success=False,
            error_message="Rate limit exceeded",
            metadata={}
        )
    
    def _create_error_response(self, element: str, error: str) -> CulturalAPIResponse:
        """Create error response."""
        return CulturalAPIResponse(
            provider=self.config.provider,
            query=element,
            results=[],
            confidence=0.0,
            response_time=0.0,
            success=False,
            error_message=error,
            metadata={}
        )


class OpenCulturalAPIsIntegration:
    """Main integration class for open cultural APIs."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.api_configs = self._initialize_api_configs()
        self.clients = self._initialize_clients()
    
    def _initialize_api_configs(self) -> Dict[CulturalAPIProvider, CulturalAPIConfig]:
        """Initialize API configurations."""
        return {
            CulturalAPIProvider.WIKIMEDIA_COMMONS: CulturalAPIConfig(
                provider=CulturalAPIProvider.WIKIMEDIA_COMMONS,
                base_url="https://commons.wikimedia.org",
                api_key=None,
                rate_limit=10,  # Conservative rate limit
                timeout=10,
                free_tier=True,
                description="Wikimedia Commons cultural media and metadata"
            ),
            CulturalAPIProvider.WIKIDATA: CulturalAPIConfig(
                provider=CulturalAPIProvider.WIKIDATA,
                base_url="https://query.wikidata.org",
                api_key=None,
                rate_limit=5,  # Conservative for SPARQL
                timeout=15,
                free_tier=True,
                description="Wikidata structured cultural knowledge"
            ),
            CulturalAPIProvider.DBPEDIA: CulturalAPIConfig(
                provider=CulturalAPIProvider.DBPEDIA,
                base_url="https://lookup.dbpedia.org",
                api_key=None,
                rate_limit=10,
                timeout=10,
                free_tier=True,
                description="DBpedia structured cultural information"
            )
        }
    
    def _initialize_clients(self) -> Dict[CulturalAPIProvider, CulturalAPIClient]:
        """Initialize API clients."""
        clients = {}
        
        for provider, config in self.api_configs.items():
            if provider == CulturalAPIProvider.WIKIMEDIA_COMMONS:
                clients[provider] = WikimediaCommonsClient(config)
            elif provider == CulturalAPIProvider.WIKIDATA:
                clients[provider] = WikidataClient(config)
            elif provider == CulturalAPIProvider.DBPEDIA:
                clients[provider] = DBpediaClient(config)
        
        return clients
    
    async def validate_cultural_context(self, 
                                       cultural_context: CulturalContext,
                                       providers: List[CulturalAPIProvider] = None) -> CulturalValidationResult:
        """
        Validate cultural context against multiple APIs.
        
        Args:
            cultural_context: Cultural context to validate
            providers: APIs to use (default: all available)
            
        Returns:
            CulturalValidationResult with validation findings
        """
        if providers is None:
            providers = list(self.clients.keys())
        
        all_responses = []
        validated_elements = []
        cross_reference_matches = {}
        contradictions = {}
        validation_flags = []
        
        # Collect all cultural elements to validate
        all_elements = (cultural_context.traditions + 
                       cultural_context.cultural_groups + 
                       cultural_context.knowledge_systems + 
                       cultural_context.linguistic_varieties)
        
        # Query each API for each element
        for element in all_elements:
            element_responses = []
            
            for provider in providers:
                if provider in self.clients:
                    try:
                        response = await self.clients[provider].search_cultural_element(element, cultural_context)
                        element_responses.append(response)
                        all_responses.append(response)
                        
                        # Track successful validations
                        if response.success and response.confidence > 0.5:
                            if element not in cross_reference_matches:
                                cross_reference_matches[element] = []
                            cross_reference_matches[element].append(provider.value)
                            
                    except Exception as e:
                        # Log error but continue with other providers
                        validation_flags.append(ValidationFlag(
                            flag_type='api_error',
                            severity='medium',
                            description=f"API {provider.value} failed for element '{element}': {str(e)}",
                            affected_dimensions=[],
                            cultural_groups=cultural_context.cultural_groups,
                            recommendation=f"Check API availability: {provider.value}"
                        ))
            
            # Determine if element is validated (multiple API confirmation)
            successful_validations = [r for r in element_responses if r.success and r.confidence > 0.4]
            if len(successful_validations) >= 2:  # Multiple APIs agree
                validated_elements.append(element)
            elif len(successful_validations) == 1 and successful_validations[0].confidence > 0.8:
                validated_elements.append(element)  # Single high-confidence validation
        
        # Calculate overall metrics
        overall_confidence = self._calculate_overall_confidence(all_responses)
        coverage_score = self._calculate_coverage_score(validated_elements, all_elements)
        
        # Check for contradictions (simplified)
        contradictions = self._detect_contradictions(all_responses)
        
        # Generate additional validation flags
        validation_flags.extend(self._generate_api_validation_flags(
            all_responses, cultural_context, coverage_score
        ))
        
        return CulturalValidationResult(
            validated_elements=validated_elements,
            api_responses=all_responses,
            overall_confidence=overall_confidence,
            validation_flags=validation_flags,
            cross_reference_matches=cross_reference_matches,
            contradictions=contradictions,
            coverage_score=coverage_score
        )
    
    def _calculate_overall_confidence(self, responses: List[CulturalAPIResponse]) -> float:
        """Calculate overall confidence from API responses."""
        if not responses:
            return 0.0
        
        successful_responses = [r for r in responses if r.success]
        if not successful_responses:
            return 0.0
        
        # Weight by response confidence and provider reliability
        total_confidence = 0.0
        total_weight = 0.0
        
        for response in successful_responses:
            # Provider reliability weights
            provider_weight = {
                CulturalAPIProvider.WIKIDATA: 1.0,
                CulturalAPIProvider.DBPEDIA: 0.9,
                CulturalAPIProvider.WIKIMEDIA_COMMONS: 0.8
            }.get(response.provider, 0.7)
            
            weighted_confidence = response.confidence * provider_weight
            total_confidence += weighted_confidence
            total_weight += provider_weight
        
        return total_confidence / total_weight if total_weight > 0 else 0.0
    
    def _calculate_coverage_score(self, validated_elements: List[str], all_elements: List[str]) -> float:
        """Calculate coverage score."""
        if not all_elements:
            return 1.0
        
        return len(validated_elements) / len(all_elements)
    
    def _detect_contradictions(self, responses: List[CulturalAPIResponse]) -> Dict[str, List[str]]:
        """Detect contradictions between API responses."""
        contradictions = {}
        
        # Group responses by query
        query_responses = {}
        for response in responses:
            if response.query not in query_responses:
                query_responses[response.query] = []
            query_responses[response.query].append(response)
        
        # Look for contradictory information
        for query, query_resps in query_responses.items():
            successful_resps = [r for r in query_resps if r.success and r.results]
            
            if len(successful_resps) >= 2:
                # Simple contradiction detection: very different confidence levels
                confidences = [r.confidence for r in successful_resps]
                if max(confidences) - min(confidences) > 0.6:
                    contradictions[query] = [
                        f"Confidence mismatch: {r.provider.value} ({r.confidence:.2f})" 
                        for r in successful_resps
                    ]
        
        return contradictions
    
    def _generate_api_validation_flags(self, 
                                     responses: List[CulturalAPIResponse],
                                     cultural_context: CulturalContext,
                                     coverage_score: float) -> List[ValidationFlag]:
        """Generate validation flags based on API responses."""
        flags = []
        
        # Low coverage flag
        if coverage_score < 0.5:
            flags.append(ValidationFlag(
                flag_type='low_api_coverage',
                severity='medium' if coverage_score < 0.3 else 'low',
                description=f"Low API coverage of cultural elements: {coverage_score:.2f}",
                affected_dimensions=[],
                cultural_groups=cultural_context.cultural_groups,
                recommendation="Consider additional cultural validation sources"
            ))
        
        # API failure flags
        failed_apis = set()
        for response in responses:
            if not response.success:
                failed_apis.add(response.provider.value)
        
        if failed_apis:
            flags.append(ValidationFlag(
                flag_type='api_failures',
                severity='medium',
                description=f"API failures: {list(failed_apis)}",
                affected_dimensions=[],
                cultural_groups=cultural_context.cultural_groups,
                recommendation="Check API availability and configuration"
            ))
        
        # No successful responses
        successful_responses = [r for r in responses if r.success]
        if not successful_responses:
            flags.append(ValidationFlag(
                flag_type='no_api_responses',
                severity='high',
                description="No successful API responses for cultural validation",
                affected_dimensions=[],
                cultural_groups=cultural_context.cultural_groups,
                recommendation="Verify cultural context accuracy manually"
            ))
        
        return flags
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get status of all configured APIs."""
        status = {
            'total_apis': len(self.api_configs),
            'available_apis': len(self.clients),
            'api_details': {}
        }
        
        for provider, config in self.api_configs.items():
            api_status = {
                'provider': provider.value,
                'description': config.description,
                'base_url': config.base_url,
                'free_tier': config.free_tier,
                'rate_limit': config.rate_limit,
                'client_available': provider in self.clients
            }
            
            status['api_details'][provider.value] = api_status
        
        return status