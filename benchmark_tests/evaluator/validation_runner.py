from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
import aiohttp
import time
import json
import statistics
from enum import Enum

from .domain_evaluator_base import DomainEvaluationResult, CulturalContext
from .evaluation_aggregator import ValidationFlag


class APIProvider(Enum):
    """Supported API providers for validation."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"


@dataclass
class APIConfig:
    """Configuration for an API provider."""
    provider: APIProvider
    endpoint: str
    api_key: Optional[str]
    rate_limit: int  # requests per minute
    timeout: int  # seconds
    free_tier_limit: int  # daily limit for free tier
    model_name: str


@dataclass
class ValidationRequest:
    """Request for validation."""
    content: str
    cultural_context: CulturalContext
    evaluation_claims: List[str]
    evaluation_dimension: str
    original_score: float


@dataclass
class APIValidationResult:
    """Result from a single API validation."""
    provider: APIProvider
    validation_score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    reasoning: str
    cultural_assessment: Dict[str, Any] = None
    cultural_elements_validated: List[str] = None  # Added for test compatibility
    potential_issues: List[str] = None  # Added for test compatibility
    response_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default values for optional fields."""
        if self.cultural_assessment is None:
            self.cultural_assessment = {}
        if self.cultural_elements_validated is None:
            self.cultural_elements_validated = []
        if self.potential_issues is None:
            self.potential_issues = []


@dataclass
class MultiModelValidationResult:
    """Aggregated validation result from multiple models."""
    consensus_score: float  # 0.0 to 1.0
    score_variance: float
    provider_results: List[APIValidationResult]
    disagreement_level: float  # 0.0 to 1.0
    validation_flags: List[ValidationFlag]
    high_confidence_providers: List[APIProvider]
    outlier_providers: List[APIProvider]
    cultural_consensus: Dict[str, Any]


class ValidationRunner:
    """Coordinates validation across multiple free API providers."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.api_configs = self._initialize_api_configs()
        self.rate_limiters = self._initialize_rate_limiters()
        self.usage_tracking = self._initialize_usage_tracking()
        
    def _initialize_api_configs(self) -> Dict[APIProvider, APIConfig]:
        """Initialize API configurations for free tiers."""
        # Default free tier configurations
        return {
            APIProvider.OPENAI: APIConfig(
                provider=APIProvider.OPENAI,
                endpoint="https://api.openai.com/v1/chat/completions",
                api_key=self.config.get('openai_api_key'),
                rate_limit=3,  # 3 requests per minute for free tier
                timeout=30,
                free_tier_limit=200,  # daily limit
                model_name="gpt-3.5-turbo"
            ),
            APIProvider.ANTHROPIC: APIConfig(
                provider=APIProvider.ANTHROPIC,
                endpoint="https://api.anthropic.com/v1/messages",
                api_key=self.config.get('anthropic_api_key'),
                rate_limit=5,  # Conservative rate limit
                timeout=30,
                free_tier_limit=100,
                model_name="claude-3-haiku-20240307"
            ),
            APIProvider.GOOGLE: APIConfig(
                provider=APIProvider.GOOGLE,
                endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
                api_key=self.config.get('google_api_key'),
                rate_limit=10,
                timeout=30,
                free_tier_limit=1000,
                model_name="gemini-pro"
            ),
            APIProvider.HUGGINGFACE: APIConfig(
                provider=APIProvider.HUGGINGFACE,
                endpoint="https://api-inference.huggingface.co/models/microsoft/DialoGPT-large",
                api_key=self.config.get('huggingface_api_key'),
                rate_limit=30,  # More generous rate limit
                timeout=30,
                free_tier_limit=10000,
                model_name="microsoft/DialoGPT-large"
            )
        }
    
    def _initialize_rate_limiters(self) -> Dict[APIProvider, Dict[str, Any]]:
        """Initialize rate limiting trackers."""
        rate_limiters = {}
        
        for provider, config in self.api_configs.items():
            rate_limiters[provider] = {
                'last_request_time': 0,
                'requests_this_minute': 0,
                'minute_start': 0,
                'daily_requests': 0,
                'day_start': time.time()
            }
        
        return rate_limiters
    
    def _initialize_usage_tracking(self) -> Dict[str, Any]:
        """Initialize usage tracking."""
        return {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'provider_usage': {provider.value: 0 for provider in APIProvider}
        }
    
    async def validate_evaluation(self, 
                                 validation_request: ValidationRequest,
                                 providers: List[APIProvider] = None) -> MultiModelValidationResult:
        """
        Validate evaluation result against multiple API providers.
        
        Args:
            validation_request: Request containing evaluation to validate
            providers: List of providers to use (default: all available)
            
        Returns:
            MultiModelValidationResult with consensus and disagreement analysis
        """
        if providers is None:
            providers = [p for p in APIProvider if self.api_configs[p].api_key]
        
        # Filter providers based on rate limits and availability
        available_providers = self._get_available_providers(providers)
        
        if not available_providers:
            return self._create_empty_validation_result("No available providers")
        
        # Create validation tasks
        validation_tasks = []
        for provider in available_providers:
            task = self._validate_with_provider(provider, validation_request)
            validation_tasks.append(task)
        
        # Execute validation requests concurrently
        try:
            provider_results = await asyncio.gather(*validation_tasks, return_exceptions=True)
        except Exception as e:
            return self._create_empty_validation_result(f"Validation failed: {str(e)}")
        
        # Process results
        successful_results = []
        for i, result in enumerate(provider_results):
            if isinstance(result, APIValidationResult) and result.success:
                successful_results.append(result)
            elif isinstance(result, Exception):
                # Log the error but continue with other results
                print(f"Provider {available_providers[i].value} failed: {str(result)}")
        
        if not successful_results:
            return self._create_empty_validation_result("All provider validations failed")
        
        # Aggregate results
        return self._aggregate_validation_results(successful_results, validation_request)
    
    async def _validate_with_provider(self, 
                                     provider: APIProvider,
                                     validation_request: ValidationRequest) -> APIValidationResult:
        """Validate with a specific API provider."""
        start_time = time.time()
        
        try:
            # Check rate limits
            if not self._check_rate_limits(provider):
                return APIValidationResult(
                    provider=provider,
                    validation_score=0.0,
                    confidence=0.0,
                    reasoning="Rate limit exceeded",
                    cultural_assessment={},
                    cultural_elements_validated=[],
                    potential_issues=["Rate limit exceeded"],
                    response_time=0.0,
                    success=False,
                    error_message="Rate limit exceeded"
                )
            
            # Create validation prompt
            prompt = self._create_validation_prompt(validation_request)
            
            # Make API request
            response = await self._make_api_request(provider, prompt)
            
            if response:
                # Parse response
                validation_result = self._parse_validation_response(response, provider)
                validation_result.response_time = time.time() - start_time
                
                # Update usage tracking
                self._update_usage_tracking(provider, True)
                
                return validation_result
            else:
                self._update_usage_tracking(provider, False)
                return APIValidationResult(
                    provider=provider,
                    validation_score=0.0,
                    confidence=0.0,
                    reasoning="API request failed",
                    cultural_assessment={},
                    cultural_elements_validated=[],
                    potential_issues=["API request failed"],
                    response_time=time.time() - start_time,
                    success=False,
                    error_message="API request failed"
                )
                
        except Exception as e:
            self._update_usage_tracking(provider, False)
            return APIValidationResult(
                provider=provider,
                validation_score=0.0,
                confidence=0.0,
                reasoning=f"Validation failed: {str(e)}",
                cultural_assessment={},
                cultural_elements_validated=[],
                potential_issues=[f"Validation failed: {str(e)}"],
                response_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _create_validation_prompt(self, validation_request: ValidationRequest) -> str:
        """Create validation prompt for API."""
        prompt = f"""Please evaluate the cultural authenticity and accuracy of this evaluation:

Content being evaluated: {validation_request.content[:500]}...

Cultural Context:
- Traditions: {', '.join(validation_request.cultural_context.traditions)}
- Cultural Groups: {', '.join(validation_request.cultural_context.cultural_groups)}
- Knowledge Systems: {', '.join(validation_request.cultural_context.knowledge_systems)}

Evaluation Claims:
{chr(10).join(f"- {claim}" for claim in validation_request.evaluation_claims)}

Original Score: {validation_request.original_score}
Dimension: {validation_request.evaluation_dimension}

Please provide:
1. Your validation score (0.0-1.0) for how accurate you think this evaluation is
2. Your confidence level (0.0-1.0) in your assessment
3. Brief reasoning for your score
4. Any cultural authenticity concerns

Respond in JSON format:
{{
    "validation_score": <float>,
    "confidence": <float>, 
    "reasoning": "<string>",
    "cultural_concerns": ["<concern1>", "<concern2>"],
    "cultural_assessment": {{
        "authenticity": <float>,
        "accuracy": <float>,
        "cultural_sensitivity": <float>
    }}
}}"""
        
        return prompt
    
    async def _make_api_request(self, provider: APIProvider, prompt: str) -> Optional[Dict[str, Any]]:
        """Make API request to specified provider."""
        config = self.api_configs[provider]
        
        if not config.api_key:
            return None
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config.timeout)) as session:
                if provider == APIProvider.OPENAI:
                    return await self._call_openai_api(session, config, prompt)
                elif provider == APIProvider.ANTHROPIC:
                    return await self._call_anthropic_api(session, config, prompt)
                elif provider == APIProvider.GOOGLE:
                    return await self._call_google_api(session, config, prompt)
                elif provider == APIProvider.HUGGINGFACE:
                    return await self._call_huggingface_api(session, config, prompt)
                
        except Exception as e:
            print(f"API request to {provider.value} failed: {str(e)}")
            return None
    
    async def _call_openai_api(self, session: aiohttp.ClientSession, config: APIConfig, prompt: str) -> Optional[Dict[str, Any]]:
        """Call OpenAI API."""
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.3
        }
        
        async with session.post(config.endpoint, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('choices', [{}])[0].get('message', {}).get('content', '')
            else:
                print(f"OpenAI API error: {response.status}")
                return None
    
    async def _call_anthropic_api(self, session: aiohttp.ClientSession, config: APIConfig, prompt: str) -> Optional[Dict[str, Any]]:
        """Call Anthropic API."""
        headers = {
            "x-api-key": config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": config.model_name,
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        async with session.post(config.endpoint, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('content', [{}])[0].get('text', '')
            else:
                print(f"Anthropic API error: {response.status}")
                return None
    
    async def _call_google_api(self, session: aiohttp.ClientSession, config: APIConfig, prompt: str) -> Optional[Dict[str, Any]]:
        """Call Google Gemini API."""
        url = f"{config.endpoint}?key={config.api_key}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "maxOutputTokens": 500,
                "temperature": 0.3
            }
        }
        
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                candidates = data.get('candidates', [])
                if candidates:
                    return candidates[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            else:
                print(f"Google API error: {response.status}")
                return None
    
    async def _call_huggingface_api(self, session: aiohttp.ClientSession, config: APIConfig, prompt: str) -> Optional[Dict[str, Any]]:
        """Call HuggingFace API."""
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 500,
                "temperature": 0.3,
                "return_full_text": False
            }
        }
        
        async with session.post(config.endpoint, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0].get('generated_text', '')
                return data
            else:
                print(f"HuggingFace API error: {response.status}")
                return None
    
    def _parse_validation_response(self, response: str, provider: APIProvider) -> APIValidationResult:
        """Parse API response into validation result."""
        try:
            # Try to parse as JSON
            if isinstance(response, str):
                # Extract JSON from response if it's embedded in text
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    response_data = json.loads(json_match.group())
                else:
                    # Fallback parsing for non-JSON responses
                    return self._fallback_parse_response(response, provider)
            else:
                response_data = response
            
            return APIValidationResult(
                provider=provider,
                validation_score=float(response_data.get('validation_score', 0.5)),
                confidence=float(response_data.get('confidence', 0.5)),
                reasoning=response_data.get('reasoning', 'No reasoning provided'),
                cultural_assessment=response_data.get('cultural_assessment', {}),
                cultural_elements_validated=response_data.get('cultural_elements_validated', []),
                potential_issues=response_data.get('potential_issues', []),
                response_time=0.0,  # Will be set by caller
                success=True,
                error_message=None
            )
            
        except Exception as e:
            return self._fallback_parse_response(response, provider)
    
    def _fallback_parse_response(self, response: str, provider: APIProvider) -> APIValidationResult:
        """Fallback parsing for non-JSON responses."""
        response_lower = str(response).lower()
        
        # Simple heuristics for validation score
        validation_score = 0.5  # Default
        if any(word in response_lower for word in ['excellent', 'accurate', 'authentic', 'good']):
            validation_score = 0.8
        elif any(word in response_lower for word in ['poor', 'inaccurate', 'problematic', 'wrong']):
            validation_score = 0.2
        
        # Simple confidence estimation
        confidence = 0.3 if len(str(response)) < 50 else 0.6
        
        return APIValidationResult(
            provider=provider,
            validation_score=validation_score,
            confidence=confidence,
            reasoning=str(response)[:200],
            cultural_assessment={},
            cultural_elements_validated=[],
            potential_issues=[],
            response_time=0.0,
            success=True,
            error_message=None
        )
    
    def _aggregate_validation_results(self, 
                                    provider_results: List[APIValidationResult],
                                    validation_request: ValidationRequest) -> MultiModelValidationResult:
        """Aggregate results from multiple providers."""
        scores = [r.validation_score for r in provider_results]
        confidences = [r.confidence for r in provider_results]
        
        # Calculate consensus metrics
        consensus_score = statistics.mean(scores)
        score_variance = statistics.variance(scores) if len(scores) > 1 else 0.0
        disagreement_level = min(1.0, score_variance * 4)  # Scale variance to 0-1
        
        # Identify outliers and high confidence providers
        mean_score = consensus_score
        outlier_threshold = 0.3  # 30% difference from mean
        
        high_confidence_providers = [r.provider for r in provider_results if r.confidence > 0.7]
        outlier_providers = [r.provider for r in provider_results 
                           if abs(r.validation_score - mean_score) > outlier_threshold]
        
        # Generate validation flags
        validation_flags = self._generate_validation_flags(
            provider_results, validation_request, disagreement_level
        )
        
        # Aggregate cultural assessments
        cultural_consensus = self._aggregate_cultural_assessments(provider_results)
        
        return MultiModelValidationResult(
            consensus_score=consensus_score,
            score_variance=score_variance,
            provider_results=provider_results,
            disagreement_level=disagreement_level,
            validation_flags=validation_flags,
            high_confidence_providers=high_confidence_providers,
            outlier_providers=outlier_providers,
            cultural_consensus=cultural_consensus
        )
    
    def _generate_validation_flags(self, 
                                 provider_results: List[APIValidationResult],
                                 validation_request: ValidationRequest,
                                 disagreement_level: float) -> List[ValidationFlag]:
        """Generate validation flags based on results."""
        flags = []
        
        # High disagreement flag
        if disagreement_level > 0.6:
            flags.append(ValidationFlag(
                flag_type='high_disagreement',
                severity='high' if disagreement_level > 0.8 else 'medium',
                description=f"High disagreement between validation providers: {disagreement_level:.2f}",
                affected_dimensions=[validation_request.evaluation_dimension],
                cultural_groups=validation_request.cultural_context.cultural_groups,
                recommendation="Consider manual review due to provider disagreement"
            ))
        
        # Low confidence flag
        avg_confidence = statistics.mean([r.confidence for r in provider_results])
        if avg_confidence < 0.5:
            flags.append(ValidationFlag(
                flag_type='low_confidence',
                severity='high' if avg_confidence < 0.3 else 'medium',
                description=f"Low validation confidence: {avg_confidence:.2f}",
                affected_dimensions=[validation_request.evaluation_dimension],
                cultural_groups=validation_request.cultural_context.cultural_groups,
                recommendation="Consider additional validation methods"
            ))
        
        return flags
    
    def _aggregate_cultural_assessments(self, provider_results: List[APIValidationResult]) -> Dict[str, Any]:
        """Aggregate cultural assessments from providers."""
        cultural_consensus = {
            'authenticity': [],
            'accuracy': [],
            'cultural_sensitivity': [],
            'concerns': []
        }
        
        for result in provider_results:
            assessment = result.cultural_assessment
            if isinstance(assessment, dict):
                for key in ['authenticity', 'accuracy', 'cultural_sensitivity']:
                    if key in assessment:
                        cultural_consensus[key].append(assessment[key])
        
        # Calculate averages
        return {
            'avg_authenticity': statistics.mean(cultural_consensus['authenticity']) if cultural_consensus['authenticity'] else 0.5,
            'avg_accuracy': statistics.mean(cultural_consensus['accuracy']) if cultural_consensus['accuracy'] else 0.5,
            'avg_cultural_sensitivity': statistics.mean(cultural_consensus['cultural_sensitivity']) if cultural_consensus['cultural_sensitivity'] else 0.5,
            'provider_count': len(provider_results)
        }
    
    def _get_available_providers(self, requested_providers: List[APIProvider]) -> List[APIProvider]:
        """Get available providers based on rate limits and configuration."""
        available = []
        
        for provider in requested_providers:
            if self._check_rate_limits(provider) and self.api_configs[provider].api_key:
                available.append(provider)
        
        return available
    
    def _check_rate_limits(self, provider: APIProvider) -> bool:
        """Check if provider is within rate limits."""
        limiter = self.rate_limiters[provider]
        config = self.api_configs[provider]
        current_time = time.time()
        
        # Check daily limit
        if current_time - limiter['day_start'] > 86400:  # New day
            limiter['day_start'] = current_time
            limiter['daily_requests'] = 0
        
        if limiter['daily_requests'] >= config.free_tier_limit:
            return False
        
        # Check per-minute limit
        if current_time - limiter['minute_start'] > 60:  # New minute
            limiter['minute_start'] = current_time
            limiter['requests_this_minute'] = 0
        
        if limiter['requests_this_minute'] >= config.rate_limit:
            return False
        
        return True
    
    def _update_usage_tracking(self, provider: APIProvider, success: bool):
        """Update usage tracking."""
        limiter = self.rate_limiters[provider]
        limiter['requests_this_minute'] += 1
        limiter['daily_requests'] += 1
        limiter['last_request_time'] = time.time()
        
        self.usage_tracking['total_requests'] += 1
        self.usage_tracking['provider_usage'][provider.value] += 1
        
        if success:
            self.usage_tracking['successful_requests'] += 1
        else:
            self.usage_tracking['failed_requests'] += 1
    
    def _create_empty_validation_result(self, reason: str) -> MultiModelValidationResult:
        """Create empty validation result when no providers available."""
        return MultiModelValidationResult(
            consensus_score=0.0,
            score_variance=0.0,
            provider_results=[],
            disagreement_level=0.0,
            validation_flags=[ValidationFlag(
                flag_type='validation_failure',
                severity='high',
                description=f"Validation failed: {reason}",
                affected_dimensions=[],
                cultural_groups=[],
                recommendation="Check API configuration and availability"
            )],
            high_confidence_providers=[],
            outlier_providers=[],
            cultural_consensus={}
        )
    
    # Alias method for test compatibility
    async def validate_with_multiple_apis(self, *args, **kwargs):
        """Alias for validate_evaluation method for test compatibility."""
        return await self.validate_evaluation(*args, **kwargs)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'usage_tracking': self.usage_tracking.copy(),
            'rate_limits_status': {
                provider.value: {
                    'requests_this_minute': limiter['requests_this_minute'],
                    'daily_requests': limiter['daily_requests'],
                    'daily_limit': self.api_configs[provider].free_tier_limit,
                    'available': self._check_rate_limits(provider)
                }
                for provider, limiter in self.rate_limiters.items()
            }
        }