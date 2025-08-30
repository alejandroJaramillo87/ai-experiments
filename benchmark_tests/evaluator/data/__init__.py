"""
Data handling and API integration systems.

This module contains data management components:
- Domain metadata extraction and analysis
- Open cultural API integrations
- Dataset management and validation
"""

from .domain_metadata_extractor import *
from .open_cultural_apis import *

__all__ = [
    # Domain metadata
    'DomainMetadataExtractor',
    'DomainMetadata',
    'MetadataField',
    
    # Cultural APIs
    'OpenCulturalAPIs',
    'CulturalAPIClient',
    'APIResponse',
    'CulturalResource',
]