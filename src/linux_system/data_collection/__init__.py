"""
Data Collection Pipeline
=======================

Comprehensive Ubuntu system data collection for LLM training and RAG.

Components:
- collectors/: System state, user activity, and configuration collectors
- storage/: Data storage, indexing, and retrieval systems  
- daemon/: Continuous collection service
- processors/: Data cleaning and formatting utilities

The pipeline collects omniscient system data while maintaining read-only access.
"""

from .collectors import *
from .storage import *
from .daemon import *

__all__ = ['collectors', 'storage', 'daemon']