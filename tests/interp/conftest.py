"""Host-runnable interp commons tests: no GPU, no network, no interp-group imports.

The commons imports interp libraries lazily, so these tests exercise pairs splitting/dedup
and judge parsing with the core environment only.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
