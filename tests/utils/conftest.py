"""Path setup for tests/utils: make repo-root packages importable.

The project is not installed as a package (tool.uv package = false), so
tests that import common/ insert the repo root, matching the pattern in
tests/interp/conftest.py and tests/training/support.py.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
