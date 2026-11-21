"""
Pytest configuration and root sys.path setup for Group G011 (MP277).
Ensures src and other project packages are importable regardless of pytest invocation method.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
