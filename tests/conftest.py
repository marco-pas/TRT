"""Pytest bootstrap shared by local development and CI.

The project uses a ``src/`` layout.  Keeping this small bootstrap means that
``python -m pytest`` works before an editable install, while CI also tests the
installed package through its explicit ``PYTHONPATH`` setting.
"""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import jax

jax.config.update("jax_enable_x64", True)
