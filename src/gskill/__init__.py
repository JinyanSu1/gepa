"""Compatibility package for the gskill agent harness.

This lets users run and import modules as ``gskill.*`` while the source files
remain under ``gepa.gskill.gskill``.
"""

from pathlib import Path

_IMPL_PATH = Path(__file__).resolve().parent.parent / "gepa" / "gskill" / "gskill"
if _IMPL_PATH.exists():
    __path__.append(str(_IMPL_PATH))  # type: ignore[name-defined]
