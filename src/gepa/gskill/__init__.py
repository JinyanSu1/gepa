"""gskill: Learning repository-specific skills for coding agents.

The implementation modules live in ``gepa.gskill.gskill`` for historical
reasons. Extending this package path keeps documented module invocations such
as ``python -m gepa.gskill.train_optimize_anything`` working without duplicating
the implementation files.
"""

from pathlib import Path

_IMPL_PATH = Path(__file__).resolve().parent / "gskill"
if _IMPL_PATH.exists():
    __path__.append(str(_IMPL_PATH))  # type: ignore[name-defined]
