"""Compatibility package for NSA-ViT module-style execution.

This project stores modules at repository root (e.g. ``sweep.py``, ``training/``),
while runtime commands use ``python -m nsa_vit.<module>``. We expose those
top-level modules by extending ``nsa_vit`` package search path to include the
repository root.
"""

from pathlib import Path

# Keep standard package path entries and add repository root so imports like
# `nsa_vit.sweep` and `nsa_vit.training.distill` resolve to existing modules.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in __path__:
    __path__.append(str(_repo_root))

