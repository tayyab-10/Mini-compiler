"""
Import-friendly wrapper around `mini-compiler.py`.

Why this exists:
- The original file is named with a hyphen (`mini-compiler.py`), which Python cannot import as a module.
- Web hosts (Render) often don't have Tkinter. The core compiler remains importable because `mini-compiler.py`
  now guards GUI imports/classes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_impl() -> ModuleType:
	root = Path(__file__).resolve().parent
	impl_path = root / "mini-compiler.py"
	if not impl_path.exists():
		raise FileNotFoundError(f"Expected compiler file at: {impl_path}")

	spec = importlib.util.spec_from_file_location("mini_compiler_impl", impl_path)
	if spec is None or spec.loader is None:
		raise RuntimeError("Failed to create import spec for mini-compiler.py")

	module = importlib.util.module_from_spec(spec)
	# Important for dataclasses: module must be in sys.modules during exec_module.
	sys.modules[spec.name] = module
	spec.loader.exec_module(module)  # type: ignore[attr-defined]
	return module


_impl = _load_impl()

# Re-export the key public API used by the FastAPI server (and potentially others).
MiniCompilerEngine = getattr(_impl, "MiniCompilerEngine")
CompilationArtifacts = getattr(_impl, "CompilationArtifacts")
Severity = getattr(_impl, "Severity")
TokenKind = getattr(_impl, "TokenKind")

# Also expose types that are handy for integrations/tests.
Diagnostic = getattr(_impl, "Diagnostic")
Token = getattr(_impl, "Token")
Span = getattr(_impl, "Span")
Position = getattr(_impl, "Position")

__all__ = [
	"MiniCompilerEngine",
	"CompilationArtifacts",
	"Severity",
	"TokenKind",
	"Diagnostic",
	"Token",
	"Span",
	"Position",
]



