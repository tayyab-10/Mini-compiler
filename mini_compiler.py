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

# AST nodes (handy for interpreter/integrations)
TypeRef = getattr(_impl, "TypeRef")
ProgramNode = getattr(_impl, "ProgramNode")
MethodDecl = getattr(_impl, "MethodDecl")
VarDecl = getattr(_impl, "VarDecl")
ConstDecl = getattr(_impl, "ConstDecl")
ClassDecl = getattr(_impl, "ClassDecl")
ClassField = getattr(_impl, "ClassField")
Parameter = getattr(_impl, "Parameter")

Statement = getattr(_impl, "Statement")
BlockStatement = getattr(_impl, "BlockStatement")
EmptyStatement = getattr(_impl, "EmptyStatement")
AssignmentStatement = getattr(_impl, "AssignmentStatement")
CallStatement = getattr(_impl, "CallStatement")
IfStatement = getattr(_impl, "IfStatement")
WhileStatement = getattr(_impl, "WhileStatement")
ReturnStatement = getattr(_impl, "ReturnStatement")
ReadStatement = getattr(_impl, "ReadStatement")
PrintStatement = getattr(_impl, "PrintStatement")

Expression = getattr(_impl, "Expression")
DesignatorExpression = getattr(_impl, "DesignatorExpression")
CallExpression = getattr(_impl, "CallExpression")
DesignatorPart = getattr(_impl, "DesignatorPart")
FieldAccess = getattr(_impl, "FieldAccess")
IndexAccess = getattr(_impl, "IndexAccess")
LiteralExpression = getattr(_impl, "LiteralExpression")
UnaryExpression = getattr(_impl, "UnaryExpression")
BinaryExpression = getattr(_impl, "BinaryExpression")
NewExpression = getattr(_impl, "NewExpression")

__all__ = [
	"MiniCompilerEngine",
	"CompilationArtifacts",
	"Severity",
	"TokenKind",
	"Diagnostic",
	"Token",
	"Span",
	"Position",
	"TypeRef",
	"ProgramNode",
	"MethodDecl",
	"VarDecl",
	"ConstDecl",
	"ClassDecl",
	"ClassField",
	"Parameter",
	"Statement",
	"BlockStatement",
	"EmptyStatement",
	"AssignmentStatement",
	"CallStatement",
	"IfStatement",
	"WhileStatement",
	"ReturnStatement",
	"ReadStatement",
	"PrintStatement",
	"Expression",
	"DesignatorExpression",
	"CallExpression",
	"DesignatorPart",
	"FieldAccess",
	"IndexAccess",
	"LiteralExpression",
	"UnaryExpression",
	"BinaryExpression",
	"NewExpression",
]



