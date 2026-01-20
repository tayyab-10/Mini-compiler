from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from mini_compiler import (
	TokenKind,
	Span,
	TypeRef,
	ProgramNode,
	MethodDecl,
	VarDecl,
	ConstDecl,
	ClassDecl,
	ClassField,
	Parameter,
	Statement,
	BlockStatement,
	EmptyStatement,
	AssignmentStatement,
	CallStatement,
	IfStatement,
	WhileStatement,
	ReturnStatement,
	ReadStatement,
	PrintStatement,
	Expression,
	DesignatorExpression,
	CallExpression,
	FieldAccess,
	IndexAccess,
	LiteralExpression,
	UnaryExpression,
	BinaryExpression,
	NewExpression,
)


class RuntimeIssue(Exception):
	def __init__(self, message: str, span: Optional[Span] = None) -> None:
		super().__init__(message)
		self.message = message
		self.span = span

	def __str__(self) -> str:
		return self.message


@dataclass
class RunArtifacts:
	output: str
	steps: int
	runtime_error: Optional[RuntimeIssue] = None


class _ReturnSignal(Exception):
	def __init__(self, value: Any) -> None:
		self.value = value


class MicroJavaInterpreter:
	"""
	A small interpreter for the AST produced by `mini-compiler.py`.

	Supported:
	- int/char scalars
	- arrays via `new T[expr]`
	- class instances via `new ClassName` with fields
	- assignments, calls, if/while, return, read, print
	"""

	def __init__(self, *, stdin: str = "", max_steps: int = 50_000) -> None:
		self._stdin_tokens = [t for t in stdin.replace("\r\n", "\n").replace("\r", "\n").split() if t]
		self._stdin_i = 0
		self._out: List[str] = []
		self._steps = 0
		self._max_steps = max_steps

		self._program: Optional[ProgramNode] = None
		self._classes: Dict[str, ClassDecl] = {}
		self._methods: Dict[str, MethodDecl] = {}
		self._globals: Dict[str, Any] = {}
		self._global_types: Dict[str, TypeRef] = {}
		self._consts: Dict[str, Any] = {}
		self._const_names: set[str] = set()

		# Call stack: list of frames (dict var -> value) and their type maps (var -> TypeRef)
		self._frames: List[Dict[str, Any]] = []
		self._frame_types: List[Dict[str, TypeRef]] = []

	def run(self, program: ProgramNode) -> RunArtifacts:
		try:
			self._prepare(program)
			main = self._methods.get("main")
			if not main:
				return RunArtifacts(output="", steps=self._steps, runtime_error=RuntimeIssue("No main() found."))
			self._call_method(main, [])
			return RunArtifacts(output="".join(self._out), steps=self._steps)
		except RuntimeIssue as issue:
			return RunArtifacts(output="".join(self._out), steps=self._steps, runtime_error=issue)
		except Exception as e:
			return RunArtifacts(output="".join(self._out), steps=self._steps, runtime_error=RuntimeIssue(f"Runtime error: {e}"))

	def _prepare(self, program: ProgramNode) -> None:
		self._program = program
		self._classes = {c.name: c for c in program.classes}
		self._methods = {m.name: m for m in program.methods}

		# consts
		for c in program.consts:
			val = c.value_token.value
			self._consts[c.name] = val
			self._const_names.add(c.name)
			self._global_types[c.name] = c.type_ref

		# global vars
		for v in program.vars:
			for name in v.names:
				self._globals[name] = self._default_value(v.type_ref)
				self._global_types[name] = v.type_ref

	def _tick(self, node_span: Optional[Span] = None) -> None:
		self._steps += 1
		if self._steps > self._max_steps:
			raise RuntimeIssue("Step limit exceeded (possible infinite loop).", node_span)

	def _call_method(self, method: MethodDecl, args: List[Any]) -> Any:
		if len(args) != len(method.parameters):
			raise RuntimeIssue(f"Method '{method.name}' expected {len(method.parameters)} args, got {len(args)}.", method.span)

		frame: Dict[str, Any] = {}
		frame_types: Dict[str, TypeRef] = {}

		# parameters
		for p, a in zip(method.parameters, args):
			frame[p.name] = a
			frame_types[p.name] = p.type_ref

		# locals
		for v in method.locals:
			for name in v.names:
				frame[name] = self._default_value(v.type_ref)
				frame_types[name] = v.type_ref

		self._frames.append(frame)
		self._frame_types.append(frame_types)
		try:
			self._exec_statement(method.body)
		except _ReturnSignal as r:
			return r.value
		finally:
			self._frames.pop()
			self._frame_types.pop()
		return None

	def _exec_statement(self, stmt: Statement) -> None:
		self._tick(getattr(stmt, "span", None))

		if isinstance(stmt, BlockStatement):
			for s in stmt.statements:
				self._exec_statement(s)
			return
		if isinstance(stmt, EmptyStatement):
			return
		if isinstance(stmt, AssignmentStatement):
			container, key, _tref = self._resolve_lvalue(stmt.target)
			if isinstance(container, dict) and isinstance(key, str) and key in self._const_names and container is self._globals:
				raise RuntimeIssue(f"Cannot assign to constant '{key}'.", stmt.target.span)
			container[key] = self._eval_expression(stmt.value)
			return
		if isinstance(stmt, IfStatement):
			cond = self._truthy(self._eval_expression(stmt.condition))
			if cond:
				self._exec_statement(stmt.then_branch)
			elif stmt.else_branch is not None:
				self._exec_statement(stmt.else_branch)
			return
		if isinstance(stmt, WhileStatement):
			while self._truthy(self._eval_expression(stmt.condition)):
				self._exec_statement(stmt.body)
			return
		if isinstance(stmt, ReturnStatement):
			val = self._eval_expression(stmt.value) if stmt.value is not None else None
			raise _ReturnSignal(val)
		if isinstance(stmt, PrintStatement):
			val = self._eval_expression(stmt.value)
			self._out.append(self._format_print(val))
			return
		if isinstance(stmt, ReadStatement):
			container, key, tref = self._resolve_lvalue(stmt.target)
			container[key] = self._read_value(tref, stmt.target.span)
			return
		if isinstance(stmt, CallStatement):
			self._eval_call(stmt.invocation)
			return

		raise RuntimeIssue(f"Unsupported statement: {stmt.__class__.__name__}", getattr(stmt, "span", None))

	def _eval_expression(self, expr: Expression) -> Any:
		if isinstance(expr, LiteralExpression):
			return expr.value
		if isinstance(expr, DesignatorExpression):
			container, key, _tref = self._resolve_lvalue(expr)
			return container[key]
		if isinstance(expr, CallExpression):
			return self._eval_call(expr)
		if isinstance(expr, UnaryExpression):
			val = self._eval_expression(expr.operand)
			if expr.operator == TokenKind.MINUS:
				return -self._to_int(val)
			raise RuntimeIssue(f"Unsupported unary operator: {expr.operator}", expr.span)
		if isinstance(expr, BinaryExpression):
			l = self._eval_expression(expr.left)
			r = self._eval_expression(expr.right)
			op = expr.operator
			if op == TokenKind.PLUS:
				return self._to_int(l) + self._to_int(r)
			if op == TokenKind.MINUS:
				return self._to_int(l) - self._to_int(r)
			if op == TokenKind.STAR:
				return self._to_int(l) * self._to_int(r)
			if op == TokenKind.SLASH:
				den = self._to_int(r)
				if den == 0:
					raise RuntimeIssue("Division by zero.", expr.span)
				return self._to_int(l) // den
			if op == TokenKind.PERCENT:
				den = self._to_int(r)
				if den == 0:
					raise RuntimeIssue("Division by zero.", expr.span)
				return self._to_int(l) % den

			# comparisons return int 1/0 (MicroJava-style)
			if op == TokenKind.EQ:
				return 1 if self._cmp_val(l) == self._cmp_val(r) else 0
			if op == TokenKind.NEQ:
				return 1 if self._cmp_val(l) != self._cmp_val(r) else 0
			if op == TokenKind.GT:
				return 1 if self._cmp_val(l) > self._cmp_val(r) else 0
			if op == TokenKind.GTE:
				return 1 if self._cmp_val(l) >= self._cmp_val(r) else 0
			if op == TokenKind.LT:
				return 1 if self._cmp_val(l) < self._cmp_val(r) else 0
			if op == TokenKind.LTE:
				return 1 if self._cmp_val(l) <= self._cmp_val(r) else 0

			raise RuntimeIssue(f"Unsupported binary operator: {op}", expr.span)
		if isinstance(expr, NewExpression):
			# `new T[expr]` => array allocation
			if expr.size is not None:
				n = self._to_int(self._eval_expression(expr.size))
				if n < 0:
					raise RuntimeIssue("Array size must be non-negative.", expr.span)
				elem_type = TypeRef(name=expr.type_ref.name, is_array=False, span=expr.type_ref.span)
				default = self._default_value(elem_type)
				return [default for _ in range(n)]

			# `new ClassName` => object allocation
			c = self._classes.get(expr.type_ref.name)
			if not c:
				raise RuntimeIssue(f"Unknown class '{expr.type_ref.name}'", expr.span)
			obj: Dict[str, Any] = {}
			for f in c.fields:
				obj[f.name] = self._default_value(f.type_ref)
			return obj

		raise RuntimeIssue(f"Unsupported expression: {expr.__class__.__name__}", getattr(expr, "span", None))

	def _eval_call(self, call: CallExpression) -> Any:
		# We only support calls to global methods: foo(...)
		if call.callee.parts:
			raise RuntimeIssue("Method calls on designators (obj.method()) are not supported.", call.span)
		name = call.callee.name
		m = self._methods.get(name)
		if not m:
			raise RuntimeIssue(f"Unknown method '{name}'.", call.span)
		args = [self._eval_expression(a) for a in call.arguments]
		return self._call_method(m, args)

	def _resolve_lvalue(self, d: DesignatorExpression) -> Tuple[Dict[Any, Any] | List[Any], Any, TypeRef]:
		# Find base variable in current frame or globals/consts.
		if self._frames and d.name in self._frames[-1]:
			container: Dict[Any, Any] | List[Any] = self._frames[-1]
			key: Any = d.name
			tref = self._frame_types[-1].get(d.name, TypeRef(name="int", is_array=False))
		elif d.name in self._globals:
			container = self._globals
			key = d.name
			tref = self._global_types.get(d.name, TypeRef(name="int", is_array=False))
		elif d.name in self._consts:
			container = self._consts  # read-only, but we still allow reading via designator
			key = d.name
			tref = self._global_types.get(d.name, TypeRef(name="int", is_array=False))
		else:
			raise RuntimeIssue(f"Undefined variable '{d.name}'.", d.span)

		# Traverse parts.
		current_val = container[key]
		current_type = tref

		for part in d.parts:
			if isinstance(part, FieldAccess):
				if not isinstance(current_val, dict):
					raise RuntimeIssue("Field access on non-object value.", part.span)
				container = current_val
				key = part.name
				if key not in container:
					raise RuntimeIssue(f"Unknown field '{key}'.", part.span)
				current_val = container[key]
				current_type = self._field_type(current_type, key) or TypeRef(name="int", is_array=False)
			elif isinstance(part, IndexAccess):
				if not isinstance(current_val, list):
					raise RuntimeIssue("Indexing non-array value.", part.span)
				idx = self._to_int(self._eval_expression(part.expression))
				if idx < 0 or idx >= len(current_val):
					raise RuntimeIssue("Array index out of bounds.", part.span)
				container = current_val
				key = idx
				current_val = container[key]
				current_type = self._array_element_type(current_type) or TypeRef(name="int", is_array=False)
			else:
				raise RuntimeIssue(f"Unsupported designator part: {part.__class__.__name__}", getattr(part, "span", None))

		return container, key, current_type

	def _field_type(self, base: TypeRef, field: str) -> Optional[TypeRef]:
		# base is class name; look up class fields
		c = self._classes.get(base.name)
		if not c:
			return None
		for f in c.fields:
			if f.name == field:
				return f.type_ref
		return None

	def _array_element_type(self, base: TypeRef) -> Optional[TypeRef]:
		# If base is already an array type, element is base.name without array
		if base.is_array:
			return TypeRef(name=base.name, is_array=False, span=base.span)
		# In `new T[expr]`, our TypeRef may not have is_array set; best-effort fallback.
		return TypeRef(name=base.name, is_array=False, span=base.span)

	def _default_value(self, tref: TypeRef) -> Any:
		if tref.is_array:
			return None  # null array
		if tref.name == "int":
			return 0
		if tref.name == "char":
			return "\0"
		# class type
		return None  # null reference

	def _read_value(self, tref: TypeRef, span: Optional[Span]) -> Any:
		if self._stdin_i >= len(self._stdin_tokens):
			raise RuntimeIssue("read() requested input but stdin is empty.", span)
		raw = self._stdin_tokens[self._stdin_i]
		self._stdin_i += 1

		# arrays/objects not supported as read targets
		if tref.is_array or (tref.name not in {"int", "char"}):
			raise RuntimeIssue("read() supports only int/char targets in this interpreter.", span)

		if tref.name == "int":
			try:
				return int(raw)
			except ValueError:
				raise RuntimeIssue(f"Invalid int input: '{raw}'.", span)
		# char
		return raw[0] if raw else "\0"

	def _format_print(self, val: Any) -> str:
		if isinstance(val, str):
			return val
		if val is None:
			return "null"
		return str(val)

	def _to_int(self, val: Any) -> int:
		if isinstance(val, int):
			return val
		if isinstance(val, str):
			return ord(val) if val else 0
		if val is None:
			return 0
		raise RuntimeIssue(f"Expected numeric value, got {type(val).__name__}.")

	def _cmp_val(self, val: Any) -> Any:
		# Normalize chars to int for comparisons
		if isinstance(val, str):
			return ord(val) if val else 0
		return val

	def _truthy(self, val: Any) -> bool:
		if isinstance(val, int):
			return val != 0
		if isinstance(val, str):
			return val != "\0" and val != ""
		return bool(val)

