"""Interactive MicroJava mini compiler with lexer, parser, semantic analysis, and GUI."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Tkinter is only required for the desktop GUI. Many Linux servers (Render) don't ship it.
# Keep the compiler core importable even when tkinter isn't available.
try:
	import tkinter as tk
	from tkinter import filedialog, messagebox, ttk
except Exception:  # pragma: no cover
	tk = None  # type: ignore[assignment]
	filedialog = None  # type: ignore[assignment]
	messagebox = None  # type: ignore[assignment]
	ttk = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Diagnostic infrastructure


class Severity(Enum):
	INFO = auto()
	WARNING = auto()
	ERROR = auto()


@dataclass
class Position:
	line: int
	column: int
	index: int


@dataclass
class Span:
	start: Position
	end: Position


@dataclass
class Diagnostic:
	severity: Severity
	message: str
	span: Optional[Span] = None
	hint: Optional[str] = None


class DiagnosticEngine:
	def __init__(self) -> None:
		self._items: List[Diagnostic] = []

	@property
	def items(self) -> List[Diagnostic]:
		return self._items

	def report(self, severity: Severity, message: str, span: Optional[Span] = None, hint: Optional[str] = None) -> None:
		self._items.append(Diagnostic(severity, message, span, hint))

	def extend(self, diagnostics: Iterable[Diagnostic]) -> None:
		self._items.extend(diagnostics)

	def clear(self) -> None:
		self._items.clear()


def combine_span(a: Span, b: Span) -> Span:
	return Span(start=a.start, end=b.end)


# ---------------------------------------------------------------------------
# Lexer


class TokenKind(Enum):
	PROGRAM = auto()
	CLASS = auto()
	FINAL = auto()
	VOID = auto()
	IF = auto()
	ELSE = auto()
	WHILE = auto()
	READ = auto()
	PRINT = auto()
	RETURN = auto()
	NEW = auto()
	INT = auto()
	CHAR = auto()
	ORD = auto()
	CHR = auto()
	LEN = auto()
	NULL = auto()
	IDENT = auto()
	NUMBER = auto()
	CHAR_CONST = auto()
	PLUS = auto()
	MINUS = auto()
	STAR = auto()
	SLASH = auto()
	PERCENT = auto()
	ASSIGN = auto()
	EQ = auto()
	NEQ = auto()
	GT = auto()
	GTE = auto()
	LT = auto()
	LTE = auto()
	LPAREN = auto()
	RPAREN = auto()
	LBRACKET = auto()
	RBRACKET = auto()
	LBRACE = auto()
	RBRACE = auto()
	SEMI = auto()
	COMMA = auto()
	DOT = auto()
	EOF = auto()
	UNKNOWN = auto()


KEYWORDS: Dict[str, TokenKind] = {
	"program": TokenKind.PROGRAM,
	"class": TokenKind.CLASS,
	"final": TokenKind.FINAL,
	"void": TokenKind.VOID,
	"if": TokenKind.IF,
	"else": TokenKind.ELSE,
	"while": TokenKind.WHILE,
	"read": TokenKind.READ,
	"print": TokenKind.PRINT,
	"return": TokenKind.RETURN,
	"new": TokenKind.NEW,
	"int": TokenKind.INT,
	"char": TokenKind.CHAR,
	"ord": TokenKind.ORD,
	"chr": TokenKind.CHR,
	"len": TokenKind.LEN,
	"null": TokenKind.NULL,
}


SYMBOLS: Dict[str, TokenKind] = {
	"+": TokenKind.PLUS,
	"-": TokenKind.MINUS,
	"*": TokenKind.STAR,
	"/": TokenKind.SLASH,
	"%": TokenKind.PERCENT,
	"=": TokenKind.ASSIGN,
	"==": TokenKind.EQ,
	"!=": TokenKind.NEQ,
	">": TokenKind.GT,
	">=": TokenKind.GTE,
	"<": TokenKind.LT,
	"<=": TokenKind.LTE,
	"(": TokenKind.LPAREN,
	")": TokenKind.RPAREN,
	"[": TokenKind.LBRACKET,
	"]": TokenKind.RBRACKET,
	"{": TokenKind.LBRACE,
	"}": TokenKind.RBRACE,
	";": TokenKind.SEMI,
	",": TokenKind.COMMA,
	".": TokenKind.DOT,
}


@dataclass
class Token:
	kind: TokenKind
	lexeme: str
	span: Span
	value: Optional[Any] = None


class Lexer:
	def __init__(self, source: str, diagnostics: DiagnosticEngine) -> None:
		self.source = source
		self.diagnostics = diagnostics
		self.length = len(source)
		self.index = 0
		self.line = 1
		self.column = 1

	def tokenize(self) -> List[Token]:
		tokens: List[Token] = []
		while not self._is_eof():
			ch = self._peek()
			if ch in " \t\r":
				self._advance()
			elif ch == "\n":
				self._advance()
				self.line += 1
				self.column = 1
			elif ch == "/" and self._peek_next() == "/":
				self._consume_comment()
			elif ch.isalpha() or ch == "_":
				tokens.append(self._consume_identifier())
			elif ch.isdigit():
				tokens.append(self._consume_number())
			elif ch == "'":
				tokens.append(self._consume_char())
			else:
				tokens.append(self._consume_symbol())
		tokens.append(self._make_token(TokenKind.EOF, "", self._current_position()))
		return tokens

	def _consume_comment(self) -> None:
		while not self._is_eof() and self._peek() != "\n":
			self._advance()

	def _consume_identifier(self) -> Token:
		start = self._current_position()
		lexeme = self._consume_while(lambda c: c.isalnum() or c == "_")
		kind = KEYWORDS.get(lexeme, TokenKind.IDENT)
		return self._make_token(kind, lexeme, start)

	def _consume_number(self) -> Token:
		start = self._current_position()
		lexeme = self._consume_while(lambda c: c.isdigit())
		try:
			value = int(lexeme)
		except ValueError:
			value = None
			self.diagnostics.report(Severity.ERROR, f"Invalid integer literal '{lexeme}'", Span(start, self._current_position()))
		return self._make_token(TokenKind.NUMBER, lexeme, start, value)

	def _consume_char(self) -> Token:
		start = self._current_position()
		self._advance()  # opening quote
		if self._is_eof():
			self.diagnostics.report(Severity.ERROR, "Unterminated character literal", Span(start, self._current_position()))
			return self._make_token(TokenKind.CHAR_CONST, "", start)
		ch = self._advance()
		value = ch
		if ch == "\\":
			if self._is_eof():
				self.diagnostics.report(Severity.ERROR, "Unterminated escape sequence", Span(start, self._current_position()))
				value = None
			else:
				esc = self._advance()
				escapes = {"n": "\n", "t": "\t", "r": "\r", "'": "'", "\\": "\\"}
				value = escapes.get(esc)
				if value is None:
					self.diagnostics.report(Severity.ERROR, f"Unknown escape sequence '\\{esc}'", Span(start, self._current_position()))
		if self._peek() != "'":
			self.diagnostics.report(Severity.ERROR, "Unterminated character literal", Span(start, self._current_position()))
		else:
			self._advance()
		lexeme = self.source[start.index : self.index]
		return self._make_token(TokenKind.CHAR_CONST, lexeme, start, value)

	def _consume_symbol(self) -> Token:
		start = self._current_position()
		ch = self._advance()
		next_ch = self._peek() if not self._is_eof() else ""
		candidate = ch + next_ch
		# Only treat as a 2-character symbol if we actually have a next character.
		if next_ch and candidate in SYMBOLS:
			self._advance()
			return self._make_token(SYMBOLS[candidate], candidate, start)
		if ch in SYMBOLS:
			return self._make_token(SYMBOLS[ch], ch, start)
		self.diagnostics.report(Severity.ERROR, f"Unexpected character '{ch}'", Span(start, self._current_position()))
		return self._make_token(TokenKind.UNKNOWN, ch, start)

	def _consume_while(self, predicate) -> str:
		start_index = self.index
		while not self._is_eof() and predicate(self._peek()):
			self._advance()
		return self.source[start_index:self.index]

	def _current_position(self) -> Position:
		return Position(self.line, self.column, self.index)

	def _make_token(self, kind: TokenKind, lexeme: str, start: Position, value: Optional[Any] = None) -> Token:
		end = self._current_position()
		span = Span(start, end)
		return Token(kind, lexeme, span, value)

	def _advance(self) -> str:
		ch = self.source[self.index]
		self.index += 1
		self.column += 1
		return ch

	def _peek(self) -> str:
		return self.source[self.index]

	def _peek_next(self) -> str:
		if self.index + 1 >= self.length:
			return ""
		return self.source[self.index + 1]

	def _is_eof(self) -> bool:
		return self.index >= self.length


# ---------------------------------------------------------------------------
# AST definitions


@dataclass
class TypeRef:
	name: str
	is_array: bool = False
	span: Optional[Span] = None


@dataclass
class ASTNode:
	span: Span


@dataclass
class ConstDecl(ASTNode):
	type_ref: TypeRef
	name: str
	value_token: Token


@dataclass
class VarDecl(ASTNode):
	type_ref: TypeRef
	names: List[str]


@dataclass
class ClassField(ASTNode):
	type_ref: TypeRef
	name: str


@dataclass
class ClassDecl(ASTNode):
	name: str
	fields: List[ClassField]


@dataclass
class Parameter(ASTNode):
	type_ref: TypeRef
	name: str


class Statement(ASTNode):
	pass


class Expression(ASTNode):
	pass


@dataclass
class BlockStatement(Statement):
	statements: List[Statement]


@dataclass
class EmptyStatement(Statement):
	pass


@dataclass
class DesignatorPart:
	span: Span


@dataclass
class FieldAccess(DesignatorPart):
	name: str


@dataclass
class IndexAccess(DesignatorPart):
	expression: Expression


@dataclass
class DesignatorExpression(Expression):
	name: str
	parts: List[DesignatorPart]


@dataclass
class CallExpression(Expression):
	callee: DesignatorExpression
	arguments: List[Expression]


@dataclass
class AssignmentStatement(Statement):
	target: DesignatorExpression
	value: Expression


@dataclass
class CallStatement(Statement):
	invocation: CallExpression


@dataclass
class IfStatement(Statement):
	condition: Expression
	then_branch: Statement
	else_branch: Optional[Statement]


@dataclass
class WhileStatement(Statement):
	condition: Expression
	body: Statement


@dataclass
class ReturnStatement(Statement):
	value: Optional[Expression]


@dataclass
class ReadStatement(Statement):
	target: DesignatorExpression


@dataclass
class PrintStatement(Statement):
	value: Expression
	width: Optional[int]


@dataclass
class LiteralExpression(Expression):
	value: Any
	literal_token: Token


@dataclass
class UnaryExpression(Expression):
	operator: TokenKind
	operand: Expression


@dataclass
class BinaryExpression(Expression):
	left: Expression
	operator: TokenKind
	right: Expression


@dataclass
class NewExpression(Expression):
	type_ref: TypeRef
	size: Optional[Expression]


@dataclass
class ProgramNode(ASTNode):
	name: str
	consts: List[ConstDecl]
	vars: List[VarDecl]
	classes: List[ClassDecl]
	methods: List["MethodDecl"]


@dataclass
class MethodDecl(ASTNode):
	return_type: TypeRef
	name: str
	parameters: List[Parameter]
	locals: List[VarDecl]
	body: BlockStatement


# ---------------------------------------------------------------------------
# Parser


class Parser:
	def __init__(self, tokens: List[Token], diagnostics: DiagnosticEngine) -> None:
		self.tokens = tokens
		self.diagnostics = diagnostics
		self.index = 0

	def parse_program(self) -> Optional[ProgramNode]:
		if not self._match(TokenKind.PROGRAM):
			self._error("A MicroJava program must start with the keyword 'program'.")
			return None
		name_token = self._expect(TokenKind.IDENT, "Expected program identifier after 'program'.")
		consts: List[ConstDecl] = []
		vars_: List[VarDecl] = []
		classes: List[ClassDecl] = []
		methods: List[MethodDecl] = []
		while not self._check(TokenKind.LBRACE) and not self._is_at_end():
			if self._match(TokenKind.FINAL):
				consts.append(self._parse_const())
			elif self._check(TokenKind.CLASS):
				classes.append(self._parse_class())
			else:
				vars_.append(self._parse_var_decl())
		lbrace = self._expect(TokenKind.LBRACE, "Expected '{' before method declarations.")
		while not self._check(TokenKind.RBRACE) and not self._is_at_end():
			methods.append(self._parse_method())
		rbrace = self._expect(TokenKind.RBRACE, "Expected closing '}' after methods.")
		span = combine_span(name_token.span, rbrace.span if rbrace else lbrace.span if lbrace else name_token.span)
		return ProgramNode(span=span, name=name_token.lexeme if name_token else "", consts=consts, vars=vars_, classes=classes, methods=methods)

	def _parse_const(self) -> ConstDecl:
		type_ref = self._parse_type()
		name_token = self._expect(TokenKind.IDENT, "Expected constant name.")
		self._expect(TokenKind.ASSIGN, "Expected '=' in constant declaration.")
		value_token = self._advance_token()
		if value_token.kind not in {TokenKind.NUMBER, TokenKind.CHAR_CONST}:
			self._error("Constant value must be a number or a character literal.", value_token)
		self._expect(TokenKind.SEMI, "Missing ';' after constant declaration.")
		span = combine_span(type_ref.span, value_token.span)
		return ConstDecl(span=span, type_ref=type_ref, name=name_token.lexeme if name_token else "", value_token=value_token)

	def _parse_var_decl(self) -> VarDecl:
		type_ref = self._parse_type()
		names: List[str] = []
		while True:
			ident = self._expect(TokenKind.IDENT, "Expected variable name.")
			if ident:
				names.append(ident.lexeme)
			if not self._match(TokenKind.COMMA):
				break
		semi = self._expect(TokenKind.SEMI, "Missing ';' after variable declaration.")
		end_span = semi.span if semi else type_ref.span
		span = combine_span(type_ref.span, end_span)
		return VarDecl(span=span, type_ref=type_ref, names=names)

	def _parse_class(self) -> ClassDecl:
		class_token = self._expect(TokenKind.CLASS, "Expected 'class'.")
		name_token = self._expect(TokenKind.IDENT, "Class name expected.")
		lbrace = self._expect(TokenKind.LBRACE, "Expected '{' after class name.")
		fields: List[ClassField] = []
		while not self._check(TokenKind.RBRACE) and not self._is_at_end():
			type_ref = self._parse_type()
			# Parse comma-separated field names
			while True:
				field_name = self._expect(TokenKind.IDENT, "Expected field name.")
				if field_name:
					field_span = combine_span(type_ref.span, field_name.span)
					fields.append(ClassField(span=field_span, type_ref=type_ref, name=field_name.lexeme))
				if not self._match(TokenKind.COMMA):
					break
			semi = self._expect(TokenKind.SEMI, "Missing ';' after field declaration.")
		rbrace = self._expect(TokenKind.RBRACE, "Expected '}' after class declaration.")
		span = combine_span(class_token.span, rbrace.span if rbrace else lbrace.span if lbrace else class_token.span)
		return ClassDecl(span=span, name=name_token.lexeme if name_token else "", fields=fields)

	def _parse_method(self) -> MethodDecl:
		if self._match(TokenKind.VOID):
			return_type = TypeRef(name="void", is_array=False, span=self._previous().span)
		else:
			return_type = self._parse_type()
		name_token = self._expect(TokenKind.IDENT, "Expected method name.")
		self._expect(TokenKind.LPAREN, "Expected '(' after method name.")
		parameters: List[Parameter] = []
		if not self._check(TokenKind.RPAREN):
			parameters.append(self._parse_parameter())
			while self._match(TokenKind.COMMA):
				parameters.append(self._parse_parameter())
		self._expect(TokenKind.RPAREN, "Expected ')' after parameters.")
		locals_: List[VarDecl] = []
		while self._check_type_start():
			locals_.append(self._parse_var_decl())
		body = self._parse_block()
		span = combine_span(return_type.span, body.span)
		return MethodDecl(span=span, return_type=return_type, name=name_token.lexeme if name_token else "", parameters=parameters, locals=locals_, body=body)

	def _parse_parameter(self) -> Parameter:
		type_ref = self._parse_type()
		name_token = self._expect(TokenKind.IDENT, "Expected parameter name.")
		span = combine_span(type_ref.span, name_token.span if name_token else type_ref.span)
		return Parameter(span=span, type_ref=type_ref, name=name_token.lexeme if name_token else "")

	def _parse_type(self) -> TypeRef:
		token = self._advance_token()
		if token.kind != TokenKind.IDENT and token.kind not in {TokenKind.INT, TokenKind.CHAR}:
			self._error("Expected type name.", token)
		is_array = False
		span = token.span
		# Array types in MicroJava are written as `Type[]`.
		# IMPORTANT: In `new Type[expr]`, the `[` belongs to the *allocation*, not the type.
		# So we only treat `[` as part of the type if it's immediately followed by `]`.
		if self._match(TokenKind.LBRACKET):
			if self._check(TokenKind.RBRACKET):
				self._advance_token()
				is_array = True
				span = combine_span(span, self._previous().span)
			else:
				# Not an array type: rewind so the caller (e.g. `new`) can parse `[expr]`.
				self._rewind()
		return TypeRef(name=token.lexeme, is_array=is_array, span=span)

	def _parse_block(self) -> BlockStatement:
		lbrace = self._expect(TokenKind.LBRACE, "Expected '{' to start a block.")
		statements: List[Statement] = []
		while not self._check(TokenKind.RBRACE) and not self._is_at_end():
			prev_index = self.index
			statements.append(self._parse_statement())
			# Error recovery: if we didn't advance, skip the problematic token
			if self.index == prev_index and not self._is_at_end():
				self._error(f"Unexpected token '{self.tokens[self.index].lexeme}', skipping.", self.tokens[self.index])
				self.index += 1
		rbrace = self._expect(TokenKind.RBRACE, "Expected '}' to close block.")
		span = combine_span(lbrace.span if lbrace else statements[0].span, rbrace.span if rbrace else lbrace.span if lbrace else statements[-1].span)
		return BlockStatement(span=span, statements=statements)

	def _parse_statement(self) -> Statement:
		if self._match(TokenKind.IF):
			self._expect(TokenKind.LPAREN, "Expected '(' after 'if'.")
			condition = self._parse_expression()
			self._expect(TokenKind.RPAREN, "Expected ')' after condition.")
			then_branch = self._parse_statement()
			else_branch = self._parse_statement() if self._match(TokenKind.ELSE) else None
			span = combine_span(condition.span, (else_branch or then_branch).span)
			return IfStatement(span=span, condition=condition, then_branch=then_branch, else_branch=else_branch)
		if self._match(TokenKind.WHILE):
			self._expect(TokenKind.LPAREN, "Expected '(' after 'while'.")
			condition = self._parse_expression()
			self._expect(TokenKind.RPAREN, "Expected ')' after condition.")
			body = self._parse_statement()
			span = combine_span(condition.span, body.span)
			return WhileStatement(span=span, condition=condition, body=body)
		if self._match(TokenKind.RETURN):
			if not self._check(TokenKind.SEMI):
				value = self._parse_expression()
			else:
				value = None
			semi = self._expect(TokenKind.SEMI, "Expected ';' after return statement.")
			span = combine_span(self._previous().span, semi.span if semi else self._previous().span)
			return ReturnStatement(span=span, value=value)
		if self._match(TokenKind.READ):
			self._expect(TokenKind.LPAREN, "Expected '(' after 'read'.")
			designator = self._parse_designator()
			self._expect(TokenKind.RPAREN, "Expected ')' after read target.")
			semi = self._expect(TokenKind.SEMI, "Expected ';' after read statement.")
			span = combine_span(designator.span, semi.span if semi else designator.span)
			return ReadStatement(span=span, target=designator)
		if self._match(TokenKind.PRINT):
			self._expect(TokenKind.LPAREN, "Expected '(' after 'print'.")
			value = self._parse_expression()
			width = None
			if self._match(TokenKind.COMMA):
				width_token = self._expect(TokenKind.NUMBER, "Print width must be a number.")
				width = width_token.value if width_token else None
			self._expect(TokenKind.RPAREN, "Expected ')' after print arguments.")
			semi = self._expect(TokenKind.SEMI, "Expected ';' after print statement.")
			span = combine_span(value.span, semi.span if semi else value.span)
			return PrintStatement(span=span, value=value, width=width)
		if self._match(TokenKind.LBRACE):
			self._rewind()
			return self._parse_block()
		if self._match(TokenKind.SEMI):
			return EmptyStatement(span=self._previous().span)
		# Check for variable declaration inside block (common mistake)
		if self._check(TokenKind.IDENT) or self._check(TokenKind.INT) or self._check(TokenKind.CHAR):
			# Look ahead to see if this looks like a type declaration
			saved_index = self.index
			if self._check_type_start():
				type_token = self._advance_token()
				if self._check(TokenKind.IDENT) and not self._check_ahead_for_assignment():
					# This looks like a variable declaration
					self.index = saved_index
					self._error("Variable declarations must appear before the method body '{}', not inside it.", type_token)
					# Try to skip the invalid declaration
					while not self._check(TokenKind.SEMI) and not self._is_at_end() and not self._check(TokenKind.RBRACE):
						self._advance_token()
					if self._check(TokenKind.SEMI):
						self._advance_token()
					return EmptyStatement(span=type_token.span)
			self.index = saved_index
		# Check if we can parse a designator (must start with IDENT)
		if not self._check(TokenKind.IDENT):
			token = self._advance_token() if not self._is_at_end() else self._previous()
			self._error(f"Invalid statement start: '{token.lexeme}'", token)
			return EmptyStatement(span=token.span)
		designator = self._parse_designator()
		if self._match(TokenKind.ASSIGN):
			value = self._parse_expression()
			semi = self._expect(TokenKind.SEMI, "Expected ';' after assignment.")
			span = combine_span(designator.span, semi.span if semi else value.span)
			return AssignmentStatement(span=span, target=designator, value=value)
		if self._check(TokenKind.LPAREN):
			call = self._finish_call(designator)
			semi = self._expect(TokenKind.SEMI, "Expected ';' after call.")
			span = combine_span(designator.span, semi.span if semi else call.span)
			return CallStatement(span=span, invocation=call)
		self._error("Invalid statement.")
		# Consume semicolon if present to avoid getting stuck
		if self._check(TokenKind.SEMI):
			self._advance_token()
		return EmptyStatement(span=designator.span)

	def _finish_call(self, designator: DesignatorExpression) -> CallExpression:
		self._expect(TokenKind.LPAREN, "Expected '(' to start argument list.")
		args: List[Expression] = []
		if not self._check(TokenKind.RPAREN):
			args.append(self._parse_expression())
			while self._match(TokenKind.COMMA):
				args.append(self._parse_expression())
		rparen = self._expect(TokenKind.RPAREN, "Expected ')' after arguments.")
		span = combine_span(designator.span, rparen.span if rparen else designator.span)
		return CallExpression(span=span, callee=designator, arguments=args)

	def _parse_designator(self) -> DesignatorExpression:
		name_token = self._expect(TokenKind.IDENT, "Expected identifier.")
		parts: List[DesignatorPart] = []
		max_depth = 100  # Prevent infinite loop in designator chain
		depth = 0
		while depth < max_depth:
			depth += 1
			if self._match(TokenKind.DOT):
				field_token = self._expect(TokenKind.IDENT, "Expected field name after '.'.")
				if field_token:
					parts.append(FieldAccess(span=field_token.span, name=field_token.lexeme))
				else:
					break  # Stop if field name is missing
			elif self._match(TokenKind.LBRACKET):
				expr = self._parse_expression()
				rbracket = self._expect(TokenKind.RBRACKET, "Expected ']' after index expression.")
				span = combine_span(expr.span, rbracket.span if rbracket else expr.span)
				parts.append(IndexAccess(span=span, expression=expr))
			else:
				break
		if name_token:
			span = name_token.span
		elif parts:
			span = parts[-1].span
		else:
			zero = Position(0, 0, 0)
			span = Span(zero, zero)
		if parts:
			span = combine_span(span, parts[-1].span)
		return DesignatorExpression(span=span, name=name_token.lexeme if name_token else "", parts=parts)

	def _parse_expression(self) -> Expression:
		return self._parse_relational()

	def _parse_relational(self) -> Expression:
		expr = self._parse_additive()
		while self._check(TokenKind.EQ) or self._check(TokenKind.NEQ) or self._check(TokenKind.GT) or self._check(TokenKind.GTE) or self._check(TokenKind.LT) or self._check(TokenKind.LTE):
			operator = self._advance_token()
			right = self._parse_additive()
			expr = BinaryExpression(span=combine_span(expr.span, right.span), left=expr, operator=operator.kind, right=right)
		return expr

	def _parse_additive(self) -> Expression:
		expr = self._parse_term()
		while self._check(TokenKind.PLUS) or self._check(TokenKind.MINUS):
			operator = self._advance_token()
			right = self._parse_term()
			expr = BinaryExpression(span=combine_span(expr.span, right.span), left=expr, operator=operator.kind, right=right)
		return expr

	def _parse_term(self) -> Expression:
		expr = self._parse_factor()
		while self._check(TokenKind.STAR) or self._check(TokenKind.SLASH) or self._check(TokenKind.PERCENT):
			operator = self._advance_token()
			right = self._parse_factor()
			expr = BinaryExpression(span=combine_span(expr.span, right.span), left=expr, operator=operator.kind, right=right)
		return expr

	def _parse_factor(self) -> Expression:
		if self._match(TokenKind.MINUS):
			operand = self._parse_factor()
			return UnaryExpression(span=combine_span(self._previous().span, operand.span), operator=TokenKind.MINUS, operand=operand)
		token = self._advance_token()
		if token.kind == TokenKind.NUMBER or token.kind == TokenKind.CHAR_CONST:
			return LiteralExpression(span=token.span, value=token.value, literal_token=token)
		if token.kind == TokenKind.IDENT:
			self._rewind()
			designator = self._parse_designator()
			if self._check(TokenKind.LPAREN):
				return self._finish_call(designator)
			return designator
		if token.kind == TokenKind.NEW:
			type_ref = self._parse_type()
			size = None
			if self._match(TokenKind.LBRACKET):
				size = self._parse_expression()
				self._expect(TokenKind.RBRACKET, "Expected ']' after array size.")
			span = combine_span(token.span, type_ref.span if not size else size.span)
			return NewExpression(span=span, type_ref=type_ref, size=size)
		if token.kind == TokenKind.LPAREN:
			expr = self._parse_expression()
			self._expect(TokenKind.RPAREN, "Missing closing ')' in expression.")
			return expr
		self._error("Invalid expression.", token)
		return LiteralExpression(span=token.span, value=None, literal_token=token)

	def _check_type_start(self) -> bool:
		return self._check(TokenKind.IDENT) or self._check(TokenKind.INT) or self._check(TokenKind.CHAR)

	def _check_ahead_for_assignment(self) -> bool:
		"""Check if the next tokens look like an assignment (ident = ...)"""
		if self.index + 1 >= len(self.tokens):
			return False
		next_token = self.tokens[self.index + 1]
		return next_token.kind in {TokenKind.ASSIGN, TokenKind.LPAREN, TokenKind.DOT, TokenKind.LBRACKET}

	# Utility parsing helpers -------------------------------------------------

	def _match(self, kind: TokenKind) -> bool:
		if self._check(kind):
			self.index += 1
			return True
		return False

	def _check(self, kind: TokenKind) -> bool:
		if self._is_at_end():
			return False
		return self.tokens[self.index].kind == kind

	def _advance_token(self) -> Token:
		if not self._is_at_end():
			self.index += 1
		return self.tokens[self.index - 1]

	def _expect(self, kind: TokenKind, message: str) -> Optional[Token]:
		if self._check(kind):
			return self._advance_token()
		if not self._is_at_end():
			self._error(message, self.tokens[self.index], hint=self._hint_for_expect(kind, self.tokens[self.index]))
		else:
			self._error(message, hint=self._hint_for_expect(kind, None))
		return None

	def _previous(self) -> Token:
		return self.tokens[self.index - 1]

	def _rewind(self) -> None:
		self.index -= 1

	def _is_at_end(self) -> bool:
		return self.tokens[self.index].kind == TokenKind.EOF

	def _error(self, message: str, token: Optional[Token] = None, hint: Optional[str] = None) -> None:
		span = token.span if token else None
		self.diagnostics.report(Severity.ERROR, message, span, hint=hint)

	def _hint_for_expect(self, expected: TokenKind, got: Optional[Token]) -> Optional[str]:
		# Friendly, “LinkedIn demo” style hints for the most common MicroJava mistakes.
		if expected == TokenKind.SEMI:
			return "Statements/declarations must end with ';'."
		if expected == TokenKind.LBRACE:
			return "Blocks start with '{'. If this is a method, ensure the header is followed by a block."
		if expected == TokenKind.RBRACE:
			return "Blocks end with '}'. Check for a missing closing brace or an extra '{' earlier."
		if expected == TokenKind.RPAREN:
			return "Missing ')'. Check function calls/conditions like: if (cond) { ... }"
		if expected == TokenKind.LPAREN:
			return "Missing '('. Calls/conditions require parentheses like: read(x); while (i < n) { ... }"
		if expected == TokenKind.IDENT and got and got.kind in KEYWORDS.values():
			return "Identifiers can’t be keywords. Rename it (e.g., 'className' instead of 'class')."
		return None


# ---------------------------------------------------------------------------
# Symbol table and type system


class TypeKind(Enum):
	INT = auto()
	CHAR = auto()
	VOID = auto()
	CLASS = auto()
	ARRAY = auto()
	NULL = auto()
	UNKNOWN = auto()


@dataclass(eq=False)
class TypeInfo:
	kind: TypeKind
	name: Optional[str] = None
	element: Optional["TypeInfo"] = None

	def __eq__(self, other: object) -> bool:
		if not isinstance(other, TypeInfo):
			return False
		if self.kind != other.kind:
			return False
		if self.kind == TypeKind.ARRAY:
			return self.element == other.element
		return self.name == other.name

	def __hash__(self) -> int:
		if self.kind == TypeKind.ARRAY:
			return hash((self.kind, self.element))
		return hash((self.kind, self.name))

	def is_reference(self) -> bool:
		return self.kind in {TypeKind.CLASS, TypeKind.ARRAY}

	def assignment_compatible(self, src: "TypeInfo") -> bool:
		if self == src:
			return True
		if self.is_reference() and src.kind == TypeKind.NULL:
			return True
		return False


INT_TYPE = TypeInfo(TypeKind.INT, name="int")
CHAR_TYPE = TypeInfo(TypeKind.CHAR, name="char")
VOID_TYPE = TypeInfo(TypeKind.VOID, name="void")
NULL_TYPE = TypeInfo(TypeKind.NULL, name="null")
UNKNOWN_TYPE = TypeInfo(TypeKind.UNKNOWN, name="<unknown>")


def array_of(base: TypeInfo) -> TypeInfo:
	return TypeInfo(TypeKind.ARRAY, element=base, name=f"{base.name}[]")


class SymbolKind(Enum):
	VARIABLE = auto()
	CONSTANT = auto()
	FIELD = auto()
	CLASS = auto()
	METHOD = auto()
	PARAMETER = auto()
	TYPE = auto()


@dataclass
class Symbol:
	name: str
	kind: SymbolKind
	type_info: TypeInfo
	node: Optional[ASTNode] = None
	attributes: Dict[str, Any] = field(default_factory=dict)


class Scope:
	def __init__(self, name: str, parent: Optional["Scope"] = None) -> None:
		self.name = name
		self.parent = parent
		self.symbols: Dict[str, Symbol] = {}

	def define(self, symbol: Symbol, diagnostics: DiagnosticEngine) -> None:
		if symbol.name in self.symbols:
			diagnostics.report(Severity.ERROR, f"Duplicate declaration of '{symbol.name}'.", symbol.node.span if symbol.node else None)
			return
		self.symbols[symbol.name] = symbol

	def resolve(self, name: str) -> Optional[Symbol]:
		scope: Optional[Scope] = self
		while scope:
			if name in scope.symbols:
				return scope.symbols[name]
			scope = scope.parent
		return None


@dataclass
class SymbolTable:
	global_scope: Scope
	class_scopes: Dict[str, Scope]
	method_scopes: Dict[str, Scope]


class SymbolTableBuilder:
	def __init__(self, diagnostics: DiagnosticEngine) -> None:
		self.diagnostics = diagnostics
		self.universe = Scope("universe", None)
		self._install_builtins()

	def _install_builtins(self) -> None:
		for builtin in [Symbol("int", SymbolKind.TYPE, INT_TYPE), Symbol("char", SymbolKind.TYPE, CHAR_TYPE), Symbol("void", SymbolKind.TYPE, VOID_TYPE)]:
			self.universe.symbols[builtin.name] = builtin
		for name, return_type, param_type in [("ord", INT_TYPE, CHAR_TYPE), ("chr", CHAR_TYPE, INT_TYPE), ("len", INT_TYPE, array_of(UNKNOWN_TYPE))]:
			sym = Symbol(name, SymbolKind.METHOD, return_type)
			sym.attributes["parameters"] = [("value", param_type)]
			self.universe.symbols[name] = sym
		self.universe.symbols["null"] = Symbol("null", SymbolKind.CONSTANT, NULL_TYPE)

	def build(self, program: Optional[ProgramNode]) -> SymbolTable:
		global_scope = Scope("global", self.universe)
		class_scopes: Dict[str, Scope] = {}
		method_scopes: Dict[str, Scope] = {}
		if not program:
			return SymbolTable(global_scope, class_scopes, method_scopes)
		for const in program.consts:
			type_info = self._resolve_type(const.type_ref, global_scope)
			symbol = Symbol(const.name, SymbolKind.CONSTANT, type_info, const)
			global_scope.define(symbol, self.diagnostics)
		for var in program.vars:
			type_info = self._resolve_type(var.type_ref, global_scope)
			for name in var.names:
				global_scope.define(Symbol(name, SymbolKind.VARIABLE, type_info, var), self.diagnostics)
		for class_decl in program.classes:
			class_type = TypeInfo(TypeKind.CLASS, name=class_decl.name)
			class_symbol = Symbol(class_decl.name, SymbolKind.CLASS, class_type, class_decl)
			global_scope.define(class_symbol, self.diagnostics)
			scope = Scope(class_decl.name, global_scope)
			for field in class_decl.fields:
				field_type = self._resolve_type(field.type_ref, global_scope)
				scope.define(Symbol(field.name, SymbolKind.FIELD, field_type, field), self.diagnostics)
			class_scopes[class_decl.name] = scope
		for method in program.methods:
			return_type = self._resolve_type(method.return_type, global_scope)
			method_symbol = Symbol(method.name, SymbolKind.METHOD, return_type, method)
			params_info = []
			for param in method.parameters:
				param_type = self._resolve_type(param.type_ref, global_scope)
				params_info.append((param.name, param_type))
			method_symbol.attributes["parameters"] = params_info
			global_scope.define(method_symbol, self.diagnostics)
			scope = Scope(method.name, global_scope)
			for param_name, param_type in params_info:
				scope.define(Symbol(param_name, SymbolKind.PARAMETER, param_type), self.diagnostics)
			for local in method.locals:
				local_type = self._resolve_type(local.type_ref, global_scope)
				for local_name in local.names:
					scope.define(Symbol(local_name, SymbolKind.VARIABLE, local_type, local), self.diagnostics)
			method_scopes[method.name] = scope
		return SymbolTable(global_scope, class_scopes, method_scopes)

	def _resolve_type(self, type_ref: TypeRef, scope: Scope) -> TypeInfo:
		base_symbol = scope.resolve(type_ref.name)
		base_type = INT_TYPE
		if base_symbol and base_symbol.type_info:
			base_type = base_symbol.type_info
		elif type_ref.name == "int":
			base_type = INT_TYPE
		elif type_ref.name == "char":
			base_type = CHAR_TYPE
		elif type_ref.name == "void":
			base_type = VOID_TYPE
		else:
			self.diagnostics.report(Severity.ERROR, f"Unknown type '{type_ref.name}'.", type_ref.span)
			base_type = UNKNOWN_TYPE
		return array_of(base_type) if type_ref.is_array else base_type


# ---------------------------------------------------------------------------
# Semantic analyzer


class SemanticAnalyzer:
	def __init__(self, diagnostics: DiagnosticEngine, symbols: SymbolTable) -> None:
		self.diagnostics = diagnostics
		self.symbols = symbols

	def analyze(self, program: Optional[ProgramNode]) -> None:
		if not program:
			return
		main = self.symbols.global_scope.symbols.get("main")
		if not main or main.kind != SymbolKind.METHOD:
			self.diagnostics.report(Severity.ERROR, "Program must declare a parameterless void main method.", program.span)
		else:
			params = main.attributes.get("parameters", [])
			if params or main.type_info != VOID_TYPE:
				self.diagnostics.report(Severity.ERROR, "main must be declared as 'void main()'.", main.node.span if main.node else program.span)
		for method in program.methods:
			scope = self.symbols.method_scopes.get(method.name)
			self._analyze_block(method.body, scope, method.return_type)

	def _analyze_block(self, block: BlockStatement, scope: Optional[Scope], return_type: TypeRef) -> None:
		for stmt in block.statements:
			self._analyze_statement(stmt, scope, return_type)

	def _analyze_statement(self, stmt: Statement, scope: Optional[Scope], return_type: TypeRef) -> None:
		if isinstance(stmt, BlockStatement):
			self._analyze_block(stmt, scope, return_type)
		elif isinstance(stmt, AssignmentStatement):
			target_type = self._resolve_designator(stmt.target, scope)
			value_type = self._analyze_expression(stmt.value, scope)
			if not target_type.assignment_compatible(value_type):
				self.diagnostics.report(Severity.ERROR, "Type mismatch in assignment.", stmt.span)
		elif isinstance(stmt, CallStatement):
			self._analyze_call(stmt.invocation, scope)
		elif isinstance(stmt, IfStatement):
			cond_type = self._analyze_expression(stmt.condition, scope)
			if cond_type != INT_TYPE:
				self.diagnostics.report(Severity.WARNING, "Conditions should be of type int (0 or non-zero).", stmt.condition.span)
			self._analyze_statement(stmt.then_branch, scope, return_type)
			if stmt.else_branch:
				self._analyze_statement(stmt.else_branch, scope, return_type)
		elif isinstance(stmt, WhileStatement):
			cond_type = self._analyze_expression(stmt.condition, scope)
			if cond_type != INT_TYPE:
				self.diagnostics.report(Severity.WARNING, "Loop condition should be of type int.", stmt.condition.span)
			self._analyze_statement(stmt.body, scope, return_type)
		elif isinstance(stmt, ReturnStatement):
			expected = self._resolve_type_ref(return_type)
			value_type = VOID_TYPE if stmt.value is None else self._analyze_expression(stmt.value, scope)
			if stmt.value is None and expected != VOID_TYPE:
				self.diagnostics.report(Severity.ERROR, "Return statement missing expression.", stmt.span)
			elif stmt.value is not None and not expected.assignment_compatible(value_type):
				self.diagnostics.report(Severity.ERROR, "Return type mismatch.", stmt.span)
		elif isinstance(stmt, ReadStatement):
			target_type = self._resolve_designator(stmt.target, scope)
			if target_type != INT_TYPE and target_type != CHAR_TYPE:
				self.diagnostics.report(Severity.ERROR, "read() target must be int or char.", stmt.span)
		elif isinstance(stmt, PrintStatement):
			value_type = self._analyze_expression(stmt.value, scope)
			if value_type != INT_TYPE and value_type != CHAR_TYPE:
				self.diagnostics.report(Severity.ERROR, "print() argument must be int or char.", stmt.span)

	def _analyze_expression(self, expr: Expression, scope: Optional[Scope]) -> TypeInfo:
		if isinstance(expr, LiteralExpression):
			if expr.literal_token.kind == TokenKind.NUMBER:
				return INT_TYPE
			if expr.literal_token.kind == TokenKind.CHAR_CONST:
				return CHAR_TYPE
			return UNKNOWN_TYPE
		if isinstance(expr, DesignatorExpression):
			return self._resolve_designator(expr, scope)
		if isinstance(expr, UnaryExpression):
			operand_type = self._analyze_expression(expr.operand, scope)
			if operand_type != INT_TYPE:
				self.diagnostics.report(Severity.ERROR, "Unary '-' expects type int.", expr.span)
			return INT_TYPE
		if isinstance(expr, BinaryExpression):
			left = self._analyze_expression(expr.left, scope)
			right = self._analyze_expression(expr.right, scope)
			if expr.operator in {TokenKind.PLUS, TokenKind.MINUS, TokenKind.STAR, TokenKind.SLASH, TokenKind.PERCENT}:
				if left != INT_TYPE or right != INT_TYPE:
					self.diagnostics.report(Severity.ERROR, "Arithmetic operations require int operands.", expr.span)
				return INT_TYPE
			if left != right:
				self.diagnostics.report(Severity.ERROR, "Operands must be of the same type.", expr.span)
			return INT_TYPE
		if isinstance(expr, CallExpression):
			symbol = self._resolve_symbol(expr.callee.name, scope)
			if not symbol or symbol.kind != SymbolKind.METHOD:
				self.diagnostics.report(Severity.ERROR, f"'{expr.callee.name}' is not a method.", expr.span)
				return UNKNOWN_TYPE
			params = symbol.attributes.get("parameters", [])
			if len(params) != len(expr.arguments):
				self.diagnostics.report(Severity.ERROR, "Argument count mismatch.", expr.span)
			for (arg_sym_name, param_type), arg_expr in zip(params, expr.arguments):
				arg_type = self._analyze_expression(arg_expr, scope)
				if not param_type.assignment_compatible(arg_type):
					self.diagnostics.report(Severity.ERROR, f"Argument for '{arg_sym_name}' has incompatible type.", arg_expr.span)
			return symbol.type_info
		if isinstance(expr, NewExpression):
			base_type = self._resolve_type_ref(expr.type_ref)
			if expr.size:
				size_type = self._analyze_expression(expr.size, scope)
				if size_type != INT_TYPE:
					self.diagnostics.report(Severity.ERROR, "Array size must be int.", expr.size.span)
				return array_of(base_type)
			if base_type.kind != TypeKind.CLASS:
				self.diagnostics.report(Severity.ERROR, "Only classes can be instantiated without size.", expr.span)
			return base_type
		return UNKNOWN_TYPE

	def _resolve_designator(self, designator: DesignatorExpression, scope: Optional[Scope]) -> TypeInfo:
		symbol = self._resolve_symbol(designator.name, scope)
		if not symbol:
			self.diagnostics.report(Severity.ERROR, f"Undeclared identifier '{designator.name}'.", designator.span)
			return UNKNOWN_TYPE
		current_type = symbol.type_info
		for part in designator.parts:
			if isinstance(part, FieldAccess):
				if current_type.kind != TypeKind.CLASS:
					self.diagnostics.report(Severity.ERROR, "Field access on non-class type.", part.span)
					return UNKNOWN_TYPE
				class_scope = self.symbols.class_scopes.get(current_type.name or "")
				if not class_scope:
					self.diagnostics.report(Severity.ERROR, f"Unknown class '{current_type.name}'.", part.span)
					return UNKNOWN_TYPE
				field_symbol = class_scope.resolve(part.name)
				if not field_symbol:
					self.diagnostics.report(Severity.ERROR, f"Class '{current_type.name}' has no field '{part.name}'.", part.span)
					return UNKNOWN_TYPE
				current_type = field_symbol.type_info
			elif isinstance(part, IndexAccess):
				if current_type.kind != TypeKind.ARRAY:
					self.diagnostics.report(Severity.ERROR, "Indexing non-array type.", part.span)
					return UNKNOWN_TYPE
				index_type = self._analyze_expression(part.expression, scope)
				if index_type != INT_TYPE:
					self.diagnostics.report(Severity.ERROR, "Array index must be int.", part.expression.span)
				current_type = current_type.element or UNKNOWN_TYPE
		return current_type

	def _resolve_symbol(self, name: str, scope: Optional[Scope]) -> Optional[Symbol]:
		if scope:
			symbol = scope.resolve(name)
			if symbol:
				return symbol
		return self.symbols.global_scope.resolve(name)

	def _resolve_type_ref(self, type_ref: TypeRef) -> TypeInfo:
		base_symbol = self.symbols.global_scope.resolve(type_ref.name)
		base_type = base_symbol.type_info if base_symbol else UNKNOWN_TYPE
		return array_of(base_type) if type_ref.is_array else base_type


# ---------------------------------------------------------------------------
# Compilation pipeline


@dataclass
class CompilationArtifacts:
	tokens: List[Token]
	ast: Optional[ProgramNode]
	symbols: SymbolTable
	diagnostics: List[Diagnostic]
	duration_ms: float


class MiniCompilerEngine:
	def compile(self, source: str) -> CompilationArtifacts:
		diagnostics = DiagnosticEngine()
		start = time.perf_counter()
		lexer = Lexer(source, diagnostics)
		tokens = lexer.tokenize()
		parser = Parser(tokens, diagnostics)
		ast = parser.parse_program()
		sym_builder = SymbolTableBuilder(diagnostics)
		symbols = sym_builder.build(ast)
		semantic = SemanticAnalyzer(diagnostics, symbols)
		semantic.analyze(ast)
		duration_ms = (time.perf_counter() - start) * 1000
		return CompilationArtifacts(tokens=tokens, ast=ast, symbols=symbols, diagnostics=diagnostics.items, duration_ms=duration_ms)


# ---------------------------------------------------------------------------
# GUI components


if tk is not None:
	class CodeEditor(tk.Frame):
		def __init__(self, master: "tk.Widget") -> None:
			super().__init__(master)
			self.text = tk.Text(
				self,
				wrap="none",
				font=("Consolas", 11),
				undo=True,
				background="#1e1e1e",
				foreground="#d4d4d4",
				insertbackground="#d4d4d4",
				selectbackground="#264f78",
				selectforeground="#ffffff",
			)
			self.line_numbers = tk.Text(self, width=4, state="disabled", background="#1f1f1f", foreground="#8f8f8f", font=("Consolas", 10))
			self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self._on_scroll)
			self.h_scroll = ttk.Scrollbar(self, orient="horizontal", command=self.text.xview)
			self.text.configure(yscrollcommand=self._on_text_scroll)
			self.text.configure(xscrollcommand=self.h_scroll.set)
			self.line_numbers.pack(side="left", fill="y")
			self.text.pack(side="left", fill="both", expand=True)
			self.v_scroll.pack(side="right", fill="y")
			self.h_scroll.pack(side="bottom", fill="x")
			self.text.bind("<KeyRelease>", self._update_line_numbers)
			self.text.bind("<MouseWheel>", lambda e: self._update_line_numbers())
			self._setup_tags()

		def _setup_tags(self) -> None:
			self.text.tag_configure("keyword", foreground="#569cd6")
			self.text.tag_configure("literal", foreground="#ce9178")
			self.text.tag_configure("identifier", foreground="#dcdcaa")
			self.text.tag_configure("diag_error", underline=1, foreground="#f14c4c")
			self.text.tag_configure("diag_warning", underline=1, foreground="#cca700")
			self.text.tag_configure("diag_info", underline=1, foreground="#4fc1ff")
			self.text.tag_configure("diag_active", background="#2a2d2e")

		def _on_text_scroll(self, *args) -> None:
			self.v_scroll.set(*args)
			self.line_numbers.yview_moveto(args[0])

		def _on_scroll(self, *args) -> None:
			self.text.yview(*args)
			self.line_numbers.yview(*args)
			self._update_line_numbers()

		def _update_line_numbers(self, event: Optional["tk.Event"] = None) -> None:
			self.line_numbers.configure(state="normal")
			self.line_numbers.delete("1.0", tk.END)
			line_count = int(self.text.index("end-1c").split(".")[0])
			numbers = "\n".join(str(i).rjust(3) for i in range(1, line_count + 1))
			self.line_numbers.insert("1.0", numbers)
			self.line_numbers.configure(state="disabled")

		def get_text(self) -> str:
			return self.text.get("1.0", tk.END)

		def set_text(self, content: str) -> None:
			self.text.delete("1.0", tk.END)
			self.text.insert("1.0", content)
			self._update_line_numbers()

		def clear_highlights(self) -> None:
			for tag in ("keyword", "literal", "identifier"):
				self.text.tag_remove(tag, "1.0", tk.END)

		def apply_highlights(self, tokens: List[Token]) -> None:
			self.clear_highlights()
			for token in tokens:
				if token.kind in {TokenKind.EOF, TokenKind.UNKNOWN}:
					continue
				if token.kind in KEYWORDS.values():
					tag = "keyword"
				elif token.kind in {TokenKind.NUMBER, TokenKind.CHAR_CONST}:
					tag = "literal"
				elif token.kind == TokenKind.IDENT:
					tag = "identifier"
				else:
					continue
				start_index = f"{token.span.start.line}.{token.span.start.column - 1}"
				end_index = f"{token.span.end.line}.{max(0, token.span.end.column - 1)}"
				self.text.tag_add(tag, start_index, end_index)

		def clear_diagnostic_marks(self) -> None:
			for tag in ("diag_error", "diag_warning", "diag_info", "diag_active"):
				self.text.tag_remove(tag, "1.0", tk.END)

		def mark_span(self, span: Optional[Span], tag: str) -> None:
			if not span:
				return
			start_index = f"{span.start.line}.{max(0, span.start.column - 1)}"
			end_index = f"{span.end.line}.{max(0, span.end.column - 1)}"
			# Avoid zero-length ranges (Text won’t show tag); expand by 1 char when possible.
			if start_index == end_index:
				end_index = f"{span.end.line}.{max(0, span.end.column)}"
			self.text.tag_add(tag, start_index, end_index)

		def goto_span(self, span: Optional[Span]) -> None:
			if not span:
				return
			start_index = f"{span.start.line}.{max(0, span.start.column - 1)}"
			self.text.mark_set("insert", start_index)
			self.text.see(start_index)
			self.text.focus_set()


	class MiniCompilerApp(tk.Tk):
		def __init__(self) -> None:
			super().__init__()
			self.title("MicroJava Mini Compiler")
			self.geometry("1300x800")
			self.engine = MiniCompilerEngine()
			self._compile_after_id: Optional[str] = None
			self._diag_by_iid: Dict[str, Diagnostic] = {}
			self._ast_canvas_nodes: Dict[int, ASTNode] = {}
			self._overview_counts: Tuple[int, int, int] = (0, 0, 0)
			self._build_ui()
			self._load_sample()

	def _build_ui(self) -> None:
		style = ttk.Style(self)
		try:
			style.theme_use("clam")
		except tk.TclError:
			pass
		style.configure(".", font=("Segoe UI", 10))
		style.configure("Treeview", rowheight=22)

		self.columnconfigure(0, weight=1)
		self.rowconfigure(0, weight=0)  # toolbar
		self.rowconfigure(1, weight=1)  # main body
		toolbar = ttk.Frame(self)
		toolbar.grid(row=0, column=0, sticky="ew")
		compile_btn = ttk.Button(toolbar, text="Compile (Ctrl+Enter)", command=self._compile)
		compile_btn.pack(side="left", padx=4, pady=4)
		self.live_var = tk.BooleanVar(value=True)
		live_toggle = ttk.Checkbutton(toolbar, text="Live Compile", variable=self.live_var, command=self._schedule_live_compile)
		live_toggle.pack(side="left", padx=8)
		open_btn = ttk.Button(toolbar, text="Open", command=self._open_file)
		open_btn.pack(side="left", padx=4)
		save_btn = ttk.Button(toolbar, text="Save", command=self._save_file)
		save_btn.pack(side="left", padx=4)
		export_btn = ttk.Button(toolbar, text="Export Report", command=self._export_report)
		export_btn.pack(side="left", padx=4)
		export_md_btn = ttk.Button(toolbar, text="Export Markdown", command=self._export_markdown_report)
		export_md_btn.pack(side="left", padx=4)
		body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
		body.grid(row=1, column=0, sticky="nsew")
		editor_frame = ttk.Frame(body)
		editor_frame.pack(fill="both", expand=True)
		self.editor = CodeEditor(editor_frame)
		self.editor.pack(fill="both", expand=True)
		self.editor.text.bind("<KeyRelease>", lambda _e: self._schedule_live_compile())
		right_notebook = ttk.Notebook(body)
		body.add(editor_frame, weight=3)
		body.add(right_notebook, weight=2)

		overview_tab = ttk.Frame(right_notebook)
		self._build_overview_tab(overview_tab)
		right_notebook.add(overview_tab, text="Overview")

		diag_tab = ttk.Frame(right_notebook)
		self._build_diagnostics_tab(diag_tab)
		right_notebook.add(diag_tab, text="Diagnostics")

		self.token_tree = self._create_tree(right_notebook, ("Kind", "Lexeme", "Line"))
		right_notebook.add(self.token_tree, text="Tokens")
		self.ast_tree = self._create_tree(right_notebook, ("Node", "Details"))
		right_notebook.add(self.ast_tree, text="AST")
		ast_graph_tab = ttk.Frame(right_notebook)
		self._build_ast_graph_tab(ast_graph_tab)
		right_notebook.add(ast_graph_tab, text="AST Graph")
		self.symbol_tree = self._create_tree(right_notebook, ("Scope", "Symbol", "Type"))
		right_notebook.add(self.symbol_tree, text="Symbols")
		self.status = ttk.Label(self, text="Ready", anchor="w")
		self.status.grid(row=2, column=0, sticky="ew")
		self.bind("<Control-Return>", lambda _event: self._compile())

	def _build_overview_tab(self, parent: ttk.Frame) -> None:
		parent.columnconfigure(0, weight=1)
		header = ttk.Label(parent, text="Compilation Dashboard", font=("Segoe UI", 12, "bold"))
		header.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))
		self.overview_stats = ttk.Label(parent, text="Compile to see stats…", justify="left")
		self.overview_stats.grid(row=1, column=0, sticky="ew", padx=10)
		self.overview_canvas = tk.Canvas(parent, height=120, background="#1e1e1e", highlightthickness=0)
		self.overview_canvas.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
		parent.bind("<Configure>", lambda _e: self._redraw_overview_chart(*self._overview_counts))

	def _build_diagnostics_tab(self, parent: ttk.Frame) -> None:
		parent.rowconfigure(0, weight=3)
		parent.rowconfigure(1, weight=2)
		parent.columnconfigure(0, weight=1)

		self.diag_tree = self._create_tree(parent, ("Severity", "Message", "Location", "Hint"))
		self.diag_tree.grid(row=0, column=0, sticky="nsew")

		detail_frame = ttk.Frame(parent)
		detail_frame.grid(row=1, column=0, sticky="nsew")
		detail_frame.rowconfigure(0, weight=1)
		detail_frame.columnconfigure(0, weight=1)
		self.diag_detail = tk.Text(
			detail_frame,
			height=8,
			wrap="word",
			background="#1e1e1e",
			foreground="#d4d4d4",
			insertbackground="#d4d4d4",
		)
		vsb = ttk.Scrollbar(detail_frame, orient="vertical", command=self.diag_detail.yview)
		self.diag_detail.configure(yscrollcommand=vsb.set)
		self.diag_detail.grid(row=0, column=0, sticky="nsew")
		vsb.grid(row=0, column=1, sticky="ns")

		tree = self._get_tree(self.diag_tree)
		tree.bind("<<TreeviewSelect>>", self._on_diag_select)
		tree.bind("<Double-1>", lambda _e: self._jump_to_selected_diagnostic())

	def _build_ast_graph_tab(self, parent: ttk.Frame) -> None:
		parent.rowconfigure(0, weight=1)
		parent.columnconfigure(0, weight=1)

		frame = ttk.Frame(parent)
		frame.grid(row=0, column=0, sticky="nsew")
		frame.rowconfigure(0, weight=1)
		frame.columnconfigure(0, weight=1)

		self.ast_canvas = tk.Canvas(frame, background="#1e1e1e", highlightthickness=0)
		vsb = ttk.Scrollbar(frame, orient="vertical", command=self.ast_canvas.yview)
		hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.ast_canvas.xview)
		self.ast_canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
		self.ast_canvas.grid(row=0, column=0, sticky="nsew")
		vsb.grid(row=0, column=1, sticky="ns")
		hsb.grid(row=1, column=0, sticky="ew")
		self.ast_canvas.bind("<Button-1>", self._on_ast_canvas_click)

	def _create_tree(self, parent: ttk.Notebook, columns: Tuple[str, ...]) -> ttk.Treeview:
		frame = ttk.Frame(parent)
		tree = ttk.Treeview(frame, columns=columns, show="headings")
		for col in columns:
			tree.heading(col, text=col)
			tree.column(col, width=150, anchor="w")
		vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
		tree.configure(yscrollcommand=vsb.set)
		tree.pack(side="left", fill="both", expand=True)
		vsb.pack(side="right", fill="y")
		return frame

	def _get_tree(self, tree_frame: ttk.Frame) -> ttk.Treeview:
		return tree_frame.winfo_children()[0]  # type: ignore[index]

	def _compile(self) -> None:
		source = self.editor.get_text()
		artifacts = self.engine.compile(source)
		self._populate_diagnostics(artifacts.diagnostics)
		self._populate_tokens(artifacts.tokens)
		if artifacts.ast:
			self._populate_ast(artifacts.ast)
			self._draw_ast_graph(artifacts.ast)
		self._populate_symbols(artifacts.symbols)
		self.editor.apply_highlights(artifacts.tokens)
		self._apply_diagnostic_marks(artifacts.diagnostics)
		self._update_overview(artifacts)
		summary = f"Diagnostics: {len(artifacts.diagnostics)} | Tokens: {len(artifacts.tokens)} | Time: {artifacts.duration_ms:.2f} ms"
		self.status.configure(text=summary)

	def _populate_diagnostics(self, diagnostics: List[Diagnostic]) -> None:
		tree = self._get_tree(self.diag_tree)
		tree.delete(*tree.get_children())
		self._diag_by_iid.clear()
		for diag in diagnostics:
			if diag.span:
				loc = f"{diag.span.start.line}:{diag.span.start.column}"
			else:
				loc = "-"
			hint = diag.hint or ""
			iid = tree.insert("", tk.END, values=(diag.severity.name, diag.message, loc, hint))
			self._diag_by_iid[str(iid)] = diag
		self.diag_detail.delete("1.0", tk.END)
		if diagnostics:
			self.diag_detail.insert("1.0", "Select a diagnostic to see details and jump to the source location.\n")

	def _populate_tokens(self, tokens: List[Token]) -> None:
		tree = self._get_tree(self.token_tree)
		tree.delete(*tree.get_children())
		for token in tokens:
			if token.kind == TokenKind.EOF:
				continue
			line = token.span.start.line
			tree.insert("", tk.END, values=(token.kind.name, token.lexeme, line))

	def _populate_ast(self, program: ProgramNode) -> None:
		tree = self._get_tree(self.ast_tree)
		tree.delete(*tree.get_children())
		root = tree.insert("", tk.END, values=("Program", program.name))
		const_node = tree.insert(root, tk.END, values=("Constants", len(program.consts)))
		for const in program.consts:
			tree.insert(const_node, tk.END, values=("Const", f"{const.type_ref.name} {const.name}"))
		var_node = tree.insert(root, tk.END, values=("Variables", len(program.vars)))
		for var in program.vars:
			tree.insert(var_node, tk.END, values=("Var", f"{var.type_ref.name} {', '.join(var.names)}"))
		class_node = tree.insert(root, tk.END, values=("Classes", len(program.classes)))
		for cls in program.classes:
			cls_node = tree.insert(class_node, tk.END, values=("Class", cls.name))
			for field in cls.fields:
				tree.insert(cls_node, tk.END, values=("Field", f"{field.type_ref.name} {field.name}"))
		methods_node = tree.insert(root, tk.END, values=("Methods", len(program.methods)))
		for method in program.methods:
			method_node = tree.insert(methods_node, tk.END, values=("Method", method.name))
			for param in method.parameters:
				tree.insert(method_node, tk.END, values=("Param", f"{param.type_ref.name} {param.name}"))

	def _populate_symbols(self, symbols: SymbolTable) -> None:
		tree = self._get_tree(self.symbol_tree)
		tree.delete(*tree.get_children())
		def insert_scope(parent, scope: Scope) -> None:
			scope_id = tree.insert(parent, tk.END, values=(scope.name, "", ""))
			for symbol in scope.symbols.values():
				tree.insert(scope_id, tk.END, values=("", symbol.name, symbol.type_info.name))
		insert_scope("", symbols.global_scope)
		for scope in symbols.class_scopes.values():
			insert_scope("", scope)
		for scope in symbols.method_scopes.values():
			insert_scope("", scope)

	def _load_sample(self) -> None:
		sample = (
			"program Sample\n"
			"  final int size = 10;\n"
			"  class Table { int[] pos; int[] neg; }\n"
			"  int x, i;\n"
			"{\n"
			"  void main()\n"
			"    int tmp;\n"
			"    Table val;\n"
			"  {\n"
			"    val = new Table;\n"
			"    val.pos = new int[size];\n"
			"    i = 0;\n"
			"    while (i < size) {\n"
			"      val.pos[i] = 0;\n"
			"      i = i + 1;\n"
			"    }\n"
			"    read(x);\n"
			"    if (x > 0) { print(x); }\n"
			"  }\n"
			"}\n"
		)
		self.editor.set_text(sample)

	def _open_file(self) -> None:
		path = filedialog.askopenfilename(title="Open MicroJava source", filetypes=[("MicroJava", "*.mj"), ("All", "*.*")])
		if not path:
			return
		content = Path(path).read_text(encoding="utf-8")
		self.editor.set_text(content)

	def _save_file(self) -> None:
		path = filedialog.asksaveasfilename(title="Save MicroJava source", defaultextension=".mj")
		if not path:
			return
		Path(path).write_text(self.editor.get_text(), encoding="utf-8")

	def _export_report(self) -> None:
		source = self.editor.get_text()
		artifacts = self.engine.compile(source)
		report = {
			"diagnostics": [
				{
					"severity": diag.severity.name,
					"message": diag.message,
					"line": diag.span.start.line if diag.span else None,
					"column": diag.span.start.column if diag.span else None,
					"hint": diag.hint,
				}
				for diag in artifacts.diagnostics
			],
			"token_count": len(artifacts.tokens),
			"duration_ms": artifacts.duration_ms,
		}
		path = filedialog.asksaveasfilename(title="Export report", defaultextension=".json")
		if not path:
			return
		Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
		messagebox.showinfo("Export", "Compilation report saved.")

	def _export_markdown_report(self) -> None:
		source = self.editor.get_text()
		artifacts = self.engine.compile(source)

		def loc(diag: Diagnostic) -> str:
			if not diag.span:
				return "-"
			return f"{diag.span.start.line}:{diag.span.start.column}"

		lines = source.splitlines()
		diag_lines: List[str] = []
		for d in artifacts.diagnostics:
			hint = f" — _{d.hint}_" if d.hint else ""
			diag_lines.append(f"- **{d.severity.name}** at `{loc(d)}`: {d.message}{hint}")

		# Optional: include a short code frame around the first error for shareability.
		frame = ""
		first = next((d for d in artifacts.diagnostics if d.span), None)
		if first and first.span and 1 <= first.span.start.line <= len(lines):
			i = first.span.start.line - 1
			start = max(0, i - 2)
			end = min(len(lines), i + 3)
			frame_lines = []
			for ln in range(start, end):
				prefix = ">> " if ln == i else "   "
				frame_lines.append(f"{prefix}{ln+1:>3} | {lines[ln]}")
			frame = "\n**Code frame (around first issue)**\n\n```text\n" + "\n".join(frame_lines) + "\n```\n"

		md = (
			"# MicroJava Mini Compiler — Compilation Report\n\n"
			f"- Time: **{artifacts.duration_ms:.2f} ms**\n"
			f"- Tokens: **{len(artifacts.tokens)}**\n"
			f"- Diagnostics: **{len(artifacts.diagnostics)}**\n\n"
			"## Diagnostics\n\n"
			+ ("\n".join(diag_lines) if diag_lines else "_No diagnostics._")
			+ "\n\n"
			+ frame
			+ "## Source\n\n```text\n"
			+ source.rstrip()
			+ "\n```\n"
		)

		path = filedialog.asksaveasfilename(title="Export Markdown report", defaultextension=".md")
		if not path:
			return
		Path(path).write_text(md, encoding="utf-8")
		messagebox.showinfo("Export", "Markdown report saved.")

	def _schedule_live_compile(self) -> None:
		if not getattr(self, "live_var", None) or not self.live_var.get():
			return
		if self._compile_after_id:
			try:
				self.after_cancel(self._compile_after_id)
			except Exception:
				pass
		# Debounce: compile ~350ms after typing stops.
		self._compile_after_id = self.after(350, self._compile)

	def _apply_diagnostic_marks(self, diagnostics: List[Diagnostic]) -> None:
		self.editor.clear_diagnostic_marks()
		for diag in diagnostics:
			if diag.severity == Severity.ERROR:
				self.editor.mark_span(diag.span, "diag_error")
			elif diag.severity == Severity.WARNING:
				self.editor.mark_span(diag.span, "diag_warning")

	def _on_diag_select(self, _event: tk.Event) -> None:
		tree = self._get_tree(self.diag_tree)
		selection = tree.selection()
		if not selection:
			return
		diag = self._diag_by_iid.get(str(selection[0]))
		if not diag:
			return
		self._show_diagnostic_details(diag)
		self._highlight_active_diagnostic(diag)

	def _jump_to_selected_diagnostic(self) -> None:
		tree = self._get_tree(self.diag_tree)
		selection = tree.selection()
		if not selection:
			return
		diag = self._diag_by_iid.get(str(selection[0]))
		if not diag:
			return
		self.editor.goto_span(diag.span)

	def _highlight_active_diagnostic(self, diag: Diagnostic) -> None:
		# Give the selected diagnostic a subtle background so it stands out.
		self.editor.text.tag_remove("diag_active", "1.0", tk.END)
		self.editor.mark_span(diag.span, "diag_active")

	def _show_diagnostic_details(self, diag: Diagnostic) -> None:
		source = self.editor.get_text()
		self.diag_detail.delete("1.0", tk.END)
		self.diag_detail.insert("1.0", self._format_diagnostic(diag, source))

	def _format_diagnostic(self, diag: Diagnostic, source: str) -> str:
		lines = source.splitlines()
		loc = ""
		frame = ""
		if diag.span and 1 <= diag.span.start.line <= len(lines):
			line_no = diag.span.start.line
			col = max(1, diag.span.start.column)
			text_line = lines[line_no - 1].rstrip("\n")
			caret = " " * (max(0, col - 1)) + "^"
			loc = f"Location: line {line_no}, col {col}\n"
			frame = f"\n{text_line}\n{caret}\n"
		hint = f"\nHint: {diag.hint}\n" if diag.hint else ""
		return f"[{diag.severity.name}] {diag.message}\n{loc}{frame}{hint}"

	def _update_overview(self, artifacts: CompilationArtifacts) -> None:
		err = sum(1 for d in artifacts.diagnostics if d.severity == Severity.ERROR)
		warn = sum(1 for d in artifacts.diagnostics if d.severity == Severity.WARNING)
		info = sum(1 for d in artifacts.diagnostics if d.severity == Severity.INFO)
		self._overview_counts = (err, warn, info)
		# Rough symbol count: sum of all scope symbol tables.
		symbol_count = len(artifacts.symbols.global_scope.symbols)
		for s in artifacts.symbols.class_scopes.values():
			symbol_count += len(s.symbols)
		for s in artifacts.symbols.method_scopes.values():
			symbol_count += len(s.symbols)

		self.overview_stats.configure(
			text=(
				f"Time: {artifacts.duration_ms:.2f} ms\n"
				f"Tokens: {len(artifacts.tokens)}\n"
				f"Symbols: {symbol_count}\n"
				f"Diagnostics: {len(artifacts.diagnostics)}  (Errors: {err}, Warnings: {warn}, Info: {info})"
			)
		)
		self._redraw_overview_chart(err, warn, info)

	def _redraw_overview_chart(self, err: int, warn: int, info: int) -> None:
		c = self.overview_canvas
		c.delete("all")
		width = max(300, c.winfo_width() or 300)
		height = max(120, c.winfo_height() or 120)
		c.configure(width=width, height=height)
		data = [("Errors", err, "#f14c4c"), ("Warnings", warn, "#cca700"), ("Info", info, "#4fc1ff")]
		maxv = max(1, max(v for _n, v, _col in data))
		padding = 16
		bar_w = (width - padding * 2) / len(data)
		for i, (name, val, color) in enumerate(data):
			x0 = padding + i * bar_w + 10
			x1 = padding + (i + 1) * bar_w - 10
			bar_h = int((height - 40) * (val / maxv))
			y0 = height - 20 - bar_h
			y1 = height - 20
			c.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
			c.create_text((x0 + x1) / 2, height - 10, text=f"{name}: {val}", fill="#d4d4d4", font=("Segoe UI", 9))

	def _draw_ast_graph(self, program: ProgramNode) -> None:
		# Simple, dependency-free AST visualization on a Canvas.
		self.ast_canvas.delete("all")
		self._ast_canvas_nodes.clear()

		def node_label(node: ASTNode) -> str:
			if isinstance(node, ProgramNode):
				return f"Program {node.name}"
			if isinstance(node, ClassDecl):
				return f"Class {node.name}"
			if isinstance(node, MethodDecl):
				return f"Method {node.name}"
			if isinstance(node, VarDecl):
				return f"Var {node.type_ref.name} " + ", ".join(node.names)
			if isinstance(node, ConstDecl):
				return f"Const {node.type_ref.name} {node.name}"
			if isinstance(node, Parameter):
				return f"Param {node.type_ref.name} {node.name}"
			return node.__class__.__name__

		def ast_children(node: Any) -> List[ASTNode]:
			children: List[ASTNode] = []
			if not hasattr(node, "__dataclass_fields__"):
				return children
			for _fname, fdef in node.__dataclass_fields__.items():  # type: ignore[attr-defined]
				val = getattr(node, fdef.name, None)
				if isinstance(val, ASTNode):
					children.append(val)
				elif isinstance(val, list):
					for item in val:
						if isinstance(item, ASTNode):
							children.append(item)
			return children

		levels: List[List[ASTNode]] = []

		def walk(n: ASTNode, depth: int) -> None:
			if len(levels) <= depth:
				levels.append([])
			levels[depth].append(n)
			for ch in ast_children(n):
				walk(ch, depth + 1)

		walk(program, 0)

		x_gap = 220
		y_gap = 90
		margin = 40
		positions: Dict[int, Tuple[int, int]] = {}

		for depth, nodes in enumerate(levels):
			for i, node in enumerate(nodes):
				x = margin + i * x_gap
				y = margin + depth * y_gap
				positions[id(node)] = (x, y)

		# Draw edges first.
		for depth, nodes in enumerate(levels[:-1]):
			for node in nodes:
				for ch in ast_children(node):
					x0, y0 = positions[id(node)]
					x1, y1 = positions[id(ch)]
					self.ast_canvas.create_line(x0 + 70, y0 + 22, x1 + 70, y1, fill="#3a3a3a", width=2)

		# Draw nodes.
		for depth, nodes in enumerate(levels):
			for node in nodes:
				x, y = positions[id(node)]
				label = node_label(node)
				rect = self.ast_canvas.create_rectangle(x, y, x + 140, y + 44, fill="#252526", outline="#569cd6", width=1, tags=("ast_node", f"ast_{id(node)}"))
				self.ast_canvas.create_text(x + 70, y + 22, text=label, fill="#d4d4d4", font=("Segoe UI", 9), tags=("ast_node", f"ast_{id(node)}"))
				self._ast_canvas_nodes[id(node)] = node

		# Make scrollable.
		all_items = self.ast_canvas.bbox("all")
		if all_items:
			self.ast_canvas.configure(scrollregion=all_items)

	def _on_ast_canvas_click(self, _event: tk.Event) -> None:
		current = self.ast_canvas.find_withtag("current")
		if not current:
			return
		tags = self.ast_canvas.gettags(current[0])
		for t in tags:
			if t.startswith("ast_"):
				try:
					node_id = int(t.replace("ast_", ""))
				except ValueError:
					continue
				node = self._ast_canvas_nodes.get(node_id)
				if node:
					self.editor.goto_span(node.span)
					self.editor.text.tag_remove("diag_active", "1.0", tk.END)
					self.editor.mark_span(node.span, "diag_active")
				return


def main() -> None:
	if tk is None:
		raise RuntimeError("Tkinter is not available in this environment. Run with --nogui or use the FastAPI web app.")
	app = MiniCompilerApp()  # type: ignore[call-arg]
	app.mainloop()


if __name__ == "__main__":
	if len(sys.argv) > 1 and sys.argv[1] == "--nogui":
		source_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
		if not source_path:
			print("Usage: mini-compiler.py --nogui <file>")
			sys.exit(1)
		source = source_path.read_text(encoding="utf-8")
		engine = MiniCompilerEngine()
		artifacts = engine.compile(source)
		for diagnostic in artifacts.diagnostics:
			line = diagnostic.span.start.line if diagnostic.span else "-"
			print(f"[{diagnostic.severity.name}] line {line}: {diagnostic.message}")
		print(f"Tokens: {len(artifacts.tokens)} | Time: {artifacts.duration_ms:.2f} ms")
	else:
		main()
