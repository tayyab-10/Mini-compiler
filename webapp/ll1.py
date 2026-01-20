from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

EPS = "ε"
EOF = "$"


@dataclass(frozen=True)
class Production:
	lhs: str
	rhs: Tuple[str, ...]

	def __str__(self) -> str:
		if len(self.rhs) == 0:
			return f"{self.lhs} -> {EPS}"
		return f"{self.lhs} -> " + " ".join(self.rhs)


@dataclass(frozen=True)
class Grammar:
	start: str
	nonterminals: Set[str]
	productions: Tuple[Production, ...]

	@property
	def terminals(self) -> Set[str]:
		terms: Set[str] = set()
		for p in self.productions:
			for s in p.rhs:
				if s == EPS:
					continue
				if s not in self.nonterminals:
					terms.add(s)
		return terms

	def productions_for(self, lhs: str) -> List[Production]:
		return [p for p in self.productions if p.lhs == lhs]


def parse_grammar_lines(*, start: str, lines: Sequence[str]) -> Grammar:
	"""
	Parse a small CFG given as production lines, e.g.:

	  E  -> T E'
	  E' -> + T E' | ε
	  T  -> F T'

	Notes:
	- Nonterminals are inferred from LHS symbols.
	- Alternatives can be separated by '|'.
	- Epsilon can be written as 'ε', 'eps', or 'epsilon' (case-insensitive).
	"""

	def norm_eps(tok: str) -> str:
		t = tok.strip()
		if t in {"ε", "eps", "epsilon", "EPS", "EPSILON"}:
			return EPS
		return t

	raw: List[Tuple[str, List[str]]] = []
	nonterminals: Set[str] = set()

	for raw_line in lines:
		line = (raw_line or "").strip()
		if not line or line.startswith("#") or line.startswith("//"):
			continue
		if "->" not in line:
			raise ValueError(f"Invalid production (missing '->'): {raw_line}")
		lhs, rhs = line.split("->", 1)
		lhs = lhs.strip()
		if not lhs:
			raise ValueError(f"Invalid production (empty LHS): {raw_line}")
		nonterminals.add(lhs)
		raw.append((lhs, [p.strip() for p in rhs.split("|")]))

	prods: List[Production] = []
	for lhs, alts in raw:
		for alt in alts:
			if not alt:
				prods.append(Production(lhs, (EPS,)))
				continue
			syms = [norm_eps(t) for t in alt.split() if t.strip()]
			prods.append(Production(lhs, tuple(syms) if syms else (EPS,)))

	return Grammar(start=start, nonterminals=nonterminals, productions=tuple(prods))


def _first_of_sequence(seq: Sequence[str], *, first: Dict[str, Set[str]], nonterminals: Set[str]) -> Set[str]:
	"""
	FIRST(seq) computed left-to-right.
	Returns terminals plus EPS (if the entire sequence can derive epsilon).
	"""
	if len(seq) == 0:
		return {EPS}

	out: Set[str] = set()
	all_eps = True

	for sym in seq:
		if sym == EPS:
			out.add(EPS)
			continue
		if sym not in nonterminals:
			out.add(sym)
			all_eps = False
			break

		# nonterminal
		f = first.get(sym, set())
		out |= (f - {EPS})
		if EPS in f:
			# continue to next symbol
			continue
		all_eps = False
		break

	if all_eps:
		out.add(EPS)
	return out


def compute_first_sets(grammar: Grammar) -> Dict[str, Set[str]]:
	first: Dict[str, Set[str]] = {nt: set() for nt in grammar.nonterminals}

	changed = True
	while changed:
		changed = False
		for p in grammar.productions:
			before = set(first[p.lhs])
			rhs = list(p.rhs)

			if len(rhs) == 0 or (len(rhs) == 1 and rhs[0] == EPS):
				first[p.lhs].add(EPS)
			else:
				first[p.lhs] |= _first_of_sequence(rhs, first=first, nonterminals=grammar.nonterminals)

			if first[p.lhs] != before:
				changed = True

	return first


def compute_first_sets_with_trace(grammar: Grammar) -> Tuple[Dict[str, Set[str]], List[Dict[str, List[str]]]]:
	"""
	Compute FIRST sets and also return an iteration log.
	The log is a list of passes; each pass maps Nonterminal -> list of newly-added symbols.
	"""
	first: Dict[str, Set[str]] = {nt: set() for nt in grammar.nonterminals}
	passes: List[Dict[str, List[str]]] = []

	changed = True
	while changed:
		changed = False
		pass_changes: Dict[str, List[str]] = {nt: [] for nt in grammar.nonterminals}
		for p in grammar.productions:
			before = set(first[p.lhs])
			rhs = list(p.rhs)

			if len(rhs) == 0 or (len(rhs) == 1 and rhs[0] == EPS):
				first[p.lhs].add(EPS)
			else:
				first[p.lhs] |= _first_of_sequence(rhs, first=first, nonterminals=grammar.nonterminals)

			added = sorted(list(first[p.lhs] - before))
			if added:
				pass_changes[p.lhs].extend(added)
				changed = True

		if any(pass_changes[nt] for nt in pass_changes):
			passes.append(pass_changes)

	return first, passes


def compute_follow_sets(grammar: Grammar, first: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
	follow: Dict[str, Set[str]] = {nt: set() for nt in grammar.nonterminals}
	follow[grammar.start].add(EOF)

	changed = True
	while changed:
		changed = False
		for p in grammar.productions:
			rhs = list(p.rhs)
			for i, sym in enumerate(rhs):
				if sym not in grammar.nonterminals:
					continue

				before = set(follow[sym])
				beta = rhs[i + 1 :]
				first_beta = _first_of_sequence(beta, first=first, nonterminals=grammar.nonterminals)

				follow[sym] |= (first_beta - {EPS})
				if len(beta) == 0 or (EPS in first_beta):
					follow[sym] |= follow[p.lhs]

				if follow[sym] != before:
					changed = True

	return follow


def compute_follow_sets_with_trace(
	grammar: Grammar, first: Dict[str, Set[str]]
) -> Tuple[Dict[str, Set[str]], List[Dict[str, List[str]]]]:
	"""
	Compute FOLLOW sets and also return an iteration log.
	The log is a list of passes; each pass maps Nonterminal -> list of newly-added symbols.
	"""
	follow: Dict[str, Set[str]] = {nt: set() for nt in grammar.nonterminals}
	follow[grammar.start].add(EOF)
	passes: List[Dict[str, List[str]]] = []

	changed = True
	while changed:
		changed = False
		pass_changes: Dict[str, List[str]] = {nt: [] for nt in grammar.nonterminals}
		for p in grammar.productions:
			rhs = list(p.rhs)
			for i, sym in enumerate(rhs):
				if sym not in grammar.nonterminals:
					continue

				before = set(follow[sym])
				beta = rhs[i + 1 :]
				first_beta = _first_of_sequence(beta, first=first, nonterminals=grammar.nonterminals)

				follow[sym] |= (first_beta - {EPS})
				if len(beta) == 0 or (EPS in first_beta):
					follow[sym] |= follow[p.lhs]

				added = sorted(list(follow[sym] - before))
				if added:
					pass_changes[sym].extend(added)
					changed = True

		if any(pass_changes[nt] for nt in pass_changes):
			passes.append(pass_changes)

	return follow, passes

Table = Dict[str, Dict[str, Production]]


def build_ll1_table(grammar: Grammar, first: Dict[str, Set[str]], follow: Dict[str, Set[str]]) -> Tuple[Table, List[str]]:
	"""
	Returns (table, conflicts).

	Table is a nested dict:
	  table[NonTerminal][TerminalOr$] = Production
	"""
	table: Table = {nt: {} for nt in grammar.nonterminals}
	conflicts: List[str] = []

	for p in grammar.productions:
		rhs = list(p.rhs)
		first_rhs = _first_of_sequence(rhs, first=first, nonterminals=grammar.nonterminals)

		for a in sorted(first_rhs - {EPS}):
			existing = table[p.lhs].get(a)
			if existing is not None and existing != p:
				conflicts.append(f"Conflict at M[{p.lhs}, {a}]: {existing} vs {p}")
			else:
				table[p.lhs][a] = p

		if EPS in first_rhs:
			for b in sorted(follow[p.lhs]):
				existing = table[p.lhs].get(b)
				if existing is not None and existing != p:
					conflicts.append(f"Conflict at M[{p.lhs}, {b}]: {existing} vs {p}")
				else:
					table[p.lhs][b] = p

	return table, conflicts


@dataclass(frozen=True)
class ParseStep:
	stack: List[str]
	remaining_input: List[str]
	action: str


@dataclass(frozen=True)
class ParseResult:
	accepted: bool
	error: Optional[str]
	steps: List[ParseStep]


def parse_tokens_ll1(grammar: Grammar, table: Table, tokens: Sequence[str], *, trace: bool = True) -> ParseResult:
	"""
	Table-driven LL(1) parsing with a stack.
	- Input is a token sequence (symbolic tokens like: id, num, +, (, ), =, ; ...)
	- Always appends EOF ($)
	- Uses EPS (ε) for epsilon-productions
	"""
	inp = [t for t in tokens if t] + [EOF]
	stack: List[str] = [EOF, grammar.start]
	steps: List[ParseStep] = []
	i = 0

	def snapshot(action: str) -> None:
		if not trace:
			return
		steps.append(ParseStep(stack=list(stack), remaining_input=list(inp[i:]), action=action))

	snapshot("init")

	while stack:
		top = stack.pop()
		cur = inp[i] if i < len(inp) else EOF

		if top == EPS:
			snapshot("pop ε")
			continue

		# Terminal or EOF
		if top not in grammar.nonterminals:
			if top == cur:
				snapshot(f"match {cur}")
				i += 1
				if top == EOF:
					return ParseResult(accepted=True, error=None, steps=steps)
				continue
			return ParseResult(
				accepted=False,
				error=f"Mismatch: expected '{top}' but found '{cur}'",
				steps=steps,
			)

		# Nonterminal: expand using table
		prod = table.get(top, {}).get(cur)
		if prod is None:
			return ParseResult(
				accepted=False,
				error=f"No rule for M[{top}, {cur}]",
				steps=steps,
			)

		rhs = list(prod.rhs)
		snapshot(str(prod))
		# Push RHS in reverse order (ignore explicit EPS on the stack)
		if len(rhs) == 0 or (len(rhs) == 1 and rhs[0] == EPS):
			stack.append(EPS)
		else:
			for sym in reversed(rhs):
				if sym == EPS:
					continue
				stack.append(sym)

	return ParseResult(accepted=False, error="Unexpected end of parse (stack exhausted).", steps=steps)


def default_assignment_expr_grammar() -> Grammar:
	"""
	Matches the lab-style examples like:
	  id = id ;
	  id = num + id ;

	Grammar:
	  S  -> id = E ;
	  E  -> T E'
	  E' -> + T E' | ε
	  T  -> F T'
	  T' -> * F T' | ε
	  F  -> ( E ) | id | num
	"""
	nts = {"S", "E", "E'", "T", "T'", "F"}
	prods = [
		Production("S", ("id", "=", "E", ";")),
		Production("E", ("T", "E'")),
		Production("E'", ("+", "T", "E'")),
		Production("E'", (EPS,)),
		Production("T", ("F", "T'")),
		Production("T'", ("*", "F", "T'")),
		Production("T'", (EPS,)),
		Production("F", ("(", "E", ")")),
		Production("F", ("id",)),
		Production("F", ("num",)),
	]
	return Grammar(start="S", nonterminals=nts, productions=tuple(prods))


