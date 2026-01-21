from __future__ import annotations

"""
Generate concrete FIRST/FOLLOW sets, LL(1) table entries, and parse traces
for the built-in demo grammar used by the LL(1) lab endpoint.

Usage:
  python -X utf8 gen_ll1_logs.py
"""

from webapp.ll1 import (
	EOF,
	EPS,
	build_ll1_table,
	compute_first_sets,
	compute_follow_sets,
	default_assignment_expr_grammar,
	parse_tokens_ll1,
)


def fmt_sym(s: str) -> str:
	return "eps" if s == EPS else s


def fmt_prod(p: object) -> str:
	return str(p).replace(EPS, "eps")


def main() -> None:
	g = default_assignment_expr_grammar()
	first = compute_first_sets(g)
	follow = compute_follow_sets(g, first)
	table, conflicts = build_ll1_table(g, first, follow)

	print("=== GRAMMAR ===")
	for p in g.productions:
		print(fmt_prod(p))

	print("\n=== FIRST ===")
	for nt, syms in sorted(first.items()):
		print(f"{nt}: {sorted(fmt_sym(s) for s in syms)}")

	print("\n=== FOLLOW ===")
	for nt, syms in sorted(follow.items()):
		print(f"{nt}: {sorted(fmt_sym(s) for s in syms)}")

	print("\n=== LL(1) TABLE (non-empty cells) ===")
	for nt in sorted(g.nonterminals):
		row = table.get(nt, {})
		for t in sorted(row.keys()):
			print(f"M[{nt}, {t}] = {fmt_prod(row[t])}")

	print("\n=== Conflicts ===")
	print(conflicts)

	print("\n=== Parse (valid): id = num + id ; ===")
	r = parse_tokens_ll1(g, table, "id = num + id ;".split(), trace=True)
	print("accepted:", r.accepted)
	print("error:", r.error)
	print("steps:", len(r.steps))
	for s in r.steps[:18]:
		print("STACK:", " ".join(s.stack), "| IN:", " ".join(s.remaining_input), "| ACT:", s.action.replace(EPS, "eps"))
	print("...")
	for s in r.steps[-12:]:
		print("STACK:", " ".join(s.stack), "| IN:", " ".join(s.remaining_input), "| ACT:", s.action.replace(EPS, "eps"))

	print("\n=== Parse (invalid): id id ; ===")
	r2 = parse_tokens_ll1(g, table, "id id ;".split(), trace=True)
	print("accepted:", r2.accepted)
	print("error:", r2.error)
	print("steps:", len(r2.steps))
	for s in r2.steps[:18]:
		print("STACK:", " ".join(s.stack), "| IN:", " ".join(s.remaining_input), "| ACT:", s.action.replace(EPS, "eps"))

	print("\n(EOF symbol is:", EOF, ")")


if __name__ == "__main__":
	main()


