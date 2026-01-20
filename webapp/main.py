from __future__ import annotations

from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the compiler engine from your existing file (core is defined before GUI).
from mini_compiler import MiniCompilerEngine  # type: ignore
from webapp.interpreter import MicroJavaInterpreter


app = FastAPI(title="MicroJava Mini Compiler", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
	app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


class CompileRequest(BaseModel):
	source: str


class RunRequest(BaseModel):
	source: str
	stdin: str = ""
	max_steps: int = 50_000


def _to_json(obj: Any, *, depth: int = 0, max_depth: int = 12) -> Any:
	"""Best-effort conversion of compiler artifacts to JSON-safe structures."""
	if depth > max_depth:
		return {"_truncated": True}
	if obj is None:
		return None
	if isinstance(obj, (str, int, float, bool)):
		return obj
	if isinstance(obj, list):
		return [_to_json(x, depth=depth + 1, max_depth=max_depth) for x in obj]
	if isinstance(obj, dict):
		return {str(k): _to_json(v, depth=depth + 1, max_depth=max_depth) for k, v in obj.items()}
	if is_dataclass(obj):
		data: Dict[str, Any] = {"_type": obj.__class__.__name__}
		for k, v in obj.__dict__.items():
			data[k] = _to_json(v, depth=depth + 1, max_depth=max_depth)
		return data
	if hasattr(obj, "name") and obj.__class__.__name__.endswith("Enum"):
		return getattr(obj, "name")
	# Enums (Severity/TokenKind/etc.)
	if hasattr(obj, "name") and hasattr(obj, "value"):
		return getattr(obj, "name")
	# Fallback
	return str(obj)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
	index_path = STATIC_DIR / "index.html"
	if index_path.exists():
		return HTMLResponse(index_path.read_text(encoding="utf-8"))
	return HTMLResponse(
		"<h2>MicroJava Mini Compiler API</h2><p>POST <code>/api/compile</code> with JSON: <code>{\"source\": \"...\"}</code></p>"
	)


@app.get("/health")
def health() -> Dict[str, str]:
	return {"status": "ok"}


@app.post("/api/compile")
def compile_source(req: CompileRequest) -> Dict[str, Any]:
	engine = MiniCompilerEngine()
	art = engine.compile(req.source)
	return {
		"duration_ms": art.duration_ms,
		"token_count": len(art.tokens),
		"diagnostic_count": len(art.diagnostics),
		"has_ast": art.ast is not None,
		"diagnostics": [
			{
				"severity": d.severity.name if hasattr(d.severity, "name") else str(d.severity),
				"message": d.message,
				"hint": d.hint,
				"span": _to_json(d.span),
			}
			for d in art.diagnostics
		],
		"tokens": [
			{
				"kind": t.kind.name if hasattr(t.kind, "name") else str(t.kind),
				"lexeme": t.lexeme,
				"value": _to_json(t.value),
				"span": _to_json(t.span),
			}
			for t in art.tokens
			if getattr(t.kind, "name", "") != "EOF"
		],
		"ast": _to_json(art.ast),
		"symbols": {
			"global": list(getattr(art.symbols.global_scope, "symbols", {}).keys()),
			"classes": {k: list(v.symbols.keys()) for k, v in getattr(art.symbols, "class_scopes", {}).items()},
			"methods": {k: list(v.symbols.keys()) for k, v in getattr(art.symbols, "method_scopes", {}).items()},
		},
	}


@app.post("/api/run")
def run_source(req: RunRequest) -> Dict[str, Any]:
	engine = MiniCompilerEngine()
	art = engine.compile(req.source)

	diags = [
		{
			"severity": d.severity.name if hasattr(d.severity, "name") else str(d.severity),
			"message": d.message,
			"hint": d.hint,
			"span": _to_json(d.span),
		}
		for d in art.diagnostics
	]
	has_errors = any((d.get("severity") == "ERROR") for d in diags)

	output = ""
	runtime_error = None
	steps = 0
	if (not has_errors) and art.ast is not None:
		interp = MicroJavaInterpreter(stdin=req.stdin or "", max_steps=req.max_steps or 50_000)
		run_art = interp.run(art.ast)
		output = run_art.output
		steps = run_art.steps
		if run_art.runtime_error is not None:
			runtime_error = {"message": run_art.runtime_error.message, "span": _to_json(run_art.runtime_error.span)}

	return {
		"duration_ms": art.duration_ms,
		"token_count": len(art.tokens),
		"diagnostic_count": len(art.diagnostics),
		"has_ast": art.ast is not None,
		"diagnostics": diags,
		"tokens": [
			{
				"kind": t.kind.name if hasattr(t.kind, "name") else str(t.kind),
				"lexeme": t.lexeme,
				"value": _to_json(t.value),
				"span": _to_json(t.span),
			}
			for t in art.tokens
			if getattr(t.kind, "name", "") != "EOF"
		],
		"ast": _to_json(art.ast),
		"symbols": {
			"global": list(getattr(art.symbols.global_scope, "symbols", {}).keys()),
			"classes": {k: list(v.symbols.keys()) for k, v in getattr(art.symbols, "class_scopes", {}).items()},
			"methods": {k: list(v.symbols.keys()) for k, v in getattr(art.symbols, "method_scopes", {}).items()},
		},
		"run": {
			"output": output,
			"steps": steps,
			"runtime_error": runtime_error,
		},
	}


