## MicroJava Mini Compiler (Desktop + Web)

This repository contains a **MicroJava-like mini compiler** (lexer + parser + basic semantic checks) and a separate **LL(1) parsing lab module** (FIRST/FOLLOW + LL(1) table + table-driven parsing with trace).

### What’s included

- **Desktop app (Tkinter GUI)**: `mini-compiler.py`
- **Import wrapper (for web/server usage)**: `mini_compiler.py`
- **Web app (FastAPI + static UI)**: `webapp/`
- **LL(1) lab module**: `webapp/ll1.py` (FIRST/FOLLOW/table + stack parser)
- **Interpreter (run the AST)**: `webapp/interpreter.py` (used by `/api/run`)
- **Reference PDF**: `MicroJava.pdf`

### Project folder layout (high-level)

- `mini-compiler.py`: compiler core + optional desktop GUI
- `mini_compiler.py`: loads `mini-compiler.py` (hyphenated filename is not importable as a module)
- `webapp/main.py`: FastAPI server (compile / run / ll1 endpoints)
- `webapp/static/index.html`: simple UI to interact with the server
- `requirements.txt`: Python dependencies

---

## Requirements

- **Python 3.10+** recommended
- Internet only required for installing dependencies

---

## Setup (Windows-friendly)

From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

If you already have a `.venv/`, just activate it and install requirements.

---

## Run locally (Web app)

Start the server from the project root:

```bash
python -m uvicorn webapp.main:app --host 127.0.0.1 --port 8000
```

Open:
- `http://127.0.0.1:8000`

---

## Run locally (Desktop app)

```bash
python mini-compiler.py
```

Notes:
- Tkinter must be available in your Python installation.
- On many Linux servers, Tkinter is not installed → use the web app instead.

---

## API Guide

All endpoints are served by `webapp/main.py`.

### `POST /api/compile`

Compiles MicroJava-like source and returns tokens, diagnostics, AST (serialized), and basic symbol info.

Example body:

```json
{ "source": "program P { void main() { } }" }
```

### `POST /api/run`

Compiles and then interprets the AST (if compilation succeeds).

Example body:

```json
{
  "source": "program P { void main() { print('H'); } }",
  "stdin": "",
  "max_steps": 50000
}
```

### `POST /api/ll1` (LL(1) Lab)

Computes **FIRST**, **FOLLOW**, builds the **LL(1) parse table**, and runs a **table-driven stack parser**.

Example body:

```json
{
  "tokens": "id = num + id ;",
  "trace": true,
  "include_working": false
}
```

Request fields:
- **`tokens`**: space-separated token stream (e.g. `id = num + id ;`)
- **`trace`**: include step-by-step parsing trace (stack, input, action)
- **`include_working`**: include FIRST/FOLLOW iteration logs (“show working”)
- **`grammar_start` + `grammar_lines`**: optionally provide your own grammar (lab sheet grammar)

#### Example token inputs (lab-sheet style)

- Valid:
  - `id = id ;`
  - `id = num + id ;`
- Invalid:
  - `id id ;`
  - `= id ;`
  - `id = ;`

---

## LL(1) lab: built-in demo grammar (assignment + expressions)

The default LL(1) grammar used by the lab endpoint (`default_assignment_expr_grammar()` in `webapp/ll1.py`) is:

- `S  -> id = E ;`
- `E  -> T E'`
- `E' -> + T E' | eps`
- `T  -> F T'`
- `T' -> * F T' | eps`
- `F  -> ( E ) | id | num`

This grammar supports:
- Assignment statement ending with `;`
- `+` and `*` with correct precedence
- Parentheses `( ... )`
- Symbolic tokens `id` and `num`

---

## Submissions / Lab Artifacts (logs + Excel)

These artifacts are **generated files** for submission and documentation. They are not “features” of the compiler; they are export outputs.

### 1) Sample run logs

Generate FIRST/FOLLOW/table + parse traces into a text file:

```bash
python -X utf8 gen_ll1_logs.py
```

Output:
- `ll1_sample_runs.txt`

### 2) Excel worksheet (parse table + FIRST/FOLLOW)

Generate Excel-friendly files (CSV + XLSX if supported):

```bash
python -X utf8 export_ll1_tables.py
```

Outputs:
- `LL1_Grammar.csv`
- `LL1_FIRST.csv`
- `LL1_FOLLOW.csv`
- `LL1_ParseTable.csv`
- `LL1_Parse_Table.xlsx` (created if `openpyxl` is available in your environment)

Open the CSV or XLSX in Excel and submit as required by your instructor.

---

## Deploy on Render (recommended)

Render is a good fit for a Python FastAPI server.

- Push this repo to GitHub
- In Render: **New → Web Service → Connect repo**
- Choose **Environment: Docker** (Render detects `Dockerfile`)
- Deploy, then share your Render URL

---

## Troubleshooting

### Import errors like `No module named 'webapp'`

Run commands from the **project root folder** (the directory that contains `webapp/`).

### Port already in use

Change the port:

```bash
python -m uvicorn webapp.main:app --host 127.0.0.1 --port 8001
```

### Tkinter missing (desktop)

If Tkinter is unavailable, use the **web app** mode instead.




