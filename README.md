## MicroJava Mini Compiler (Desktop + Web)

This project contains:
- **Desktop app**: `mini-compiler.py` (Tkinter GUI)
- **Web app**: FastAPI backend + static UI in `webapp/`

---

## Run locally (Web)

From the project root:

```bash
python -m pip install -r requirements.txt
python -m uvicorn webapp.main:app --host 127.0.0.1 --port 8000
```

Open in browser:
- `http://127.0.0.1:8000`

API:
- `POST /api/compile` with JSON: `{"source":"program P { void main() { } }"}`

---

## Deploy on Render (recommended)

Render is a better fit than Vercel because this is a Python server.

1. Push this repo to GitHub.
2. In Render: **New → Web Service → Connect your repo**
3. Choose:
   - **Environment**: Docker
   - Render will detect `Dockerfile`
4. Deploy.

After deploy, share your Render URL on LinkedIn.

---

## Run locally (Desktop)

```bash
python mini-compiler.py
```

If Tkinter isn’t available (common on Linux servers), use the web app instead.




