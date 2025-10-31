## ⚙️ Setup with `uv`

This project uses [`uv`](https://github.com/astral-sh/uv) — a fast modern Python package manager — to manage dependencies and virtual environments.

### 1. Install `uv`
If you don’t have `uv` yet, install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

and create (and activate) a virtual environment:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```