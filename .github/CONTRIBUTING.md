# Contributing

Thanks for your interest! For this initial single-file release:
- Please open an issue to propose changes.
- Keep the CLI flags stable.
- Avoid adding heavy dependencies.

## Development Setup
```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python cl_sak.py --quickstart
```

## Code Style
- Follow PEP 8.
- Add concise docstrings and type hints for public functions.
