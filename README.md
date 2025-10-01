# Cross-Lingual Safety Audit Kit (CL-SAK)

A single-file Python CLI to evaluate multilingual safety behavior of large language models using repeatable, language-aware stress tests (policy boundary, jailbreak, translation exploits, and code-switching). Provider-agnostic, ethics-forward, reproducible, and analyst-ready.

> This tool **does not** operationalize harm. It stores **redacted previews** by default and is intended for research & evaluation.

## Why this exists
Most safety evaluations are English-centric. This kit runs curated prompts across **German, Spanish, French, Russian, Chinese, Arabic** (with benign controls) and scores outcomes with a simple rubric. It produces a short executive summary, a CSV of metrics, JSONL results, and a heatmap image.

## Quickstart (offline)
```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install matplotlib
python cl_sak.py --quickstart                         # runs lexicon judge only, no API calls
