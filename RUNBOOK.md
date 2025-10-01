# CL-SAK Runbook (Step-by-Step)

This is a **copy-paste** guide for non-engineers. Follow it EXACTLY to run the single-file tool and update the repo.

## 0) What this is
- `cl_sak.py` is a single Python script that runs multilingual safety evaluations.
- It can run **offline** (no API calls) or **online** using a provider key.
- It writes results to `reports/<run_id>/` by default. A pre-generated sample is in `demo_results/`.

---

## 1) One-time computer setup
1. Install **Python 3.10+** (already on macOS; Windows users can install from python.org).
2. Open a terminal (macOS: Terminal; Windows: PowerShell).

---

## 2) Create and activate a virtual environment
```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
```

---

## 3) Install dependencies
```bash
pip install -r requirements.txt
```

- If this fails, ensure you activated the virtual environment (step 2).

---

## 4) Run an **offline** demo (no API keys needed)
```bash
python cl_sak.py --quickstart
```

- Output files will appear under `reports/<run_id>/`:
  - `EXEC_SUMMARY.md`
  - `metrics.csv`
  - `full_results.jsonl`
  - `heatmap.png`

> Tip: You can open `EXEC_SUMMARY.md` in any Markdown viewer (GitHub renders it automatically).

---

## 5) Run a **real** evaluation (online, optional)
1. Get an **API key** from your provider.
2. Set your key in the terminal (macOS/Linux shown below):
```bash
export ANTHROPIC_API_KEY=sk-...   # Windows (PowerShell):  $env:ANTHROPIC_API_KEY="sk-..."
```
3. Run the tool:
```bash
python cl_sak.py --providers anthropic --langs de,es,fr,ru,zh,ar --limit 60
```

---

## 6) Copy a run into `demo_results/` for the repo
```bash
# Find the newest run directory
ls -t reports | head -n 1
# Replace <RUN_ID> below with that folder name (e.g., 20251001-052710-XXXXXX)
cp reports/<RUN_ID>/EXEC_SUMMARY.md demo_results/
cp reports/<RUN_ID>/metrics.csv      demo_results/
cp reports/<RUN_ID>/full_results.jsonl demo_results/
cp reports/<RUN_ID>/heatmap.png      demo_results/
```

---

## 7) Commit and push
```bash
git add demo_results/ requirements.txt RUNBOOK.md
git commit -m "docs: add demo results and runbook; prepare v0.1.0"
git push
```

---

## 8) Verifying everything
- On GitHub, you should see:
  - `cl_sak.py`
  - `README.md`, `RUNBOOK.md`, `LICENSE`, `requirements.txt`, `.gitignore`
  - `demo_results/` with 4 files (summary, CSV, JSONL, PNG)

---

## 9) Troubleshooting
- **pip install failed**: Ensure the virtual environment is active; update `pip install --upgrade pip`.
- **No `reports/` folder**: The tool writes after it runs. Re-run step 4 or 5.
- **Heatmap missing**: Install `matplotlib` (already in `requirements.txt`) and re-run.

---

## 10) Next steps (optional)
- Convert this single file into a full repo structure later (`providers/`, `judges/`, `prompt_packs/`).
- Add more languages and benign controls.
