# Proofs vis-à-vis the STELE

This directory links formal proofs to the constitutional metrics and standards.

## Layout

- `paper/stele_core_proofs.md`  
  Human-readable mathematical proofs for core claims used in the constitution.

- `coq/stele_ensembles.v`  
  Lemmas about OR-ensembles of judges and false-negative bounds.

- `coq/stele_parity.v`  
  Lemmas about parity metrics such as L1_PARITY_GAP and L1_PARITY_RATIO.

- `coq/stele_risk.v`  
  Lemmas about the composite risk definition used in metrics.yaml.

- `logs/proof_runs.jsonl`  
  JSONL records of proof runs (lemma, tool, commit, status).

## Linking to the Constitution

Metrics and standards can reference proof IDs such as:

- `ENSEMBLE_OR_FN_BOUND`
- `L1_PARITY_GAP_BOUND`
- `COMPOSITE_RISK_MONOTONE`

These IDs correspond to lemmas stated in `paper/stele_core_proofs.md` and formalized in `coq/*.v`. When a lemma is machine-checked, a row should be added to `logs/proof_runs.jsonl` indicating the tool, version, commit, and status.
