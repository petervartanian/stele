# The STELE: *S*overeign *T*estbed for *E*valuating *L*anguage-model *E*xternalities

The STELE is a "sovereign testbed" for evaluating how language models externalize risk and power across jurisdictions, languages, and infrastructures. It treats models as *geo-technological* actors: systems whose behavior must be measured against the mandates of states, alliances, regulators, firms, and civil-society compacts—not just against a single lab’s internal policy.

At its core, the STELE is a Python CLI plus a machine-readable *constitution graph*:
a versioned set of YAML-based constitutions (for states, regulators, standards bodies, and institutional buyers), an explicit mapping layer (languages ↔ jurisdictions ↔ harm domains ↔ infrastructures), and formally defined metrics and standards backed by proofs. In its purple configuration, a single run can evaluate one model against many constitutions, or many models against one constitution, and compare the resulting externalities.

Each run produces:

- a **signed manifest** with hashes, seeds, model identities, routing, and tool versions,
- a **snapshot of all constitutions and mappings used** (jurisdictions, alliances, corporate policies, frameworks),
- **metrics in CSV and JSONL** keyed to constitutional metric IDs, with per-constitution and cross-constitution views,
- **language × domain × jurisdiction heatmaps** and parity reports,
- a **human-readable briefing** tied back to named standards (for example, `STELE_CORE_V1`, `EU_DSA_SAFETY_V1`, `PROCURE_GOV_001`),
- and a **graph export** (for example, JSON/GraphML) exposing the nodes and edges of the run’s concept space: which harms, laws, safeguards, and infrastructures were actually engaged.

The STELE is provider-agnostic, sovereignty-respecting, reproducible, and analyst-ready. It is designed as shared infrastructure: labs, regulators, procurement teams, and researchers can all run the same experiments, but against *their* own named constitutions.

> This testbed does not operationalize live harm mitigation; by default does it store **redacted previews** and signed evidence bundles for audit, research, and governance, and it is not a drop-in content moderation system.

## Why this exists

Most safety work today is:
English-centric, platform-defined, and buried in code or policy PDFs. It often confuses two different questions:

1. *What does this model do to people, languages, and institutions around the world?*  
2. *According to whom is that behavior acceptable?*

The STELE separates and makes both layers explicit.

- **Multilingual, jurisdiction-aware, and tiered.**  
  The testbed runs curated suites across a registry of languages arranged in tiers (anchor, imperial, regional lingua francas, historically marginalized), and across **jurisdictional tiers** (nation-states, alliances, regulators, corporate standards, civil charters). It measures both absolute safety and **parity**: which languages, regions, and constituencies incur more risk.

- **Constitutional and plural, not hard-coded and monolithic.**  
  Harms, threat models, governance tiers, metrics, standards, tests, and model registries live under `constitution/` as explicit YAML files. In purple mode, there is no single “STELE policy”: there is a **graph of constitutions**, each owned, versioned, and signed by the mandate that claims it. The engine reads and enforces these; it does not embed hidden policies in code.

- **Proof-aware and mandate-aware metrics.**  
  Core quantitative claims (OR-ensembles of judges, parity gaps, composite risk scores, confidence intervals) are backed by proofs and lemmas under `proofs/` and summarized in `THEORY.md`. Each metric carries:
  - its formal definition,
  - the theorems it depends on,
  - the normative assumptions it encodes (for example, how to trade off false negatives vs false positives for a given regulator).

- **Explicit geo-linguistic and regulatory mapping.**  
  The prompts and surface strings used to probe models are mapped to harm domains, threat intents, infrastructures, and jurisdictions via `indices/lexicon.yaml` and `MAPPING.md`. This layer is non-normative by design: it is a documented set of heuristics and alignments that can be inspected, critiqued, forked, and replaced by other actors.

- **Reproducible, contestable evidence bundles.**  
  Every run emits a self-contained bundle with fixed seeds, prompts, constitutions, mappings, and outcomes. Bundles can be:
  - re-validated against newer constitutions or mappings,
  - compared across models and providers,
  - attached to procurement, certification, or enforcement processes as structured evidence.

The goal is not to bless any model as “safe”. The goal is to give sovereign and institutional actors a **common, inspectable instrument** for asking: *Safe according to whom, for whom, in which languages, and at what externalized cost?*

## Quickstart (reference runner, local)

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run a local reference scenario:
# - one model
# - one reference constitution (STELE core)
# - a small multilingual, multi-jurisdiction slice
python stele.py run \
  --config configs/reference_core.yaml \
  --output-dir demo_results/
