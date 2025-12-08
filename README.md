# The STELE: *S*overeign *T*estbed for *E*valuating *L*anguage-model *E*xternalities

The STELE models the contemporary AI (or, \[large-\]language-model) stack as a **global dependency graph of normative and technical objects**. Nodes in this graph are laws, regulations, standards, corporate policies, model families, routing schemes, datasets, prompts, and deployment surfaces; edges record how they constrain, implement, or violate each other. A STELE run instantiates a slice of this graph as a multilingual experiment on one or more language models.

Concretely, the STELE is a Python CLI that:

1. parses a versioned, machine-readable **constitution** under `constitution/` (governance tiers, harms, threat models, metrics, standards, and model registries),
2. binds that constitution to a **mapping layer** (`indices/lexicon.yaml` and related files) that aligns languages, jurisdictions, harm domains, and infrastructures, and
3. executes language- and tier-aware stress tests (policy-boundary probes, jailbreaks, translation exploits, code-switching, and related families), using formally specified metrics and decision rules.

Each run produces a signed evaluation bundle:

- a **manifest** with hashes, seeds, model identities, routes, and tool versions;
- a **snapshot** of the constitution and mappings actually used in the run;
- **metrics in CSV and JSONL**, keyed to stable constitutional metric IDs, with views over languages, domains, and jurisdictions;
- **heatmaps and parity reports** over language × domain × jurisdiction;
- and a **graph export** (for example, JSON/GraphML) describing the nodes and edges of the instantiated dependency graph: which harms, safeguards, legal instruments, and technical components were engaged.

The quantitative layer is **proof-aware**. Core metrics—ensemble risk, parity gaps, coverage and calibration objects, composite scores—are defined in `metrics.yaml` and justified by lemmas in `proofs/` and summarized in `THEORY.md`. Each metric is explicit about its normative commitments (for example, how it trades false negatives against false positives for a given mandate) and about the conditions under which its guarantees hold.

The mapping layer is **non-normative by construction**. It documents the heuristics that map natural-language law and policy onto prompts, tags, and evaluation targets, and is intended to be inspected, critiqued, and forked. Competing actors—regulators, firms, civil-society coalitions—can supply their own constitutions and mappings while reusing the same experimental machinery.

The object of the exercise is not to certify any model as “safe.” It is to give institutional actors a shared, inspectable instrument for asking, with evidence:

> Safe according to whom, for whom, in which languages, and at what externalized cost?

By design, the STELE targets **evaluation and governance**, not live intervention. It stores redacted previews and signed bundles suitable for audit, research, procurement, and enforcement workflows; it is not a content-moderation service.

## Quickstart (local reference run)

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Reference scenario:
# - one model
# - one reference constitution
# - a small multilingual, multi-jurisdiction slice
python stele.py run \
  --config configs/reference_core.yaml \
  --output-dir demo_results/
