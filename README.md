# ResidualGate

We introduce **ResidualGate**, a selective decision framework that combines capability-based policy enforcement with a lightweight language model while preserving a strict human-review safety floor.

## Layout

```
src/agentgate/
  labels.py        # Label / PolicyLabel enums and disposition rules
  compose.py        # tighten / rescue / release_bounded composition rules
  backends/          # pluggable LLM backends (vLLM, HF, OpenAI-compatible)
  camel_bridge/       # integration with the CaMeL agent runtime (engines, replay, synthesis)
  multimodal/         # policy/runner/triage for the multimodal case study
  analysis/           # statistics used to report results (bootstrap, McNemar, Wilson, ...)
tests/                 # pytest suite
data/multimodal/        # DocVQA-based task data for the multimodal case study
```

## Requirements

- Python >= 3.10
- The `camel_bridge` package depends on the external `camel` and `agentdojo` packages, which are not installed by default; some tests require them and will fail without them.

## Running tests

```bash
PYTHONPATH=src pytest tests/
```
