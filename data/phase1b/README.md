# Phase 1B real Google Flow data

Place the frozen confirmatory transition table at:

`data/phase1b/flow_transitions.csv`

Do **not** derive this file from Phase 1A synthetic outputs. Every row must correspond to an actual Google Flow generation/edit operation and include the provenance columns defined in `flow_transitions.template.csv`.

Until the real dataset is present, the Phase 1B workflow intentionally returns `WAITING_FOR_REAL_DATA` rather than fabricating a confirmatory result.
