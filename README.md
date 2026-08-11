# Flow — GF-OCT Falsification

Reproducible falsification framework for **Google Flow Optimal Control Theory (GF-OCT)**.

## Phase 1A

Phase 1A is a **simulation falsification / analysis-engine validation**, not a claim about real Google Flow behavior. It runs two synthetic worlds:

- `nominal`: mechanisms expected by GF-OCT are present.
- `null`: model-tier advantage, useful history dependence, delta-prompt protection, and learned-policy advantage are removed.

The same preregistered H1–H6 decision rules are applied to both worlds. This checks that the analysis can both support and reject claims instead of being hard-wired toward confirmation.

### H1–H6

- **H1** — higher-tier model reduces restricted mean time-to-target (RMST).
- **H2** — higher-tier model improves episode-level quality gain per credit.
- **H3** — edit is macro-state preserving within a frozen equivalence margin.
- **H4** — delta prompting reduces macro drift relative to full restatement.
- **H5** — learned operator policy reduces total resource + terminal-loss objective `J`.
- **H6** — history features improve held-out next-quality prediction over a Markov-only model.

## Run

```bash
python -m pip install -e ".[test]"
pytest -q
gf-oct-phase1a --prereg preregistration/phase1a.freeze.json --out results/phase1a
```

To analyze real Flow transitions using the same frozen rules:

```bash
gf-oct-phase1a \
  --prereg preregistration/phase1a.freeze.json \
  --data data/flow_transitions.csv \
  --out results/observed
```

## Required real-data columns

`episode_id, step, model_tier, operator, prompt_policy, policy_type, q_macro_before, q_macro_after, q_micro_before, q_micro_after, q_total_before, q_total_after, credit_cost, reached_target, terminal`

## Interpretation guardrail

Synthetic results are **sanity checks only**. Real-world support for GF-OCT requires a frozen Phase 1B dataset collected from Google Flow under matched tasks, blinded evaluation, fixed controller policy, and platform/model-version logging.
