# GF-OCT Phase 1B — Real-Flow Confirmatory Execution

## Purpose

Test the frozen GF-OCT hypotheses H1–H6 on **real Google Flow transitions only**. Phase 1B must not use synthetic rows, hand-edited quality outcomes, or post-hoc threshold changes.

## Frozen hypotheses

- **H1:** High-tier model reduces restricted mean time-to-target (RMST) by at least 0.50 steps.
- **H2:** High-tier model improves episode quality gain per credit by at least 0.0005.
- **H3:** Edit preserves macro state with mean absolute macro drift <= 0.05.
- **H4:** Delta prompting reduces macro drift versus full restatement by at least 0.02.
- **H5:** Learned operator policy reduces total objective J by at least 5 cost units versus heuristic policy.
- **H6:** History-aware prediction yields RMSE ratio <= 0.90 versus Markov-only prediction.

All confirmatory decision rules are frozen in `preregistration/phase1b.freeze.json`.

## Minimum collection target

- 120 unique episodes minimum.
- 30 episodes minimum represented in each confirmatory contrast arm.
- Maximum 10 logged transitions per episode.
- Target quality: 0.90.
- No outcome-dependent early stopping.

## Required transition columns

The Phase 1A analytical columns remain unchanged. Phase 1B additionally requires provenance:

- `source` must equal `real_google_flow`
- exact `model_name`
- fixed `controller_id`
- fixed `evaluator_id`
- `timestamp_utc`
- `artifact_uri` linking the output or immutable artifact record

Use `data/phase1b/flow_transitions.template.csv` as the schema.

## Controller discipline

The controller must not be told whether the current condition is the nominal high/low label. It may receive the target specification, current observed output/evaluator feedback, and allowed operator. Controller prompt/version is fixed before collection.

## Evaluator discipline

The evaluator is condition-blinded. It returns `q_macro`, `q_micro`, and `q_total` in [0,1] from the frozen rubric. Human override is not allowed in the confirmatory dataset; any invalid evaluator run is logged as a platform/evaluation failure and rerun under the same frozen rule.

## Real-Flow collection sequence

1. Randomize episode assignment before generation.
2. Generate or edit in Google Flow using the exact assigned model/operator/prompt policy.
3. Record the actual Flow credit charge shown for the operation; do not infer from a remembered pricing table.
4. Save immutable output provenance and timestamp.
5. Score the output using the frozen evaluator.
6. If target quality is not reached and step < 10, continue under the assigned controller policy.
7. Stop at target, step 10, or documented external platform failure.

## Confirmatory lock

The analysis is not considered confirmatory until collection completeness is satisfied. Partial datasets can be processed only as `INTERIM_NONCONFIRMATORY`; they cannot change thresholds, sample targets, or the frozen controller/evaluator.

## Run

```bash
gf-oct-phase1b \
  --prereg preregistration/phase1b.freeze.json \
  --data data/phase1b/flow_transitions.csv \
  --out results/phase1b \
  --require-complete
```

Expected final artifact: `results/phase1b/decision.json`.
