# Why quantization-aware training failed at W8A8

This directory is the complete research record for one negative result, kept
whole rather than summarized, because the failure turned out to be more useful
than a success would have been.

**Start here:** [`goal4_final_analysis.md`](goal4_final_analysis.md) — the
mechanism, the causal chain, and the English formulation for citation.

| File | Contents |
|---|---|
| [`goal4_final_analysis.md`](goal4_final_analysis.md) | Established facts, the mechanism, rejected hypotheses, deployment recommendation, research conclusion |
| [`goal4_hypotheses.md`](goal4_hypotheses.md) | Every hypothesis with its prediction, test, and verdict — including seven that were falsified |
| [`goal4_experiment_log.md`](goal4_experiment_log.md) | Chronological log; each entry records what it *ruled out* |
| [`goal4_results.md`](goal4_results.md) | Result tables |
| [`collect_results.py`](collect_results.py) | Regenerates the tables from each run's own `metrics.jsonl` — no hand transcription |

The one-line version:

> QAT did not fail because 8-bit activations cannot represent the policy. It
> failed because fake-quantization makes the policy a discontinuous function of
> its weights, so PPO's adaptive learning-rate controller reads quantization
> noise as policy divergence and throttles the learning rate to its floor —
> for 2000 of 2000 iterations. The run never took a meaningful gradient step,
> and its reward curve looked healthy throughout.
