# Why quantization-aware training failed at W8A8

This directory is the complete research record for one negative result, kept
whole rather than summarized, because the failure turned out to be more useful
than a success would have been.

**Start here:** [`goal6_root_cause.md`](goal6_root_cause.md) — the resolved
answer. The goal-4 and goal-5 documents are kept as the record of how it was
reached, including two mechanisms that were reproducible, predictive, and not
the cause.

| File | Contents |
|---|---|
| [`goal4_final_analysis.md`](goal4_final_analysis.md) | Established facts, the mechanism, rejected hypotheses, deployment recommendation, research conclusion |
| [`goal4_hypotheses.md`](goal4_hypotheses.md) | Every hypothesis with its prediction, test, and verdict — including seven that were falsified |
| [`goal4_experiment_log.md`](goal4_experiment_log.md) | Chronological log; each entry records what it *ruled out* |
| [`goal4_results.md`](goal4_results.md) | Result tables |
| [`goal4_experiment_designs.md`](goal4_experiment_designs.md) | Designs for the two remaining interventions — frozen-anchor residual QAT and FP32-teacher behavior preservation — with the precondition the mechanism finding forces on both |
| [`goal5_mechanism_confirmation.md`](goal5_mechanism_confirmation.md) | The perturbation-response study, and the intervention that falsified the learning-rate mechanism it had just confirmed |
| [`goal6_root_cause.md`](goal6_root_cause.md) | **The answer.** Encoder freezing and FP32-teacher preservation both falsified; the RL-free capacity measurement that replaced argument by elimination; and the calibration finding that retired most of the collapse |
| [`collect_results.py`](collect_results.py) | Regenerates the tables from each run's own `metrics.jsonl` — no hand transcription |

The one-line version, twice revised — each revision by an experiment that
killed the previous one:

> QAT did not fail because PPO's controller misreads quantization noise. That
> was real, reproducible and predictive, and fixing it restored nothing. It
> failed because 8-bit activations cannot represent this policy to the accuracy
> the controller demands — measured directly, with the RL removed. And most of
> the collapse that motivated the whole investigation was not a precision limit
> at all: it was an activation-scale heuristic that left three quarters of the
> integer range unused. Correcting it takes W8A8 from 0.041 to 1.000 on the same
> checkpoint, with no retraining.

Superseded along the way, and kept because the path matters:

* *"QAT recovers 86–101%"* — held only on a task where the robot rested on its
  chassis (goal 4).
* *"The learning-rate collapse is the cause"* — the collapse is real and pins
  the rate for 2000/2000 iterations; removing it restored nothing (goal 5).
* *"W8A8 collapses"* — at the shipped calibration only (goal 6).
