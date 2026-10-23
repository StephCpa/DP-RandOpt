# Completed 1024-token paired pilot

Date: 2026-10-22, Asia/Shanghai.

Update 2026-10-23: the 10% cap-hit gate referenced below is withdrawn, and
1024 is fixed as the working cap. See [current analysis](../selection_decomposition/REPORT.md)
for trajectory-prefix evidence and the updated baselines. This report retains
its original numerical results and historical decision rationale.

All 32 preselected candidates completed at approximately 14:19 on gpu-lzn
GPU 0. The successful run took 11,903.5 seconds (3 h 18 min), including engine
startup, under changing GPU contention. All 49,600 response records and their
lengths/finish reasons were validated and copied locally. Source numeric
results are in [paired_analysis.json](paired_analysis.json).

The comparison uses the same original candidates 0–31, item seed 20261013,
candidate/noise seeds, sigmas, model and 1,550 items. Candidate selection did
not depend on test accuracy. Reference values below are restricted to these
32 rows of the 512-token table, not the complete 1,001-row pool.

| Metric | 512-token reference | 1024-token pilot |
|---|---:|---:|
| Base test accuracy, 500 items | 16.80% | 19.00% |
| Mean single-candidate test accuracy, 31 perturbations | 15.55% | 15.99% |
| All-response cap-hit rate | 43.20% | 33.54% |
| Base all-response cap-hit rate | 42.13% | 28.77% |
| Median perturbed-candidate cap-hit rate | 42.26% | 33.10% |

All 16,637 cap hits at 1024 also have `finish_reason=length`, directly
confirming truncation. The 512-token table stores lengths but not finish
reasons; its values are cap-hit diagnostics.

The base accuracy difference is +2.20 percentage points, with a paired test-item
bootstrap 95% interval of [-1.40, 5.80] points. The mean difference for the 31
perturbed candidates is +0.43 points [-0.21, 1.07]. Seventeen candidates
improved, two were unchanged, and twelve worsened. These intervals do not
support a clear accuracy gain at the present sample size.

The perturbation summary averages the correctness difference across the fixed
31 candidates per test item, then resamples the 500 items together (5,000
replicates). It does not treat 15,500 candidate/item predictions as independent,
and does not estimate variability across fresh candidate pools. These are
single-candidate averages, not ensemble-vote or D1 accuracies.

Longer outputs reduce truncation, but the remaining rate greatly exceeds the
predeclared 10% resource gate. It is therefore premature to build a full
1024-token pool or claim that truncation alone explains earlier performance.
Low-level execution differs from the original 512-token run (shared-GPU
memory accounting, context limit and batch limits), so bitwise-identical
decoding prefixes cannot be verified; the reference lacks text/token records.

## Next action

Update 2026-10-23: the action below has completed. See the
[completed 2048-token analysis](../length_pilot_32_max2048/REPORT.md) for current
results and the revised next step. The remaining text records the original plan.

Run the same 32 candidates at 2048 tokens in a new directory. The saved
manifest differs from the 1024-token manifest only in `max_tokens`; candidate
and item hashes are unchanged. Set the context budget to 4096 so it includes
both prompts and the full output allowance. Retain CPU base weights,
128 concurrent sequences, 2048 batched tokens, per-candidate checkpoints and
full response/finish-reason records. Keep all 512/1024 artifacts immutable.

The 2048 pilot was launched on GPU 0 at 15:33 (wrapper PID 1299614). Its
live status and subsequent startup verification are recorded in
`../length_pilot_32_max2048/README.md`. No full candidate-pool run or GSM8K
experiment was launched.
