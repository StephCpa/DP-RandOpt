# Completed batched controls and baseline comparison

Date: 2026-10-23. The GPU run finished at 09:02 Asia/Shanghai after 45 min
37 s. All 14,900 generated responses were downloaded: eight 1550-item greedy
calls and 2500 base-model temperature samples. The 12 source files match the
server transfer inventory. CPU reconstruction reproduces the saved metrics.

## Fixed-configuration repeatability

Base and candidates 1–3 were scored twice in the 1024-token production batch
configuration. Across 6200 paired responses, reward, correctness, voting
answer and token-sequence flip rates are all zero. The same four candidates
also match the original 1024-token pilot exactly in full response text and
scores. This includes private-pool, public and test splits.

All eight recorded generation checks preserve complete parameter-byte
checksums. Every reset returns to the same base weights; each repeated
perturbation produces identical weights. This rules out observed weight drift
and scoring instability for these controls. It does not establish universal
batch invariance or zero variability across hardware and configurations.

## Self-consistency versus perturbation ensembles

All results use the same 500 Countdown test items, Qwen2.5-3B-Instruct and
1024 output tokens. Sampling uses temperature 0.7, top_p=1, top_k=-1 and seed
20261023, with five samples per item. It is one fixed five-sample realization,
not an average over independently repeated self-consistency experiments.

| Method | Mean member accuracy | Vote accuracy | Checked pass@5 |
|---|---:|---:|---:|
| Base greedy, single response | 19.00% | — | — |
| Random five distinct perturbations | 15.99% | 41.19% | 47.26% |
| Base self-consistency, five samples | 17.80% | 45.80% | 51.60% |
| Public top-5, Q=50 | 20.96% | 53.40% | 58.40% |
| Nonprivate top-5, n=1000 | 20.24% | 48.20% | 58.00% |
| D1, epsilon=8, delta=1e-5, n=1000 | 20.02% | 48.34% | 57.80% |

Perturbation baselines use the frozen 32-candidate pool, with base excluded
from random draws. Random vote averages 2000 draws; random checked pass@5 is
the exact finite-population expectation. D1 averages 400 selection draws.
Sampling length finishes account for 9.04% of its 2500 responses; this is a
diagnostic, not a new truncation gate.

| Paired contrast | Vote difference, percentage points | Conditional 95% interval |
|---|---:|---:|
| Self-consistency minus random perturbations | +4.61 | [0.79, 8.49] |
| D1 minus self-consistency | +2.54 | [-2.68, 7.75] |
| Public top-5 minus self-consistency | +7.60 | [2.80, 12.60] |
| Nonprivate top-5 minus self-consistency | +2.40 | [-2.80, 7.60] |

Intervals resample the 500 paired test items, 5000 replicates, seed 20261022.
They condition on the saved sample realization, candidate pool and selection
data. They exclude variability from fresh sampling seeds, fresh candidate
pools and other decoding configurations. These exploratory contrasts are
not adjusted for multiple comparisons.

Self-consistency also improves checked pass@5 over random perturbations by
4.34 points [0.67, 8.06]. D1 improves checked pass@5 over self-consistency by
6.20 points [1.35, 10.99]. Countdown supplies a prompt-visible correctness
check, so checked pass is an available label-free decoding baseline here,
not a hidden-label oracle. On GSM8K, pass@5 would instead be a retrospective
oracle metric and must not be presented as a deployable label-free checker.

## Interpretation and next experiment

Unselected weight perturbations do not show an advantage over ordinary
sampling diversity in this setting. Their random-ensemble vote is lower than
the observed base self-consistency vote. D1's vote advantage over sampling is
uncertain, and public top-5 remains the strongest tested voting baseline in
the 32-candidate pilot. These controls support keeping Countdown secondary
for the private-selection claim.

The full 1001-candidate 512-token result remains distinct; this pilot does
not establish that its selection gain disappears. The next deciding task
is GSM8K with cap 1024, matched random-vote, self-consistency, public-selection
and nonprivate-selection baselines. Compare D1 against those baselines, not
only against a greedy base response. Preserve item assignments while varying
candidate seeds. See the [GSM8K execution plan](../gsm8k_ready/PLAN.md).

No new GPU run was started during this analysis. At the resource check, lzn
GPU 0 had about 32 GiB free but an active unrelated workload; GPU 1 was fully
busy. The previous control process has exited.

## Reproduction

The published [comparison vectors](comparison_vectors.npz) reproduce every
paired contrast without large response archives:

```text
python scripts/analyze_batched_controls.py --comparisons-only
```

Full archive validation additionally needs the nine gzip output bundles and
the original four 1024-token candidate bundles, kept in the experiment workspace:

```text
python scripts/analyze_batched_controls.py
```

[analysis.json](analysis.json) contains all contrasts, reconstructed flip
rates and cross-session comparisons. [COMPLETE.json](COMPLETE.json) contains
the original run summary. `progress.json` is an intermediate eight-call
snapshot; completion is determined by `COMPLETE.json`, not that snapshot.
