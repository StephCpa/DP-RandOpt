# GSM8K selection diagnostics: first frozen pool

Date: 2026-10-25. All work is CPU-only. N=1001, fixed candidate seed 20261024, item seed 20261023, K=5, sigma grid unchanged. The primary D1 setting remains epsilon=8, delta=1e-5, n=1000. No new pool, GPU job, test-set tuning or model-scale change was introduced.

## Decision

Do not expand the claim that DP-RandOpt beats data-free baselines in this 3B GSM8K setting. Selection primarily rejects damaging perturbations and concentrates on low-diversity, base-level members. However, the stronger statement that there is no selectable spread anywhere is false: substantial lower-tail degradation is predictable. D4 remains untested because private-pool answer identities were not saved. No positive ensemble-level reliability result is available to justify a new D4 run.

## Sigma composition and group votes

| Ensemble | sigma=0 (base) | 0.0005 | 0.001 | 0.002 |
|---|---:|---:|---:|---:|
| random5 | 0.00% | 32.29% | 33.84% | 33.87% |
| public | 0.00% | 80.00% | 20.00% | 0.00% |
| nonprivate | 0.00% | 100.00% | 0.00% | 0.00% |
| D1 | 0.05% | 81.80% | 18.15% | 0.00% |

The candidate pool has 332/333/335 perturbations by increasing sigma (33.2/33.3/33.5%). The random row is the realized composition of 2000 ensembles, not the exact population proportion. D1 averages 400 draws; public and nonprivate are the frozen stable-sort top-5 sets.

| Sigma | Candidates | Mean member accuracy | Within-group random-5 vote | Pairwise answer disagreement | Correctness flips vs base |
|---|---:|---:|---:|---:|---:|
| 0.0005 | 332 | 85.640% | 88.177% | 13.205% | 7.944% |
| 0.001 | 333 | 85.096% | 88.855% | 15.647% | 9.828% |
| 0.002 | 335 | 82.135% | 88.458% | 21.719% | 13.198% |

Each within-sigma vote uses 2000 new data-independent ordered ensembles, with five distinct members. This is a descriptive analysis, not selection of a new sigma. The largest sigma is most diverse, but sigma=0.001 has the best within-group vote. Thus the prediction that sigma=0.002 votes best is not supported.

## Diversity and accuracy

| Ensemble | Member accuracy | Vote accuracy | Answer disagreement | Pairwise joint error |
|---|---:|---:|---:|---:|
| public | 85.534% | 88.324% | 13.859% | 9.712% |
| nonprivate | 85.398% | 87.642% | 13.161% | 10.129% |
| D1 | 85.402% | 88.046% | 13.634% | 9.999% |
| random5 | 84.282% | 88.739% | 17.341% | 9.707% |
| self_consistency | 84.230% | 89.310% | 18.241% | 9.553% |

Disagreement averages the ten unordered member pairs and all 1319 items; -1 is an abstention category. The JSON also reports disagreement restricted to pairs where both answers are valid, and correctness disagreement. Valid-only rates give the same qualitative pattern. Self-consistency is one saved sample realization, not 2000 independent runs.

Among 2000 random mixed-sigma ensembles, the raw Pearson correlation of disagreement with vote is only 0.008; after linearly residualizing both on mean member accuracy it is 0.680. For D1 these correlations are 0.113 and 0.605. Within sigma groups the raw correlations are negative (-0.081, -0.199, -0.253), while member-adjusted correlations are positive (0.390, 0.470, 0.652). More disagreement alone does not guarantee a better vote. These are descriptive associations on overlapping ensembles and common test items, not independent-replicate tests or causal identification. They support an accuracy/diversity tradeoff, not a one-variable explanation.

## Candidate spread and pool/test reliability

| Candidates | Observed SD (pp) | Marginal binomial SE (pp) | Shared-item noise SD (pp) | Estimated latent SD (pp) | Pool/test Pearson | Pool/test Spearman |
|---|---:|---:|---:|---:|---:|---:|
| 0.0005 | 0.578 | 0.966 | 0.580 | 0.000 | -0.029 | -0.024 |
| 0.001 | 0.735 | 0.981 | 0.637 | 0.367 | 0.112 | 0.075 |
| 0.002 | 3.106 | 1.052 | 0.752 | 3.014 | 0.940 | 0.306 |
| all_perturbations | 2.428 | 1.000 | 0.672 | 2.333 | 0.914 | 0.697 |

The naive subtraction of each candidate's approximately 1-pp binomial SE overstates noise in across-candidate differences because candidates share the same items. For the candidate-by-item correctness matrix C (M candidates, T items), the paired noise-variance estimate is [trace(S) - sum(S)/M] / [(M-1)T], where S is the sample covariance of candidate correctness across items. Subtract it from the observed sample variance of candidate accuracies. This projects out common item difficulty.

The correction assumes iid sampling of items from a population and treats the stored response function as fixed. It cannot identify decoder rerun variability. Negative latent variance is retained in the JSON; sigma=0.0005 gives -1.659e-7 in accuracy-squared units, with its displayed SD clipped to zero. This is not proof of exactly zero spread, nor a confidence bound. No independent test corpus or controlled repeated generation is available.

Overall estimated latent SD is 2.333 pp, not below 1 pp. The large-sigma group has a pronounced lower tail (minimum 37.68%, median 82.49%, maximum 84.99%); its Pearson 0.940 versus Spearman 0.306 reveals substantial tail influence. The five lowest candidates and group quantiles are saved in tail_sensitivity.json. Restricting descriptively to each sigma group's upper half by pool score yields pool/test Pearson correlations -0.060, 0.102 and 0.047. This restriction is a diagnostic, not a new selection protocol.

In contrast, the smallest-sigma group has negligible estimated spread and pool/test Pearson -0.029. Its mean 85.640% essentially equals the table base 85.595%. Across all candidates, 24.9% have an observed score above base, but none exceeds it by 5 pp; the observed maximum is 87.036% (+1.441 pp), selected retrospectively on test and not evidence of a deployable expert. The valid conclusion is weak upper-end expert-selection signal alongside strong ability to detect damaged candidates.

## Flips and split difficulty

Correctness flips against the score-table base are 7.944%, 9.828% and 13.198% by increasing sigma, versus 5.155% between the two base batch configurations. Even the smallest sigma exceeds that observed reference. However, one cross-configuration comparison is neither a repeated-run noise distribution nor a matched control. These figures cannot identify what fraction is numerical noise or establish which sigma is statistically distinguishable from rerun noise. Do not subtract 5.155% from the flip rates as an effect estimator.

| Base split | Correct / total | Accuracy |
|---|---:|---:|
| pool | 933 / 1000 | 93.300% |
| public | 46 / 50 | 92.000% |
| test | 1129 / 1319 | 85.595% |

The pool/test gap is +7.705 pp; a descriptive independent-item normal interval is approximately [5.257,10.153] pp. Public has only 50 items, and its 92% point estimate is imprecise. The 49/50 selection cutoff is additionally subject to selecting among 1001 candidates, so it should not be read as the base model's public accuracy.

Training-split items are clearly easier in the observed pool. This is consistent with exposure/memorization, but also with intrinsic difficulty or distribution differences; no pretraining membership evidence or unseen matched control is available. It is not a confirmed contamination finding. Results concern selection on a public benchmark's training split treated as private for mechanism evaluation, not demonstrated performance on genuinely unseen sensitive records.

## D4 and claim boundaries

D4 cannot be tested exactly with these artifacts: pool/public store binary rewards, while answer strings/codes are test-only, including in the checkpoint. Different wrong-answer agreements produce different votes with identical rewards. A private ensemble-score/test-vote correlation is therefore unavailable. See D4_FEASIBILITY.md for the evidence, fixed-library exponential-mechanism formulation and the per-item stability assumption needed for an end-to-end privacy argument. No D4 result or private-ensemble utility has been invented.

| Hypothesis / claim | Audit outcome |
|---|---|
| Selection over-represents the smallest sigma | Supported: 80% public, 100% nonprivate, 81.8% D1 |
| Largest sigma produces the best vote | Not supported: intermediate sigma is best within-group |
| Selected members sacrifice diversity | Supported descriptively; causal mechanism not established |
| No selectable spread exists anywhere | Not supported: predictable lower-tail damage is substantial |
| High-quality small-sigma candidates have strong ranking signal | Not supported by current correlations/spread estimates |
| Small perturbations are just numerical reruns | Unresolved; no matched rerun noise estimate |
| GSM8K contamination is established | Unresolved; split difficulty gap alone is insufficient |
| D4 reliability is positive | Unmeasured; required private answers absent |

The prior Countdown findings span different model variants, output caps and pool sizes. This GSM8K audit does not establish that every Countdown gap is explained only by public-data sufficiency. The current decision is to stop expanding the 3B GSM8K superiority claim and retain the negative result with the theory and audits; it is not a universal impossibility claim. A 7B run would need its own written rationale and authorization.

## Reproduction and verification

Run `python -m scripts.diagnose_gsm8k_selection` from the repository root; it requires the archived full score table and the local self-consistency gzip bundle. Compact draws.npz stores all ensemble member indices and per-draw metrics plus self-consistency answer codes. Primary random/public/nonprivate/D1 vote means exactly reproduce the previous audit. The source hash is verified before analysis. Four unittest checks cover covariance equivalence, shared item difficulty, deterministic candidate spread and abstentions in disagreement.

Mode: numeric and claim audit. No-invention status: measured and unidentifiable quantities are separated. No literature or publication-readiness assessment was performed.
