# GSM8K first full-pool audit

Completed CPU audit: 2026-10-25. One candidate pool (seed 20261024), fixed item seed 20261023. N=1001, n=1000, Q=50, test=1319, K=5, epsilon=8, delta=1e-5. All percentages below use the same official test items.

## Main finding

The first full pool does not establish a utility benefit from private selection. D1 improves member accuracy relative to random perturbations, but loses vote accuracy. The nonprivate selection control also underperforms random voting; privacy noise alone therefore cannot explain the deficit. Self-consistency has the highest point estimate among the fixed-K primary comparisons.

| Method | Mean member accuracy | Vote accuracy | Oracle pass@5 |
|---|---:|---:|---:|
| Random perturbations | 84.285% | 88.739% | 94.343% |
| Public top-5 (stable ties) | 85.534% | 88.324% | 93.783% |
| Nonprivate top-5 | 85.398% | 87.642% | 93.480% |
| D1 | 85.402% | 88.046% | 93.605% |
| Base self-consistency | 84.230% | 89.310% | 94.845% |

Standalone greedy is 85.747%; the score-table base is 85.595%. Random-5 excludes base and samples distinct perturbations (2000 ensembles). D1 uses 400 selection draws. All selected methods use the full candidate pool including base. Oracle pass requires reference labels and is not a label-free deployment baseline.

## Paired comparisons

| Vote difference | Difference (pp) | Conditional 95% interval (pp) |
|---|---:|---:|
| D1 minus random5 | -0.693 | [-1.280, -0.118] |
| D1 minus self consistency | -1.264 | [-2.578, +0.001] |
| D1 minus public | -0.278 | [-1.227, +0.619] |
| D1 minus nonprivate | +0.404 | [-0.137, +0.957] |
| public minus self consistency | -0.986 | [-2.426, +0.455] |
| random5 minus self consistency | -0.571 | [-1.561, +0.440] |
| nonprivate minus self consistency | -1.668 | [-3.108, -0.227] |

Intervals use 5000 paired test-item bootstrap draws and condition on one frozen pool, one sampling realization and the selection data. They are exploratory and unadjusted. They do not measure candidate-pool variability or full DP-selection Monte Carlo uncertainty. The D1 versus self-consistency interval just includes zero; its endpoint is especially sensitive to Monte Carlo variation. Do not claim a replicated ranking or universal method inferiority.

## Selection and sensitivity checks

Public/private top-5 overlap is 1/5 (candidate 129). Public members average 85.534%, private members 85.398%; private selection does not yield better members in this draw. D1 members average 85.402% versus random 84.285%, yet vote falls from 88.739% to 88.046% and oracle pass falls from 94.343% to 93.605%. Better individual scores do not ensure complementary ensemble errors. This pattern motivates a diversity diagnosis, but does not prove a causal mechanism.

The public cutoff is 49/50 with 45 candidates tied. Randomized tie-breaking (400 draws) gives 88.767% rather than stable-order 88.324%. The D1 difference is -0.720 pp, conditional interval [-1.431,-0.030]. Tie-breaking is material and must be reported.

| Private n | D1 vote, epsilon=8 |
|---|---:|
| 200 | 88.598% |
| 500 | 88.438% |
| 1000 | 88.031% |

Each n uses 400 draws; n<1000 additionally resamples the private subset. The n=1000 redraw gives 88.031% versus the primary 88.046%, illustrating Monte Carlo variation. These results do not support a positive private-data equivalent or justify tuning n on test performance. Rewards are already binary answer correctness, so an answer-only rescoring is identical.

## Decoding and execution audit

All 2,371,369 generated-length entries are present and within 1–1024 tokens. Cap-hit rates are 0.084% for base and 0.175% for perturbations across all splits. The source table and metadata hashes match the server; candidate seeds/sigmas reproduce the frozen RNG construction. Both baseline bundles reproduce their saved correctness, member, vote and oracle-pass results.

Across base calls, 68/1319 item correctness decisions differ (5.155%): 33 are correct only in the score table and 35 only in the standalone baseline. The table batches 2369 prompts; the standalone call batches 1319. This is a cross-configuration discrepancy, not an estimate of same-configuration repeatability or proof of a weight-reset bug. Any future reproducibility control should match the 2369-prompt workload exactly.

The GPU run completed at 2026-10-25 08:24 Asia/Shanghai after starting at 2026-10-23 16:08. An OOM at 2026-10-24 23:45 triggered one automatic restart. The 8.60 hours in the final metadata measures only the resumed attempt; wall time was approximately 40h16m. Preserve this distinction.

Descriptive K curves are saved in secondary.json. Larger K changes the inference budget; those values are not matched-cost evidence against five-sample self-consistency. Primary K remains 5.

## Claim audit and next decision

| Claim | Evidence status |
|---|---|
| D1 beats random perturbation voting on this pool | Not supported; -0.693 pp |
| Private data adds value beyond public selection | Not established |
| DP noise is the sole cause of the shortfall | Contradicted by the nonprivate control |
| Output truncation is a major limitation here | Not supported by cap-hit rates |
| One pool establishes a general design ranking | Not supported |

Do not automatically expand to the other two full pools on this evidence. The next useful diagnostic is matched-workload repeatability plus CPU ensemble-error/diversity analysis if further investigation is authorized. Preserve this negative result and the frozen protocol; do not change K, sigma, n or prompts to optimize the official test set. No new GPU job was launched during this audit.

## Reproduction

Run `python -m scripts.analyze_gsm8k_pool --comparisons-only` to reproduce the seven primary paired intervals from committed compact vectors. Full analysis additionally needs the locally archived baseline gzip responses and the frozen prepared GSM8K test JSONL under results/cache/gsm8k_20261023. Run `python -m scripts.analyze_gsm8k_pool` from the repository root. The full pool table and metadata, compact vectors, analysis and source hashes are retained with this report.
