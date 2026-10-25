# Private selection and data-free voting: research conclusions and stopping decision

Evidence cutoff: 25 October 2026. This report synthesizes the completed experiments; it adds no model generations or protocol changes.

## Research conclusion

Private selection can closely match nonprivate member selection at epsilon=8 in the evaluated settings. That observation does not imply superiority over data-free ensembles. On the first full Qwen2.5-3B-Instruct GSM8K pool, D1 reaches 88.05%, compared with 88.74% for random perturbation voting and 89.31% for base-model self-consistency. Selection improves average member quality while reducing ensemble diversity. The resulting vote does not improve.

Countdown provides a qualified positive counterexample. Its full 512-token pool shows a substantial private-selection gain over both public-only selection and random voting. The longer-cap 32-candidate controls do not replicate that full-pool comparison: public-only is stronger there, and D1's advantage over self-consistency is unresolved. Consequently, neither “private selection always helps” nor “all Countdown gains are explained by public-data sufficiency” fits the evidence.

The current decision is to stop expanding the 3B GSM8K superiority claim, preserve the positive Countdown setting with its limits, and retain the theory, reproducibility work and negative diagnostics as the research outcome. This is a stopping decision for the current empirical direction, not an impossibility result about private selection.

## Comparable results and distinct settings

All selected-method rows below use K=5, private n=1000, public Q=50, epsilon=8 and delta=1e-5. Countdown uses 500 test items; GSM8K uses all 1319 official test items. Random-5 uses five distinct perturbations excluding base. Self-consistency uses five samples at temperature 0.7. Selection can include base. Accuracies are percentages.

| Setting / candidate seed | N | Cap | Base | Random-5 | Public | Nonprivate | D1 | Self-consistency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Countdown base-model legacy / 20261005 | 1001 | 128 | 3.60% | — | 27.80% | 21.80% | 21.94% | — |
| Countdown base-model legacy / 20261008 | 1001 | 128 | 4.20% | — | 28.00% | 25.80% | 25.34% | — |
| Countdown Instruct nested prefix / 20261013 | 550 | 512 | 16.80% | — | 50.80% | 58.00% | 57.92% | — |
| Countdown Instruct full pool / 20261013 | 1001 | 512 | 16.80% | 39.07% | 43.40% | 57.20% | 57.16% | — |
| Countdown Instruct first 32 / 20261013 | 32 | 512 | 16.80% | 40.84% | 46.60% | 47.20% | 47.20% | — |
| Countdown Instruct first 32 / 20261013 | 32 | 1024 | 19.00% | 41.19% | 53.40% | 48.20% | 48.34% | 45.80% |
| Countdown Instruct first 32 / 20261013 | 32 | 2048 | 17.00% | 41.46% | 49.00% | 47.00% | 46.93% | — |
| GSM8K Instruct full pool / 20261024 | 1001 | 1024 | 85.60% | 88.74% | 88.32% | 87.64% | 88.05% | 89.31% |

The two legacy Countdown runs use Qwen2.5-3B, whereas Route 1, the cap controls and GSM8K use Qwen2.5-3B-Instruct. Legacy item and candidate seeds change together. Route 1 holds item seed 20261013; GSM8K uses item seed 20261023 and candidate seed 20261024. These are different model/data/cap settings, not repeated seeds of one experiment.

The 550- and 32-candidate Countdown tables are prefixes of the full pool, not independent replications. Missing cells mean that a matched measurement is unavailable in this synthesis. In particular, self-consistency at 1024 tokens is not inserted into a 512-token row. The legacy random comparator included base; it is omitted here rather than relabelled as perturbation-only.

The full Countdown row uses the original completed-route audit's 400 D1 draws (57.159%). A later decomposition uses another draw stream (57.172%). Both are preserved; the approximately 0.013-point difference is Monte Carlo variation, not a revision to the score table. The [machine-readable evidence ledger](../results/gpu-smoke/cross_task_synthesis_20261025/evidence.json) records each row's source.

## What Countdown establishes

**Legacy 128-token setting.** Public-only exceeds D1 by 5.86 and 2.66 points in the two runs. D1-minus-public conditional 95% intervals are [-9.64,-2.23] and [-6.80,1.58] points. The second interval does not resolve the ordering. These short-cap base-model runs support neither a universal private-data benefit nor a clean causal conclusion about output length.

**Full 512-token Instruct setting.** D1 exceeds public-only by 13.76 points, with a conditional paired interval [9.57,18.12]. It differs from nonprivate selection by -0.04 points [-0.48,0.41]. The perturbation-only random-vote decomposition gives a D1 gain of approximately 18.10 points [14.64,21.79]. Randomized public-score ties leave the original public result unchanged. Answer-only reward selection reduces the D1/public gap to 5.52 points [1.78,9.33], but does not remove it. This positive result cannot be dismissed as public-data sufficiency or privacy-noise artifacts.

Its scope is one pool at a 512-token cap. About 43.7% of outputs reach that cap, and no matched full-pool self-consistency comparison exists. The available public-size curve ends at Q=50 and is nonmonotone; it does not identify a numeric public-data equivalent. The nested N=550 result is supportive within the same pool, not a second independent success.

**Longer-cap candidate subset.** At 1024 tokens, random vote is 41.19%, D1 48.34%, public-only 53.40% and self-consistency 45.80%. D1 adds 7.15 points over random voting, but its +2.54-point difference from self-consistency has interval [-2.68,7.75]. Public-only uses no private examples and has the highest vote point estimate. The 2048 subset has the same broad ordering of public versus D1. These subset results cannot invalidate the full 512-token pool, or establish that a full 1024-token pool would behave the same way.

Countdown's arithmetic target and allowed numbers are visible in the prompt. A checker can attain checked pass@5 without hidden labels: at 1024 tokens it reaches 47.26% for random perturbations, 51.60% for self-consistency and 57.80% for D1. This task therefore tests a narrower selection question than tasks without a free checker.

The prefix audit supports 1024 as a practical cap for the saved subset trajectories: tokens after 1024 add just one correct perturbed response among 15,500. Cap hits largely reflect repetitive generation, so the former 10% cap-hit stopping gate is retired. Same-configuration repeats of base and candidates 1–3 have zero flips across 6200 response pairs, with matching weight resets. These controls apply to their tested configuration, not all workloads.

Sources: [full-pool audit](../results/gpu-smoke/completed_route1_audit/REPORT.md), [vote decomposition](../results/gpu-smoke/selection_decomposition/REPORT.md), and [completed controls](../results/gpu-smoke/batched_controls_20261023/REPORT.md).

## What GSM8K establishes

D1-minus-random vote is -0.69 points, with conditional interval [-1.28,-0.12]. D1-minus-self-consistency is -1.26 points with interval approximately [-2.578,+0.001]; its upper endpoint is near zero and sensitive to Monte Carlo variation. D1 and nonprivate differ by +0.40 points [-0.14,0.96]. These comparisons support a small observed privacy cost relative to member selection, not statistical equivalence or a universally negligible privacy cost.

The public cutoff has 45 candidates tied at 49/50. Random tie-breaking raises public vote from 88.32% to 88.77%. D1 is 88.60%, 88.44% and 88.03% at n=200,500,1000 in the secondary size analysis. Smaller n changes both the empirical ranking and the effective impact of DP noise; this trend alone does not identify the cause.

Selection strongly favors sigma=0.0005: 100% of nonprivate members, 80% of public members and 81.8% of D1 members. D1's pairwise answer disagreement is 13.63%, versus random voting's 17.34% and self-consistency's 18.24%. Within-sigma random vote peaks at sigma=0.001 (88.85%), not at the most diverse sigma=0.002 (88.46%). Across random ensembles, disagreement has almost no raw association with vote, but a positive association after controlling for mean member accuracy. This is evidence of an accuracy/diversity tradeoff, not proof that maximizing disagreement would improve the method.

Candidate spread is not globally absent. Accounting for shared test-item covariance gives an estimated latent SD of 2.33 points across perturbations, dominated by damaging large-sigma candidates. Among sigma=0.0005 candidates, estimated latent spread is near zero and pool/test Pearson correlation is -0.029. The observed evidence supports screening out bad candidates more strongly than finding transferable experts above base.

Base accuracy is 93.3% on the pool, 92.0% on public items and 85.60% on test. This establishes a split-performance gap, not pretraining contamination. Correctness flips relative to base range from 7.94% to 13.20% by sigma. The 5.16% difference between two base batch configurations is not a matched rerun noise estimate and cannot be subtracted to isolate perturbation effects.

Sources: [full-pool comparison](../results/gpu-smoke/gsm8k_pool_audit_20261025/REPORT.md) and [selection diagnostics](../results/gpu-smoke/gsm8k_selection_diagnostics_20261025/REPORT.md).

## Theory and what remains conditional

The existing [theory note](theory.md) separates privacy accounting, density-dependent member-selection utility and ensemble voting. Its member-utility guarantees do not ensure improved vote accuracy; the real-data mismatch is consistent with that distinction. The synthesis does not re-prove the note's theorems, establish novelty or certify publication readiness. Simulation results remain design evidence, not additional real-model replications.

Privacy statements concern fixed candidates and bounded per-example utilities, with only the accounted selected output released. They do not cover publishing genuinely private score tables. Here the selection split is synthetic Countdown data or public GSM8K benchmark data treated as private for evaluation. Production batch dependence also requires attention in an end-to-end proof: replacing one record must not alter other records' utilities beyond the assumed sensitivity. A frozen-table mechanism audit alone does not establish that property for every inference backend.

## Stopping decision and conditions for reopening

1. Do not launch the other two GSM8K pools, a 7B migration or test-driven changes to K, sigma, n or prompts. The existing primary protocol stays fixed.
2. Stop asserting that DP-RandOpt beats data-free baselines in the evaluated 3B GSM8K setting. Preserve the full Countdown positive result and state its unreplicated cap/configuration limits.
3. Retain D4 as an untested hypothesis. Exact ensemble vote utilities cannot be reconstructed from private-pool correctness rewards; private answer identities were discarded. A fixed, data-independent ensemble library and one exponential-mechanism selection would target the vote directly, but no pool/test ensemble reliability measurement exists here. See the [D4 feasibility assessment](../results/gpu-smoke/gsm8k_selection_diagnostics_20261025/D4_FEASIBILITY.md).
4. Reopening GPU work requires a written, bounded question, resource estimate and success criterion fixed before new test results. A D4 proposal must address missing pool answers and matched-workload repeatability. A 7B proposal must explain why scaling should produce useful selection signal rather than merely consume more compute. Neither is scheduled or authorized by this report.

A defensible research summary is: *At epsilon=8, private member selection closely tracks nonprivate selection in the evaluated configurations. Its benefit over data-free inference is task- and configuration-dependent: a full short-cap Countdown pool shows a positive selection effect, while the full 3B GSM8K pool does not outperform random perturbation voting or self-consistency. The latter failure is associated with weak upper-end ranking signal and reduced ensemble diversity.*

## Reproduction and uncertainty

From the repository root, `python -m scripts.summarize_frozen_evidence` rebuilds the eight-setting ledger and table solely from saved audit JSON. It does not modify source score tables or launch inference.

The compact paired comparisons can be checked with:

```text
python -m scripts.analyze_gsm8k_pool --comparisons-only
python -m scripts.analyze_batched_controls --comparisons-only
```

Full diagnostics use `python -m scripts.diagnose_gsm8k_selection`; it additionally needs the locally archived self-consistency gzip bundle. Full GSM8K rescoring also requires the frozen prepared test JSONL. Text-prefix reconstruction needs the archived Countdown response bundles, tokenizer and production parser. These dependencies are listed in the [artifact guide](../results/gpu-smoke/ARTIFACTS.md); compact records in Git do not substitute for missing raw text.

Reported paired intervals resample test items and condition on the saved candidate pool, selection data and sampling realization. Selection repeats are not independent candidate pools. Multiple exploratory comparisons are unadjusted. No cross-task average, universal ranking, exact public-data equivalent or contamination conclusion is inferred. The GSM8K run recovered from one OOM restart; its 8.60-hour metadata field describes only the resumed segment, whereas wall time was about 40h16m.
