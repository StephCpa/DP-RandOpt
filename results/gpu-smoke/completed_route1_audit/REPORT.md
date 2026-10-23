# Completed Route 1: CPU audit and next experiment

Date: 2026-10-22 (Asia/Shanghai)

## Main finding

For the completed 512-token Countdown setting, D1 at fixed K=5, epsilon=8,
delta=1e-5 and n=1,000 achieves 57.16% test vote accuracy, compared with
43.40% for public-only selection using 50 examples and 57.20% for non-private
selection on the same private pool. This is evidence of private-selection
utility in this one setting, conditional on its candidate pool and output cap.
It does not establish a general advantage across seeds, models, or tasks.

## Evidence and integrity

- Model: Qwen2.5-3B-Instruct; item seed and candidate seed: 20261013.
- N=1,001 including base; private/public/test sizes: 1,000/50/500.
- Generation completed on 2026-10-21 at 10:02 on gpu-lzn GPU 1 (L40).
- The resumed segment took 11.94 hours, not the entire interrupted run.
- Local and remote SHA-256 match:
  `687959e266f5317c557679229e2e032e5c8caf843a7f19c5d067ee2ba8070551`.
- All 1,551,550 lengths are present. All six checked arrays in the first
  550 rows match the archived partial table exactly, including answer codes.
- This audit generated no model responses and launched no GPU experiments.
- Analysis source: `../../../scripts/audit_completed_route1.py`.
- Numeric source: [audit.json](audit.json); [diagnostic figure](diagnostics.png).

Reproduce from the repository root with `PYTHONPATH=.`:

```bash
python scripts/audit_completed_route1.py \
  results/gpu-smoke/lzn_route1_n1001_max512.npz \
  results/gpu-smoke/partial/lzn_route1_n550.npz \
  results/gpu-smoke/server_3b_n1000.npz \
  results/gpu-smoke/server_3b_n1000_seed20261008.npz \
  --partial results/gpu-smoke/partial/lzn_route1_n550.npz \
  --out results/gpu-smoke/completed_route1_audit --reps 400
```

K=5 remains the frozen main comparison from the earlier protocol. The K curve
below is descriptive and must not be used to choose a test-optimized K for
a confirmatory claim. All main comparisons use the same K and original reward.
D1 uses no public pre-screen. Each stochastic audit uses 400 repeats, except
the descriptive K curve, which uses 100 per K. These repeats are not new seeds
or independent test sets.

## Matched comparison

| Setting | Base | Public-only K=5 | Non-private K=5 | D1 K=5, eps=8, n=1000 |
|---|---:|---:|---:|---:|
| 128 tokens, seed 20261005, N=1001 | 3.60% | 27.80% | 21.80% | 21.94% |
| 128 tokens, seed 20261008, N=1001 | 4.20% | 28.00% | 25.80% | 25.34% |
| 512 tokens, seed 20261013, N=550 | 16.80% | 50.80% | 58.00% | 57.92% |
| 512 tokens, seed 20261013, N=1001 | 16.80% | 43.40% | 57.20% | 57.16% |

The full-table D1-minus-public difference is **13.76 percentage points**, with
a paired test-item bootstrap 95% interval of **[9.57, 18.12] points**. The
non-private-minus-public difference is 13.80 points [9.40, 18.20]. D1 differs
from non-private by -0.04 points [-0.48, 0.41]. Privacy noise is therefore not
the main source of the public/private gap at epsilon=8 in this setting.

Bootstrap uses 5,000 paired resamples of the 500 test items and the per-item
mean over 400 D1 draws. It estimates test-sample uncertainty for expected
selection performance conditional on this pool and selection data, not
uncertainty across candidate pools, datasets, or output caps. Earlier intervals
that resampled one mechanism draw per bootstrap replicate targeted a different
quantity and are retained in the historical artifacts.

Increasing N from 550 to 1001 reduced public-only accuracy by 7.4 points and
D1 by about 0.76 points. More candidates did not help both selectors equally
in this realization. This alone does not prove a general candidate-count effect
or identify its cause. The partial and full runs are nested, not independent.

The old and new runs change candidate seeds, items, and output caps together.
An additional metadata check during migration preparation found that the
original `gpu-gfkd:~/dprandopt_run/tables/server_3b_n1000.npz.meta.json`
records `models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`,
a base-model path, whereas Route 1 records Qwen2.5-3B-Instruct. This additional
checkpoint-identity difference must also be retained; model-family naming
does not establish identical checkpoints. The reversal cannot identify an
item-seed effect or a token-budget effect. The 512/1024/2048 paired pilots all
use the Route 1 Instruct reference and are unaffected by this legacy mismatch.

## Length-cap diagnostic

| Quantity | Fraction with length exactly 512 |
|---|---:|
| All responses | 43.745% (678,729 / 1,551,550) |
| Base, all 1,550 items | 42.129% |
| Base, 500 test items | 42.800% |
| Median perturbed candidate, all items | 43.645% |

These are cap-hit rates. Finish reasons and full texts were not saved, so
length equality does not distinguish coincidental EOS at the boundary from
forced truncation. The frequency nevertheless strongly motivates a longer-cap
check. Adding the remaining candidates did not remove this limitation.

## Selection diagnostics

Original public scores have a single candidate at the K=5 cutoff. Repeating
random tie-breaking 400 times leaves public-only at 43.40%. The public top-5
overlaps the private top-50 in 4/5 candidates.

With answer-only selection rewards (`reward >= 1.0` on both public and private
data), public-only is 51.20%, non-private is 56.60%, and D1 is 56.72%. The
D1-public difference becomes 5.52 points [1.78, 9.33]. Reward choice materially
affects the gap, but the matched comparison still favors D1 here. Answer-only
public selection has six candidates tied at the cutoff; random tie-breaking
gives a mean 51.34% and a 2.5th–97.5th percentile range of 48.60%–53.60%.

For the old seed 20261008, answer-only selection reverses the point ordering
(D1 35.86%, public-only 32.60%), but the conditional difference interval
[-0.32, 6.84] points includes zero. The two old seeds cannot establish a stable
design ranking or prove that private data is generally unhelpful.

D8 releases in 99.5% of full-table repeats and falls back in 0.5%. Its raw
accuracy is 55.32%; public fallback gives 55.46%. On the old seed 20261008,
release/fallback rates are 2.25%/97.75%, and raw/public-fallback accuracies are
4.59%/27.86%. These are 400-repeat estimates; earlier 100-repeat estimates are
not overwritten. No D1/D8 ranking across candidate pools is established.

## K curve: earlier conclusion does not transfer

| K | Public-only | Non-private | D1 eps=8 |
|---:|---:|---:|---:|
| 1 | 13.40% | 28.20% | 28.20% |
| 5 | 43.40% | 57.20% | 57.12% |
| 10 | 57.20% | 66.40% | 66.29% |
| 20 | 67.60% | 72.60% | 72.34% |
| 50 | 74.60% | 76.60% | 76.50% |
| 100 | 76.60% | 77.40% | 77.95% |

The K=5 curve entry uses a separate 100-draw batch, hence its small difference
from the main 400-draw estimate. The old claim that K=5–10 beats K=50 is
supported only in the earlier 128-token private-selection results. It fails
in this 512-token setting. Larger ensembles also require more inference, so
these values are not compute-matched comparisons. Do not promote K=100 on
the basis of this test curve; future K changes require public validation with
stored public answers if the tuning objective is ensemble vote accuracy.

## Public-data equivalence: crossover not identified

| Public examples Q | Public-only mean over random subsets/ties |
|---:|---:|
| 0, base reference | 16.80% |
| 0, random five-candidate ensemble | 38.98% |
| 5 | 41.87% |
| 10 | 43.30% |
| 25 | 45.11% |
| 50 | 43.40% |

| Private examples n | D1 eps=8 | Non-private |
|---:|---:|---:|
| 200 | 55.19% | 55.27% |
| 500 | 55.93% | 56.00% |
| 1000 | 57.13% | 57.20% |

Public subsets are sampled only from the 50 available public examples, not
from a new population. Private and non-private curves use matched pool subsets.
The n=1000 row uses a separate random batch from the main comparison. Shading
in the figure is repeat variation, not a test confidence interval.

None of the observed public-only mean points reaches D1 for n=200, 500 or
1000. A numeric equivalent Q cannot be inferred: the public curve is noisy,
non-monotone, and stops at 50. In particular, do not extrapolate an exact Q or
a general lower bound. More public items would be needed for a measured
crossover. The random-five baseline also shows that part of the base-to-D1
gain comes from ensembling rather than private selection alone.

## Next experiment: paired length-cap pilot before another full table

1. Freeze item seed 20261013, the exact saved candidate seed/sigma manifest,
   model, prompts, precision and greedy decoding. Use base plus the first
   31 perturbations (indices 0–31) chosen by index, not by test performance.
   Reuse all 1,550 items. Do not simply change `--n-candidates` to 31: that
   changes the random draws and can change the sigma assignments.
2. Generate a separate 1,024-token pilot table; keep the completed 512-token
   table immutable. Save finish reasons, token lengths, answer text/codes,
   candidate identifiers, and complete launch metadata. Verify identical
   item/candidate manifests before generation.
3. Compare cap-hit and finish-reason rates, base accuracy, and paired per-item
   accuracy changes for these candidates against their 512-token rows.
   The pilot diagnoses truncation; it cannot replace a full-pool D1 comparison.
4. If base or median candidate cap-hit rate remains above 10%, test 2,048 on
   the same pilot before committing to a full table. This is a resource gate,
   not proof that a lower cap-hit rate removes all bias.
5. Only after the pilot, complete a matched long-cap table if warranted. Keep
   K=5 as the primary comparison and original/answer-only rewards clearly
   separated. Afterwards vary only candidate seed while holding item seed
   20261013 and the selected cap fixed to measure candidate-pool variability.

Implementation update (2026-10-22): the dedicated `build_length_pilot.py`
now supports the explicit candidate prefix and saved finish reasons, with
CPU tests and matching local/server preflight. See
`../length_pilot_32_max1024/README.md`. The model pilot has not run yet.
At the current read-only check, lzn GPU 1 held 35,354 MiB with 100% utilization;
no new generation was launched. GSM8K/Route 2 remains deferred.

## Claim-evidence audit

| Claim | Status | Evidence / limit |
|---|---|---|
| Resume preserved the first 550 candidates | Supported | Exact array equality and complete lengths |
| D1 beats public-only at fixed K=5 here | Supported conditionally | Matched setting, conditional bootstrap |
| Epsilon=8 noise is the main cause of the gap | Not supported | D1 nearly equals non-private |
| K=5–10 universally beats K=50 | Contradicted in new setting | Descriptive K curve |
| 512 tokens resolves truncation | Not supported | 43.745% overall cap hits |
| New/old reversal is caused by item seed | Unidentified | Multiple factors changed together |
| n private examples equal a known Q public examples | Unidentified | No measured crossover |
| Results rank DP designs across settings | Unsupported | Too few independent pools/settings |

Audit mode: numeric-audit and claim-audit. No citations were added or audited.
No results were invented; historical failure evidence is retained. Highest
remaining scientific concern: output-cap confounding. Next work: paired
long-cap pilot implementation and experiment execution when resources allow.
