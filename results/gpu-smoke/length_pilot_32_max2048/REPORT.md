# Completed length-pilot analysis and decision

Updated: 2026-10-23, Asia/Shanghai. Mode: numeric and claim audit.

Superseding update: the 10% cap-hit gate and batch-size-one proposal below
have been withdrawn. CPU trajectory-prefix and voting decompositions support
fixing cap=1024 and using random-K/checker baselines. Production-batched
controls are now running; see [the current decision](../selection_decomposition/REPORT.md).
All numerical observations below remain archived evidence.

The 2048-token pilot completed all 32 candidates and 49,600 responses on
2026-10-22 at 18:50. Generation took 11,811.9 seconds, including engine startup.
All 34 downloaded source files (32 response bundles, manifest, score table)
match live server SHA-256 hashes. Both pilot tables were reconstructed from
their response bundles and matched exactly in rewards, answer codes, lengths,
candidate seeds and sigmas. No new GPU experiment was started for this audit.

## Matched candidate subset

All three columns below use original candidate indices 0–31, including base,
with the same 1000/50/500 pool/public/test items and item seed 20261013.
The 512-token column is restricted to these 32 candidates; it is not the
previous full 1001-candidate result. Model: Qwen2.5-3B-Instruct.

| Metric | 512 tokens | 1024 tokens | 2048 tokens |
|---|---:|---:|---:|
| Base test accuracy | 16.80% | 19.00% | 17.00% |
| Mean single-candidate accuracy, 31 perturbations | 15.55% | 15.99% | 16.31% |
| All-response cap-hit rate | 43.20% | 33.54% | 33.01% |
| Public-only K=5, Q=50 | 46.60% | 53.40% | 49.00% |
| Nonprivate K=5, n=1000 | 47.20% | 48.20% | 47.00% |
| D1 K=5, n=1000, epsilon=8 | 47.20% | 48.33% | 46.92% |
| Data-free random K=5 | 40.71% | 41.19% | 41.44% |

D1 uses delta=1e-5, the existing reward bound of 1.1, and 400 selection
repeats. Random-K values also average 400 draws. K=5 was fixed before this
audit; no best-K or best-cap claim is made. The point estimates do not license
choosing 1024 tokens because its public-only test accuracy happens to be largest.

The 2048-minus-512 mean single-candidate difference is +0.75 percentage points,
with conditional paired test-item bootstrap 95% interval [0.15, 1.38] points.
The 2048-minus-1024 difference is +0.32 points [-0.31, 0.99]. For base, the
2048-minus-512 difference is +0.20 points [-3.40, 3.80]. Bootstrap uses 5,000
replicates, seed 20261022, resampling 500 test items together across fixed
candidates. These are exploratory, unadjusted intervals, conditional on the
candidate pool and selection data; they do not measure across-pool uncertainty.

## Private-data value and sensitivity checks

At 2048 tokens, D1 minus public-only is -2.08 points with conditional test-item
95% interval approximately [-4.35, 0.15]. D1 minus nonprivate is only -0.08
points. The observed public advantage is therefore not primarily a privacy
noise penalty in this subset. At 1024 tokens D1 minus public is -5.07 points,
with interval approximately [-8.34, -1.84]. These are conditional comparisons,
not general claims that private selection is worse.

| Private examples n | D1 at 512 | D1 at 1024 | D1 at 2048 |
|---|---:|---:|---:|
| 200 | 46.09% | 48.31% | 46.27% |
| 500 | 46.71% | 48.58% | 46.61% |
| 1000 | 47.20% | 48.32% | 46.93% |

These rows use 400 independently sampled subsets/selection draws per n;
small differences from the preceding n=1000 table are Monte Carlo variation.
The public comparator always has Q=50. This audit does not estimate an exact
public-data equivalent or extrapolate beyond the observed 32 candidates.

Randomly breaking public-score ties leaves original-reward public accuracy
unchanged for all three caps. With answer-only rewards at 2048, deterministic
public accuracy is 48.40%, random-tie public accuracy is 47.78%, and D1 is
46.81%; D1 minus random-tie public is -0.98 points [-3.02, 1.05]. Reward choice
and tie handling affect the gap but do not reveal a robust D1 advantage here.
The full 1001-candidate, 512-token D1 result remains separate evidence.

## Why simply increasing the cap is not the next step

All 16,373 cap hits at 2048 have `finish_reason=length`. The rate remains
well above the existing 10% resource gate. Among these truncated responses,
only 5.70% contain a closing answer tag and 0.91% yield a valid voting answer.
Among normally stopped responses, the corresponding rates are 99.95% and
40.66%. Normal termination therefore also does not guarantee a valid formula.

A reproducible repetition heuristic flags 83.96% of truncated 2048 responses,
versus 4.67% of normally stopped responses. It detects a whitespace-normalized
line of at least 20 characters repeated at least three times. This is evidence
of frequent repeated text, not a validated loop classifier or proof that every
truncated response is unproductive. The corresponding 1024 rates are 79.89%
and 4.21%.

Only 7,868 of 32,963 responses that stopped naturally at 1024 have identical
complete text at 2048 (23.87%). Furthermore, 7,090 previously stopped responses
now finish at the longer cap, while 7,354 previously truncated responses now
stop naturally. These are different decoding trajectories, not cleanly nested
continuations. The manifests match except for the cap and rendered prompt
hashes agree, but context limits and runtime execution differ. The cause of
the text divergence is unresolved; numerical execution differences are a
hypothesis, not a demonstrated diagnosis. The original 512 table has no texts.

## Decision and next bounded diagnostic

Do not launch a 1001-candidate 2048-token expansion, a 4096-token escalation,
or GSM8K on the strength of these results. Longer output has not passed the
truncation gate, and this subset does not show a robust private-data benefit.

The next useful GPU work is a small decoding reproducibility diagnostic,
using only the existing 50 public items, base first. Freeze context=4096,
batch size=1, prompt order, model weights, tokenizer, seed and sampling settings.
Save prompt/output token IDs and execution metadata. Repeat identical 2048-token
generation twice without changing settings; check token equality before
comparing caps. If exact repeats fail, investigate execution/weight-reset
reproducibility before expanding the experiment. If they pass, score the same
saved trajectory at token prefixes 512/1024/2048 using the existing parser.
This estimates the effect of allowing more of that recorded trajectory,
without assuming independent runs are nested. It is not proof of equivalence
to a separate capped decode. Repeat for original candidates 1–3 only after
the base check passes, with before/after-reset weight checksums.

This proposed diagnostic is not running. It uses no test items to tune a
prompt or decoding configuration. Any later prompt change must be selected
on public data and confirmed on newly frozen test items, since the existing
test set has been repeatedly inspected.

## Claim-evidence audit

| Claim | Status | Evidence / limit |
|---|---|---|
| Both pilots completed and archives are internally consistent | Supported | Completion manifests, 34 server hashes, exact table reconstruction |
| A 2048-token cap resolves truncation | Unsupported | 33.01% length finishes; 10% gate failed |
| Increasing the cap alone caused an accuracy gain | Unsupported | Cross-run text divergence and runtime confounding |
| D1 beats public-only at long caps in this subset | Unsupported | Fixed-K comparisons and sensitivity checks |
| Privacy noise explains most of the observed gap | Unsupported here | D1 and nonprivate nearly coincide |
| Repeated output contributes to the failure pattern | Partially supported | Explicit repeated-line heuristic; no causal attribution |
| These pilots rank designs or establish performance at N=1001 | Unsupported | One fixed subset and one item seed |

Severity: material limits on causal and general utility claims; no detected
archive/table inconsistency. No new citations or external claims were added.
No-invention status: all reported results derive from saved outputs and the
CPU audit. Next owner: experiment implementation for the bounded reproducibility
check; manuscript claims should remain conditional until it is resolved.

## Reproduction and artifacts

Run from the repository with its package on PYTHONPATH:

```text
python scripts/analyze_length_pilot.py --reference results/gpu-smoke/lzn_route1_n1001_max512.npz --pilot-dir results/gpu-smoke/length_pilot_32_max2048 --previous-pilot-dir results/gpu-smoke/length_pilot_32_max1024
python scripts/audit_length_series.py
```

- [Paired response analysis](paired_analysis.json)
- [Selection, sensitivity, repetition and integrity audit](series_audit.json)
- [Server SHA-256 inventory](server_sha256.txt)
- [Launch and historical startup record](README.md)

Selection RNG seed: 20261023. Completed source response bundles, manifests,
and score tables were preserved. Four existing pilot unit tests passed locally.
