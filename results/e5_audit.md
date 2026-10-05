# E5: auditing selection mechanisms

Simulated Countdown-like table, N = 2000 candidates, rewards in [0, 1.1].

## (a) Membership inference with natural examples

Statistic for example j: mean reward of the selected members on j minus the mean over all candidates. Members = the n private examples; non-members = n held-out pool examples. AUC averaged over 10 private subsamples (0.5 = no signal).

| n | K | plain RandOpt | D1 eps=8 | D1 eps=4 | D1 eps=1 |
|---|---|---|---|---|---|
| 200 | 1 | 0.511 | 0.511 | 0.511 | 0.512 |
| 200 | 50 | 0.529 | 0.529 | 0.531 | 0.513 |
| 1000 | 1 | 0.508 | 0.508 | 0.508 | 0.508 |
| 1000 | 50 | 0.513 | 0.513 | 0.512 | 0.514 |

## (b) How stable is plain RandOpt's top-K set?

g_K = U_(K) - U_(K+1), the sum-score gap at the selection boundary. One replaced record moves a score difference by at most 2R, so the top-K set can change with one record only if g_K < 2R; d = ceil(g_K / 2R) is the distance to instability. 'PTR release' is the probability that propose-test-release (Dwork & Lei 2009; Thakurta & Smith 2013) would publish the exact non-private top-K set: d + Lap(1/eps) > ln(1/delta)/eps, which is (eps, 1e-5)-DP. 200 private subsamples.

| n | K | P(g_K < 2R) | median d | PTR release eps=1 | eps=4 | eps=8 | candidates within 2R of the boundary (median) |
|---|---|---|---|---|---|---|---|
| 200 | 1 | 0.17 | 3 | 0.01 | 0.60 | 0.83 | 1 |
| 200 | 5 | 0.65 | 1 | 0.00 | 0.11 | 0.36 | 3 |
| 200 | 10 | 0.90 | 1 | 0.00 | 0.01 | 0.11 | 5 |
| 200 | 50 | 1.00 | 1 | 0.00 | 0.00 | 0.01 | 18 |
| 1000 | 1 | 0.08 | 5 | 0.11 | 0.77 | 0.92 | 1 |
| 1000 | 5 | 0.10 | 5 | 0.09 | 0.72 | 0.90 | 1 |
| 1000 | 10 | 0.38 | 2 | 0.00 | 0.28 | 0.62 | 2 |
| 1000 | 50 | 0.81 | 1 | 0.00 | 0.02 | 0.21 | 4 |

PTR release probability = P[Lap(1/eps) > thr - d] (exact Laplace tail). Plain RandOpt's set is usually one-record stable for small K, and unstable for K = 50, where many candidates crowd the boundary.

## (c) Crafted worst-case canary, many runs

A replace-one pair (D, D') moving the scores of the candidates ranked K+1..2K up by R and all others down by R. Test: at least h of the K selected members lie in that set (h chosen on half the runs, evaluated on the other half); Clopper-Pearson 95%. 'exact delta' is the Plackett-Luce privacy curve of the same pair at the accounted eps (Monte Carlo, 4000 samples); the accountant promises <= 1e-5. 'max loss' is the largest privacy loss sampled.

| design | K | accounted eps | empirical eps lower bound (20k runs) | exact delta at accounted eps | max loss |
|---|---|---|---|---|---|
| plain RandOpt | 1 | inf | 0 for this pair (gap > 2R); not DP in general | - | - |
| D1 | 1 | 1 | 0.68 | 0.0e+00 | 0.69 |
| D1 | 1 | 4 | 3.39 | 0.0e+00 | 3.76 |
| D1 | 1 | 8 | 5.13 | 0.0e+00 | 7.92 |
| plain RandOpt | 10 | inf | 0 for this pair (gap > 2R); not DP in general | - | - |
| D1 | 10 | 1 | 0.00 | 0.0e+00 | 0.57 |
| D1 | 10 | 4 | 0.16 | 0.0e+00 | 1.07 |
| D1 | 10 | 8 | 0.70 | 0.0e+00 | 0.99 |
| plain RandOpt | 50 | inf | inf (output changes deterministically) | - | - |
| D1 | 50 | 1 | 0.00 | 0.0e+00 | 0.71 |
| D1 | 50 | 4 | 0.80 | 0.0e+00 | 1.55 |
| D1 | 50 | 8 | 0.30 | 0.0e+00 | 2.32 |

## (d) What a one-run audit can certify

A one-run audit (Steinke, Nasr & Jagielski 2023) with r guesses certifies at most the eps below, even if every guess is right. Only canaries whose target candidate sits within 2R of the selection boundary can change a selection output, so r is capped by the last column of table (b) (single-target canaries), whatever the true leakage.

| r (all correct) | certified eps (95%) |
|---|---|
| 5 | 0.20 |
| 10 | 1.05 |
| 20 | 1.82 |
| 50 | 2.78 |
| 100 | 3.49 |
| 500 | 5.11 |

Plain RandOpt is not DP for any finite eps, yet a one-run audit with a few dozen informative canaries certifies eps < 3. For RQ4 use many runs on the stored table (c) or the exact privacy curve, and report natural-example AUCs (a) separately as average-case leakage.
