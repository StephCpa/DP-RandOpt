# E0: checks of the plan's arithmetic

## 1. The 'What the bound implies' table

Slack t in accuracy points; density gamma = 0.05; 95% guarantee for all K members; delta' = 1e-5.

**(a) Plan as written** (simple zCDP conversion, density unchanged across picks, reward range 1):

| eps | K | eps0 | n=200 | n=1000 | n=5000 |
| --- | --- | --- | --- | --- | --- |
| 4 | 1 | 4.00 | 1.5 | 0.3 | 0.1 |
| 4 | 10 | 0.49 | 17.0 | 3.4 | 0.7 |
| 4 | 50 | 0.22 | 45.4 | 9.1 | 1.8 |
| 8 | 1 | 8.00 | 0.7 | 0.1 | 0.0 |
| 8 | 10 | 0.92 | 9.1 | 1.8 | 0.4 |
| 8 | 50 | 0.41 | 24.2 | 4.8 | 1.0 |

**(b) Corrected**: tight CKS conversion; peeling correction with N = 1000 (the plan's experiment) and N = 5000; Countdown reward range R = 1.1. `inf` means gamma*N < K, so the guarantee is void.

| eps | K | eps0 (CKS) | n=200,N=1k | n=200,N=5k | n=1000,N=1k | n=1000,N=5k | n=5000,N=1k | n=5000,N=5k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1.000 | 6.6 | 6.6 | 1.3 | 1.3 | 0.3 | 0.3 |
| 1 | 10 | 0.156 | 59.7 | 58.6 | 11.9 | 11.7 | 2.4 | 2.3 |
| 1 | 50 | 0.070 | 216.6 | 159.1 | 43.3 | 31.8 | 8.7 | 6.4 |
| 2 | 1 | 2.000 | 3.3 | 3.3 | 0.7 | 0.7 | 0.1 | 0.1 |
| 2 | 10 | 0.294 | 31.7 | 31.1 | 6.3 | 6.2 | 1.3 | 1.2 |
| 2 | 50 | 0.132 | 115.1 | 84.5 | 23.0 | 16.9 | 4.6 | 3.4 |
| 4 | 1 | 4.000 | 1.6 | 1.6 | 0.3 | 0.3 | 0.1 | 0.1 |
| 4 | 10 | 0.546 | 17.1 | 16.8 | 3.4 | 3.4 | 0.7 | 0.7 |
| 4 | 50 | 0.244 | 62.0 | 45.5 | 12.4 | 9.1 | 2.5 | 1.8 |
| 8 | 1 | 8.000 | 0.8 | 0.8 | 0.2 | 0.2 | 0.0 | 0.0 |
| 8 | 10 | 0.991 | 9.4 | 9.2 | 1.9 | 1.8 | 0.4 | 0.4 |
| 8 | 50 | 0.443 | 34.2 | 25.1 | 6.8 | 5.0 | 1.4 | 1.0 |

**(c) Designs that escape the K bottleneck** (R = 1.1, N = 5000, gamma = 0.05, 95%): D1 = K Gumbel picks (CKS); D7 = one EM over K-subsets (pure eps, all members good simultaneously); D3 = one pick (pure eps), neighbours data-free. Add/remove adjacency with monotone scores halves every entry.

| eps | K | n | D1 slack | D7 slack | D3 expert slack |
| --- | --- | --- | --- | --- | --- |
| 1 | 10 | 200 | 58.6 | 36.4 | 6.6 |
| 1 | 10 | 1000 | 11.7 | 7.3 | 1.3 |
| 1 | 10 | 2000 | 5.9 | 3.6 | 0.7 |
| 1 | 50 | 200 | 159.1 | 173.6 | 6.6 |
| 1 | 50 | 1000 | 31.8 | 34.7 | 1.3 |
| 1 | 50 | 2000 | 15.9 | 17.4 | 0.7 |
| 2 | 10 | 200 | 31.1 | 18.2 | 3.3 |
| 2 | 10 | 1000 | 6.2 | 3.6 | 0.7 |
| 2 | 10 | 2000 | 3.1 | 1.8 | 0.3 |
| 2 | 50 | 200 | 84.5 | 86.8 | 3.3 |
| 2 | 50 | 1000 | 16.9 | 17.4 | 0.7 |
| 2 | 50 | 2000 | 8.5 | 8.7 | 0.3 |
| 4 | 10 | 200 | 16.8 | 9.1 | 1.6 |
| 4 | 10 | 1000 | 3.4 | 1.8 | 0.3 |
| 4 | 10 | 2000 | 1.7 | 0.9 | 0.2 |
| 4 | 50 | 200 | 45.5 | 43.4 | 1.6 |
| 4 | 50 | 1000 | 9.1 | 8.7 | 0.3 |
| 4 | 50 | 2000 | 4.6 | 4.3 | 0.2 |
| 8 | 10 | 200 | 9.2 | 4.6 | 0.8 |
| 8 | 10 | 1000 | 1.8 | 0.9 | 0.2 |
| 8 | 10 | 2000 | 0.9 | 0.5 | 0.1 |
| 8 | 50 | 200 | 25.1 | 21.7 | 0.8 |
| 8 | 50 | 1000 | 5.0 | 4.3 | 0.2 |
| 8 | 50 | 2000 | 2.5 | 2.2 | 0.1 |

## 2. Per-pick eps0 for K picks at (eps, 1e-5)

| eps | K=1 | K=5 | K=10 | K=50 |
| --- | --- | --- | --- | --- |
| 1 | 1.000 / 1.000 | 0.200 / 0.221 | 0.129 / 0.156 | 0.058 / 0.070 |
| 2 | 2.000 / 2.000 | 0.400 / 0.416 | 0.253 / 0.294 | 0.113 / 0.132 |
| 4 | 4.000 / 4.000 | 0.800 / 0.800 | 0.488 / 0.546 | 0.218 / 0.244 |
| 8 | 8.000 / 8.000 | 1.600 / 1.600 | 0.916 / 0.991 | 0.410 / 0.443 |

Each cell: simple conversion / CKS conversion. K=1 uses pure DP (eps0 = eps).

## 3. Population guarantee, optimal temperature, and lower bound

| n | optimal eps0 | Gibbs slack at optimum | Gibbs slack at eps0=8 | argmax slack N=1k | argmax slack N=5k |
| --- | --- | --- | --- | --- | --- |
| 100 | 1.54 | 34.0 | 66.6 | 46.0 | 49.4 |
| 200 | 1.09 | 24.0 | 61.4 | 32.6 | 34.9 |
| 500 | 0.69 | 15.2 | 57.0 | 20.6 | 22.1 |
| 1000 | 0.49 | 10.8 | 54.9 | 14.6 | 15.6 |
| 2000 | 0.34 | 7.6 | 53.4 | 10.3 | 11.0 |

Population slack in points (gamma_pop = 0.05, beta = 0.05, replace-one). The Gibbs bound does not depend on N; the argmax union bound does.

| eps | gamma | n needed (lower bound, any eps-DP selector) | n sufficient (EM, prob 1/2) |
| --- | --- | --- | --- |
| 0.25 | 0.001 | 477 | 1216 |
| 0.25 | 0.01 | 293 | 848 |
| 0.25 | 0.05 | 164 | 590 |
| 0.5 | 0.001 | 229 | 608 |
| 0.5 | 0.01 | 136 | 424 |
| 0.5 | 0.05 | 72 | 295 |
| 1.0 | 0.001 | 104 | 304 |
| 1.0 | 0.01 | 58 | 212 |
| 1.0 | 0.05 | 26 | 148 |
| 2.0 | 0.001 | 42 | 152 |
| 2.0 | 0.01 | 19 | 106 |
| 2.0 | 0.05 | 3 | 74 |

Target slack t = 5 points. The lower bound is vacuous once eps > ln(1/(2 gamma)); there the EM's remaining gap to optimal is at most the additive constant.

## 4. Information ceiling for plain RandOpt (seeds only)

| N | K | n | log2 C(N,K) bits | bits/example | members inferable at 90% | at 99% |
| --- | --- | --- | --- | --- | --- | --- |
| 5000 | 50 | 200 | 400 | 2.00 | 753 | 435 |
| 1000 | 50 | 1000 | 282 | 0.28 | 532 | 307 |
| 1000 | 1 | 1000 | 10 | 0.01 | 19 | 11 |

Releasing the top-K train rewards, base train reward and per-sigma means (as RandOpt's results.json does) voids this ceiling.

## 5. Compute budget re-derivation

| item | tokens/s | per candidate | GPU-h |
| --- | --- | --- | --- |
| 0.5B, two tasks | 10000 | 75 s | 42 |
| 1.5B, two tasks | 6000 | 125 s | 69 |
| 3B, two tasks | 3500 | 214 s | 119 |
| 7B, Countdown | 1700 | 441 s | 123 |

Scoring subtotal 353 GPU-h + other items 290 = 643 GPU-h (plan: ~645).

Raising N to 2000 (needed for K = 50 at gamma = 0.05, Section 1b) doubles scoring: 995 GPU-h. Scoring the test set only for candidates the mechanisms can reach saves up to a third of scoring.

## 6. Power of the primary comparison

| discordant share | MDE alpha=0.05 | alpha=0.025 | alpha=0.0125 (Holm first step) |
| --- | --- | --- | --- |
| 0.2 | 3.9 | 4.3 | 4.7 |
| 0.3 | 4.8 | 5.3 | 5.8 |
| 0.4 | 5.6 | 6.1 | 6.7 |

Minimum detectable accuracy difference (points) at 80% power, 1000 test items, McNemar normal approximation. Plan: ~5 points (6 after Holm) at 30% discordance -- confirmed.
