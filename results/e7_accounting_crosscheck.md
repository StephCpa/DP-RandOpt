# E7: independent cross-check of the accountant

Reference implementation: AutoDP 0.2.3.1 (the accountant DP-ES uses), run from source
because its wheel does not build on this image. Target delta' = 1e-5.

| mechanism | AutoDP eps | dprandopt eps | note |
|---|---|---|---|
| EM eps0 = 0.443, 50 picks | 7.9878 | 7.9941 | BR-RDP vs BR-zCDP + CKS: agree to 0.1% |
| EM eps0 = 0.991, 10 picks | 7.9918 | 7.9983 | agree to 0.1% |
| EM eps0 = 0.070, 50 picks | 1.0012 | 1.0012 | identical |
| EM eps0 = 1.6, 5 picks | 9.3669 | 8.0000 | dprandopt also takes pure composition (5 x 1.6 = 8), which is tighter here |
| Gaussian sigma = 1 (rho = 0.5) | 4.3772 | 4.7289 | AutoDP uses the exact Gaussian curve; a generic rho-zCDP conversion must be looser |

DP-ES's own bound also reproduces exactly with AutoDP 0.2.3.1: `scripts/compute_privacy.py`
prints epsilon = 0.705168 at delta = 1e-05 (18 releases, sampling rate 0.05).
