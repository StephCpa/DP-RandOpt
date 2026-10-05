# E1: bf16 drift of the base weights in RandOpt's perturb/restore loop

RMS drift of the base weights, in units of sigma, after k perturb+restore cycles (1,250 cycles = N 5,000 candidates on 4 engines). 'lost' = share of weights the perturbation leaves unchanged after bf16 rounding.

| weight std | sigma | k=1 | k=10 | k=100 | k=1250 | lost |
|---|---|---|---|---|---|---|
| 0.01 | 0.0005 | 0.016 | 0.047 | 0.113 | 0.278 | 0.04 |
| 0.01 | 0.001 | 0.011 | 0.031 | 0.074 | 0.209 | 0.02 |
| 0.01 | 0.002 | 0.007 | 0.020 | 0.049 | 0.155 | 0.01 |
| 0.02 | 0.0005 | 0.022 | 0.067 | 0.171 | 0.368 | 0.07 |
| 0.02 | 0.001 | 0.016 | 0.047 | 0.113 | 0.278 | 0.04 |
| 0.02 | 0.002 | 0.011 | 0.031 | 0.074 | 0.209 | 0.02 |
| 0.05 | 0.0005 | 0.034 | 0.099 | 0.274 | 0.563 | 0.18 |
| 0.05 | 0.001 | 0.024 | 0.076 | 0.194 | 0.406 | 0.09 |
| 0.05 | 0.002 | 0.018 | 0.053 | 0.129 | 0.303 | 0.04 |

Fix: evaluate candidate i as copy(base) + sigma_i * noise(seed_i) (RandOpt's own `apply_perturbation`, which restores from a stored base copy) so every scored model is exactly reproducible from its seed.
