"""E1: weight drift from RandOpt's in-place perturb/restore loop in bfloat16.

RandOpt's sampling loop calls perturb_self_weights(seed, sigma) then
restore_self_weights(seed, sigma) on the same engine for every candidate
(randopt.py, run_sampling). Each call rounds to bf16, so the base weights
random-walk. This emulates the loop with exact bf16 rounding in numpy.
Writes results/e1_bf16_drift.md.
"""
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "e1_bf16_drift.md")


def bf16(x):
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    u = ((u + ((u >> 16) & 1) + 0x7FFF) & 0xFFFF0000).astype(np.uint32)
    return u.view(np.float32)


def run(w_std, sigma, cycles, d=200_000, seed=0, marks=(1, 10, 100, 1250)):
    rng = np.random.default_rng(seed)
    w0 = bf16(rng.normal(0, w_std, d))
    w = w0.copy()
    out = {}
    for k in range(1, cycles + 1):
        noise = bf16(rng.standard_normal(d))          # torch.randn(..., dtype=bf16)
        step = bf16(np.float32(sigma) * noise)       # sign * scale * noise in bf16
        w = bf16(w + step)                           # p.data.add_(...)
        w = bf16(w - step)                           # restore
        if k in marks:
            out[k] = float(np.sqrt(np.mean((w - w0) ** 2)) / sigma)
    lost = float(np.mean(bf16(w0 + bf16(np.float32(sigma) * bf16(rng.standard_normal(d)))) == w0))
    return out, lost


lines = ["# E1: bf16 drift of the base weights in RandOpt's perturb/restore loop\n",
         "RMS drift of the base weights, in units of sigma, after k perturb+restore cycles "
         "(1,250 cycles = N 5,000 candidates on 4 engines). 'lost' = share of weights the "
         "perturbation leaves unchanged after bf16 rounding.\n",
         "| weight std | sigma | k=1 | k=10 | k=100 | k=1250 | lost |", "|---|---|---|---|---|---|---|"]
for w_std in (0.01, 0.02, 0.05):
    for sigma in (5e-4, 1e-3, 2e-3):
        o, lost = run(w_std, sigma, 1250)
        lines.append(f"| {w_std} | {sigma} | " + " | ".join(f"{o[k]:.3f}" for k in (1, 10, 100, 1250))
                     + f" | {lost:.2f} |")
lines.append("\nFix: evaluate candidate i as copy(base) + sigma_i * noise(seed_i) (RandOpt's own "
             "`apply_perturbation`, which restores from a stored base copy) so every scored model is "
             "exactly reproducible from its seed.\n")
open(OUT, "w").write("\n".join(lines))
print("\n".join(lines))
