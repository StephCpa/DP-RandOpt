# DP-RandOpt Project Status

Date: 2026-10-23

## Current status

Latest update 2026-10-23: fixed future Countdown cap at 1024 and removed the
10% truncation gate. CPU scoring of saved-trajectory prefixes found only one
extra correct perturbed test response beyond 1024 (out of 15,500). Random-5
perturbation voting is now the principal data-free comparator, with checked
pass@5 alongside vote. At N=32/cap=1024, random vote is 41.19%, public 53.40%,
D1 48.34%; these are distinct from the full N=1001/512 results. Production
batched repeatability and base self-consistency controls started on lzn GPU 0
at 08:16, PID 1727851. See `selection_decomposition/REPORT.md` for the current
decision, prefix caveats, member-quality breakdown and running-control protocol.
Earlier conclusions below are retained as dated history and are superseded
where they rely on the removed cap-hit gate or batch-size-one proposal.

Update 2026-10-23: both 32-candidate length pilots are complete and audited
locally. The 2048-token pilot finished on 2026-10-22 at 18:50 Asia/Shanghai;
all 49,600 records passed validation and 34 downloaded source-file hashes
match the server. Truncation remains 33.01%. At fixed K=5, N=32, n=1000,
Q=50, epsilon=8 and delta=1e-5, D1 is 46.92% versus public-only 49.00%.
This is a subset diagnostic, not the full N=1001 result. Cross-cap text
divergence prevents attributing changes solely to the cap. No new GPU job
was launched during this audit. Full expansion is deferred; the next bounded
step is a public-item decoding reproducibility check. See
`length_pilot_32_max2048/REPORT.md` for evidence, uncertainty and the protocol.

Historical launch record: the paired 32-candidate, 1,024-token pilot started on gpu-lzn GPU 0 on
2026-10-22 at 11:00 Asia/Shanghai (wrapper PID 1136710). Initialization passed
after adapting KV profiling to the shared GPU. See
`length_pilot_32_max1024/README.md` for the unchanged manifest, initial failed
attempt, actual memory allocation, runtime limits and log location.

Route 1 completed all 1001 candidates on 2026-10-21 at 10:02 Asia/Shanghai.
The final table has been downloaded and its SHA-256 matched the server copy.
All 1,551,550 output lengths are present. The first 550 rows exactly match
the archived partial table in rewards, answer codes, seeds, sigmas, and lengths.
The updated CPU audit and decision record are in
`completed_route1_audit/REPORT.md`. The dated sections below retain earlier
snapshots and are not descriptions of an active job.

Status refresh: 2026-10-17. No new DP-RandOpt experiment was started. A live
read-only check found both GPU 0 and GPU 1 occupied by separate VLLM engine
processes at about 35--36 GiB each and 100% utilization, so the prepared
resume is waiting for real headroom.

## Historical server run (2026-10-13)

Route 1 was running on `gpu-lzn` / `gfkd-R2280-V6-F201`, physical GPU 1,
sharing the card with the user's existing P6-RolltheDice process. The DP-RandOpt
job used one engine and `CUDA_VISIBLE_DEVICES=1`:

- Qwen2.5-3B-Instruct, 1001 candidates, pool/public/test `1000/50/500`
- `seed=20261013`, `item_seed=20261013`, `max_tokens=512`, bfloat16
- Remote PID: `3848917` (exited on CUDA OOM)
- Log: `~/dprandopt_run/logs/lzn_gpu1_route1_3b_n1000_seed20261013_max512.log`
- Output: `~/dprandopt_run/tables/lzn_gpu1_route1_3b_n1000_seed20261013_max512.npz`

The run stopped on 2026-10-14 after the log reached `561/1001`; the latest
partial checkpoint safely contains `550/1001` completed candidates and 852,500
recorded output lengths. The failure was a CUDA OOM during `apply_perturbation`
after the shared P6 process grew to about 15.3 GiB. The remaining 451
candidates would take roughly 23 hours at the previous rate, but resumption
requires a lower vLLM memory reservation or a less memory-intensive shared
workload.

## Scope

This document records the current implementation, execution validation, and
evidence status for the DP-RandOpt repository. The target problem is private
selection among data-independent weight perturbations. The present real-model
evidence uses Qwen2.5 on generated Countdown tasks.

The completed runs establish that the end-to-end score-table pipeline works.
They do not yet establish a general utility claim across models or tasks.

## Repository and environment

- Repository: `项目H-DP-RandOPT/`
- Upstream RandOpt checkout: `third_party/RandOpt/`
- Server staging path: `~/dprandopt_run` on `gpu-gfkd`
- Server environment: conda environment `llm`
- Server stack: PyTorch 2.6.0+cu118, vLLM 0.8.5.post1, Ray 2.46.0,
  Transformers 4.51.3
- Server hardware: eight NVIDIA RTX A6000 GPUs
- Local hardware: NVIDIA RTX 2080 Ti through WSL2
- Local precision: float16, because the 2080 Ti has compute capability 7.5
- Server precision: bfloat16

The score-table builder now creates the parent directory of `--out` before
writing checkpoints. It also accepts `--precision bfloat16|float16`, so the
same entry point works on both the A6000 and the 2080 Ti.

The builder now separates the candidate/noise seed from the dataset item seed:
`--seed` controls candidate perturbations and decoding, while `--item-seed`
controls generated items and their split. If omitted, `--item-seed` defaults to
`--seed` for backward compatibility.

New score tables also store optional per-response `output_lengths` in the
`.npz` file and in partial checkpoints, so the next run can test the truncation
hypothesis directly.

## Execution validation

The local 2080 Ti run completed end to end with Qwen2.5-0.5B, three candidates,
pool/public/test sizes 2/1/1, and 32 maximum generation tokens. Model loading,
perturbation application, generation, reset, checkpointing, and final table
writing all succeeded. The tiny test accuracy was zero; this is an execution
check, not a utility result.

The server smoke tests then validated the same path on one A6000:

| Model | Candidates | Pool / public / test | Result |
|---|---:|---:|---|
| Qwen2.5-0.5B | 3 | 2 / 1 / 1 | Execution path passed; accuracy zero |
| Qwen2.5-0.5B | 11 | 10 / 5 / 5 | Throughput/scaling check; accuracy zero |
| Qwen2.5-3B | 5 | 8 / 4 / 8 | Nonzero base accuracy; exploratory G1 failed |
| Qwen2.5-3B | 11 | 10 / 5 / 10 | Exploratory G1 passed; too small for a claim |

## Expanded Qwen2.5-3B results

Two independent score tables were generated at the larger scale. Each used
1001 candidates (base plus 1000 perturbations), pool/public/test sizes
1000/50/500, and 128 maximum generation tokens.

| Seed | Runtime | Throughput | Base test accuracy | Best plain-RandOpt K | Best accuracy | G1 gain |
|---|---:|---:|---:|---:|---:|---:|
| 20261005 | 11.20 h | 4480 tok/s | 0.036 | 10 | 0.248 | 12.4 points |
| 20261008 | 11.08 h | 4550 tok/s | 0.042 | 5 | 0.258 | 10.6 points |

Both seeds passed the exploratory G1 check. Both also showed gains at small K
and a decline at larger K. The existing `gate_G1` field is a legacy gate at
`K=50`; it should be read as `G1_at_K50`, not as the primary gate for the
public protocol. The public-only audit below fixes `K=5`, so future formal
tables should report the gate at that frozen public choice.

Artifacts:

- [Seed 20261005 score table](server_3b_n1000.npz)
- [Seed 20261005 Phase 0](server_3b_n1000_phase0.json)
- [Seed 20261005 Phase 1](server_3b_n1000_phase1.csv)
- [Seed 20261008 score table](server_3b_n1000_seed20261008.npz)
- [Seed 20261008 Phase 0](server_3b_n1000_seed20261008_phase0.json)
- [Seed 20261008 Phase 1](server_3b_n1000_seed20261008_phase1.csv)

## Public-protocol audit

The initial `--select-on public` analysis was identified as a diagnostic
substitution: it replaces the private pool with public rewards, but the script
still uses test labels to report the K curve. It therefore cannot be presented
as a test-free K-selection protocol.

The corrected CPU-only audit uses a deterministic 25/25 split of the 50 public
columns. It selects the smallest K with the highest validation reward, then
freezes that K for private evaluation. The pooled choice is K=5 for both
seeds.

At epsilon 8, private vote accuracy over 20 selection repeats was:

| Design | Seed 20261005 | Seed 20261008 |
|---|---:|---:|
| D1 | 0.219 | 0.251 |
| D7 | 0.236 | 0.194 |
| D8 | 0.233 | 0.042 |

D8's second-seed fallback to the base model is a real instability signal and
must remain in the failure analysis. It should not be hidden by reporting only
the first seed.

The four CPU-only follow-up checks were then run on the two existing tables
(100 selection repeats; no model generation):

| Seed | Public-only K=5 | Non-private K=5 | D1 n=200 / 500 / 1000 | D8 raw | D8 + public fallback | D8 release / fallback |
|---|---:|---:|---|---:|---:|---:|
| 20261005 | 0.278 | 0.218 | 0.223 / 0.222 / 0.220 | 0.237 | 0.237 | 100% / 0% |
| 20261008 | 0.280 | 0.258 | 0.234 / 0.249 / 0.254 | 0.047 | 0.278 | 3% / 97% |

The public-only baseline is higher than D1 at every tested private pool size in
both seeds. This setting therefore does not yet show a private utility gain at
epsilon 8. The reward-bin truncation proxy found 842/846 of the base pool
responses at reward `<=0.01`, 109/107 at `0.05--0.11`, and 49/47 at `>=1.0`;
these bins are consistent with a truncation concern but cannot prove it because
the score tables do not store generated token lengths.

- [Public protocol audit JSON](public_protocol_audit.json)
- [Existing-table CPU diagnostics](existing_table_diagnostics.json)
- [Protocol boundary and next-step note](protocol_audit.md)
- [Audit script](../../scripts/audit_public_protocol.py)
- [Existing-table diagnostic script](../../scripts/audit_existing_tables.py)

## Current limitations

1. The real-model evidence covers one model family and one generated task.
2. The public split contains only 50 examples and is split again for tuning
   and validation, so public K selection is noisy.
3. The current expanded results are not a complete comparison against the
   closest prior methods or a second task.
4. The legacy G1 field is evaluated at K=50; it must be renamed or recomputed
   at the public-frozen K before it is used as a primary claim.
5. The base reward bins are only a truncation proxy; exact length diagnostics
   require saving per-response lengths in a future score table.
6. Two seeds swap the D1/D7 ranking, so they are not enough to rank designs.
7. The D8 result is seed-sensitive and requires explicit release/fallback
   reporting.

## 550-candidate partial-run audit (CPU-only, 2026-10-17)

The interrupted Route 1 run was converted from its checkpoint into a valid
550-candidate table using candidate seed and item seed `20261013`. The first
550 candidates are a prefix of the pre-drawn random candidate list, so this is
an audit of the finished data rather than a new model run.

At `max_tokens=512`, length-cap hits remain substantial: 42.8% of base test
responses, 41.4% of pool responses, and a median 43.8% of candidate responses
have length exactly 512. This setting therefore still needs a longer-token
check before the result can be treated as an unconstrained answer-quality
measurement.

For this new item seed, base test accuracy is 0.168. Public-only K=5 accuracy
is 0.508, while D1 at epsilon 8 is 0.5792 (paired bootstrap 95% CI for
D1-minus-public: [0.034, 0.108]). The answer-only reward re-selection gives
0.516 versus 0.5813, and the public top-5 overlaps the private top-50 in 3/5
candidates (4/5 with answer-only rewards). The earlier 128-token runs still
show the opposite ordering on both seeds. Items, candidate pools, and output
caps changed together, so this contrast cannot isolate an item-seed effect
or a token-budget effect.

The new equivalent-size curve is in `partial_public_equivalence.csv`, with
the plot in `partial_public_equivalence.png`; the complete machine-readable
diagnostics are in `partial_public_diagnostics.json`.

## Memory-safe resume preparation

The remote Route 1 staging tree now has the following changes, but the job is
intentionally not resumed yet:

- `--gpu-memory-utilization` is exposed and can be set to `0.40`;
- `--base-weights-device cpu` stores the pristine model copy in pinned CPU
  memory, saving roughly 6 GB of GPU memory;
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` can be set in the launch
  environment;
- `scripts/run_resumable.py` retries a failed command while preserving the
  existing `.partial.npz` checkpoint;
- checkpoint validation reloads and shape-checks `output_lengths`, so the
  first 550 candidates retain their saved lengths on resume.

The applied worker patch is recorded in
`scripts/patches/worker_extn_cpu_base.patch`. The recommended resume uses
`--checkpoint-every 10`, `--gpu-memory-utilization 0.40`, and
`--base-weights-device cpu`; it still must be scheduled only after confirming
the shared GPU has sufficient headroom.

## Historical recommendation (before Route 1 completion)

The four CPU-only checks are complete, but they do not justify route 2 yet:
the public-only K=5 baseline currently beats D1, and the 128-token tables have
an unresolved truncation signal. The next GPU run should therefore be route 1
with a fixed item seed, `--max-tokens >= 512`, and all eight GPUs
(`--num-engines 8`). It will save response lengths and report public-only,
non-private, D1/D7/D8, and D8 release/fallback under the same frozen K=5 rule.
Do not start that GPU run until its command and output schema are updated.

The present state contains enough evidence to validate the implementation and
motivate the next test. It does not yet support a broad claim that DP-RandOpt
is superior across tasks or models.

## Reproducibility entry points

The main execution entry point is:

```bash
PYTHONPATH=/path/to/DP-RandOpt:/path/to/RandOpt \
python scripts/build_score_table.py \
  --dataset countdown --model /path/to/model \
  --n-candidates N --pool POOL --public PUBLIC --test TEST \
  --num-engines 1 --tp 1 --max-tokens 128 \
  --item-seed 20261005 \
  --out tables/run.npz
```

CPU analysis is performed with:

```bash
python scripts/analyze_score_table.py tables/run.npz \
  --out results/run --ns 200,500,1000 --ks 1,5,10,50,100 \
  --eps 1,4,8 --designs D1,D7,D8 --reps 10
```

The public-only audit is:

```bash
python scripts/audit_public_protocol.py \
  results/gpu-smoke/server_3b_n1000.npz \
  results/gpu-smoke/server_3b_n1000_seed20261008.npz \
  --out results/gpu-smoke/public_protocol_audit.json
```

## Resume verified — 2026-10-20 22:09 Asia/Shanghai

Resumed the existing 512-token run on gpu-lzn physical GPU 1 (NVIDIA L40).
Wrapper PID: 40020. Log: /home/lzn/dprandopt_run/logs/route1_resume_20261020.log.
Checkpoint preflight confirmed 550 completed rows and 852,500 nonzero lengths.
New progress: 551/1001 at 0.02 h and 552/1001 at 0.05 h; throughput 6,548 generated tokens/s.
GPU 1 uses 17,482 MiB of 46,068 MiB with 81% utilization. No startup/OOM failure observed.
Worker environment confirms RANDOPT_BASE_WEIGHTS_DEVICE=cpu and PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
Uses gpu-memory-utilization=0.40, checkpoint-every=10, and at most three automatic retries.
449 candidates remain; initial estimate is 11–13 hours, approximately October 21 09:00–11:00 Asia/Shanghai, subject to candidate lengths and contention.
The two newly completed candidates are in memory; next scheduled checkpoint is at 560/1001.
This completes the existing 512-token setting and does not resolve its length-cap concern. Keep its outputs separate from any future longer-token experiment.
