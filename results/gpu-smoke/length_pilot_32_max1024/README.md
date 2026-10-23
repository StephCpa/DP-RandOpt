# Paired 1024-token pilot: completed

Updated: 2026-10-22, Asia/Shanghai.

Completion update: all 32 candidates finished at 14:19. All 49,600 response
records were validated and downloaded. See [REPORT.md](REPORT.md) for the
complete paired analysis. Dated startup notes below retain the earlier state.

Current launch: wrapper PID `1136710`, started 11:00 Asia/Shanghai, log
`/home/lzn/dprandopt_run/logs/length_pilot_32_max1024_20261022_110025.log`.
The first launch (PID 1128385) failed during KV-cache profiling and was stopped
before candidate generation. A second unrelated approximately 8 GiB workload
had joined GPU 0; the local vLLM 0.8.5 profiler counts other processes in its
non-torch allocation estimate. Therefore 0.40 left no KV budget.

The revised launch uses `gpu_memory_utilization=0.65`, pinned CPU base weights,
`max_model_len=2048`, `max_num_seqs=128`, and `max_num_batched_tokens=2048`.
Prompts plus the 1024-token output budget are checked against the context limit.
Initialization succeeded with 173,616 KV-cache tokens. Observed pilot GPU
allocation was 13,284 MiB, leaving 15,870 MiB free while both other workloads
remained intact. The fraction is a profiler budget, not a claim that this
process holds 65% of the physical card. Runtime settings are saved per attempt.
The original 32-candidate manifest is unchanged. Later dated sections preserve
earlier preparation/launch snapshots.

Verified at approximately 11:08: candidate 0 completed and its 1,550 response
records passed manifest/content validation. Log: `[1/32] candidate=0
checkpoint_saved elapsed_s=396.7 tok_s=2105`. The elapsed time includes engine
startup; generation itself took roughly five minutes. Worker PID 1137851 was
observed actively using the GPU (70% SM in the sample). A provisional remaining
budget is 3–4 hours, sensitive to sharing and candidate response lengths.

First-candidate diagnostic (not a complete pilot result): base test accuracy
is 19.0% vs the reference 16.8%; the paired 95% difference interval is
[-1.4, 5.8] percentage points. 28.774% of all 1,550 responses both hit 1,024
tokens and have `finish_reason=length`, versus 42.129% cap hits at 512. This
supports continued truncation concern, not a significant base-accuracy gain.
Local snapshot: `paired_analysis_first_candidate.json`.

The pilot keeps the original Qwen2.5-3B-Instruct model, greedy decoding seed
20261013, item seed 20261013, and all 1,550 items (1,000 private / 50 public /
500 test). It evaluates candidate indices 0–31 from the completed 1,001-row
512-token table. Seeds and sigmas are loaded from that table and checked
against the original full-size random draw; they are not redrawn for N=32.

The only intended generation-protocol change is the output cap from 512 to
1,024 tokens. vLLM batching/kernel behavior may differ with output lengths;
we do not claim bitwise-identical decoding prefixes, which the old table
cannot verify because it did not save response texts or token IDs.

## Frozen manifest

- Manifest: [manifest.json](manifest.json).
- Manifest SHA-256:
  `e03e55dac244d7876bb49d2e928f1b15330f4de824f2a96752c4178cccf9ef00`.
- Reference SHA-256:
  `687959e266f5317c557679229e2e032e5c8caf843a7f19c5d067ee2ba8070551`.
- Local and remote CPU preflight produced the same manifest fingerprint.
- Remote output: `/home/lzn/dprandopt_run/tables/length_pilot_32_max1024/`.

## Implemented safeguards and outputs

`scripts/build_length_pilot.py` validates the reference table and metadata,
regenerates and hashes the complete items/messages, checks rendered prompts
on resume, and refuses an incompatible manifest in an existing directory.

Each completed candidate is stored as a validated, atomic `.json.gz` file.
It includes candidate ID, seed, sigma, item index, response text, token count,
finish reason, stop reason, extracted voting answer and reward. Incomplete
temporary files are not considered finished checkpoints. Completed bundles
are verified before being skipped on resume. There is one checkpoint per
candidate, so a failure loses at most the current candidate's work.

After all 32 bundles are present, the script writes `table.npz` and
`COMPLETE.json`. The CPU analysis script can also analyze a completed subset
and labels it explicitly as partial. It compares answer correctness rather
than independently assigned wrong-answer codes.

Runtime records include package versions, GPU visibility, memory fraction,
prompt fingerprint and start time. The model uses pinned CPU base weights,
vLLM memory utilization 0.40, and expandable allocator segments. The launcher
uses a file lock to prevent concurrent writers and permits up to three retries.

## Validation actually completed

- Three CPU tests passed locally and in the server's `opcd` environment:
  original full-draw prefix preservation, rejection of mismatched resume
  settings, and atomic checkpoint integrity including text/finish reasons.
- A separate temporary synthetic fixture passed checkpoint-to-final-table
  assembly and paired analysis. It was removed and is not experimental data.
- Python compilation and shell syntax checks passed.
- No actual model inference with the new pilot script has run yet.

## Current resource outcome

The one-shot launcher checked GPU 1 and did not start: only 1,791 MiB was free
at 100% utilization. It requires at least 24,576 MiB free and utilization at
most 20%. No other job was stopped or changed. No background queue, monitor,
or automatic future launch was created.

To launch once GPU 1 has headroom:

```bash
ssh gpu-lzn 'bash /home/lzn/dprandopt_run/scripts/launch_length_pilot_lzn.sh'
```

To analyze saved candidates on the server (CPU only):

```bash
cd /home/lzn/dprandopt_run
PYTHONPATH=. /home/lkx/miniconda3/envs/opcd/bin/python scripts/analyze_length_pilot.py \
  --reference tables/lzn_gpu1_route1_3b_n1000_seed20261013_max512.npz \
  --pilot-dir tables/length_pilot_32_max1024
```

This pilot diagnoses the length cap; it cannot establish full-pool D1 utility.
After completion, inspect base and median candidate cap/length-finish rates
before deciding whether to test 2,048 tokens or build a full 1,024-token table.

## GPU 0 launch — 2026-10-22

The user redirected execution from gpu-gfkd GPU 6 to gpu-lzn GPU 0.
GPU 0 had 37,288 MiB free with 13–38% sampled utilization and an existing
approximately 8 GiB workload. The launch retained a 24,576 MiB free-memory
floor and used a 60% utilization ceiling for this explicitly shared run.
The 32-candidate 1024-token pilot started at 10:56 Asia/Shanghai, wrapper PID
1128385, log `/home/lzn/dprandopt_run/logs/length_pilot_32_max1024_20261022_105612.log`.
The reference, model, environment, and frozen manifest remain on gpu-lzn;
the proposed A6000 migration and additional same-host 512-token rerun are not
part of this launch. GPU inference/checkpoint verification is pending below.
