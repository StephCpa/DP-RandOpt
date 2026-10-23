# 2048-token paired pilot

Completed: 2026-10-22 18:50 Asia/Shanghai, 32/32 candidates, 49,600 responses.
Full local validation and CPU analysis completed 2026-10-23. See [REPORT.md](REPORT.md).
The startup observations and ETA below are historical, not current job status.

Started: 2026-10-22 15:33 Asia/Shanghai, gpu-lzn physical GPU 0 (L40).

- Wrapper PID: `1299614`.
- Log: `/home/lzn/dprandopt_run/logs/length_pilot_32_max2048_20261022_153336.log`.
- Remote output: `/home/lzn/dprandopt_run/tables/length_pilot_32_max2048/`.
- Manifest SHA-256:
  `5ae3573e1a4ba9405e7698dcab9d4ce88c23d8f3ae6baafe67db4beda7580c8d`.
- Candidates: base plus perturbation indices 1–31 from the original table.
- Private/public/test items: 1000/50/500, item seed 20261013.
- The manifest differs from the completed 1024 pilot only in `max_tokens`.

Context limit is computed from the longest rendered prompt plus the requested
output budget, rounded up to a power of two (4096 for this run). Other runtime
settings remain CPU pinned base weights, 128 maximum sequences, 2048 batched
tokens, expandable allocator segments, and vLLM memory fraction 0.65.
The fraction is interpreted by this vLLM version's shared-GPU memory profiler;
actual GPU occupation must be monitored separately.

Four tests passed locally and remotely, including rejection of a 2048-token
context that leaves no room for the prompt. Manifest checks passed on both
hosts. The launcher has a shared lock across cap settings, validates available
memory, and preserves independent output directories. One atomic gzip response
bundle is written per candidate, with up to three retries on failure.

The goal is a length-cap diagnostic, not a new full-pool D1 result. The 10%
cap-hit gate remains a resource decision threshold, not proof of absent bias.
No larger generation run is chained to this pilot.

## Startup and first-candidate verification

At approximately 15:41, candidate 0 had completed and its 1,550 records passed
validation. The log recorded `elapsed_s=357.4 tok_s=3637`, including startup.
Worker 1300888 was observed actively computing. Initialization allocated a
636,256-token KV cache for the 4096-token context. GPU memory occupied about
29,582 MiB with 15,876 MiB free. No failure/retry was observed for this launch.
The provisional remaining time is 2.5–4 hours; this is based on one candidate
and can change with response lengths and other GPU jobs.

The first base result is 17.0% test accuracy (1024: 19.0%; 512: 16.8%) with
449/1550 responses (28.97%) finishing at the 2048 length cap. It is a partial
diagnostic, not evidence that a longer cap improves accuracy.

Of the 1104 base responses that finished normally at 1024, only 261 (23.64%)
have exactly identical text in the 2048 run. These observations cannot be
treated as literal extensions of identical decoding trajectories. Context
and batch execution can change numerical decoding behavior; its precise cause
has not been isolated. The remaining candidates continue as an exploratory
cap comparison. A causal truncation analysis would need matched trajectories,
for example recorded output token IDs and prefix scoring in a future run.

Source: [paired_analysis_first_candidate.json](paired_analysis_first_candidate.json).
