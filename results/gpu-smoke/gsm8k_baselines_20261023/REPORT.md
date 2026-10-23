# GSM8K full-test baselines

Completed 2026-10-23 14:38 Asia/Shanghai on lzn GPU 0 (L40).
Recorded elapsed time: 527.36 seconds including engine initialization.
All 1,319 test items have one greedy response and five stochastic responses
(7,914 total). Both post-generation weight checks passed before output save.
All five downloaded files match the server SHA-256 hashes.

| Metric | Greedy | Five samples |
|---|---:|---:|
| Mean single-response accuracy | 85.7468% | 84.2305% |
| Plurality vote accuracy | 85.7468% | 89.3101% |
| Oracle pass@5 | n/a | 94.8446% |
| Token-cap finish rate | 0.1516% | 0.1365% |

Self-consistency improves vote accuracy by 3.5633 percentage points.
This is one fixed sampling realization, not replicated sampling evidence.
GSM8K pass@5 requires reference labels and is only an oracle diagnostic.
No random-perturbation or private-selection performance is yet available.

The next comparison uses the frozen pool protocol in ../gsm8k_ready/PLAN.md.
The first full pool launched at 16:08; no other candidate seed was launched.
