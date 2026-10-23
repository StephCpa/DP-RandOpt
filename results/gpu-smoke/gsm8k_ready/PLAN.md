# GSM8K execution plan

Date: 2026-10-23. Mode: internal experiment design and execution preparation.
No GSM8K model outputs or accuracy results have been generated.

## Question

Does private selection among data-independent perturbations improve utility
over data-free sampling and random-perturbation voting, and what is its cost
relative to public-only and nonprivate selection? Countdown remains secondary
because its prompt-visible arithmetic checker provides a label-free alternative.

## Data and scoring

Use the official [GSM8K source](https://github.com/openai/grade-school-math),
revision `3101c7d5072418e28b9008a6636bde82a006892c`: 7473 training problems and
1319 test problems. The source stores the final numeric label after `####`.
Freeze item seed 20261023. Its permutation of training rows assigns 1000
private-pool examples and 50 disjoint public examples. Use all 1319 official
test problems in source order. Exact-question overlap between train and test
is zero. No reference solution or target is inserted into the prompt.

Append one fixed instruction requesting step-by-step work and a final
`#### <answer>` line. Use the Qwen chat template with a generation prompt.
The model is Qwen2.5-3B-Instruct in bf16, output cap 1024. CPU tokenization
verified a maximum prompt length of 244 tokens across the 2369 selected
items, so context 2048 accommodates the complete output allowance.

The deployed GSM8K handler's reward fallback was inconsistent with its voting
extractor: a wrong strict answer could receive credit for a different trailing
number. The new `canonical_voting_answer_v1` scoring uses the same strict-first
extracted answer for both selection and voting, falling back only if strict
extraction is absent. Numeric normalization removes commas and insignificant
decimal zeros. Selection rewards are binary, with sensitivity/reward bound 1.
The legacy 300-character extraction window is retained and recorded.
This scoring choice is fixed before any model outputs are inspected.

Data are staged at `/home/lzn/dprandopt_run/data/gsm8k/`, with train/test Parquet
files compatible with the existing handler. The CPU integration check passed
the exact split mapping, answer/reward agreement and prompt budget. See
[protocol.json](protocol.json) and [preflight.json](preflight.json).

## Methods and comparisons

Use the existing D1 Gumbel top-K implementation and CKS calibration, epsilon=8,
delta=1e-5, K=5 and n=1000 as the primary configuration. Build the full pool
of base plus 1000 perturbations with sigma grid {0.0005,0.001,0.002}.
Candidate seeds 20261024, 20261025 and 20261026 vary the perturbations while
the item assignment remains fixed. K is not selected on test accuracy.

| Method | Selection information | Primary output |
|---|---|---|
| Greedy base | None | Exact-match accuracy |
| Base self-consistency | None; five samples, temperature 0.7, top_p=1 | Plurality vote accuracy |
| Random five perturbations | None; distinct members excluding base | Expected vote accuracy |
| Public top-5 | 50 public examples | Vote and mean-member accuracy |
| Nonprivate top-5 | 1000 pool examples | Vote and mean-member accuracy |
| D1 | Same pool under epsilon=8, delta=1e-5 | Expected vote accuracy |

Report paired differences against random voting, self-consistency and
public-only, plus the D1-to-nonprivate privacy cost. Use 2000 random draws and
400 D1 selection draws per frozen table. Report selection/member overlap,
mean member accuracy and descriptive K curves separately from the fixed-K
comparison. CPU secondary analyses can vary n in {200,500,1000} and epsilon
without extra model generation.

Pass@5 on GSM8K is a retrospective oracle diagnostic requiring labels; do
not describe it as an available test-time label-free checker. The primary
metric is voting accuracy. No learned verifier is introduced in this protocol.

## Execution order and resources

1. Run the greedy-base and five-sample self-consistency baselines on all 1319
   test items once a suitable GPU is available. These are full-task baselines,
   not a reduced test-set smoke run. Keep temperature and prompt fixed.
2. Run the first full candidate pool with seed 20261024. It requires
   2,371,369 responses; measure throughput and memory from actual GSM8K
   outputs before predicting duration or allocating more engines.
3. Audit the first table and then run the other fixed candidate seeds for
   pool variability. Report all completed seeds, including negative results.
   Do not count the three pools as three independent test datasets.

Use the established production settings: max 128 sequences, 2048 batched
tokens, context 2048, eager execution, CPU base-weight copy, and checkpoint
interval 10 candidates. Restrict engines to explicitly available GPUs. Do
not reduce the candidate pool or test set and present that as the main result.

No new GPU experiment is running. At preparation time GPU 0 had an active
unrelated workload and about 32 GiB free; GPU 1 was busy. The baseline runner
uses GSM8K prompts and extraction, not the Countdown-specific controls script.
`scripts/run_gsm8k_baselines.py` and its guarded lzn launcher are prepared.
Only CPU preflight has run; the new GPU generation path remains untested
until the full-test baseline executes. The generic score-table builder is
prepared for the new data and scoring path.

## Evidence and interpretation

The central utility claim requires improvement over the strong data-free
and public baselines, not merely over greedy base. Conditional paired
test-item intervals describe a fixed pool/sample realization; independent
candidate seeds and sampling realizations address different variability.
If the first full-pool result does not support the comparison, preserve it
and reassess the claim rather than tune on the official test set.

All GSM8K result entries remain unmeasured. No new method, verifier or model
architecture is substituted. The earlier Countdown subset remains diagnostic
evidence, not a replacement for this full-task comparison.
