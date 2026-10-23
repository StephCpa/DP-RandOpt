# Voting, selection and the length-cap decision

Date: 2026-10-23 (Asia/Shanghai). This update supersedes the 10% truncation
gate and batch-size-one reproducibility proposal in earlier reports.

## Decision

Fix future Countdown runs at 1024 output tokens, the setting in the deployed
RandOpt Countdown handler. Remove the 10% cap-hit resource gate. Cap hits
remain a behavioral diagnostic, not a prerequisite for useful selection.
Selecting perturbations that produce fewer invalid or repetitive responses
is a legitimate utility gain, provided it is measured against data-free
ensembles and described without claiming improved reasoning in general.

Saved-trajectory prefix analysis shows essentially no additional accuracy
from tokens 1025–2048. This supports a practical cap choice, not a proof that
every long response is hopeless or that no possible larger budget could help.
The 1024 choice is not based on maximizing observed test-set vote accuracy.

## CPU prefix rescoring

All 49,600 saved 2048-token responses were re-tokenized using the same local
Qwen tokenizer. Prefixes were decoded with cleanup disabled and scored by
the deployed RandOpt reward and voting parser. Full-text re-scoring exactly
matched all saved rewards and voting answers. Text encode/decode round trips
were exact for all records.

| Same saved trajectory | Prefix 512 | Prefix 1024 | Full 2048 output |
|---|---:|---:|---:|
| Base accuracy | 16.60% | 17.00% | 17.00% |
| Mean accuracy, 31 perturbations | 15.65% | 16.303% | 16.310% |
| Random-5 vote, excluding base | 40.60% | 41.48% | 41.46% |
| Exact expected random checked pass@5 | 45.93% | 47.39% | 47.41% |

From prefix 512 to 1024, mean perturbed accuracy gains 0.652 percentage points,
conditional paired test-item bootstrap 95% interval [0.523, 0.800]. From
1024 to full output, it gains only 0.00645 points [0.000, 0.01935]: exactly
one additional correct candidate/item response among the 15,500 perturbed
test responses, with no correct-to-incorrect changes. Base is unchanged.
The sole gain is candidate 14, test item 63 (global item 1113): saved length
1335, re-tokenized length 1334. It is not near the 1024 boundary.

Re-tokenization is approximate: 99.51% of length-finished responses have
exactly the saved token count; normal stops typically lose one token,
consistent with a removed stop token. Only 75/49,600 records (0.151%) differ
in length by more than two tokens, but the largest difference is 174 tokens.
Original token identities cannot be recovered from count agreement alone.
Moving the reconstructed cut by +/-2 tokens changes a score or voting answer
on 0.4375% of all responses at 512 and 0.0343% at 1024. This is a local
sensitivity check, not an upper bound on all re-tokenization error. Independent
capped executions need not follow identical trajectories under batched
floating-point execution. The prefix analysis does not require that assumption.

Sources: [prefix diagnostics](../prefix_2048_cpu/diagnostics.json) and
[prefix accuracy analysis](../prefix_2048_cpu/analysis.json).

## Most gain over base is already available without selection data

All rows below use K=5. Random ensembles contain five distinct perturbations
and exclude base. Earlier random-K summaries sampled from all rows, including
base; this is a clarified comparator. Random vote averages 2000 uniformly
ordered draws. Expected random pass@5 is computed exactly per test item as
`1 - choose(M-c,5)/choose(M,5)`, where M excludes base and c counts correct
perturbations. It is not the independent-draw formula `1-(1-p)^5`.

| 32-candidate setting | Base | Random vote | Random checked pass@5 | Public vote | Public checked pass@5 | D1 vote | D1 checked pass@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | 16.80% | 40.84% | 45.96% | 46.60% | 56.20% | 47.20% | 53.87% |
| 1024 | 19.00% | 41.19% | 47.26% | 53.40% | 58.40% | 48.34% | 57.80% |
| 2048 | 17.00% | 41.46% | 47.41% | 49.00% | 58.00% | 46.93% | 56.60% |

D1 uses epsilon=8, delta=1e-5, n=1000, 400 selection draws; public selection
uses Q=50. Random-vote Monte Carlo standard error is approximately 0.065–0.072
percentage points, separate from uncertainty across test items and fresh pools.

At 1024 tokens, random voting adds 22.19 points over base, while D1 adds
7.15 points over random voting. The expected single random perturbation is
15.99%, below base. Thus most of D1's gain over base is already obtained by
the data-free ensemble on this subset. This is a descriptive decomposition;
it does not independently isolate all effects of perturbation diversity.

Checked pass@5 requires only that at least one member supplies a valid formula
equal to the prompt-visible target. A test-time checker can validate the
numbers and arithmetic without private labels. These are implementable
label-free baselines, not an oracle granted hidden test answers. Checking
every already-tagged formula is different from introducing a solver that
searches for a new formula. Vote and checked-pass use the existing parser.

The checker adds about 5–10 points on these ensembles. Together with the
prompt-visible correctness test, this makes Countdown a secondary task for
the private-selection claim. GSM8K should become the deciding task after the
bounded controls; using 1024 tokens there remains an empirical choice to
verify rather than a guarantee that all reasoning chains fit.

## Member quality versus composition

| Cap | Public mean member accuracy | Private mean member accuracy | Public vote | Private vote | Set overlap |
|---|---:|---:|---:|---:|---:|
| 512 | 19.76% | 18.68% | 46.60% | 47.20% | 4/5 |
| 1024 | 20.96% | 20.24% | 53.40% | 48.20% | 3/5 |
| 2048 | 21.00% | 20.32% | 49.00% | 47.00% | 4/5 |

Private here means nonprivate top-5 on the private pool, isolating selection
from privacy noise. At 1024, public members are slightly better on this test
set; the proposed explanation "private has better members but loses only
through composition" is not observed. However, the 0.72-point member gap is
much smaller than the 5.20-point vote gap. At 512, private members are worse
but its vote is slightly better, illustrating the role of composition and
rank-dependent tie breaking. Neither observation establishes a general ranking.
Exact member lists, individual accuracies, overlaps and paired intervals are
recorded in [audit.json](audit.json).

The full N=1001, 512-token pool differs materially: random vote 39.07%, D1
57.17%, and public 43.40%. D1's improvement over random is 18.10 points, not
the 6–8-point subset gain. Nonprivate selected mean member accuracy is 25.44%
versus public 19.88%, and their checked pass@5 is 63.60% versus 55.00%.
The subset therefore does not invalidate the full-pool selection result.
That result still has the cap/configuration and task-specific limitations
already documented; it is not a long-cap replication.

## Batched controls now running

Started 2026-10-23 at 08:16 on lzn GPU 0, PID 1727851.
Log: `logs/batched_controls_20261023_081626.log`.
Output: `tables/batched_controls_20261023/`.

The repeatability control uses the actual 1024 pilot configuration: bf16,
eager vLLM, no prefix caching, context 2048, max 128 sequences, 2048 batched
tokens, temperature zero and the original decoding seed. Each call submits
the original 1550 items in order. Base and candidates 1–3 run in production
order, then the entire block repeats. It records per-item reward, correctness,
voting-answer and token-sequence flip rates separately for pool/public/test.
All parameter bytes are SHA-256 checked before/after reset, before/after
generation and across repeated perturbations. Checksums test weight identity;
cross-configuration output divergence alone is not evidence of a reset bug.
Flip rates estimate within-configuration noise for these candidates, not a
universal uncertainty bound for every design.

The same bounded job then generates 5 base-model samples on each of 500 test
items (2500 completions), with temperature=0.7, top_p=1, top_k=-1, seed=20261023
and cap=1024, fixed before observing results. It reports mean member accuracy,
plurality vote, checked pass@5 and paired comparisons with the perturbation
baseline. There is no temperature sweep. Matching random-perturbation voting
would weaken the evidence for an added benefit from weight perturbations under
this configuration; it would not establish equivalence for all sampling choices.

These GPU results are pending. No full-pool expansion or GSM8K run is chained
to the controls. Archived score tables and response bundles are unchanged.

## Claim audit update

| Claim | Status | Evidence / boundary |
|---|---|---|
| The vote, not selection, provides most gain over base on this subset | Supported descriptively | Random vote about 41%, base 17–19%, D1 47–48% |
| A low cap-hit rate is required for a valid RandOpt gain | Removed | Loop avoidance/valid-answer yield may be part of the gain |
| 1024 is an adequate working cap for these saved trajectories | Supported with approximation limits | One extra correct perturbation/item beyond 1024; retokenization caveat |
| Private top-5 has better members but only loses through voting | Not observed here | Public member mean is slightly higher at all three caps |
| Selection is useless in the full candidate pool | Unsupported | Full 512 pool shows +18.10 points over random vote |
| Perturbations outperform ordinary sampling diversity | Pending | Base self-consistency control running |
| Cross-cap text divergence implies weight-reset failure | Unsupported | Batch execution differs; direct checksum control is separate |

Reproduce the CPU decomposition with `python scripts/audit_vote_selection.py`
and prefix summary with `python scripts/analyze_prefix_scores.py`, from the
repository root with the root on PYTHONPATH. Prefix reconstruction itself uses
`scripts/rescore_text_prefixes.py` on a machine with the frozen Qwen tokenizer,
RandOpt parser and saved response bundles. Two finite-population/voting tests
pass; the earlier four pilot tests remain available. All new numbers derive
from the frozen synthetic Countdown artifacts; no external citations were added.
