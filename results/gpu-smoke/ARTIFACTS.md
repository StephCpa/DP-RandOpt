# Frozen experiment artifacts

The committed score tables include synthetic Countdown items (Route 1 item
seed 20261013) and public GSM8K benchmark items (item seed 20261023), evaluated
with Qwen2.5-3B-Instruct. Legacy audit summaries separately identify the older
base-model runs. These are not confidential records: "private" identifies the
selection split and mechanism experiment.

- `lzn_route1_n1001_max512.npz` and metadata: full 1001-candidate reference.
- `length_pilot_32_max1024/table.npz` and `length_pilot_32_max2048/table.npz`:
  matched first-32-candidate pilots, plus frozen manifests and summaries.
- `prefix_2048_cpu/prefix_{512,1024,2048}.npz`: CPU approximate prefix rescoring
  of the same saved 2048-token trajectories; see diagnostics for tokenization error.
- `selection_decomposition/audit.json`: members, votes, checked pass@5 and overlap.
- `selection_decomposition/REPORT.md`: voting decomposition and links to completed controls.
- `batched_controls_20261023/analysis.json` and `comparison_vectors.npz`:
  completed sampling/repeatability controls and compact paired-comparison inputs.
- `gsm8k_ready/protocol.json` and `preflight.json`: prepared official-data
  assignments and CPU checks, retained as the frozen preparation record.

Set PYTHONPATH to the repository root and run:

```text
python scripts/audit_vote_selection.py
python scripts/analyze_prefix_scores.py
python scripts/analyze_batched_controls.py --comparisons-only
python -m unittest discover -s tests -p test_vote_decomposition.py
python -m unittest discover -s tests -p test_pilot.py
```

The first two commands use only the committed score tables and package
dependencies. They regenerate summary JSON, not the frozen source NPZ files.

Full text-prefix reconstruction and cross-cap text comparisons also require
the 32 gzip response bundles per pilot, which remain archived in the original
experiment workspace rather than Git. Obtain those unchanged bundles and the
Qwen tokenizer before running `rescore_text_prefixes.py` or
`analyze_length_pilot.py`; do not manufacture them from score codes. The former
also requires the production RandOpt parser on PYTHONPATH. `server_sha256.txt`
records the 2048 source-bundle hashes for transfer verification.

GPU launch scripts are specific to the documented lzn staging environment.
The batched-controls runner has an immutable protocol and refuses to reuse
a directory that already contains a runtime record. A crash requires an
explicit new control-session directory, preserving the failed session.

Aggregate summaries, synthetic Countdown and public-benchmark GSM8K score
tables, compact comparison arrays and provenance are published. The old dated status sections may refer to additional local-only
logs or checkpoints; absence of those files is not a missing generation result.

## Completed GSM8K and cross-task synthesis (2026-10-25)

- `gsm8k_n1001_seed20261024_items20261023_max1024.npz`: the full candidate pool;
  its metadata records resumed-segment time, not total wall time.
- `gsm8k_baselines_20261023/COMPLETE.json`: full-test baseline results;
  the two raw response gzip files remain locally archived.
- `gsm8k_pool_audit_20261025/`: primary comparisons, compact item vectors and report.
- `gsm8k_selection_diagnostics_20261025/`: sigma/diversity/spread diagnostics,
  ensemble draws, self-consistency answer codes and D4 feasibility assessment.
- `cross_task_synthesis_20261025/`: eight-setting ledger assembled from archived
  audits. See the [research conclusions](../../docs/RESEARCH_CONCLUSIONS_20261025.md).

From the repository root:

```text
python -m scripts.summarize_frozen_evidence
python -m scripts.analyze_gsm8k_pool --comparisons-only
python -m scripts.analyze_batched_controls --comparisons-only
```

These compact reproduction paths need only the committed artifacts and package
requirements. Full `python -m scripts.analyze_gsm8k_pool` additionally requires
both raw baseline gzip bundles and prepared test JSONL at
`results/cache/gsm8k_20261023/test.jsonl`. Full
`python -m scripts.diagnose_gsm8k_selection` needs the self-consistency gzip
bundle. Their raw archives are not silently reconstructed from correctness codes.
Private-pool GSM8K answer identities were never saved, so D4 ensemble vote
utilities cannot be recovered by downloading a missing committed file.
