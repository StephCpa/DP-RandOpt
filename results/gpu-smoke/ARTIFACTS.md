# Frozen Countdown artifacts

The committed NPZ tables contain synthetic Countdown instances generated with
item seed 20261013, evaluated with Qwen2.5-3B-Instruct. These are research
score tables, not a release of confidential private examples. The word
"private" identifies the selection split and mechanism experiment.

- `lzn_route1_n1001_max512.npz` and metadata: full 1001-candidate reference.
- `length_pilot_32_max1024/table.npz` and `length_pilot_32_max2048/table.npz`:
  matched first-32-candidate pilots, plus frozen manifests and summaries.
- `prefix_2048_cpu/prefix_{512,1024,2048}.npz`: CPU approximate prefix rescoring
  of the same saved 2048-token trajectories; see diagnostics for tokenization error.
- `selection_decomposition/audit.json`: members, votes, checked pass@5 and overlap.
- `selection_decomposition/REPORT.md`: current interpretation and pending controls.
- `batched_controls_20261023/analysis.json` and `comparison_vectors.npz`:
  completed sampling/repeatability controls and compact paired-comparison inputs.
- `gsm8k_ready/protocol.json` and `preflight.json`: prepared official-data
  assignments and CPU checks; no GSM8K model results.

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

Only aggregate summaries, small synthetic score tables and provenance are
published. The old dated status sections may refer to additional local-only
logs or checkpoints; absence of those files is not a missing generation result.
