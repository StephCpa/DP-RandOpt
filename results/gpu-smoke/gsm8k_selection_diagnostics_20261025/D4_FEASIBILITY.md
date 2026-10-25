# D4 feasibility on the existing GSM8K artifacts

Decision: not executable from the saved private-pool data. No GPU run was started.

The final score table has pool_reward[1001,1000], public_reward[1001,50],
and test_answer[1001,1319]. The server checkpoint contains rewards[1001,1050],
answers[1001,1319], done[1001], and output_lengths[1001,2369]. Its answers
are test-only. The builder stores numeric pool/public rewards and discards
those responses; no full-pool response bundles were found in the server tables.
The standalone greedy/self-consistency bundles also contain test items only.

Binary correctness is insufficient for plurality vote: with two correct
members and three incorrect members, the vote can be correct if the three
wrong answers differ, or wrong if they agree. Both cases have identical
per-member rewards. Ties additionally depend on member order. We must not
assign artificial wrong-answer identities or use test answers as private
utilities. Thus the requested pool-vote/test-vote reliability correlation
cannot be computed and no D4 accuracy is reported.

For a future authorized experiment, the proposed mechanism has a simple
fixed-library formulation. Draw ordered five-member ensembles without any
private data and freeze this library before scoring. For each ensemble E,
let u(E,D) be its number of correct private-example votes with a fixed
abstention/tie rule. With fixed candidate models and prompts evaluated
independently of the private dataset, u has sensitivity at most 1 under
replacement of one example (also under add/remove adjacency for sum scores).
A single exponential mechanism samples E with probability proportional to
exp(epsilon * u(E,D) / 2). This is one pure epsilon-DP release, independent
of K in its accounting; utility and computational cost still depend on the
ensemble library. Do not release the private utility table as an additional
output or tune the library using unaccounted private evaluations.

The independence condition matters here: production batching can affect
answers on other examples. The score-table simulator treats candidate
response functions as fixed; a deployment privacy proof must either enforce
per-item/batch-invariant predictions or justify that replacing one private
record cannot alter other records' votes. Existing cross-batch base flips
show why this assumption must be stated. The algebraic sensitivity of a
frozen table alone is not an end-to-end proof for an arbitrary vLLM pipeline.

Reconstructing the missing private answers would require additional inference.
It is outside the current CPU-only request. The existing candidates, K,
sigma grid, private n, prompt and test protocol have not been changed.
