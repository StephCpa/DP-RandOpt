# DP-RandOpt: theory

This note states and proves the results the research plan asks for (RQ1), with the corrections found while checking the plan. Every bound has a calculator in `dprandopt/bounds.py`, and `experiments/e0_plan_checks.py` prints the numbers. Section 7 lists what is new here and what is standard.

## 0. Setting

- **Candidates.** N candidates θ_1, …, θ_N are drawn before any private data is touched: seeds, scales σ_i from a public grid, and the base model θ_0. In RandOpt's code σ_i is drawn per candidate (`rng.choice(sigma_list)`), so σ is already part of each candidate. The script's "best σ" is only logged and saved; selection never uses it. It is still a data-dependent release, though, so DP-RandOpt must drop it.
- **Data.** The private set is D = (z_1, …, z_n). Each example gives candidate i a reward r(i, z) ∈ [0, R]. For RandOpt's Countdown reward R = 1.1 (answer + 0.1·format). For GSM8K, R = 1.
- **Scores.** The sum score is U_i(D) = Σ_j r(i, z_j). The mean score û_i = U_i/(nR) lies in [0, 1].
- **Adjacency.** Under replace-one, |U_i(D) − U_i(D′)| ≤ R for all i. Under add/remove, U_i(D′) − U_i(D) ∈ [0, R] for all i when D′ = D ∪ {z}. In that case the scores are *monotone*: every score moves in the same direction.
- **Density.** For a threshold τ, γ(τ) = |{i : û_i ≥ τ}|/N is the empirical density. γ*(τ) is the same with population scores p_i = E_z r(i, z)/R. The plan's δ(m) is γ*(base + m). We write γ to avoid a clash with the DP parameter δ.

The exponential mechanism (EM) with parameter ε_0 samples i with probability ∝ exp(ε_0 U_i/(cR)). Here c = 2 under replace-one and c = 1 under add/remove with monotone scores. In mean-score units this is a Gibbs distribution with inverse temperature λ = ε_0 n / c.

## 1. Privacy

**Lemma 1 (one pick).** The EM pick is ε_0-DP (McSherry & Talwar 2007). It is also ε_0-bounded-range, hence ε_0²/8-zCDP (Cesar & Rogers 2021).

*Proof sketch of the BR property.* The log-ratio of output probabilities is (ε_0/(cR))(U_i − U′_i) − log(Z/Z′). Across outputs i its range is (ε_0/(cR))·max_{i,j}|(U_i − U′_i) − (U_j − U′_j)|. That is at most ε_0 under replace-one (c = 2, differences in [−R, R]). Under add/remove with c = 1 it is also at most ε_0, because all differences lie in [−R, 0]. ∎

**Theorem 1 (D1: K Gumbel picks).** Gumbel top-K with per-pick ε_0 has the same output distribution as K adaptive EM picks without replacement (Durfee & Rogers 2019). It is therefore Kε_0-DP and (Kε_0²/8)-zCDP. For a target (ε, δ′), `accounting.calibrate_eps0` returns the largest ε_0 whose better bound satisfies the target. It converts zCDP to (ε, δ′) with the bound of Canonne, Kamath & Steinke (2020, Prop. 12):

  ε = min_{α>1} [ρα + log(1 − 1/α) − (log δ′ + log α)/(α − 1)].

That conversion is never worse than ρ + 2√(ρ log(1/δ′)), the bound the plan used. It raises ε_0 by 8–20% at K ≥ 10 (results/e0, §2). The tests check it against the exact privacy curve of the Gaussian mechanism (`tests/test_accounting.py`).

**Proposition 2 (other designs).**
- **D2.** Picks on disjoint, data-independent shards are ε-DP by adaptive parallel composition. Excluding earlier picks is allowed because each shard's mechanism is ε-DP given the earlier outputs.
- **D3.** One pure ε-DP pick. The neighbours are post-processing because they use only public randomness and the released expert.
- **D7.** One EM over K-subsets with utility q(S) = min_{i∈S} U_i. This utility has sensitivity R, and it is monotone under add/remove, so the mechanism is ε-DP. The sampler is exact: the number of K-subsets whose lowest-ranked member sits at rank r is C(r, K−1). The tests check this against brute-force enumeration.
- **D0.** Releasing all N sum scores with Gaussian noise. The L2 sensitivity is R√N, so σ = R√N/√(2ρ).
- **D8, stability release.** Release the exact top-K set if d + Lap(1/ε) > ln(1/δ′)/ε, otherwise ⊥. Here d = ⌈g_K/(2R)⌉ and g_K = U_(K) − U_(K+1). This is (ε, δ′)-DP by propose-test-release (Dwork & Lei 2009; Thakurta & Smith 2013).

**What is released.** Only the selected seeds and σ's. RandOpt's `results.json` also writes per-candidate train rewards, the base train reward, per-σ mean rewards and `best_sigma`. Each is a data-dependent statistic, so a DP-RandOpt run must suppress them or privatize them and count them in the budget.

## 2. Empirical utility

**Theorem 2 (single pick).** Suppose γ(τ) ≥ γ. Then for every β ∈ (0, 1),

  Pr[û_I < τ − t] ≤ β  for  t = c (ln(1/γ) + ln(1/β)) / (ε_0 n).

*Proof.* Let B = {i : û_i < τ − t} and G = {i : û_i ≥ τ}. Then Pr[I ∈ B] = Σ_B w_i / Σ_all w_i ≤ |B| e^{λ(τ−t)} / (|G| e^{λτ}) ≤ γ^{−1} e^{−λt}, with λ = ε_0 n/c. ∎

With c = 2 and R = 1 this is the plan's formula. The plan's table is reproduced exactly (`tests/test_bounds.py`). Countdown's R = 1.1 is already inside û, so in points of *answer accuracy* the slack is 1.1× larger.

**Corollary 3 (K peeling picks; corrected).** If γN ≥ K, then with probability at least 1 − β all K picks have û ≥ τ − t, where

  t = c (ln((N − K + 1)/(γN − K + 1)) + ln(K/β)) / (ε_0 n).

*Why the correction.* The plan applied Theorem 2 to each pick with the density unchanged. But each pick without replacement can remove a good candidate. By the last pick, at most γN − K + 1 good candidates remain among N − K + 1. With the plan's own experiment (N = 1000, γ = 0.05) there are exactly 50 good candidates, so the 50th pick can face density 1/951 instead of 1/20. The guarantee needs N ≳ 2K/γ (N ≥ 2000 for K = 50 at γ = 0.05), or else a smaller K.

**Proposition 4 (D7: all members good at once).** If ⌊γN⌋ ≥ K, then with probability at least 1 − β every member of the D7 subset has û ≥ τ − t, where

  t = c (ln[C(N, K)/C(⌊γN⌋, K)] + ln(1/β)) / (ε n).

*Proof.* Theorem 2 applied to the EM over the C(N, K) subsets: all C(⌊γN⌋, K) subsets of the good set have utility at least τ. ∎

Since ln[C(N, K)/C(γN, K)] ≈ K ln(1/γ), D7 pays K·ln(1/γ) at the full ε. D1 instead pays about √K·(ln(1/γ) + ln K) at its zCDP-calibrated ε_0. D7's bound is better for moderate K (K = 10: 36 vs 59 points at ε = 1, n = 200) but not at K = 50 (results/e0 §1c).

**Lower bound for K picks (context).** For approximate DP, selecting K items that are each near the top requires n = Ω(√K·log N/(εα)) in the worst case (Bafna & Ullman 2017; Steinke & Ullman 2017). So the √K price of distinct private picks is unavoidable in general. Escaping it requires structure that worst-case lower bounds ignore: D3's assumption that neighbours of an expert are good, or D8's large gaps.

## 3. Population utility without N or d (new)

The plan proposed bounding population performance through the generalization that DP already gives. That route fails at the ε_0 values that matter. DP transfer theorems give a generalization error of order e^{ε_0} − 1, which is vacuous at ε_0 = 4–8. A union bound over all N candidates works but brings back ln N. The PAC-Bayes view of the EM avoids both problems.

**Theorem 5 (Gibbs bound).** Let the private examples be i.i.d. Let γ* be the population density at threshold τ*, and β ∈ (0, 1). With probability at least 1 − β over D, the EM pick I satisfies

  E_I[p_I] ≥ τ* − √(ln(4/β)/(2n)) − (ln(2/γ*) + ln(2/β))/λ − λ/(8n),  where λ = ε_0 n/c.

The expectation is over the mechanism's coins only. N and the parameter count d do not appear.

*Proof.*
1. *Donsker–Varadhan / PAC-Bayes.* For each i, Hoeffding's lemma gives E_D exp(λ(p_i − û_i)) ≤ exp(λ²/(8n)). Average over the uniform prior π on candidates and apply Markov's inequality: with probability at least 1 − β/2, E_π exp(λ(p − û)) ≤ exp(λ²/(8n))·2/β. By the Donsker–Varadhan formula, every distribution ρ then satisfies E_ρ[p] ≥ E_ρ[û] − (KL(ρ‖π) + ln(2/β))/λ − λ/(8n).
2. *The EM is the Gibbs posterior.* For ρ = EM, E_ρ[û] − KL(ρ‖π)/λ = λ^{−1} ln E_π e^{λû} ≥ τ′ − ln(1/γ(τ′))/λ for any τ′.
3. *Empirical vs. population density.* Each i in the population-good set has Pr[û_i < τ* − a] ≤ e^{−2na²}. By Markov's inequality on the failing fraction, with probability at least 1 − 2e^{−2na²} we have γ(τ* − a) ≥ γ*/2. Take a = √(ln(4/β)/(2n)) and τ′ = τ* − a, then take a union bound over the two events. ∎

**Corollary 6 (temperature).** The bound is optimized at λ* = √(8nL) with L = ln(2/γ*) + ln(2/β), i.e. ε_0* = c√(8L/n). There the slack is √(ln(4/β)/(2n)) + √(L/(2n)). With n = 200 and γ* = 0.05, ε_0* ≈ 1.1. The noiseless argmax (ε_0 → ∞) has no N-free guarantee. Its union-bound guarantee is 2√(ln(2N/β)/(2n)) (results/e0 §3).

**Status of the "selection as regularizer" prediction.** Corollary 6 is a worst-case statement. On the simulated score tables (results/e3), expected test accuracy *increases* essentially monotonically with ε_0. No finite ε_0 beats argmax by more than 0.22 points, and the bound's optimum ε_0* ≈ 1 costs up to 8 points at n = 50–200. Lower temperature narrows the train–test gap only by choosing worse candidates. Since the test costs nothing on the real score table, keep it as a check, but expect a null result. Do not claim it in advance.

## 4. Lower bound (optimality of the single-pick rate)

**Theorem 7 (packing, pure DP).** Fix γ ∈ (0, 1/2], t > 0 and n. Suppose an ε-DP selector M (replace-one), on every instance whose good set at τ has density γ, returns a candidate with û ≥ τ − t with probability at least 1/2. Then

  n ≥ (ln(⌊1/γ⌋/2)/ε − 1)/t.

*Proof.* Split the candidates into M = ⌊1/γ⌋ disjoint blocks B_1, …, B_M of size γN. Let z_b give reward R exactly on B_b and z_0 give reward 0 everywhere. Let D_b consist of k = ⌊tn⌋ + 1 copies of z_b and n − k copies of z_0, and let D_0 consist of n copies of z_0. In D_b, the block B_b has û = k/n > t and every other candidate has û = 0 < k/n − t. So success on D_b means outputting from B_b. By group privacy over k replacements, Pr[M(D_0) ∈ B_b] ≥ e^{−εk}/2 for every b. Summing over the disjoint blocks gives M e^{−εk}/2 ≤ 1, hence k ≥ ln(M/2)/ε. Finally k ≤ tn + 1. ∎

Theorem 2 with β = 1/2 needs n = c(ln(1/γ) + ln 2)/(εt). The EM is therefore optimal up to the constant c and an additive 1/t in the regime ε ≲ ln(1/γ), which is the regime where privacy binds (results/e0 §3). For approximate DP the analogous bound replaces ln(1/γ) by min(ln(1/γ), ln(1/δ′)) up to constants (Steinke & Ullman 2017).

## 5. The information ceiling for plain RandOpt, stated correctly

The plan bounds plain RandOpt's leakage by H(output) ≤ log₂ C(N, K) bits and divides by n. Two refinements:

1. **What the bits bound actually bounds.** Let the membership bits S_j be independent and uniform. Then Σ_j I(S_j; O) ≤ I(S; O) ≤ log₂ C(N, K). By Fano's inequality, an attacker who guesses S_j correctly with probability p extracts at least 1 − h(p) bits. So at most log₂C(N,K)/(1 − h(p)) members can be inferred with accuracy p: 753 at 90% for N = 5000, K = 50 (results/e0 §4). This bounds the *number* of exposed members. It says nothing about how badly any single member is exposed.
2. **The ceiling only holds if nothing else is released.** RandOpt's default outputs include real-valued train rewards. For a DP-RandOpt release, drop those fields or privatize them.

Plain RandOpt is not DP for any finite ε. When the gap at the selection boundary is below 2R, a single crafted record can change the output with certainty. On simulated tables that is the case for 81–100% of private sets at K = 50, but for only 8–17% at K = 1 (results/e5 §b).

## 6. Majority vote

No assumption-free bound links member accuracy to vote accuracy. Plurality voting can fail even when every member is above 50%, if members share errors. The empirical quantities to report are the per-item agreement among members and the share of items where the gold answer wins a plurality. On simulated tables the vote's gain over K = 1 peaks at K ≈ 5–10. That shape depends on the error correlation and must be measured on the real table before K is fixed.

## 7. What is new here and what is standard

| Item | Status |
|---|---|
| Theorem 1 (D1 privacy), Lemma 1, Theorem 2 | Standard (McSherry–Talwar; Durfee–Rogers; Cesar–Rogers; CKS conversion) |
| Corollary 3 (peeling correction, N ≥ 2K/γ) | Elementary but missing from the plan; it changes the experiment's N |
| Proposition 4 (D7 threshold-subset EM, exact sampler) | The mechanism is a special case of the EM over subsets; the min-utility, rank-count sampler and the comparison to D1 appear to be new in this setting (cf. Gillenwater et al. 2022's joint EM) |
| Theorem 5 / Corollary 6 (Gibbs bound) | Standard PAC-Bayes (Catoni 2007; McAllester 1999), applied here to make the population guarantee N-free; the density form is the new part |
| Theorem 7 (packing) | Standard technique (Hardt–Talwar; Steinke–Ullman); the density parametrisation is ours |
| §5 Fano reading of the bits ceiling | Standard information theory; corrects how the plan uses the ceiling |
