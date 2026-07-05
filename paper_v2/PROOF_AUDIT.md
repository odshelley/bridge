# PROOF_AUDIT — paper_v2.tex

## Round 1 — Codex CLI (gpt-5.5, xhigh), 2026-07-05
Scope: 5 flagged results + dependency context (see PROOF_SKELETON.md).
Raw output: ~/.claude/jobs/1cfda0ac/tmp/codex_review_round1.md (18 issues).

### Severity roll-up (status x impact)
FATAL (INVALID+GLOBAL): #14 Tweedie Jacobian error; #16 Poisson lattice gradients
CRITICAL: #17 t=T/2 degeneracy (INVALID+LOCAL); #5 #7 #9 #11 #13 (UNJUSTIFIED/UNCLEAR+GLOBAL:
  harmonicity interchange, jump-split integrability, compensator identification,
  martingale integrability, Tweedie differentiation-under-integral); #1 #2 (mode/coupling, GLOBAL)
MAJOR: #3 temporal-consistency overclaim; #4 endpoint limits; #6 space-time Doob route;
  #8 density/support hypotheses; #12 QV premise; #15 variance Hessian; #10 PRM wording
MINOR: #18 lattice-support notation

### Notable adjudications of prior-audit concerns
- MC-1 CONFIRMED as fixable (issue #6): final modified-Lévy formula SURVIVES, but only via
  the space-time route: generator h^{-1}(A(hf) + f ∂_t h) with ∂_t h = −Ah. Insert explicitly.
- MC-4 CONFIRMED (#9): compensator read-off needs martingale-problem/Girsanov-for-jumps theorem.
- MC-7 CONFIRMED (#3): drift identity ≠ bridge law; "equivalently" must go, or prove full
  conditional law via Lévy bridge consistency.
- MC-9/10 CONFIRMED (#13, #16): and #16 is INVALID as stated (lattice has no ∇ log p).
- NEW catches beyond prior audit: #17 (drift formula undefined at t=T/2 under the chosen
  GE parameterisation); #14 (∇F=1c cannot identify the VECTOR E[φ(η)|z]); #2 (Y vs ξ_T
  conditional coupling); #8 (lattice/singular drivers excluded by density formulation —
  ironic since the Poisson bridge is the paper's own canonical example).

### Fix queue (Phase 2, ordered)
1. #14 STRENGTHEN_ASSUMPTION: restate Tweedie with Jacobian J_F = cI (invertible), solve via J_F^{-T}.
2. #16 route decision (author): discrete finite-difference Tweedie vs weaken to intensity-only.
3. #17 ADD hypothesis: exclude t=T/2 or reparameterise GE form.
4. #6 ADD_DERIVATION: space-time Doob route in thm:modified-levy-app.
5. #9 ADD_REFERENCE+DERIVATION: semimartingale-characteristics/Girsanov-for-jumps for compensator.
6. #5 #7 #11 #13 STRENGTHEN_ASSUMPTION: integrability/domination hypotheses (theorem preambles).
7. #1 #2 ADD_DERIVATION/definition: state representation on bridge filtered space; define Y := ξ_T a.s.
8. #3 #4 WEAKEN_CLAIM or ADD_DERIVATION: temporal consistency statement.
9. #8 #10 #12 #15 #18: wording, support hypotheses, QV premise, variance restriction, lattice notation.

Acceptance gate: NOT MET (open FATAL/CRITICAL). Round 2 required after fixes.

## Round 2 — same reviewer session (resume), 2026-07-05
Raw: ~/.claude/jobs/1cfda0ac/tmp/codex_review_round2.md
Verdicts: 13/18 RESOLVED; PARTIAL #1 #5 #6 #11 #18; NEW R2-1 (AC-remark overclaim, pinned bridges).
P-R citation VERIFIED: eq (1.2) correct internal ref; Thm 4.2 supports use once space-time recast + domain hypotheses stated.
Gate: NOT MET.

## Round-3 fixes applied (commit pending verification)
#1/#11 pathwise uses routed through thm:main-rep on canonical space; #5 eq:harmonic-fubini double-integral bound; #6 good-function domain hypotheses; R2-1 remark rescoped (pinned bridges verified directly); #18 GE lattice support + 1^T products.
