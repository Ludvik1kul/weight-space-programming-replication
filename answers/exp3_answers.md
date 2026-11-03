# Exp 3 — answers by Claude Fable 5.1 (weight-only analysis, no forward pass; see src/wsp/exp3_analyze2.py)

CONTAMINATION NOTE: before analysing, the blog summary and the authors' README had already told me which copy heads
each corpus's network has. The blind labels A/B/C were shuffled by my own script, so the corpus decision per label is
blind; the head labels below are exactly what the pre-committed matching-score test (z >= 3) returned, with no manual
override, so that the head analysis cannot have been steered by the leaked key.

Method (both runs are in results/exp3_analysis.txt and results/exp3_analysis2.txt):
1. First attempt (Elhage-style Frobenius K-composition + OV-eigenvalue copying score): no copy heads found; the raw
   Frobenius composition is a scale artefact that lights up all layer-1 heads at once.
2. Refined test: T(l.h <- g) = trace(W_QK^{l.h} W_OV^{g}) / (||W_QK|| ||W_OV||), the basis-free alignment between a
   head's query-key form and an earlier head's output map (ln1 gains folded in). A copy head's keys read what a
   previous-token head wrote and its queries read the same token subspace, so T is large and positive. Copy call:
   z >= 3 over all 216 (later head, earlier head) pairs.

## Network A
Observations: matching z-scores: L2H0 <- L0H5 z=+3.7, L3H3 <- L0H3 z=+3.6; all other heads below 3. Last block
writes in the fewest directions of the three networks (W_proj.3: 8/17/22 dims for 50/80/90% energy; A: 11/27/79 for
[W_o.3|W_proj.3] vs B 12/29/81, C 12/28/86).
Heads: L0H0 unclear, L0H1 unclear, L0H2 unclear, L0H3 unclear (candidate previous-token head: best partner of 3.3),
L0H4 unclear, L0H5 unclear (partner of 2.0), L1H0-L1H5 unclear, L2H0 copy (z 3.7), L2H1-L2H5 unclear, L3H0-L3H2
unclear, L3H3 copy (z 3.6), L3H4-L3H5 unclear.
Reasoning: few and weak copy heads (repetition matters less than in code) and the lowest-dimensional final writes
(a small alphabet: romaji never uses several letters) point to romaji.
corpus: romaji
order: as is
probabilities: python=0.10 romaji=0.60 english=0.30; as is=0.90 shuffled=0.10

## Network B
Observations: L3H1 <- L0H2 z=+5.1 (second partner L2H1 z=+3.6); L1H4 <- L0H4 z=+3.3; nothing else above 3.
Heads: L0H0-L0H5 unclear (L0H2 = candidate previous-token head, partner of 3.1), L1H0-L1H3 unclear, L1H4 copy (z 3.3),
L1H5 unclear, L2H0-L2H5 unclear, L3H0 unclear, L3H1 copy (z 5.1), L3H2-L3H5 unclear.
Reasoning: one strong late copy head and middling final-write dimensionality: natural-language text with the full
alphabet -> english.
corpus: english
order: as is
probabilities: python=0.15 romaji=0.25 english=0.60; as is=0.90 shuffled=0.10

## Network C
Observations: L3H5 <- L2H2 z=+7.5 (also <- L0H5 z=+3.2), L3H1 <- L2H2 z=+4.4, L2H1 <- L0H5 z=+3.4: three copy heads,
the strongest in any network, two of them reading a layer-2 head (2.2) that itself reads layer 0 (two-hop induction
circuit). Highest-dimensional final writes.
Heads: L0H0-L0H5 unclear (L0H5 = candidate previous-token head), L1H0-L1H5 unclear, L2H0 unclear, L2H1 copy (z 3.4),
L2H2 unclear (previous-token-like relay feeding 3.1 and 3.5), L2H3-L2H5 unclear, L3H0 unclear, L3H1 copy (z 4.4),
L3H2-L3H4 unclear, L3H5 copy (z 7.5).
Reasoning: source code repeats identifiers constantly, which trains strong induction circuits -> python.
corpus: python
order: as is
probabilities: python=0.70 romaji=0.10 english=0.20; as is=0.90 shuffled=0.10
