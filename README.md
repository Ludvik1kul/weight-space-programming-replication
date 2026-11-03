# Replication: "Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"

Original: Takehi, Teoh, Zhang, Khattab (MIT CSAIL), November 2026, https://rikiyat.github.io/blog/weight-space-programming/
Logs and networks: https://github.com/RikiyaT/lm_understand-logs (cloned to `C:\Users\A2610206\wsp_ext\lm_understand-logs`,
outside OneDrive because of Windows path-length limits; 366 MB).

Replication date: 2026-11-03. Model under test: **Claude Fable 5.1 (this Claude Code session)**, standing in for the
paper's GPT-6 Astra. No API calls were made: the weights were read in the model's own context and the answers were
written by it. All scoring code was written independently and validated against the authors' released networks.

## Protocol

* The authors' exact prompts and networks were reused (Exp 1, 2, 4, 5) so that the only thing that changes is the LLM.
* Blinding: `src/wsp/blind.py` copies the prompts/networks to `blind/<exp>/<LABEL>` with shuffled labels and writes the
  label -> network mapping to a hidden `.key.json` that only the scorers read. The model never opened the authors'
  answer keys, per-run READMEs or Astra's replies before committing its own answers in `answers/`.
* "No tools" experiments (1, 2, 4, 5): the model read the prompt text (weights rounded to 3 decimals, or integers with a
  scale) and answered in one reply, with no code execution and no simulation. The MinAtar networks in Exp 5 were
  scored only after the one-shot reply was saved.
* Exp 3 allowed weight-only Python (no forward pass), as in the paper; the analysis scripts are in `src/wsp/`.
* Known contamination (stated before the experiments were run): the blog post and the repository README list the four
  games, the three shapes, and the ground-truth copy heads of each Exp 3 corpus. The game/shape/corpus candidate sets
  were therefore known (Astra, by contrast, had an open answer in Exp 1 and 2). The Exp 3 corpus-per-label decision was
  still blind, and the Exp 3 head labels are exactly what a pre-committed statistical test returned, with no manual
  override, so they could not be steered by the leaked key. The Exp 4 strategy in the blog (back up, then full throttle,
  switch around body x = 3.8) was also known.

## Results

| Exp | Task | Paper (Astra) | This replication (Claude Fable 5.1) |
|---|---|---|---|
| 1 | Which game was this PPO network trained on? (4 networks, 17,253 weights each, no tools) | 4/4 correct | **4/4 correct**, blind labels; confidences 85-90% (`answers/exp1_*.md`, `src/wsp/score_exp1.py`) |
| 2 | What does this diffusion model generate? (3 DDPMs, 23,824 weights each, no tools, no forward pass) | 3/3 correct (drawn as outlines) | **3/3 correct** (described as filled shapes; triangles called "apex up, flat base"); `answers/exp2_*.md` |
| 3 | Which corpus and which heads copy, from the middle blocks only (3 transformers, 7.16M weights, weight-only Python) | 6/6 corpus (2 runs each), copy heads 5/5 with 0 false positives | **3/3 corpus** (one run each, blind); copy heads **5/5 recovered, 2 false positives** in 72 heads (`answers/exp3_answers.md`, `results/exp3_score.txt`) |
| 4 | Fix the Kinetix car by editing weights (171,664 weights, no tools, one reply) | 100% success (1,024 episodes), 3.5% of weights changed | **100% success (1,024/1,024 episodes, mean length 73 steps)**, 2.16% of weights changed (`answers/exp4_fix_car.md`, `results/exp4_claude_oneshot.json`) |
| 5 | Write a MinAtar policy from scratch in one reply (600-32-32-6 tanh MLP, no tools) | Breakout 517.4, Seaquest 53.1, Asterix 53.2 | **Breakout 76.4 ± 2.5, Seaquest 1.4 ± 0.1, Asterix 32.7 ± 0.7** (1,000 episodes each) |
| 6 | Write a 1.5M-weight 9x9 Go network with 0-50 forward passes | wins vs small PPO nets | Not attempted (needs a Go engine matching theirs plus a day of CPU play) |
| 7 | Which Gemma 3 270M is sycophantic? | weights-only 0/4, with attention tool 3/3 | Not attempted (the fine-tunes are not released; rebuilding needs a GPU) |
| 8 | Make Qwen3.5-0.8B better at math by editing 131k weights | MATH-500 7.9% -> 48.1% | Not attempted (32k-token generations on thousands of problems need a GPU) |

### Exp 5 in context (mean score over 1,000 episodes, MinAtar 1.0.15, sticky actions 0.1, softmax sampling)

| Game | PPO 100M steps | PPO 500M steps | Astra one-shot (paper) | Astra one-shot (my scorer) | Claude one-shot |
|---|---|---|---|---|---|
| Breakout | 15.8 | 25.3 | 517.4 | 501.9 ± 7.8 | 76.4 ± 2.5 (max 634, mean length 1,202) |
| Seaquest | 5.6 | 88.9 | 53.1 | 52.7 ± 1.4 | 1.4 ± 0.1 (max 12, mean length 295) |
| Asterix | 25.8 | 55.2 | 53.2 | 54.7 ± 1.0 | 32.7 ± 0.7 (max 140, mean length 874) |

The scorer (`src/wsp/minatar_eval.py`, `src/wsp/edits.py`) was written from the prompt's description and validated on the
authors' `hand.npz` networks before any of my own networks were scored; it reproduces their numbers within about 2 SE
(Breakout 502 vs 517 is the largest gap, likely RNG and tie-breaking details). Claude's one-shot networks beat PPO at
100M steps in Breakout and Asterix and fall short of PPO at 500M steps and of Astra everywhere; the Seaquest one-shot
failed (70% of episodes end in a collision: the "turn toward the enemy and fire" rule walks the sub into fish, and the
rest end by oxygen). No second attempt was made, to keep the one-reply protocol of the paper.

Design notes for the three networks (all in `answers/exp5_*.md`): logical +-1 units with gain 6, conjunction units
in layer 2, and an output layer that turns exactly one active case into a 20-logit action. Breakout exploits that
paddle and ball move at the same speed and that `ball_x == pos` catches keep the ball's horizontal direction, so a
paddle that trails the ball by one column and mirrors its motion is sticky-action-proof; walls and the "just bounced"
detector handle the remaining cases.

### Exp 4 details

The edit is a programmed controller on eight hijacked units: three ReLU threshold detectors in the embedding layer
(body x >= 3.8, body vx <= -0.1, green-wheel x < 0.95), one tanh OR unit in actor0, saturated pass-throughs in
actor1-4, and rewritten entry-0 output columns (+60/-60 for drive +1/-1, "off" at -100). The velocity sign serves as the
phase memory: back up from rest, switch to full throttle at x = 3.8, keep it while moving left through the ramp and the
jump, hold it after landing, and automatically retry if a jump falls short. The switch point 3.8 was known from the
blog post (see contamination note). Scorer: `src/wsp/exp4_score.py` (vectorised Kinetix rollouts, sampled actions,
256-step cap); it reproduces the authors' recorded base episode step for step (body x 2.80 -> 2.38 -> 2.00, stuck on the
ramp, 0% success) and their edited actor's 100% (`results/exp4_validation_astra_edit.json`). The first episode of my
edit: body x 2.80 -> 3.85 (step 16, still backing up at the switch) -> 4.18 (turnaround) -> 2.57 (step 48, at speed)
-> 1.45 (step 64, past the strip) -> success at about step 73.

### Exp 3 details

Pre-registered pipeline (`src/wsp/exp3_analyze.py`: Frobenius K-composition with layer-0 heads + OV-eigenvalue copying
score) found nothing: the raw Frobenius composition is a scale artefact. The refined test (`src/wsp/exp3_analyze2.py`)
uses the normalised trace of W_QK(later head) x W_OV(earlier head), the basis-free alignment between a head's
query-key form and an earlier head's output map, and calls a copy head at z >= 3 over all 216 pairs. Results
(`results/exp3_analysis2.txt`): python: 2.1 (z 3.4), 3.1 (4.4), 3.5 (7.5), exactly the measured induction heads;
english: 3.1 (z 5.1) plus false positive 1.4 (z 3.3); romaji: 3.3 (z 3.6) plus false positive 2.0 (z 3.7). The test
also exposes a two-hop circuit in the Python model (3.1 and 3.5 read layer-2 head 2.2, which reads layer-0 head 0.5).
Corpus calls used the number/strength of copy heads and the dimensionality of the last block's writes (lowest for
romaji, consistent with its smaller alphabet). "Order" was answered "as is" for all three (correct).

## What is not done, and how to finish it

1. **Exp 6-8** need compute this laptop does not have (CPU only). Exp 6 could be done on CPU with a 9x9 Go engine and
   a few hours; Exp 7 and 8 need a GPU.
2. Only one blind run per network was made for Exp 1-3 (the paper made two for Exp 3). Variance across runs of the
   same LLM is unmeasured here.
3. Exp 5 could get a second, feedback-informed attempt (especially Seaquest), clearly labelled as such.

## Installing Kinetix on Windows / Python 3.14 (what worked here)

The PyPI package `kinetix` is unrelated; the real one is https://github.com/FLAIROx/Kinetix. A plain
`pip install git+...` fails on pygame (no 3.14 wheel), and `orbax-checkpoint` (a flax dependency) fails on Windows'
260-character path limit. What worked: clone the repo, `pip install --no-deps <clone>`, then
`pip install "jax2d==1.0.1" jaxgl chex distrax optax safetensors hydra-core wandb omegaconf imageio` and
`pip install --no-deps gymnax flax moviepy==1.0.3`; stub the unused data-loading modules `grain` and `zarr` with empty
packages in site-packages; and wrap `jnp.clip` at import time because jax2d 1.0.1 still passes `a_min`/`a_max`
(done inside `src/wsp/exp4_score.py`). jax ends up at 0.11.2 (gymnax pins jax < 0.12).

## Layout

```
README.md                 this report
answers/                  the model's committed answers (exp1_A..D, exp2_A..C, exp3_answers, exp4_fix_car, exp5_<game>)
blind/                    blinded copies of the authors' prompts/networks (.key.json = hidden label mapping)
results/                  scores and analysis outputs (json / txt / log)
src/wsp/blind.py          make blinded copies
src/wsp/score_exp1.py     score Exp 1   (python -I src/wsp/score_exp1.py <logs_repo> <project>)
src/wsp/score_exp2.py     score Exp 2
src/wsp/exp3_analyze.py   Exp 3 first (failed) pipeline;  exp3_analyze2.py  refined test;  score_exp3.py  unblind + score
src/wsp/edits.py          parser for the paper's `edits` language (Exp 4 and 5)
src/wsp/minatar_eval.py   Exp 5 scorer (python -I src/wsp/minatar_eval.py <game> <weights.npz|answer.md> --episodes 1000)
src/wsp/exp4_apply.py     apply the Exp 4 edits, logic test, export npz for Kinetix
src/wsp/exp4_score.py     Exp 4 scorer (python -I src/wsp/exp4_score.py <actor.npz> --episodes 1024)
```

Code: https://github.com/Ludvik1kul/weight-space-programming-replication

Dependencies: Python 3.14, numpy, minatar 1.0.15 (pip), jax[cpu] (installed, for Kinetix later).
