# Does Learning to Stop Scale?

In [Where the Thinking Stops Mattering](https://github.com/Herz1c/WhereTheThinkingStopsMattering) I taught
Qwen3-0.6B to stop reasoning once its own thinking stops mattering. The result was 43% fewer generated tokens at
about −1 pp accuracy, with the change being in the weights only. This repository asks whether the same recipe
works on larger models without re-tuning it: Qwen3-1.7B, Qwen3-4B and Qwen3.8-27B. The short answer: **yes on
short tasks at every size, and no on long reasoning**, where the adapter turns out to have learned how long
reasoning usually is and not when its own answer is settled.

**Answer for 1.7B is yes, and it is even slightly better.** Over the same four benchmarks, the 1.7B model trained with
the same recipe generates **47.4% fewer tokens at −0.14 pp average accuracy** (mean of two full evaluations),
against 43.4% / −0.97 pp at 0.6B. On HumanEval, which played no part in any decision, it saves 58% at
−0.8 pp.

**Answer for 4B is yes as well.** Same recipe, same criterion: **43.7% fewer tokens at −1.05 pp average accuracy**
(mean of two full evaluations), close to the 0.6B result benchmark by benchmark. On HumanEval it saves 62% at
+2.1 pp. The accuracy loss is small but real: its interval excludes zero in both runs.

| Model | Tokens saved | Accuracy change | HumanEval (held-out) |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B | 43.4% | −0.97 pp | 65.0% / −1.22 pp |
| Qwen3-1.7B | 47.4% | −0.14 pp | 58.4% / −0.76 pp |
| Qwen3-4B | 43.7% | −1.05 pp | 61.7% / +2.13 pp |
| Qwen3.8-27B | 60.4% | +0.04 pp | 67.8% / +2.74 pp |

**Answer for 27B on the same four benchmarks is yes: 60.4% fewer tokens at +0.04 pp** (mean of two full
evaluations, both intervals cover zero). But that average hides two opposite effects, and they matter for the
next result ([details](#result-qwen38-27b)).

**On long reasoning the answer is no.** The same 27B adapter, run on six long-reasoning sets (AIME 2026, two
HMMT rounds, LiveCodeBench v6, IFBench, MMLU-Pro; base replies of tens of thousands of thinking tokens), cuts
**90.5% of the thinking and loses 36.9 pp of accuracy**. A second version of the method, built for long
reasoning and judged by a rule fixed before its run, **also failed** at its pilot (−4.1 pp overall, −20 pp on
the longest questions), and the work stopped there as the rule said. Both are reported in full
[below](#where-it-breaks-long-reasoning).

Four sizes, one training run each, are not enough to claim a trend with size, and the 27B model belongs to a
newer generation. What they show is that the recipe keeps working on short tasks without re-tuning, and that
this says little about long ones.

Student project, October–November 2026. 1.7B: one RTX 5060 Ti (16 GB), about 70 GPU-hours. 4B: an RTX PRO 6000
(96 GB) provided and run by [mrtineu](https://github.com/mrtineu), about 20 GPU-hours. 27B: two H200 cards in a
datacentre, run by the same collaborator; about 12 hours of wall clock for training and evaluation, 15 for the
long-reasoning measurement and 9 for the pilot.

<p align="center"><img src="figures/scaling3_report_benchmarks.png" alt="Tokens saved and accuracy change per benchmark, Qwen3-0.6B, Qwen3-1.7B and Qwen3-4B" width="900"></p>

## Result: Qwen3-1.7B

| Benchmark | What it tests | Tokens saved, 1.7B | Accuracy change, 1.7B | 0.6B (two-run mean) |
| --- | --- | ---: | ---: | ---: |
| MBPP+ | writing a Python function | **55.7%** | **+4.12 pp** | 64.9% / −0.98 pp |
| IFEval | following format instructions | **50.3%** | −0.23 pp | 38.9% / −0.61 pp |
| BFCL (3 subsets) | function calling | **33.7%** | −0.84 pp | 30.8% / −0.94 pp |
| SVAMP | grade-school arithmetic | **47.4%** | −3.60 pp | 30.8% / −1.33 pp |
| **Average** | | **47.4%** | **−0.14 pp** | 43.4% / −0.97 pp |

Mean of two full evaluations of the same adapter, 8 samples per question, 22,072 responses each:
+47.3% [45.9, 48.7] / −0.21 pp [−0.95, +0.59] and +47.4% [46.0, 48.7] / −0.07 pp [−0.83, +0.74]
(95% paired bootstrap intervals over questions). Tokens saved per benchmark is one minus the ratio of
summed tokens, thinking included, a response cut off at the cap counted in full and as wrong; the
average is the geometric mean over benchmarks, accuracy change the arithmetic mean. vLLM is not
bit-reproducible: only 34% of responses are identical between the two runs. The 0.6B column is the mean of
its two full runs as well; the per-benchmark table in the 0.6B repository shows one of the two, so those
numbers differ slightly from these.

**Held-out sets** (never looked at during development):

| Set | Tokens saved | Accuracy change |
| --- | ---: | ---: |
| HumanEval (164 × 4) | **58.4%** [54.5, 62.2] | −0.76 pp [−4.57, +3.05] |
| GSM8K test (500 × 2) | 35.9% [32.5, 39.1] | +34.1 pp (mostly the requested answer format, learned) |
| MATH-500, all levels (313 × 4) | 11.4% [9.6, 13.5] | +4.8 pp [+2.7, +7.0] (graded format-independently) |

**Criterion fixed in advance** for "the method transfers": at least 30% saved and at least −1.5 pp average
accuracy on the two-run mean, and a HumanEval interval that covers zero. All three are met. (Written in my
research log before the 1.7B runs. The log is private, so this is self-reported; see
[PREREGISTRATION.md](PREREGISTRATION.md), where later criteria are committed before their results.)

### What did not go well

- **SVAMP loses 3.6 pp, and about 90% of it is answer format, not arithmetic.** The trained model more often
  writes the computation into the answer line ("Final answer: 5 + 3 = 8") or closes its reasoning twice;
  SVAMP's grader rightly rejects both. Wrong values rose only from 388 to 414 of 8,000. The habit appears
  after the supervised stage and the RL stage amplifies it while shortening SVAMP from 24% to 47% saving.
- **BFCL irrelevance loses 4.7 pp:** the model calls a function more often where it should decline.
- Both were fixed before the 4B run (below).

## Result: Qwen3-4B

| Benchmark | Tokens saved, 4B | Accuracy change, 4B | 1.7B | 0.6B |
| --- | ---: | ---: | ---: | ---: |
| MBPP+ | **64.3%** | −1.47 pp | 55.7% / +4.12 pp | 64.9% / −0.98 pp |
| IFEval | **41.8%** | −1.37 pp | 50.3% / −0.23 pp | 38.9% / −0.61 pp |
| BFCL (3 subsets) | **29.5%** | −0.18 pp | 33.7% / −0.84 pp | 30.8% / −0.94 pp |
| SVAMP | **31.6%** | −1.19 pp | 47.4% / −3.60 pp | 30.8% / −1.33 pp |
| **Average** | **43.7%** | **−1.05 pp** | 47.4% / −0.14 pp | 43.4% / −0.97 pp |

Two full evaluations of the same adapter: +43.9% [42.5, 45.3] / −1.14 pp [−1.68, −0.57] and
+43.6% [42.1, 45.0] / −0.97 pp [−1.46, −0.44]; 32% of responses are identical between the two runs. The
supervised stage alone gives 31.1% / −0.78 pp; the RL stage adds the rest, mostly on SVAMP (8% → 32%) and MBPP.
The model still reasons on every question: no response in any run has fewer than ten thinking tokens.

| Held-out set | Tokens saved | Accuracy change |
| --- | ---: | ---: |
| HumanEval (164 × 4) | **61.7%** [58.6, 64.9] | +2.13 pp [−1.37, +5.95] |
| GSM8K test (500 × 2) | 32.4% [30.0, 34.9] | +6.6 pp [+4.7, +8.7] (graded format-independently) |
| MATH-500, all levels (313 × 4) | 11.5% [10.0, 13.2] | +10.4 pp [+8.0, +12.9] (graded format-independently) |

The **criterion fixed in advance** (the same as for 1.7B, written down in the research log before any 4B
training result; self-reported, as above) is met:
43.7% ≥ 30%, −1.05 pp ≥ −1.5 pp, and the HumanEval interval covers zero. HumanEval is a weak test at this
size, though: the 4B base already solves 89.5% of it. GSM8K and MATH share their source with the training pools.

### What changed for 4B

- **The two 1.7B negatives were addressed first.** Math answers now count as correct in training only if the
  last line is "Final answer: N" with a bare number, and every supervised math example closes that way. On SVAMP
  the "number alone" errors stay at the base model's level (22 → 19 of 8,000; at 1.7B they went 18 → 200). The
  function-calling pool now includes requests where the right function has been removed, and BFCL irrelevance
  loses 1.0 pp with an interval covering zero (1.7B: −4.7 pp).
- **Harder pools, by the same matching rule:** GSM8K with MATH levels 3–4, KodCode hard with some medium,
  function calling with more offered functions, instructions with four constraints instead of one or two. Share of
  correct base attempts: math 66% (target 71%), code 54% (52%), instructions 69% (67%). The function-calling
  pool stayed far too easy (89% against 64%) and gave the RL stage almost no signal.
- **Two bugs found by the calibration run and fixed:** the function-calling labels read only the first assistant
  turn, so some "no call" examples were mislabeled; and the checker for "end with a question / exclamation"
  failed on a trailing emoji. Both existed in the 0.6B and 1.7B data as well.
- The pool composition and both fixes were decided after seeing the 4B base on calibration questions, not
  before; the training recipe, its hyperparameters and the criterion were fixed before any 4B training result.
- Base and trained 4B model were generated on the same card, a different one from the smaller studies.

### What did not go well at 4B

- **SVAMP still loses 1.2 pp, now through wrong values** (238 → about 357 of 8,000), concentrated in the
  longest fifth of questions (−4.5 pp), partly offset by capped responses almost disappearing (75 → 11).
- **A new format defect:** about 35 SVAMP responses open the answer with a second `</think>` and are rejected
  although the number is usually right (≈ 0.4 pp). The reward gives such answers nothing, yet the RL stage did
  not remove them.
- **IFEval loses 1.4 pp** (interval excludes zero in one run of two), **MBPP 1.5 pp** (borderline), the latter on
  the questions the 0.6B base never solves.
- Only 16 give-up examples for math qualified under the stricter answer format, where up to 300 were allowed.

## Result: Qwen3.8-27B

Same recipe, same protocol (temperature 0.6, cap 4,096 tokens, MBPP+ 8,192), same criterion.

| Benchmark | Tokens saved, 27B | Accuracy change, 27B | Base → trained: mean tokens, capped replies |
| --- | ---: | ---: | --- |
| MBPP+ | **82.6%** | +1.54 pp | 1,906 → 330, 11.0% → 0.2% |
| IFEval | **70.0%** | −1.31 pp | 1,359 → 411, 10.3% → 1.0% |
| BFCL simple / multiple / irrelevance | −2.3% / 10.3% / 23.1% | +1.49 / −1.10 / −1.15 pp | 130 → 133, 150 → 135, 323 → 249 |
| SVAMP | **46.0%** | −0.22 pp | 266 → 144, 1.3% → 0.0% |
| **Average** | **60.4%** | **+0.04 pp** | |

Two full evaluations of the same adapter: +60.3% [58.5, 62.0] / +0.01 pp [−0.85, +0.88] and
+60.4% [58.7, 62.0] / +0.06 pp [−0.81, +0.94]. The supervised stage alone gives 42.4% / +1.73 pp; the RL stage
buys 18 points of saving for 1.7 pp of accuracy. No reply in either run has fewer than 20 thinking tokens.

| Held-out set | Tokens saved | Accuracy change |
| --- | ---: | ---: |
| HumanEval (164 × 4) | **67.8%** [61.6, 72.2] | +2.74 pp [−0.30, +6.10] |
| GSM8K test (500 × 2) | 51.6% | +0.50 pp [−0.80, +1.90] |
| MATH-500, all levels (313 × 4) | 39.8% | +7.43 pp [+5.03, +9.82] |

The **criterion fixed in advance** (the same as before, written in the research log on 2026-11-18 before any
27B training data existed; self-reported) is met: 60.4% ≥ 30%, +0.04 pp ≥ −1.5 pp, HumanEval interval covers zero.

### What the 27B average hides

- **A third to a half of the saving is replies the base model never finished.** Replies that the base ran into
  the cap account for 52% of the MBPP+ saving, 40% of SVAMP's and 32% of IFEval's; 43–72% of those are now
  right. That source of saving exists only because of the 4,096 / 8,192 cap.
- **The flat accuracy is a sum of a loss and a gain.** Bucketing questions by how long the base model thinks on
  them: where it thinks 0.5–2k tokens the trained model loses 6.7 pp; where the base runs into the cap it gains
  11 to 30 pp, because a short answer beats an unfinished one. The three smaller sizes lose 1–2.5 pp fairly
  evenly instead.

  | Base median thinking | 0.6B | 1.7B | 4B | 27B | Base replies capped (27B) |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | < 0.5k tokens | −0.9 pp | −1.0 | −0.5 | −0.6 | 0.1% |
  | 0.5–2k | −0.8 | −1.9 | −2.5 | **−6.7** | 2.2% |
  | 2–8k | −1.1 | +0.1 | −1.1 | **+11.0** | 37.6% |
  | > 8k | −2.5 | +4.7 | +4.4 | **+30.1** | 80.4% |

  (The buckets are defined from the base run's own samples, a small selection effect in the base's favour.)
- **The doubled `</think>` is back, on IFEval:** about 80 replies per run (1.8%) start the answer, close the
  thinking a second time and restart; every one is graded wrong. That alone is more than the whole IFEval loss.
- **BFCL irrelevance loses 0.7–1.6 pp** (interval excludes zero in one run); function calling saves little,
  because the 27B base hardly thinks there to begin with.
- The base model is evaluated at temperature 0.6, this study's protocol; its vendor recommends 1.0 for thinking
  mode. The capped tail was measured under this sampling only.
- The 27B model is of a newer generation than the three Qwen3 sizes: a transfer test, not a fourth point on
  one curve.

## Where it breaks: long reasoning

The short benchmarks have replies of hundreds to a few thousand tokens. To see what the 27B adapter does where
the base model reasons for tens of thousands, I ran both on six public long-reasoning sets at the vendor's
sampling (temperature 1.0, up to 253,952 generated tokens, 4 samples per question, MMLU-Pro 1). GPQA-Diamond
was planned as a seventh; its download failed on the cluster and it was not measured.

| Set | Accuracy, base | Accuracy, adapter | Change | Mean thinking tokens, base → adapter |
| --- | ---: | ---: | ---: | ---: |
| AIME 2026 | 96.7% | 58.3% | −38.3 pp [−51.7, −25.8] | 19,449 → 5,659 |
| HMMT February 2026 | 93.2% | 41.7% | −51.5 pp [−63.6, −37.9] | 34,936 → 1,772 |
| HMMT November 2025 | 99.2% | 48.3% | −50.8 pp [−65.8, −35.8] | 21,601 → 1,640 |
| LiveCodeBench v6 | 92.7% | 54.0% | −38.7 pp [−44.9, −32.6] | 28,155 → 1,401 |
| IFBench | 82.7% | 45.3% | −37.4 pp [−41.8, −32.6] | 8,031 → 316 |
| MMLU-Pro (1,000) | 84.3% | 79.7% | −4.6 pp [−6.6, −2.6] | 5,210 → 343 |
| **Unweighted mean** | 91.5% | 54.6% | **−36.9 pp** | **−90.5%** |

The base model's accuracy is within 3 pp of what BottleCapAI publishes for the same base on each of these sets,
so the prompts and graders are close enough to theirs to be used. For orientation: their own efficient variant
of this model, ThinkingCap-Qwen3.8-27B, reports about 37% less thinking at under 1 pp. This adapter is nowhere
near that.

By how long the base model thinks on the same question (six sets pooled):

| Base median thinking | Questions | Accuracy change | Thinking, base → adapter |
| --- | ---: | ---: | ---: |
| < 0.5k tokens | 511 | −0.8 pp | 240 → 177 |
| 0.5–2k | 294 | −7.6 pp | 1,107 → 799 |
| 2–8k | 299 | −21.4 pp | 4,383 → 489 |
| 8–32k | 333 | −31.9 pp | 16,971 → 787 |
| > 32k | 131 | −55.9 pp | 57,410 → 1,733 |

**The adapter learned an absolute stopping length, not "stop when the answer is settled".** It stops after one
to two thousand thinking tokens whatever the question needs (90th percentile about 3k on mathematics and code).
Every training example of "stop" was short, because the training rollouts were capped at 6,144 tokens; the
give-up examples at 2,048 tokens are a minor part (9–12% of its replies end near that budget). I had predicted
65–80% saved at −8 to −15 pp before the measurement: the direction was right and the size wrong by a factor
of about three.

Two grading faults of mine were found and fixed on the way (a `from math import *` that shadowed the built-in
three-argument `pow`, and time limits on the cluster that did not reproduce locally); the numbers above are from
the regraded records.

### Second attempt: a method for long reasoning, and its failure

Version 2 trains from the base model on long rollouts of new sources (competition mathematics, rated
programming problems, multiple choice, instructions with constraints, function calling), with no give-up
examples and no thinking budget. A calibration of the base model put the attainable saving at about 38% of
thinking tokens on mathematics and 37% on code (thinking after the point from which a forced answer is right
and stays right). Two objectives competed in a pilot: **trace** (the model's own thinking up to that point,
then `</think>` and the answer) and **decision** (whole rollouts; only the decision to close the thinking is
trained, the rest held to the base distribution). The rule, written before the run: an adapter passes when the
overall accuracy change on 300 validation questions is at least −1.0 pp and every length bucket is at least
−3.0 pp; if neither passes, the work stops.

| | trace | decision |
| --- | ---: | ---: |
| Overall accuracy change | −4.08 pp [−6.83, −1.42] | −15.92 pp [−18.67, −12.83] |
| Overall saving of thinking tokens | 67.3% [62.9, 70.6] | 62.0% [54.8, 66.7] |
| Base thinking < 2k tokens (190 questions) | +3.42 pp | −5.79 pp |
| 2–8k (47) | −11.70 pp | −18.62 pp |
| 8–32k (35) | −21.43 pp | −42.86 pp |
| > 32k (28) | −20.54 pp | −46.43 pp |

**Neither passes.** As the rule said, no further data were generated, no main training was run and the
long-reasoning sets were not opened for these adapters.

What went wrong, found after the fact and therefore a diagnosis, not a test:

- **Trace learned a typical length per kind of task again.** On mathematics it thinks for a median of 216, 559
  and 725 tokens on questions where the base needs under 2k, 2–8k and 8–32k. It saves 67% where the
  calibration said 38% was attainable: it stops well before its own training targets.
- **Decision moves the reasoning into the answer.** It closes its thinking early and keeps working after
  `</think>` (code answers up to 23,000 tokens, 15% accuracy). After training it still gave about a 15% chance
  of stopping at each boundary where it should go on, which compounds over tens of boundaries.
- **The likely reason for the first:** in half of the mathematics rollouts that settle early, the base model
  already answers right twice after about one paragraph. The training rows therefore say "one paragraph is
  enough" for half of mathematics, and nothing in a row tells the model which half a question is in.
- Where reasoning is short, version 2 holds up: +3.4 pp at 65% saved in the shortest bucket.

One caveat on the word "pre-registered": the rule was written before the run and not changed by it, but
[PREREGISTRATION.md](PREREGISTRATION.md) was pushed about seven hours after the pilot had started on the
cluster. For the pilot it is therefore self-reported, like the 1.7B, 4B and 27B criteria; the file says so, and
has the full protocol and outcome.

## Harder data for a stronger model

The 0.6B recipe trains on GSM8K and easy coding tasks. For 1.7B those are nearly solved: 80% of GSM8K
questions are answered right in all four tries, and 87% of finished attempts at the easy coding tasks pass
their tests. Yet on the benchmarks 1.7B spends 26% *more* tokens than 0.6B, mostly in a long tail of coding
answers that run out of budget. Training only on easy questions risks a model that learns "short is fine"
and then stops too early on hard ones.

So the training pools were **matched in difficulty to what 0.6B saw**, by a rule fixed before any 1.7B
training result: mix the easy pool with a harder one (MATH levels 3–5; KodCode medium and hard) in the
proportion that gives the 1.7B base the same share of correct attempts as 0.6B had on its original pool
(math 70.8%, code 51.9%). Calibration on 200 hard questions per domain gave 40% hard math and 80% hard code.

Did it work? On the questions the base model gets right at most half the time, the trained model is *more*
accurate on all four benchmarks (+5.6 to +9.3 pp), and on HumanEval's longest fifth +14.8 pp. (Those subsets
are picked from the base run's own samples, which flatters any second run a little; the 0.6B model under the
same selection gained nothing on code or function calling, so the gap between the sizes is not just the
selection.) On
MATH-500, all three model sizes save the least on the hardest levels and gain accuracy there:

<p align="center"><img src="figures/scaling3_report_ladder.png" alt="MATH-500 by level: tokens saved and accuracy change for Qwen3-0.6B, Qwen3-1.7B and Qwen3-4B" width="900"></p>

The model does not cut its reasoning uniformly; it cuts where the extra reasoning was not paying off.

## The method, briefly

Both stages train a LoRA (r = 16) on the model's own outputs, never on evaluation data:

1. **Stop-point distillation.** Sample the base model's reasoning on training tasks, force `</think>` at
   paragraph boundaries, find the earliest cut where the forced answer is verifiably right (unit tests, the
   gold number, the right function call, a checked format constraint), and train on *own reasoning up to the
   cut + `</think>` + answer*. Three rules from the 0.6B work keep accuracy: on code, stop only after the
   function is drafted in the reasoning; after a long unproductive attempt, give up with a best guess; for
   instruction following, use verifiable constraints of types IFEval does not use.
2. **GRPO** from that adapter, reward `1[correct] · (1 − α · length / domain mean length)`; a wrong or capped
   answer earns nothing whatever its length.

The 0.6B repository explains the method and its failures in detail; nothing about it was re-tuned here.

## What changed from the 0.6B setup, and why

All written into the research log before the results they could affect (the log is private; see
[PREREGISTRATION.md](PREREGISTRATION.md) for what that does and does not prove):

- **Own evaluation module** (`evaluation/`) instead of the reference harness. Same protocol; the benchmark
  authors' graders (EvalPlus, BFCL's AST checker, Google's IFEval verifiers) are downloaded at pinned
  revisions and called directly. Checked against the reference on the 0.6B baseline: byte-identical question
  file, 22,070 of 22,072 stored responses graded the same (the two others hinge on a random draw inside an
  IFEval verifier), and a full re-run gave −0.3% tokens / −0.11 pp with every interval covering zero.
- **Difficulty-matched training pools** (above).
- **A fixed code grader.** 1.7B often follows its solution with a usage-example block; the training grader
  read the last block and called correct solutions wrong. It now reads the block that defines the function.
- **Hard MATH solutions in the supervised data** close with `\boxed{N}` rather than the requested
  "Final answer: N"; only that closing line is rewritten, the reasoning stays the model's own.
- **Difficulty-aware reporting:** MATH-500 by level, and accuracy by how often a fixed reference (the 0.6B
  base) solved each question, so the same bins can be laid over every model size.

Deviations that happened rather than were planned: the run was interrupted twice by Windows Update and once
by a pause; every step resumed from its saved state, except that GRPO continued from step 30 with a fresh
optimizer state. The hard coding pool came out easier than its calibration sample (56.8% correct attempts
against a 51.9% target). Details, timings and every number are kept in a private research log.

## Reproduce

Requirements: Python 3.12, a CUDA GPU with ≥ 16 GB, vLLM, Transformers 4.57, PEFT, bubblewrap for running
model-written code. Set `BASE_MODEL=Qwen/Qwen3-1.7B` for every training script. The 4B run (a card with far
more memory) is scripted end to end in `chains/`: `setup.sh`, `calibrate_4b.sh`, `generate_4b.sh`, `train_4b.sh`;
the 27B run in `calibrate_27b.sh`, `generate_27b.sh`, `train_27b.sh`, the long-reasoning measurement in
`reference_27b.sh` and the pilot in `calibrate_long_27b.sh` and `long_27b.sh`.

```bash
python -m evaluation.sources --bench bench                      # benchmarks + their graders, pinned
python -m evaluation.run --out runs/base --model Qwen/Qwen3-1.7B
python stopping/base_data.py                                     # GSM8K etc. -> data/train.jsonl
python stopping/code_tasks.py build                              # KodCode easy
python stopping/code_tasks.py build --difficulty medium,hard --prefix code_hard
python stopping/math_tasks.py                                    # MATH train, levels 3-5
python stopping/tool_tasks.py build                              # Glaive function calling
python analysis/overlap_check.py --train data/tool_train.jsonl --sources glaive     --questions bench/questions.jsonl --write-clean data/tool_train_clean.jsonl
python stopping/constraint_tasks.py --out data/constraint_train.jsonl --count 3000 --questions bench/questions.jsonl
python stopping/constraint_tasks.py --out data/wrapper_train.jsonl --count 2500 --wrappers --questions bench/questions.jsonl
```

Then, per domain: `generate_rollouts.py` → `mix_pools.py calibrate/merge` → `find_stop_points.py` →
`verified_examples.py --mode draft` → `give_up_examples.py` → `build_shortest.py` → `train_lora.py` →
`train_grpo.py --pool math=… --pool code=…` → `evaluation.run --adapter …` and `evaluation.compare`.
Each script's docstring gives its arguments; the values used are those of the 0.6B run, plus the
difficulty-matching steps above.

## Data and licences

Training data: GSM8K (MIT), MATH train split (MIT), KodCode-Light-RL-10K (CC BY-NC 4.0, used for
non-commercial research and not redistributed), Glaive function calling v2 (Apache-2.0),
databricks-dolly-15k (CC BY-SA 3.0) with own format constraints. Every training source was checked for
8-gram overlap with all evaluation questions. Benchmarks and their graders are downloaded from their authors
and not redistributed. Base models: Qwen3-1.7B, Qwen3-4B and Qwen3.8-27B (Apache-2.0). The long-reasoning
pools (DeepMath-103K, HARP, CodeContests, MedQA and others) and their overlap filter are listed in
[PREREGISTRATION.md](PREREGISTRATION.md).

The evaluation protocol (prompts, sampling, token caps, metric) follows the
[ThinkingCap Test](https://github.com/BottleCapAI/ThinkingCap-Test) by BottleCapAI (Apache-2.0), where the
0.6B work started; this repository re-implements it and contains none of its code.

## Next

The open problem is the one the 27B work exposed: a stop that follows the question and not the typical length
of its kind. Searching for it on the 27B model costs a day of two large cards per attempt, so the failed pilot
is being repeated unchanged on Qwen3-4B and Qwen3-0.6B to see whether it fails the same way there. Both
replications, and how their results will be read, were committed in [PREREGISTRATION.md](PREREGISTRATION.md)
before they started; they are not reported here yet. A new method goes back to the 27B model only after it
passes the same rule on 4B in every length bucket, including lengths it was not trained on.

Still open from the short-task studies: the doubled `</think>`, a function-calling pool that is actually hard
for the base model, held-out sets for function calling and instruction following, a harder held-out coding
set, and more than one training seed.
