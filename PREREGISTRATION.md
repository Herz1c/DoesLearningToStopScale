# Pre-registration

What counts as success, and how choices are made, written down **before** the results they apply to. The git
history of this file is the timestamp: a rule counts as pre-registered from the commit that first contains it.

Changes are appended as dated amendments; nothing is rewritten after a result is in.

## Status of earlier claims (stated plainly)

The README says the 1.7B and 4B success criterion was fixed before the results. It was, in my private research
log (1.7B: 2026-10-12; 4B: 2026-11-15, before any 4B training result). That log is not public, and in this
repository's history the criterion first appears in the same commits as the results (`d66f054`, `d6e7ffe`).
**So for 1.7B and 4B the pre-registration is self-reported and cannot be verified from outside.** The criterion
was the same both times: at least 30% of tokens saved and at least −1.5 pp average accuracy on the mean of two
full evaluations, and a HumanEval interval that covers zero.

From this file on, criteria are committed here before the runs they judge.

---

## Version 2: long reasoning on Qwen3.8-27B

First committed 2026-11-24. Written 2026-11-20 (criteria A–F, plan) and 2026-11-23 (method, validation rule,
amendment 1) in the research log; the text below is taken from there.

### Why a version 2

The recipe of this repository was developed on tasks whose reasoning is short (hundreds to a few thousand
tokens). On long-reasoning benchmarks (competition mathematics, harder programming problems, graduate-level
multiple choice; replies of tens of thousands of tokens) an adapter trained that way stops far too early and
loses a large amount of accuracy: it learned roughly how long reasoning usually is, not when its own answer is
settled. Version 2 is meant to give one adapter that shortens reasoning at every length without that loss.
The measurements behind this paragraph will be reported in full together with the version-2 results.

### What had been seen when this was written

- The base model (Qwen3.8-27B, no adapter) on the reference sets below and on the short benchmarks.
- The earlier short-task adapter on the same sets (the failure described above).
- A calibration of the base model on new training sources: 962 questions × 2 rollouts, with answers forced at
  up to 12 paragraph boundaries per rollout to see from where the answer is right and stays right.
  Result: of all thinking tokens, 38.0% (mathematics) and 36.9% (code) lie after that point in rollouts that
  end right. The rule for going on was "both at least 20%"; it is met.
- A plumbing test of the whole chain on Qwen3-0.6B with stand-in data (see amendment 1).

Not seen: any result of any version-2 training on the 27B model. The instructions for the first such run (the
pilot below) had been prepared when this file was committed; none of its output had been looked at.

### Success criteria (one adapter, trained from the base model)

Reference sets: AIME 2026, HMMT February 2026, HMMT November 2025, GPQA-Diamond, LiveCodeBench v6, MMLU-Pro
(1,000 questions), IFBench. Temperature 1.0, top-p 0.95, top-k 20, up to 253,952 generated tokens, 4 samples per
question (MMLU-Pro 1).

- **A.** Mean over the seven sets of the reduction in mean thinking tokens: at least 30%.
- **B.** Mean accuracy change over the seven sets: at least −1.5 pp.
- **C.** No single set worse than −4 pp.
- **D.** The four short benchmarks of this repository, two full runs: at least 40% of tokens saved and at least
  −1.5 pp on the two-run mean.
- **E.** Accuracy by length: questions bucketed by the base model's median thinking (< 0.5k, 0.5–2k, 2–8k,
  8–32k, > 32k tokens): change of at least −2 pp in every bucket.
- **F.** The share of replies with fewer than 20 thinking tokens does not rise against the base model.

Full success = A–F. Lower level = A at least 20% with B–F. Anything else is reported as not met, with the
numbers. The reference sets are opened once for the final supervised adapter and, if an RL stage follows, once
more for its adapter; no other adapter is run on them.

My guess before any data: full success 35–45%, at least the lower level 65–75%.

Known gap, declared now: criterion E was meant to include the short benchmarks measured without their token
cap. The evaluation chain does not do that yet; if it still does not at the end, E is reported on the reference
sets and the validation questions only, and named as a gap.

For orientation only, not a criterion: the vendor's own efficient variant of the same base model
(ThinkingCap-Qwen3.8-27B, a full-weight fine-tune) reports, on these seven sets, a mean thinking reduction of
35.7% at −1.25 pp (my arithmetic from its published table).

### Method (fixed before any training)

Definitions. A rollout is *usable* when it closed its thinking and two answers forced at its own end are both
right. Its *settled point* is the earliest read-out boundary from which every later one is right twice.

Two training objectives compete in a pilot:

- **trace** — the earlier recipe on long data: the model's own thinking up to the settled point, `</think>`,
  and the shortest of four answers sampled there that passes the strict grader. A usable rollout that settles
  only at its own end is kept whole. Cross-entropy on the completion.
- **decision** — every base rollout whole; only the decision to end the thinking is trained: −log p(`</think>`)
  at the paragraph boundaries at or after the settled point, KL divergence to the base model's own
  distribution everywhere else (after Settle, arXiv 2609.38997; the choice of boundaries and weights is mine).

Both: no give-up examples, no fixed thinking budget. LoRA r 16, α 32, dropout 0.05 on the attention and MLP
projections; AdamW, learning rate 1e-4, 5% warm-up then cosine, 2 epochs, 16 sequences per step, gradient clip
1.0, sequences up to 40,000 tokens (trace rows above that are dropped, decision rows cut to their first 40,000
tokens). One setting per objective; nothing is tuned on the 27B model.

Data. Calibration pool (962 questions) for the pilot; for the main training additionally 1,641 new questions
(mathematics 550, code 370, multiple choice 350, instructions with three format constraints 271, function
calling 100), two rollouts each. Sources: DeepMath-103K and HARP (no AIME problems), CodeContests (Codeforces
problems with a rating, checked against a reference solution), MedQA and easy science multiple choice, Dolly
instructions with own constraints (none of the emoji type, which IFBench uses), Glaive function calling.
Every question is filtered for 8-gram overlap with the reference sets and the short benchmarks. Planned
deviation from the first plan: no new constraint types were built; the calibration showed the existing
three-constraint prompts are not saturated (62% base accuracy).

### Validation measure and selection rule

300 validation questions (60 per domain, disjoint from all training questions), 4 rollouts per question for the
base model and for every adapter, temperature 1.0, up to 65,536 tokens.

- Accuracy = the strict grade; a reply that runs into the cap is wrong.
- Per domain: accuracy change in pp, and saving = 1 − mean thinking tokens / the base model's mean thinking
  tokens. Overall = unweighted mean over the five domains, for both.
- Length buckets by the base model's mean thinking per question, all domains together: < 2k, 2–8k, 8–32k,
  > 32k. A bucket with fewer than 20 questions joins the next shorter one. Bucket change = mean over its
  questions.

**Rule: an adapter passes when the overall change is at least −1.0 pp and every bucket is at least −3.0 pp.**
Point estimates decide; bootstrap intervals are reported next to them.

- Pilot: both objectives are trained on the calibration data. The winner is the passing objective with the
  larger overall saving. **If neither passes, the work stops there and that is reported.**
- Main training: the winner, on calibration + new data.
- **The reference sets are opened only if the main adapter passes the same rule.** Otherwise they stay closed
  and the validation result is the result.
- An RL stage (reward per question, relative to the base model's length on the same question, only for right
  answers) is decided after the supervised result and will be registered here before it runs.

### Amendment 1 (2026-11-23, before any 27B training)

Made after the plumbing test on Qwen3-0.6B with stand-in data, which is not one of the evaluation sets.

The decision objective as first written (one KL mean over all non-stop positions) let the small model stop at
the first paragraph boundary: the "go on" signal at a boundary was one position among thousands, the stop term
a mean over a handful. It is now three terms, each a mean over its own positions: KL at ordinary positions
+ 1.0 × −log p(`</think>`) at stop positions + 4.0 × KL at *hold* positions (boundaries up to the last read-out
that was still wrong in a usable rollout, and every boundary of a rollout that is not usable). Three settings
of that weight were tried on the stand-in (none, 1, 4) and the last was kept; no further tuning.

On the stand-in only the trace objective passed the rule; the decision objective did not in any of the three
settings. The pilot on the 27B model decides between them by the rule above, as written.

### Note on timing (2026-11-25)

The section "What had been seen" says the instructions for the pilot had been prepared when this file was
committed. That was what I knew; it was not the whole picture. The pilot was started on the cluster on
2026-11-24 at 08:58 (cluster clock, CET) by the collaborator who runs the cards. This file was pushed the same
day at 16:16, while the second objective was still training. The comparison against the base model was
computed at 18:06, and I first looked at any output of the run on 2026-11-25.

So the rules above were written before any result existed and were not changed by the run, but **the public
timestamp of this file is about seven hours after the pilot started, not before it.** For the pilot the
pre-registration is therefore self-reported in the same sense as for 1.7B and 4B. Anything registered from
here on is committed before its run starts.

### Outcome of the pilot (2026-11-25)

Run of 2026-11-24, 08:58–18:06, two H200 cards, no error or restart. Every code reply was graded again on a
second machine with identical verdicts. 300 validation questions × 4 rollouts.

| | trace | decision |
| --- | ---: | ---: |
| Overall accuracy change | −4.08 pp [−6.83, −1.42] | −15.92 pp [−18.67, −12.83] |
| Overall saving of thinking tokens | 67.3% [62.9, 70.6] | 62.0% [54.8, 66.7] |
| Bucket < 2k (190 questions) | +3.42 pp | −5.79 pp |
| Bucket 2–8k (47) | −11.70 pp | −18.62 pp |
| Bucket 8–32k (35) | −21.43 pp | −42.86 pp |
| Bucket > 32k (28) | −20.54 pp | −46.43 pp |

**Neither objective passes the rule** (overall at least −1.0 pp and every bucket at least −3.0 pp). As
registered, the work stopped there: no further data were generated, no main training was run, and the reference
sets were not opened. Criteria A–F are not met.

What the adapters do, found after the fact and therefore not evidence for any rule:

- The trace objective learned a typical length per kind of task instead of when its answer is settled. On
  mathematics it thinks for a median of 216, 559 and 725 tokens on questions where the base model needs under
  2k, 2–8k and 8–32k. It saves 67% where the calibration put the attainable saving at 38%.
- Where reasoning is short it holds up: the < 2k bucket gains 3.4 pp at 65% saved.
- The decision objective closes its thinking early and continues reasoning inside the answer (code answers up
  to 23,000 tokens, 15% accuracy). After training it still gave about a 15% chance of stopping at each boundary
  where it should go on, which compounds over tens of boundaries.
- Suspected common cause, not tested: the settled point marks where the base model can still finish the work in
  its answer, which is earlier than where the reasoning is done.

Any continuation is a new plan and will be registered in this file before it runs.

---

## Replication of the pilot on Qwen3-4B

Registered 2026-11-25, before the run. The commit that adds this section is its timestamp; the run is started
only after that commit is public.

### Purpose

The 27B pilot failed in a specific way (an adapter that learns a typical length per kind of task). Searching
for a better method on the 27B model costs a day and many card-hours per attempt. This run asks one question:
**does the same pilot, unchanged, fail the same way on Qwen3-4B?** If it does, 4B becomes the test bed for the
next method and the 27B model is used again only for a method that has passed there. No new method is tried in
this run.

### What had been seen

- Everything listed for version 2 above, and the outcome of the 27B pilot.
- One analysis of the 27B calibration data made after that outcome: in 49% of the mathematics rollouts that
  settle early (code 21%, instructions 45%, multiple choice 85%) the answer is already right twice at the first
  read-out boundary, about the second paragraph. The earlier suspicion, that the base model finishes the work
  in a longer answer, was not supported (forced answers at the settled point are as long as the model's own).
- Qwen3-4B on the short benchmarks and with the short-task recipe (the README's 4B result).

Not seen: Qwen3-4B on any of the long-reasoning pools, with or without an adapter.

### Protocol

The 27B pilot, with the model replaced: the same 962 calibration questions (2 rollouts, forced answers at up to
12 boundaries), the same 300 validation questions (4 rollouts), the same two objectives with the same
hyper-parameters, the same validation measure and rule. Differences, all forced by the model or the machine:

- generation is capped at 32,768 tokens instead of 65,536 (the model's context is 40,960 tokens), so a reply
  that would have been longer counts as capped, i.e. wrong, for the base model and the adapters alike;
- one card (RTX PRO 6000, 96 GB) instead of two;
- Qwen3-4B belongs to the Qwen3 family, the 27B model to a later one; prompt template and tokenizer differ.

### Reading of the result (fixed now)

- **Enough long reasoning to judge:** at least 20 of the 300 validation questions must have a base mean
  thinking of 8,000 tokens or more. If fewer do, these pools do not test long reasoning on this model; that is
  reported and 4B is not used as the test bed without a new, registered set of questions.
- **Failure reproduced:** the trace objective does not pass the rule, and at least one length bucket from
  2,000 tokens up is below −3.0 pp. Then 4B with these pools is the test bed.
- **Failure not reproduced:** the trace objective passes the rule. That is reported as a difference between
  the two models; 4B is then not a test bed for this failure, and no method search is run on it.
- Anything in between (fails only overall, or only in the < 2k bucket) is reported as "fails differently" and
  treated as not reproduced.
- The decision objective is run and reported; it has no part in the verdict.
- My guess before the run: enough long reasoning 80%; failure reproduced, given that, 75%.

### Condition for going back to the 27B model

A new method goes to the 27B model only after, on the 4B test bed, it passes the same rule in every length
bucket and passes a test on reasoning lengths it was not trained on. That test and the method are registered
here before they are run.

---

## Replication of the pilot on Qwen3-0.6B

Registered 2026-11-25, before the run. The commit that adds this section is its timestamp; the run is started
only after that commit is public. It runs at the same time as the 4B replication, and neither result had been
seen when this was written.

### Purpose

The same question as for 4B, asked of the smallest model: **does the unchanged pilot fail the same way on
Qwen3-0.6B?** A 0.6B run is cheap enough to be repeated many times, so if it reproduces the failure it is the
fastest place to try a new method. It is not a substitute for 4B (see the last part of this section).

### What had been seen

Everything listed for the 4B replication above. Qwen3-0.6B on the short benchmarks and with the short-task
recipe (the first study). Not seen: Qwen3-0.6B on any of the long-reasoning pools, with or without an adapter.

### Protocol

Identical to the 4B replication (`chains/long_4b.sh` with `BASE_MODEL=Qwen/Qwen3-0.6B`, `TAG=06b`; the same
questions, cap of 32,768 tokens, objectives, hyper-parameters, measure and rule), run from
`notebooks/long_06b_colab.ipynb` on one GPU with bf16. Two differences, neither meant to change any number:

- program judging may use the fallback sandbox (`SANDBOX=user` in `stopping/code_tasks.py`) if bubblewrap cannot
  run on that machine; it must pass the same self-test and the chain's 24/24 judge check first;
- library versions are those of the 27B environment (vLLM 0.34, Transformers 5.22, PEFT 0.21).

### Reading of the result (fixed now)

- **Enough long reasoning to judge:** at least 20 of the 300 validation questions with a base mean thinking of
  8,000 tokens or more, **and at least 10 of those answered right in at least one of the 4 base rollouts**.
  The second condition is new for 0.6B: a small model may think long on questions it never gets right, and a
  bucket with nothing right cannot lose accuracy. If either fails, these pools do not test long reasoning on
  this model; that is reported and 0.6B is not used as a test bed for this failure.
- **Failure reproduced / not reproduced / fails differently:** exactly as for 4B (trace objective fails the
  rule with a bucket from 2,000 tokens up below −3.0 pp / trace passes / anything in between, treated as not
  reproduced). The decision objective is reported and has no part in the verdict.
- My guess before the run: enough long reasoning 55%; failure reproduced, given that, 70%.

### How 0.6B and 4B are used together

- A method tried on 0.6B goes to 4B before anything else. The condition for the 27B model registered above
  (passes the rule in every bucket and a held-out-length test on 4B) is unchanged; a pass on 0.6B alone never
  counts toward it.
- If 0.6B reproduces the failure and 4B does not, the 0.6B result is reported as a property of the small model,
  and 0.6B is not used to search for a method meant for 27B.

---

## Next method on the test bed: two candidates

Registered 2026-11-26, before any code for either candidate has been run on a model. The commit that adds this
section is its timestamp. When this was written I had not seen a result from either the 4B or the 0.6B
replication.

### When this is run

This runs only on a model where the replication above reads **failure reproduced** (4B or 0.6B). If neither
model reproduces the failure, this section is not run, and that is reported. A model that does reproduce it is
called the *test bed* below. If both do, 0.6B may be used first, to find faults in the code; only the 4B result
counts (see "How 0.6B and 4B are used together").

### The idea both candidates share

The 27B pilot learned a typical length for each kind of task. Both candidates replace the oracle cut with a
training signal that depends on the question itself and is measured on the model's own answers. A question the
model needs long reasoning for should therefore keep it.

- **A. Length reward by difficulty** (after LASER-D, arXiv 2505.15612, Liu et al. 2025). RL in which a short
  right answer earns a bonus only up to a target length. The target is set separately for easy, medium and hard
  questions, from the model's own success rate.
- **B. Online preference pairs per question** (after the online DPO of A. le Moigne, "ThinkingCap", 2026, and
  his public code). The model answers new questions in rounds, and pairs are built only between answers to the
  same question: shorter right over longer right, and right over a shorter wrong one. A brake stops shortening
  for a kind of task that loses accuracy or saves more than its calibrated limit.

Both are my adaptations to long reasoning and to this repository; the differences from the originals are listed
under each candidate. Neither original was tested on reasoning of this length.

### Data (both candidates)

- **Training questions:** the 962 calibration questions (mathematics 250, multiple choice 250, code 233,
  instructions 129, function calling 100). For these, the test bed's base model already has two rollouts and the
  forced read-outs. Before training, one sixth of each domain (42, 42, 39, 21, 16; 160 in all; drawn with seed 0)
  is set apart as the **selection set**. The remaining 802 are the training questions. No other questions are
  added.
- **Validation:** the same 300 questions as the pilot, with 4 rollouts, the same measure and the same rule
  (overall change at least −1.0 pp, every length bucket at least −3.0 pp). They are opened **once per
  candidate**, for the one checkpoint the selection set picked.
- **Token cap:** 32,768 generated tokens for training and validation alike, as in the replication. A capped reply
  counts as wrong.
- **Adapter:** LoRA r 16, α 32, dropout 0.05 on the attention and MLP projections, trained from the base model
  with no supervised stage first.
- **Sampling for training answers:** the validation sampling (temperature 1.0, top-p 0.95, top-k 20).

### Candidate A: length reward by difficulty

- **Training:** GRPO as in `stopping/train_grpo.py`.
  - 16 questions per step, 8 answers each, one on-policy update per batch.
  - Advantage = reward − group mean.
  - Learning rate 1e-5, no KL term.
  - 50 steps; the questions of a step are drawn without replacement from all training questions (seed 0), so
    each is used at most once. Checkpoints every 10 steps.
- **Reward:** +1 right, −0.5 wrong or capped, −1 when no answer can be read. A right answer whose length
  (thinking + answer tokens) is at most the target L_d of its question's difficulty bucket gets +0.5 more. Long
  right answers are not penalised; they only miss the bonus.
- **Difficulty bucket** of a question at a step, from the share of its 8 answers that are right: easy above 2/3,
  medium 1/3 to 2/3, hard below 1/3.
- **Target L_d:** the smallest l in 2,048, 3,072, …, 32,768 such that (share of the bucket's answers not longer
  than l) × c_d ≥ 1, with c_d = 6, 3, 1 for easy, medium, hard. This is the paper's expected-correct-responses
  rule for 8 answers. Before the first update every L_d is 32,768, so no bonus can push length down yet. After
  that, L_d is recomputed every 10 steps.
- **Differences from the paper:**
  - L_d is recomputed from the answers of the last 10 training steps instead of from a separate monitoring set
    (no extra generation).
  - The lower end of the search is 2,048 instead of 4,096 or 8,192.
  - The base model is Qwen3, not an R1 distillation.
  - The tasks cover five domains, not mathematics only.
  - LoRA, not full weights.

### Candidate B: online preference pairs per question

- **Rounds:** 8. The training questions are split (seed 0) into 8 parts of about 100, each with the same domain
  proportions. Each round takes the next part and 8 answers to each question from the current adapter. The
  first round answers with the base model. With about 10 function-calling and 13 instruction questions per round,
  the brake for those two domains is noisy; that is accepted, not tuned.
- **Base statistics per question,** from the two base rollouts of the calibration:
  - p0 = share right;
  - b = mean thinking of the right rollouts (if neither was right, the mean thinking of both).
- **Pairs per question** are built from the current answers only, at most 2 per question:
  - **shorten:** only if at least 2 answers are right and clean (finished, readable) and the share right is at
    least min(p0, 0.75).
    - Chosen = the right answer with the shortest thinking; rejected = the right answer with the longest.
    - The pair is kept only if the rejected answer thinks at least 1.3 times as long.
    - The chosen answer must think at least 0.4 × b.
  - **right over shorter wrong:** only if some answers are right and some wrong.
    - Rejected = the wrong answer with the shortest thinking.
    - Chosen = the shortest right answer that thinks at least as long as the rejected one. If no right answer
      thinks that long, chosen = the longest right answer.
- **Brake per domain,** measured on the round's own questions against their base statistics:
  - A domain is held for the next round when its saving of thinking tokens reaches its cap, or when its accuracy
    change is −3.0 pp or worse.
  - A held domain gets no shorten pairs; right-over-wrong pairs are still built.
  - The cap per domain is the share of thinking after the settled point in the test bed's own calibration
    (rollouts that end right). It is computed by `stopping/readouts.py --report` before training.
- **Long brake:** if the round's questions with base thinking of 8,000 tokens or more lose 3.0 pp or more, no
  shorten pairs are built in the next round for questions with base thinking of 2,000 tokens or more.
- **DPO settings:** β 0.1; reference = the base model (adapter off); learning rate 1.4e-5 with 2 warm-up steps;
  16 pairs per step; each pair seen once; gradient clip 1.0; the same adapter continued across rounds, with no
  merging.
- **Differences from the original:**
  - Fewer, longer rounds.
  - Base statistics come from 2 rollouts per question, not 8.
  - No loop, format or counting pairs.
  - The right-over-wrong pair always rejects the shortest wrong answer.
  - The long brake is new.
  - The caps come from the calibration, not from earlier test results.
  - The kinds of task are the five domains here.

### Choosing a checkpoint and a winner

- **Checkpoints judged on the selection set:** 4 rollouts per question, the same measure. Length buckets use the
  base's mean thinking from its two calibration rollouts. Candidate A is judged after steps 20, 30, 40 and 50;
  candidate B after rounds 2, 4, 6 and 8.
- **Each candidate's checkpoint** is the one that passes the rule on the selection set with the largest overall
  saving. **If none passes, that candidate fails, and its validation is not run.**
- **Validation:** the chosen checkpoint is run once on the validation questions. A candidate **passes** if it
  passes the rule there. If both pass, the winner is the one with the larger overall saving.
- **No tuning after this commit:** nothing in either candidate is tuned after it. A fault found in the code (a
  crash, a wrong grade) is fixed and noted. Any change of a number or rule above is added here as a dated
  amendment before the run it affects.

### Held-out lengths (the condition for 27B registered above)

This applies only to a candidate that passes. It is trained again, unchanged, on the training questions whose
base mean thinking is below 8,000 tokens, and its checkpoint is chosen by the selection set in the same way.

It passes the held-out test if, on the validation questions, the rule holds and both buckets from 8,000 tokens up
(or the one bucket they merge into) are at least −3.0 pp. Only a candidate that passes this goes to the 27B
model, under a new registration.

### My guess before any run

Given a test bed:
- A passes 30%;
- B passes 35%;
- at least one passes 50%;
- a candidate that passes also passes the held-out lengths 50%.
