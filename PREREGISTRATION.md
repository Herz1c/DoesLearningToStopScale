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
