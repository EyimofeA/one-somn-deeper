# GPT-5.6 Pro research review prompt

You are acting as a principal AI researcher reviewing an ongoing research
project. Your role is adversarial: do not accept our narrative or conclusions.
Treat all conclusions as hypotheses, distinguish evidence from interpretation,
and do not write code.

Your objective is to identify the highest-information, highest-expected-value
next research move.

Repository: <https://github.com/EyimofeA/one-somn-deeper>

I have attached the review packet directory containing source, reports, charts,
and the research log. Read these first:

1. `README.md`
2. `RESEARCH_LOG.md`
3. `STATUS.md`
4. `predictions.md`
5. `FAILED_EXPERIMENTS.md`
6. `vdf_final_label_submission.py`
7. `vdf_t_curriculum_submission.py`
8. Everything under `reports/`
9. `train_vdf_trace_ablation.py`
10. `train_vdf_depth_curriculum.py`

## Research question

Can a neural network learn a generalizable sequential VDF computation?

Input: `(N, x, T)`

Output:

`x^(2^T) mod N`

The intended learned computation is:

`s0 = x`

`s(t+1) = F_theta(s(t), N)`

`output = s(T)`

T should control how many times the **same tied transition** executes. The
model should not learn or predict T as an arithmetic quantity.

## Current working hypothesis — challenge it

We do **not** think recurrence itself is refuted. We suspect the failure may
be treating prompt/answer token positions as an algorithmic workspace instead
of maintaining a genuine evolving latent state `h_t`.

The competition model currently resembles:

```text
prompt tokens -> answer-aligned token register -> output tokens
```

Its proposed alternative would be:

```text
(N, x) -> encoder -> h0
                     |
                     v
              tied F_theta(h_t, N), exactly T times
                     |
                     v
                 decoder -> output tokens
```

However, do not assume this diagnosis is correct. It may be a state-interface
issue, a transition-cell issue, an objective issue, a data issue, insufficient
compute, or a fundamentally unsuitable approach.

## Evaluate these hypotheses

- **H1 Architecture failure:** the current model cannot represent the needed
  computation.
- **H2 State representation failure:** its token register does not expose a
  useful evolving computation.
- **H3 Training-objective failure:** final-label supervision does not identify
  a reusable local transition.
- **H4 Data-distribution failure:** current training data does not induce the
  algorithmic behavior we need.
- **H5 Scaling failure:** the mechanism is correct but undertrained or too
  small.

For each: evidence for, evidence against, confidence, and the cheapest
discriminating experiment.

## Facts that must not be lost

- Structured serial diagnostics with learned LSD-first comparator/subtractor
  modules and real recurrent GRU state succeeded strongly in controlled
  reduction tasks.
- The final-label prompt-tail-register VDF model did not generalize.
- Legal final-label curriculum produced a small T=1 bump but no T>=2 result.
- A research-only genuine T=1 -> T<=2 -> T<=4 final-label curriculum reached
  near-perfect phase training fit yet 0% held-out-x exact at T=1/2/4.
- Diagnostic intermediate trace supervision improved in-batch fitting but had
  0% held-out final exact for the same prompt-tail-register VDF architecture
  in the short e1 condition.
- A direct Transformer control did not establish a depth mechanism either.

## Specific questions

1. Did we actually diagnose the failure correctly? Is the conclusion that the
   current VDF architecture cannot learn a reusable transition justified, or
   are we missing an optimization/training explanation?
2. Is a true latent recurrent workspace the right next move, or an
   architectural rabbit hole? Compare recurrent Transformers, Universal
   Transformers, latent workspace models, RNN/state-space models, and
   recurrent memory tokens.
3. With one GPU hour, design the smallest decisive proof-of-life experiment
   for latent recurrent state. Specify dataset, architecture, loss, expected
   result, and kill condition.
4. Did we misuse teacher forcing? Rank teacher forcing, scheduled sampling,
   professor forcing, consistency losses, latent-state matching, and
   self-supervised rollout losses for this setting.
5. Should T only control recurrence count, or also be represented in learned
   state? Which choice is most likely to extrapolate to unseen T?
6. Is `Square + Reduce` a productive decomposition, or should the modular VDF
   transition be learned end-to-end?
7. Attack the strongest claimed positive result: is the learned serial
   comparator/reducer actually algorithmic, or explainable by distribution
   coverage or another confound?
8. Which literature on neural algorithmic reasoning, Universal Transformers,
   scratchpads, recurrent memory, learned arithmetic circuits, or latent
   computation genuinely transfers here?
9. Given one week and limited GPUs, what three experiments would a serious
   research group run, and what would it stop doing?
10. Give probabilities, expected time, and research value for: keep tuning the
    current architecture; replace the state representation; use explicit
    algorithmic modules; submit a simpler leaderboard model; or abandon this
    line.

## Highest-value question

> You have seen our diagnostics. The reducer learned a reusable transition
> with explicit state supervision, but the competition model failed. Design
> the minimum experiment that proves whether the missing ingredient is latent
> state representation.

## Required output

1. Independent reconstruction of the problem.
2. Evidence table: observation; what it proves; what it does **not** prove.
3. Diagnosis and confidence (low/medium/high, with why).
4. What we got right; what we got wrong; surviving hypotheses.
5. Five experiments ranked by information gain. For each: hypothesis,
   predicted result if true/false, implementation difficulty, cost, and kill
   condition.
6. Then act as three researchers:
   - A believes latent recurrent computation is viable;
   - B believes the architecture is fundamentally wrong;
   - C believes scale/optimization is the main issue.

   Have them debate, then name the strongest argument, weakest assumption, and
   single experiment that resolves the disagreement.

Be willing to conclude this research direction is dead.
