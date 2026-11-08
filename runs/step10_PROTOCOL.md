# Step 10 protocol — weight editing on a new model (DeepSeek-R1-Distill-Qwen-1.5B)

Written 2026-11-07, before any edit was made and before any edited model saw a test problem.

## Roles
- Editor: Claude (Opus 5.5) in place of GPT-6 Astra. Astra is not available and is not used.
- The editor reads weights and activations with Python on the Mac, proposes edits, and scores them on practice problems.

## Data
- Search uses TRAIN splits only: GSM8K train (7,473), MATH train (7,498 with a boxed answer; none equal to a test problem).
  - `data/train_dev.jsonl`: 100 GSM8K + 100 MATH (random.Random(1)) — scores candidate edits.
  - `data/train_probe.jsonl`: 300 + 300 other problems — for looking inside the model.
- Test: the same 100 GSM8K test and 100 MATH-500 problems as Steps 2–9. Not opened during the search.

## Edit form
- Same family as the paper: low-rank (rank 1–2) changes to MLP down_proj in at most 2 layers, written in bf16.
  No training, no gradient descent on the test or dev answers.

## Budget and selection
- At most 8 candidate edits scored on the dev set (same settings as the test: vLLM order, temperature 0.6,
  top_p 0.95, top_k 20, presence penalty 1.5, cap 32,768, seed 0).
- The candidate with the highest dev accuracy is frozen (SHA-256 recorded) before any test run.
- The frozen edit is tested even if it does not beat the original on dev (no silent dropping).
- Control: a random direction of the same size in the same places (as Steps 3 and 5).

## Test and judgement
- Original (Step 10 check runs), frozen edit, and random control on GSM8K test 100 and MATH-500 100.
- Paired McNemar test against the original. A "real" improvement needs p < 0.05 on at least one test set
  and the random control not to show the same gain.
