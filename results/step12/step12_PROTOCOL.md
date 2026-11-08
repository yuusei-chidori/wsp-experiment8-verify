# Step 12 protocol — MATH-500 target 90%: same-problem right-vs-wrong directions (Track A)

Written 2026-11-08 ~08:55, while only the ORIGINAL model was generating (hard200 samples, MATH-500 rest),
before any direction was computed and before any edited model generated text.

Model DeepSeek-R1-Distill-Qwen-1.5B (dl/r1d15). Sampling as Steps 10-11 (mlx-lm, vLLM order, T 0.6, top_p 0.95,
top_k 20, presence 1.5, stop from tokenizer). Grading grade.py (math-verify). Editing wsh 0.1.0 ProjectOut, side "out".

## 1. Yardstick (fixed now)
- Final test = ALL 500 MATH-500 problems, one sample each, seed 0, cap 32,768.
- Original model's 500 = Step 10's 100 (runs10/r1d15_math.jsonl, same settings) + the other 400 (runs12/base_math400.jsonl).
- Target asked by the user: >= 90% (450/500); also reported on the usual 100.
- Success = paired McNemar p < 0.05 vs original on the 500 AND the same-size random edit does not show the same gain.
- Test problems are not used to find, choose or tune anything.

## 2. Finding the direction (Track A)
- hard200 = 200 Level-5 MATH TRAIN problems not used before (data/hard200.jsonl); 4 samples each (seeds 0-3, cap 16,384).
- Mixed problems = 1 to 3 of 4 samples right. Split by problem (random, seed 0): 60% "find", 40% "check".
- Pairs: inside one problem, each wrong, not-cut sample with the right sample closest in length (reuse allowed).
  Same problem => difficulty and topic are equal in each pair; length is matched as closely as possible.
- Sites: 28 layer outputs, 28 mlp.down_proj, 28 self_attn.o_proj (as Step 11). Two poolings over generated tokens
  only: "early" = first 1,024 tokens, "whole" = first 4,096 tokens.
- Direction = unit mean of paired differences (wrong - right) on "find" pairs.
- Score on "check" pairs: paired AUC (share of pairs where wrong projects higher), shuffled-pair baseline (sign flips,
  200 times) and correlation of the pair's projection gap with its log-length gap.
- Candidates: the 2 editable sites (down_proj or o_proj, any pooling) with the highest check-pair AUC. They go ahead
  even if they do not beat the shuffled baseline; that is reported.

## 3. Edit, screen, dev, freeze
- Edit = wsh.ProjectOut(site, d, side="out", strength s), s in {1, 2}  -> 4 candidates.
- Screen: "check" problems, 2 samples each (seeds 0, 1, cap 16,384), against the original's seeds 0, 1 on the same
  problems. Per site, the strength with more right answers goes on (tie -> smaller s).
- Dev: the 100 MATH problems of data/train_dev.jsonl (cap 32,768), against runs10/dev_base.jsonl.
  The higher dev score is frozen (tie -> smaller ||dW||); SHA-256 recorded.
- Test: frozen edit and a random direction at the same site with the same ||dW||_F (orthogonal, seed 0) on MATH-500 500.
