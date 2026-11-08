# Step 11 protocol — the same experiment with fursund/weight-space-harness (wsh 0.1.0, commit 45e3efc)

Written 2026-11-07 ~21:05, after the harness's inspect and probe steps and before any edited model generated text.

Model, data, sampling, grading: as Step 10 (DeepSeek-R1-Distill-Qwen-1.5B; search on TRAIN only; vLLM order,
T 0.6, top_p 0.95, top_k 20, presence 1.5, seed 0; math-verify). Test sets unopened until a candidate is frozen.

Harness steps used: inspect (summarize / svd_report / row_report), probe (collect / divergence / direction),
edit (ProjectOut in an EditLog, side "out"), verify (sweep over strengths on a failing set + control set, plateau).

Candidates (5, all wsh.ProjectOut, rank 1, one matrix):
- H1 = harness top site: model.layers.26.self_attn.o_proj, direction = wsh.direction(wrong, right) on the probe set;
  strengths 0.5, 1, 2.
- H2 = site that survived the length check best: model.layers.18.mlp.down_proj, direction from wrong vs
  length/set-matched right answers; strengths 1, 2.

Verify (harness sweep): screen64 (38 probe-wrong = failing set, 26 probe-right = control), cap 16,384.
"Before" = the unedited model's rerun on the same 64 (Step 10 base0: 16/38 and 23/26).

Selection (fixed now):
1. Per family, the strength with the highest screen64 total goes to dev (tie -> smaller strength).
2. Of the two, the higher dev score (200 problems, cap 32,768) is frozen (tie -> smaller weight change); SHA-256 recorded.
3. Frozen edit and a random direction at the same site with the same ||dW||_F (seed 0) run on the GSM8K 100 and
   MATH-500 100 test sets; paired McNemar vs the Step 10 original runs.
Success = p < 0.05 on a test set and the random control not showing the same gain.
