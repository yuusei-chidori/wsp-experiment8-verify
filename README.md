# Verifying Experiment 8 of "Sparks of Weight-Space Programming"

A local check (Mac mini M4 Pro, mlx-lm) of Experiment 8 in
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/):
an AI made a rank-1 edit to two `down_proj` matrices of Qwen3.5-0.8B to stop reasoning loops.

## Final reports (slide decks, open in a browser)
| | Japanese | English |
|---|---|---|
| Executives (15 slides) | [reports/final-executive.html](reports/final-executive.html) | [reports/final-executive-en.html](reports/final-executive-en.html) |
| Engineers (44 slides) | [reports/final-engineer.html](reports/final-engineer.html) | [reports/final-engineer-en.html](reports/final-engineer-en.html) |

Per-step reports (Japanese) are in `reports/step*.html`.

## Findings in brief
- The published edit is exactly what the blog describes (2 tensors, 0.015% of weights).
- GSM8K 31% → 70%, MATH-500 16% → 53% on 100 problems each; with the authors' sampling order 18% → 70% and 5% → 57%.
- Three random-direction controls on the same columns make the model worse (15–20%), so the direction matters.
- The gain comes from breaking reasoning loops, not from added math skill.
- Steps 10–12: on DeepSeek-R1-Distill-Qwen-1.5B (no loops), edits found by Claude, including with
  [weight-space-harness](https://github.com/fursund/weight-space-harness), had no effect.
  Full MATH-500: base 431/500, edit 426, random edit 428. The 90% target was not reached.

## Layout
- `scripts/`, `runs/`: generation, grading and control scripts (Steps 1–9)
- `results/`: JSON summaries, generated answers, protocols, and per-step folders `step10/`–`step12/` (with their own scripts)
- `data/`: problem subsets used

Model weights are not included; they can be downloaded from Hugging Face.
