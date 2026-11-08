#!/bin/zsh
# Step 12 base-model generations only (no edited model): 4 samples of hard200 (seeds 0-3, cap 16384), then the
# 400 MATH-500 problems not yet run (seed 0, cap 32768) so the original model has a full 500-problem score.
cd "${0:A:h}/.."
gen() { # model problems cap out batch seed
  for i in {1..8}; do
    [ "$(wc -l < $4 2>/dev/null || echo 0)" -ge $(wc -l < $2) ] && break
    ./venv/bin/python scripts/run_gen.py --model $1 --problems $2 --cap $3 --batch $5 --seed $6 --sampler vllm --stop-from-tokenizer --out $4 > $4.run$i.log 2>&1
  done; echo "DONE $4 $(wc -l < $4) $(date +%H:%M)"; }
for s in 0 1 2 3; do gen dl/r1d15 data/hard200.jsonl 16384 runs12/hard_s$s.jsonl 32 $s; done
gen dl/r1d15 data/math400_rest.jsonl 32768 runs12/base_math400.jsonl 24 0
echo BASE12_DONE
