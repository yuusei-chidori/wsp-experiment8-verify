#!/bin/zsh
# usage: run_dev.sh <model_dir> <name>   — scores a model on the 200 dev (train) problems
cd "${0:A:h}/.."
for i in {1..8}; do
  [ "$(wc -l < runs10/dev_$2.jsonl 2>/dev/null || echo 0)" -ge 200 ] && break
  ./venv/bin/python scripts/run_gen.py --model $1 --problems data/train_dev.jsonl --cap 32768 --batch 24 --sampler vllm --stop-from-tokenizer --out runs10/dev_$2.jsonl > runs10/dev_$2.run$i.log 2>&1
  echo "dev $2 run $i exit $? -> $(wc -l < runs10/dev_$2.jsonl) $(date +%H:%M)"
done
