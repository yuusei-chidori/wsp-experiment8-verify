#!/bin/zsh
cd "${0:A:h}/.."
for set in gsm8k:data/subset100.jsonl math:data/math100.jsonl; do
  name=${set%%:*}; probs=${set#*:}
  for i in {1..8}; do
    [ "$(wc -l < runs10/r1d15_$name.jsonl 2>/dev/null || echo 0)" -ge 100 ] && break
    ./venv/bin/python scripts/run_gen.py --model dl/r1d15 --problems $probs --cap 32768 --batch 24 --sampler vllm --stop-from-tokenizer --out runs10/r1d15_$name.jsonl > runs10/r1d15_$name.run$i.log 2>&1
    echo "$name run $i exit $? -> $(wc -l < runs10/r1d15_$name.jsonl 2>/dev/null) done $(date +%H:%M)"
  done
done
echo ALL_DONE
