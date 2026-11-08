#!/bin/zsh
# usage: run_test.sh <frozen candidate name>  — freeze, build random control, run both on the TEST sets, grade
cd "${0:A:h}/.."
c=$1
shasum -a 256 dl/cand_$c/model.safetensors > runs10/FROZEN_$c.sha256; date >> runs10/FROZEN_$c.sha256
./venv/bin/python -I scripts/make_random10.py edits10/$c.json dl/r1d15 edits10/random_$c.json --seed 0
[ -f dl/rand_$c/edit_report.json ] || ./venv/bin/python -I scripts/make_edit.py dl/r1d15 dl/rand_$c edits10/random_$c.json > /dev/null
for m in cand_$c rand_$c; do
  for set in gsm8k:data/subset100.jsonl math:data/math100.jsonl; do
    name=${set%%:*}; probs=${set#*:}
    for i in {1..8}; do
      [ "$(wc -l < runs10/test_${m}_$name.jsonl 2>/dev/null || echo 0)" -ge 100 ] && break
      ./venv/bin/python scripts/run_gen.py --model dl/$m --problems $probs --cap 32768 --batch 24 --sampler vllm --stop-from-tokenizer --out runs10/test_${m}_$name.jsonl > runs10/test_${m}_$name.run$i.log 2>&1
    done
    ./venv/bin/python scripts/grade.py runs10/r1d15_$name.jsonl runs10/test_${m}_$name.jsonl logs/none out/step10_test_${m}_$name.json > /dev/null 2>&1
    python3 -c "import json;d=json.load(open('out/step10_test_${m}_$name.json'));print('TEST $m $name', d['base']['correct'],'->',d['edited']['correct'],'better',d['paired']['rescues'],'worse',d['paired']['losses'],'p',round(d['paired']['mcnemar_p'],3))"
  done
done
echo TEST_DONE
