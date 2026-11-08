#!/bin/zsh
# Step 11: build 5 harness edits -> screen sweep -> dev for 2 -> freeze -> test with random control
cd "${0:A:h}/.."
PY=./venv_wsh/bin/python
B() { [ -f dl/h_$1/edit_report.json ] || $PY scripts/build11.py "$@" | tail -1; }
B H1s05 model.layers.26.self_attn.o_proj plain 0.5
B H1s1  model.layers.26.self_attn.o_proj plain 1
B H1s2  model.layers.26.self_attn.o_proj plain 2
B H2s1  model.layers.18.mlp.down_proj matched 1
B H2s2  model.layers.18.mlp.down_proj matched 2
gen() { # model name problems cap out
  for i in {1..8}; do
    [ "$(wc -l < $5 2>/dev/null || echo 0)" -ge $(wc -l < $3) ] && break
    ./venv/bin/python scripts/run_gen.py --model $1 --problems $3 --cap $4 --batch ${6:-24} --sampler vllm --stop-from-tokenizer --out $5 > $5.run$i.log 2>&1
  done; }
for c in H1s05 H1s1 H1s2 H2s1 H2s2; do
  gen dl/h_$c $c data/screen64.jsonl 16384 runs11/screen_$c.jsonl 32
  ./venv/bin/python scripts/screen11_score.py $c; echo "  $(date +%H:%M)"
done
DEV=($(./venv/bin/python scripts/pick11.py screen)); echo "TO_DEV $DEV $(date +%H:%M)"
for c in $DEV; do
  gen dl/h_$c $c data/train_dev.jsonl 32768 runs11/dev_$c.jsonl
  ./venv/bin/python scripts/grade.py runs10/dev_base.jsonl runs11/dev_$c.jsonl logs/none out11/dev_$c.json > /dev/null 2>&1
  python3 -c "import json;d=json.load(open('out11/dev_$c.json'));print('DEV $c', d['edited']['correct'],'/200 vs base',d['base']['correct'],'better',d['paired']['rescues'],'worse',d['paired']['losses'],'p',round(d['paired']['mcnemar_p'],3))"
done
F=$(./venv/bin/python scripts/pick11.py dev $DEV); echo "FROZEN $F $(date +%H:%M)"
shasum -a 256 dl/h_$F/model.safetensors > runs11/FROZEN_$F.sha256; date >> runs11/FROZEN_$F.sha256
B R_$F x random:0:$F 0
for m in h_$F h_R_$F; do
  for set in gsm8k:data/subset100.jsonl math:data/math100.jsonl; do
    n=${set%%:*}; p=${set#*:}
    gen dl/$m $m $p 32768 runs11/test_${m}_$n.jsonl
    ./venv/bin/python scripts/grade.py runs10/r1d15_$n.jsonl runs11/test_${m}_$n.jsonl logs/none out11/test_${m}_$n.json > /dev/null 2>&1
    python3 -c "import json;d=json.load(open('out11/test_${m}_$n.json'));print('TEST $m $n', d['base']['correct'],'->',d['edited']['correct'],'better',d['paired']['rescues'],'worse',d['paired']['losses'],'p',round(d['paired']['mcnemar_p'],3))"
  done
done
echo STEP11_DONE $(date +%H:%M)
