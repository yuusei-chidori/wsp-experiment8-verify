#!/bin/zsh
# Step 12 after probe12.py: build 4 edits -> screen on check problems (2 seeds) -> dev MATH 100 -> freeze -> test 500
cd "${0:A:h}/.."
PY=./venv_wsh/bin/python
B() { [ -f dl/h_$1/edit_report.json ] || $PY scripts/build12.py "$@" | tail -1; }
gen() { # model problems cap out batch seed
  for i in {1..8}; do
    [ "$(wc -l < $4 2>/dev/null || echo 0)" -ge $(wc -l < $2) ] && break
    ./venv/bin/python scripts/run_gen.py --model $1 --problems $2 --cap $3 --batch $5 --seed $6 --sampler vllm --stop-from-tokenizer --out $4 > $4.run$i.log 2>&1
  done; }
B A0s1 0 1; B A0s2 0 2; B A1s1 1 1; B A1s2 1 2
for c in A0s1 A0s2 A1s1 A1s2; do
  for s in 0 1; do gen dl/h_$c data/check12.jsonl 16384 runs12/scr_${c}_s$s.jsonl 32 $s; done
  ./venv/bin/python scripts/score12.py SCREEN_$c runs12/hard_s0.jsonl,runs12/hard_s1.jsonl runs12/scr_${c}_s0.jsonl,runs12/scr_${c}_s1.jsonl out12/screen_$c.json 2>/dev/null
  echo "  $(date +%H:%M)"
done
DEV=($(python3 -c "
import json
r={c:json.load(open(f'out12/screen_{c}.json'))['edited'] for c in ['A0s1','A0s2','A1s1','A1s2']}
print(*[('A%ds1'%i if r['A%ds1'%i]>=r['A%ds2'%i] else 'A%ds2'%i) for i in (0,1)])"))
echo "TO_DEV $DEV $(date +%H:%M)"
for c in $DEV; do
  gen dl/h_$c data/dev_math100.jsonl 32768 runs12/dev_$c.jsonl 24 0
  ./venv/bin/python scripts/score12.py DEV_$c runs10/dev_base.jsonl runs12/dev_$c.jsonl out12/dev_$c.json 2>/dev/null
done
F=$(python3 -c "
import json
c='$DEV'.split(); d={x:json.load(open(f'out12/dev_{x}.json'))['edited'] for x in c}; w={x:json.load(open(f'dl/h_{x}/edit_report.json'))['change_fro'] for x in c}
print(sorted(c,key=lambda x:(-d[x],w[x]))[0])")
echo "FROZEN $F $(date +%H:%M)"
shasum -a 256 dl/h_$F/model.safetensors > runs12/FROZEN_$F.sha256; date >> runs12/FROZEN_$F.sha256
B R_$F random:0:$F 0
cat runs10/r1d15_math.jsonl runs12/base_math400.jsonl > runs12/base_math500.jsonl
for m in $F R_$F; do
  gen dl/h_$m data/math500.jsonl 32768 runs12/test_$m.jsonl 24 0
  ./venv/bin/python scripts/score12.py TEST_$m runs12/base_math500.jsonl runs12/test_$m.jsonl out12/test_$m.json 2>/dev/null
done
echo STEP12_DONE $(date +%H:%M)
