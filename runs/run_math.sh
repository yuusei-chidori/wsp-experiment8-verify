#!/bin/zsh
cd "${0:A:h}/.."
for m in edited base; do
  for i in {1..8}; do
    [ "$(wc -l < runs/math_$m.jsonl 2>/dev/null || echo 0)" -ge 100 ] && break
    ./venv/bin/python scripts/run_gen.py --model dl/$m --problems data/math100.jsonl --cap 32768 --batch 24 --out runs/math_$m.jsonl > runs/math_$m.run$i.log 2>&1
    echo "$m run $i exit $? -> $(wc -l < runs/math_$m.jsonl 2>/dev/null) done $(date +%H:%M)"
  done
done
echo ALL_DONE
