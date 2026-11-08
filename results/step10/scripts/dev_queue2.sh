#!/bin/zsh
cd "${0:A:h}/.."
./venv/bin/python scripts/build_cand.py wrongL15a05 '[{"kind":"wrong","layer_dir":15,"layer_edit":15,"alpha":0.5}]'
./venv/bin/python scripts/build_cand.py wrongL15a2 '[{"kind":"wrong","layer_dir":15,"layer_edit":15,"alpha":2}]'
for c in wrongL15a05 wrongL15a2; do
  ./runs10/run_dev.sh dl/cand_$c $c
  ./venv/bin/python scripts/grade.py runs10/dev_base.jsonl runs10/dev_$c.jsonl logs/none out/step10_dev_$c.json > /dev/null 2>&1
  python3 -c "import json;d=json.load(open('out/step10_dev_$c.json'));print('DEV $c', d['edited']['correct'], '/200 vs base', d['base']['correct'], 'better', d['paired']['rescues'], 'worse', d['paired']['losses'], 'p', round(d['paired']['mcnemar_p'],3))"
done
F=$(./venv/bin/python scripts/pick_frozen.py waitL19a3 wrongL15a1 wrongL15L19a6 waitL19am1 wrongL15a05 wrongL15a2)
echo "FROZEN $F $(date +%H:%M)"
./runs10/run_test.sh $F
