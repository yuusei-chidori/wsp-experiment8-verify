#!/bin/zsh
cd "${0:A:h}/.."
./venv/bin/python scripts/screen.py waitL19am1 '[{"kind":"wait","layer_dir":19,"layer_edit":19,"alpha":-1}]'
for c in waitL19a3 wrongL15a1 wrongL15L19a6 waitL19am1; do
  ./runs10/run_dev.sh dl/cand_$c $c
  ./venv/bin/python scripts/grade.py runs10/dev_base.jsonl runs10/dev_$c.jsonl logs/none out/step10_dev_$c.json > /dev/null 2>&1
  python3 -c "import json;d=json.load(open('out/step10_dev_$c.json'));print('DEV $c', d['edited']['correct'], '/200 vs base', d['base']['correct'], 'better', d['paired']['rescues'], 'worse', d['paired']['losses'], 'p', round(d['paired']['mcnemar_p'],3))"
done
echo DEV1_DONE
