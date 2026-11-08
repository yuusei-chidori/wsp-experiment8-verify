"""Step 10: practice sets from the TRAIN splits only (GSM8K train, MATH train). Never touches test problems.
dev = 100 GSM8K + 100 MATH used to score candidate edits; probe = 300 + 300 used to look inside the model."""
import json, random, sys
import pyarrow.parquet as pq
sys.path.insert(0, 'scripts')
from grade import last_boxed

g = pq.read_table('data/train_dl/main/train-00000-of-00001.parquet').to_pylist()
m = pq.read_table('data/train_dl/math/data/train-00000-of-00001.parquet').to_pylist()
test_text = {json.loads(l)['problem'].strip() for f in ('data/math100.jsonl', 'data/subset100.jsonl') for l in open(f)}
G = [{'id': f'gsm8k_train_{i}', 'problem': r['question'], 'answer': r['answer'].split('####')[-1].strip().replace(',', '')}
     for i, r in enumerate(g)]
M = []
for i, r in enumerate(m):
    a = last_boxed(r['solution'])
    if a and r['problem'].strip() not in test_text:
        M.append({'id': f'math_train_{i}', 'problem': r['problem'], 'answer': a, 'level': r.get('level'), 'subject': r.get('type')})
print('gsm8k train', len(G), 'math train usable', len(M), 'of', len(m))
rng = random.Random(1)
rng.shuffle(G); rng.shuffle(M)
sets = {'dev': G[:100] + M[:100], 'probe': G[100:400] + M[100:400]}
for k, v in sets.items():
    with open(f'data/train_{k}.jsonl', 'w') as f:
        for r in v: f.write(json.dumps(r) + '\n')
    print(k, len(v))
