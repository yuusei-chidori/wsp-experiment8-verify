"""Step 12 data. (1) data/math500.jsonl: all 500 MATH-500 test problems, id math500_<index> (same ids as math100).
(2) data/hard200.jsonl: 200 Level-5 MATH TRAIN problems not in Step 10's dev/probe sets (same shuffle as
make_train_sets.py, taken from M[400:]) and not equal to any MATH-500 problem. Used only to find directions."""
import json, random, sys
import pyarrow.parquet as pq
sys.path.insert(0, 'scripts')
from grade import last_boxed
t = [json.loads(l) for l in open('step4/bench/math500.jsonl')]
with open('data/math500.jsonl', 'w') as f:
    for i, r in enumerate(t):
        f.write(json.dumps({'id': f'math500_{i}', 'problem': r['problem'], 'answer': r['answer'], 'subject': r.get('subject'),
                            'level': r.get('level'), 'unique_id': r.get('unique_id')}) + '\n')
m100 = {json.loads(l)['id']: json.loads(l)['problem'] for l in open('data/math100.jsonl')}
full = {f'math500_{i}': r['problem'] for i, r in enumerate(t)}
assert all(full[k] == v for k, v in m100.items()), 'math100 ids do not match'
m = pq.read_table('data/train_dl/math/data/train-00000-of-00001.parquet').to_pylist()
old_test = {json.loads(l)['problem'].strip() for f in ('data/math100.jsonl', 'data/subset100.jsonl') for l in open(f)}
M = []
for i, r in enumerate(m):
    a = last_boxed(r['solution'])
    if a and r['problem'].strip() not in old_test:
        M.append({'id': f'math_train_{i}', 'problem': r['problem'], 'answer': a, 'level': r.get('level'), 'subject': r.get('type')})
rng = random.Random(1); rng.shuffle(M)
used = {json.loads(l)['id'] for f in ('data/train_dev.jsonl', 'data/train_probe.jsonl', 'data/screen64.jsonl') for l in open(f)}
tt = {p.strip() for p in full.values()}
H = [r for r in M[400:] if r['level'] == 'Level 5' and r['id'] not in used and r['problem'].strip() not in tt][:200]
with open('data/hard200.jsonl', 'w') as f:
    for r in H: f.write(json.dumps(r) + '\n')
print('math500', len(t), 'hard200', len(H))
