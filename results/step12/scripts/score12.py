"""Step 12: paired score of an edited run against the original model on the same problems (and seeds).
python score12.py <label> <base.jsonl[,base2...]> <edited.jsonl[,edited2...]> [out.json]   (files paired in order)"""
import json, re, sys
sys.path.insert(0, "scripts")
from grade import grade, mcnemar_exact
lab, B, E = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(",")
b, e, toks, waits, cut = {}, {}, 0, 0, 0
for k, (pb, pe) in enumerate(zip(B, E)):
    for l in open(pb):
        r = json.loads(l); b[(k, r["id"])] = bool(grade(r)[1])
    for l in open(pe):
        r = json.loads(l); e[(k, r["id"])] = bool(grade(r)[1]); toks += r["n_tokens"]; cut += r["cut"]
        waits += len(re.findall(r"\bWait\b", r["text"].split("</think>")[0]))
keys = [k for k in e if k in b]
better = sum(e[k] and not b[k] for k in keys); worse = sum(b[k] and not e[k] for k in keys)
res = {"label": lab, "n": len(keys), "base": sum(b[k] for k in keys), "edited": sum(e[k] for k in keys), "better": better,
       "worse": worse, "p": mcnemar_exact(better, worse), "cut": cut, "mean_tokens": toks / len(e), "wait_per_1k": 1000 * waits / toks}
print(json.dumps(res))
if len(sys.argv) > 4: json.dump(res, open(sys.argv[4], "w"), indent=1)
