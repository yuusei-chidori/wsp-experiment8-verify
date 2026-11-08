import json,re,sys
sys.path.insert(0,"scripts")
from grade import grade
def st(p):
    rows=[json.loads(l) for l in open(p)]
    ids=[r["id"] for r in rows]
    ok=sum(grade(r)[1] for r in rows); t=sum(r["n_tokens"] for r in rows)
    w=sum(len(re.findall(r"\bWait\b",r["text"].split("</think>")[0])) for r in rows)
    return dict(n=len(rows),uniq=len(set(ids)),ok=ok,cut=sum(r["cut"] for r in rows),mean_tok=round(t/len(rows)),wait1k=round(1000*w/t,2))
out={}
for p in sys.argv[1:]:
    out[p]=st(p); print(p,out[p])
json.dump(out,open("out11/gen_stats.json","w"),indent=1)
