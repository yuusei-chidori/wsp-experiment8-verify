"""Score a screen64 run: failing set (38 probe-wrong) and control (26 probe-right), like wsh.before_after."""
import json, re, sys
sys.path.insert(0, "scripts")
from grade import grade
name = sys.argv[1]; WR = set(json.load(open("data/screen64_wrong.json")))
rows = [json.loads(l) for l in open(f"runs11/screen_{name}.jsonl")]
ok = [grade(r)[1] for r in rows]; toks = sum(r["n_tokens"] for r in rows)
waits = sum(len(re.findall(r"\bWait\b", r["text"].split("</think>")[0])) for r in rows)
res = {"name": name, "n": len(rows), "total": sum(ok), "fail_set": sum(o for o, r in zip(ok, rows) if r["id"] in WR),
       "control": sum(o for o, r in zip(ok, rows) if r["id"] not in WR), "cut": sum(r["cut"] for r in rows),
       "mean_tokens": toks / len(rows), "wait_per_1k": 1000 * waits / toks}
print(json.dumps(res)); open("runs11/screen_results.jsonl", "a").write(json.dumps(res) + "\n")
