"""Step 10: quick screen of a candidate edit on 64 PROBE (train) problems — model health, not selection.

python screen.py <name> '<json list of {kind, layer_dir, layer_edit, alpha}>'
  kind: "wait" or "wrong" (directions.unit), layer_dir: where the direction is measured, layer_edit: whose down_proj is edited.
Builds dl/cand_<name>, generates (cap 16384, same sampling), prints accuracy / mean tokens / "Wait" per 1k tokens.
"""
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, "scripts")
from directions import unit  # noqa: E402
from grade import grade  # noqa: E402

name, edits = sys.argv[1], json.loads(sys.argv[2])
spec = {"edits": [{"layer": e["layer_edit"], "alpha": e["alpha"],
                   "u": unit("runs10/acts_probe_base.npz", e["kind"], e["layer_dir"]).tolist()} for e in edits]}
os.makedirs("edits10", exist_ok=True)
json.dump(spec, open(f"edits10/{name}.json", "w"))
json.dump(edits, open(f"edits10/{name}.desc.json", "w"))
out_dir = f"dl/cand_{name}"
if not os.path.exists(os.path.join(out_dir, "edit_report.json")):
    subprocess.run(["./venv/bin/python", "-I", "scripts/make_edit.py", "dl/r1d15", out_dir, f"edits10/{name}.json"],
                   check=True, stdout=subprocess.DEVNULL)
out = f"runs10/screen_{name}.jsonl"
for i in range(4):
    if os.path.exists(out) and sum(1 for _ in open(out)) >= 64:
        break
    subprocess.run(["./venv/bin/python", "scripts/run_gen.py", "--model", out_dir, "--problems", "data/screen64.jsonl",
                    "--cap", "16384", "--batch", "32", "--sampler", "vllm", "--stop-from-tokenizer", "--out", out],
                   stdout=open(f"runs10/screen_{name}.log", "a"), stderr=subprocess.STDOUT)
rows = [json.loads(l) for l in open(out)]
ok = [grade(r)[1] for r in rows]
toks = sum(r["n_tokens"] for r in rows)
waits = sum(len(re.findall(r"\bWait\b", r["text"].split("</think>")[0])) for r in rows)
res = {"name": name, "edits": edits, "n": len(rows), "correct": sum(ok), "cut": sum(r["cut"] for r in rows),
       "mean_tokens": toks / len(rows), "wait_per_1k": 1000 * waits / toks,
       "wrong_set_correct": sum(o for o, r in zip(ok, rows) if r["id"] in WRONG) if (WRONG := set(json.load(open("data/screen64_wrong.json")))) else 0}
print(json.dumps(res))
with open("runs10/screen_results.jsonl", "a") as f:
    f.write(json.dumps(res) + "\n")
