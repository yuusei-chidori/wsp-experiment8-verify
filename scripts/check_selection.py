"""Step 4b: was the published edit chosen without the test sets, and was its "held-out" confirmation really held out?

python -I check_selection.py <runs/p50c dir> <gsm8k_test.jsonl> <math500.jsonl> <out.json>

Reads run p50c's tool_calls.jsonl (every Python run and forward pass, in order), requests/ and work_notes/.
  - the last save_edit() call: the edit is frozen from there on
  - which forward requests came before it, which failed (a failed request did not count as a pass)
  - whether the 1,152 confirmation problems appear in any request before the confirmation began
  - whether any 13-word window of a GSM8K-test or MATH-500 problem appears in the run's code, outputs or notes
"""
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower().replace("’", "'")).split())


def grams(s, n=13):
    w = norm(s).split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def main(run, gsm_test, math500, out_p):
    calls = [json.loads(l) for l in open(os.path.join(run, "tool_calls.jsonl"))]
    last_save = max(i for i, c in enumerate(calls) if c["kind"] == "python" and "save_edit(" in (c.get("code") or ""))
    fwd = [(i, int(c["k"])) for i, c in enumerate(calls) if c["kind"] == "forward"]
    ids, ok, kind = {}, {}, {}
    for f in glob.glob(os.path.join(run, "requests", "fwd_*.json.claimed")):
        k = int(re.findall(r"fwd_(\d+)", f)[0])
        q = json.load(open(f))
        ids[k] = set(q.get("problems") or [])
        kind[k] = f'{q.get("type")}/{q.get("model")}'
        ok[k] = json.load(open(f.replace(".json.claimed", ".done.json"))).get("ok") is True
    before = sorted(k for i, k in fwd if i < last_save)
    after = sorted(k for i, k in fwd if i > last_save)
    conf = defaultdict(set)
    for r in csv.DictReader(open(os.path.join(run, "work_notes", "final_confirmation.csv"))):
        conf[int(r["run"])].add(r["id"])
    conf_ids = set().union(*conf.values())
    first_conf = min(conf)
    reuse = {k: len(ids[k] & conf_ids) for k in ids if k < first_conf and ids[k] & conf_ids}

    tg = {}
    for p in (gsm_test, math500):
        for l in open(p):
            t = json.loads(l)
            for x in grams(t["problem"]):
                tg[x] = t.get("id") or t.get("unique_id")
    seen = open(os.path.join(run, "tool_calls.jsonl")).read()
    for f in glob.glob(os.path.join(run, "work_notes", "*")) + glob.glob(os.path.join(run, "explanations.md")):
        seen += open(f, errors="ignore").read()
    test_hits = sorted({tg[x] for x in grams(seen) if x in tg})

    out = {
        "requests": len(ids), "failed_requests": sorted(k for k in ok if not ok[k]),
        "passes_counted": sum(ok.values()),
        "last_save_edit_tool_call": last_save,
        "requests_before_freeze": [before[0], before[-1]], "passes_before_freeze": sum(ok[k] for k in before),
        "requests_after_freeze": [after[0], after[-1]], "passes_after_freeze": sum(ok[k] for k in after),
        "distinct_problems_in_successful_requests_before_freeze": len(set().union(*[ids[k] for k in before if ok[k]])),
        "confirmation": {"requests": sorted(conf), "problems": len(conf_ids),
                         "requests_match_csv": all(conf[k] == ids[k] for k in conf),
                         "all_confirmation_requests_after_freeze": all(k in after for k in conf),
                         "confirmation_problems_in_earlier_requests": {str(k): {"n": v, "request_ok": ok[k], "after_freeze": k in after}
                                                                       for k, v in sorted(reuse.items())}},
        "test_problem_13gram_hits_in_run_record": test_hits,
        "request_kinds": {str(k): kind[k] + ("" if ok[k] else " (failed)") for k in sorted(kind)},
    }
    json.dump(out, open(out_p, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "request_kinds"}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
