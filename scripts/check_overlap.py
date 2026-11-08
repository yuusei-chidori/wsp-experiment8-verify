"""Step 4a: do the practice problems the editor could tune on overlap with the test sets?

python -I check_overlap.py <practice_dir> <gsm8k_test.jsonl> <math500.jsonl> <requests_dir> <subset100.jsonl> <out.json>

For every test problem, find its most similar practice problem by three measures:
  - exact match after normalisation (lower case, punctuation and spaces collapsed)
  - 13-word window shared (the authors' own decontamination rule)
  - word 5-gram Jaccard similarity, also with every number replaced by "#" (catches the same template with new numbers)
Then check whether the closest practice problems were actually sent through a forward pass in run p50c.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict


def norm(s):
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9#]+", " ", s)
    return " ".join(s.split())


def mask_numbers(s):
    return re.sub(r"\d+(?:[.,]\d+)*", "#", s)


def grams(words, n):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def best_matches(tests, practice, masked):
    key = (lambda s: norm(mask_numbers(s))) if masked else norm
    pw = {p["id"]: key(p["problem"]).split() for p in practice}
    pg = {i: grams(w, 5) for i, w in pw.items()}
    index = defaultdict(set)
    for i, g in pg.items():
        for x in g:
            index[x].add(i)
    out = {}
    for t in tests:
        tg = grams(key(t["problem"]).split(), 5)
        cand = defaultdict(int)
        for x in tg:
            if len(index[x]) < 200:  # skip very common phrases
                for i in index[x]:
                    cand[i] += 1
        best = (0.0, None)
        for i, c in cand.items():
            j = c / (len(tg) + len(pg[i]) - c) if tg else 0.0
            if j > best[0]:
                best = (j, i)
        out[t["id"]] = best
    return out


def main(practice_dir, gsm_test, math500, req_dir, subset_p, out_p):
    practice = []
    for f in ("gsm8k_train.jsonl", "math_train.jsonl", "aime.jsonl"):
        practice += [json.loads(l) for l in open(os.path.join(practice_dir, f))]
    ptext = {p["id"]: p for p in practice}
    gsm = [json.loads(l) for l in open(gsm_test)]
    m5 = [json.loads(l) for l in open(math500)]
    for i, r in enumerate(m5):
        r.setdefault("id", r.get("unique_id", f"math500_{i}"))
    subset = {json.loads(l)["id"] for l in open(subset_p)}

    # problems used in any forward pass of p50c (generation or teacher-forced) and in which pass
    used = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(req_dir, "fwd_*.json.claimed")), key=lambda s: int(re.findall(r"fwd_(\d+)", s)[0])):
        k = int(re.findall(r"fwd_(\d+)", f)[0])
        txt = open(f).read()
        for pid in set(re.findall(r"(?:gsm8k_train|math_train_[a-z_]+?|aime)_\d+", txt)):
            used[pid].append(k)

    all_norm = {norm(p["problem"]): p["id"] for p in practice}
    p_grams13 = defaultdict(set)
    for p in practice:
        for g in grams(norm(p["problem"]).split(), 13):
            p_grams13[g].add(p["id"])

    res = {"practice_problems": len(practice), "used_in_p50c_passes": len(used), "sets": {}}
    for name, tests, pool in (("gsm8k_test", gsm, [p for p in practice if p["source"] == "gsm8k"]),
                              ("math500", m5, [p for p in practice if p["source"] != "gsm8k"])):
        raw = best_matches(tests, pool, masked=False)
        msk = best_matches(tests, pool, masked=True)
        rows = []
        for t in tests:
            n = norm(t["problem"])
            g13 = set().union(*[p_grams13.get(g, set()) for g in grams(n.split(), 13)]) if len(n.split()) >= 13 else set()
            jr, ir = raw[t["id"]]
            jm, im = msk[t["id"]]
            rows.append({"id": t["id"], "exact": all_norm.get(n), "shared13": sorted(g13),
                         "jaccard5": round(jr, 3), "nearest": ir, "jaccard5_masked": round(jm, 3), "nearest_masked": im,
                         "nearest_used_in": used.get(im, []) if im else [], "in_my_subset100": t["id"] in subset})
        th = [0.3, 0.5, 0.7]
        summ = {"n": len(rows), "exact": sum(r["exact"] is not None for r in rows),
                "shared_13gram": sum(bool(r["shared13"]) for r in rows)}
        for x in th:
            hit = [r for r in rows if r["jaccard5_masked"] >= x]
            summ[f"masked_jaccard>={x}"] = len(hit)
            summ[f"masked_jaccard>={x}_and_nearest_used_in_p50c"] = sum(bool(r["nearest_used_in"]) for r in hit)
            summ[f"masked_jaccard>={x}_in_subset100"] = sum(r["in_my_subset100"] for r in hit)
        top = sorted(rows, key=lambda r: -r["jaccard5_masked"])[:12]
        res["sets"][name] = {"summary": summ,
                             "top_pairs": [dict(r, test=next(t["problem"] for t in tests if t["id"] == r["id"]),
                                                practice=ptext[r["nearest_masked"]]["problem"] if r["nearest_masked"] else None)
                                           for r in top],
                             "rows": rows}
        print(name, json.dumps(summ, indent=1))
    json.dump(res, open(out_p, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main(*sys.argv[1:7])
