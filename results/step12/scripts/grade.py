"""Step 2 grading and paired statistics.

python grade.py runs/base.jsonl runs/edited.jsonl logs/gsm8k out/step2_summary.json

Grade: the last \\boxed{...} (brace matched) of the text after the last </think> (whole text when there is none),
checked against the gold answer with math-verify, as the post's mathqa.is_correct does. A plain numeric comparison
is reported beside it as a cross-check of the grader.
"""
import glob
import json
import math
import sys

from math_verify import parse, verify


def last_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = i + len("\\boxed")
    while j < len(text) and text[j] == " ":
        j += 1
    if j >= len(text) or text[j] != "{":
        rest = text[j:].split("$")[0].split("\n")[0].strip()
        return rest or None
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
        k += 1
    return None


def answer_part(text):
    return text.split("</think>")[-1].strip() if "</think>" in text else text


def num(s):
    try:
        return float(s.replace(",", "").replace("\\$", "").replace("$", "").replace("\\%", "").strip())
    except Exception:
        return None


def grade(rec):
    pred = last_boxed(answer_part(rec["text"]))
    if pred is None:
        return pred, False, False
    try:
        ok = bool(verify(parse("\\boxed{" + rec["answer"] + "}"), parse("\\boxed{" + pred + "}")))
    except Exception:
        ok = False
    a, b = num(pred), num(rec["answer"])
    return pred, ok, (a is not None and b is not None and abs(a - b) < 1e-6)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def summarize(recs):
    n = len(recs)
    acc = sum(r["ok"] for r in recs)
    return {"n": n, "correct": acc, "acc": acc / n, "acc_ci95": wilson(acc, n),
            "acc_numeric": sum(r["ok_num"] for r in recs) / n,
            "cut": sum(r["cut"] for r in recs), "cut_share": sum(r["cut"] for r in recs) / n,
            "mean_tokens": sum(r["n_tokens"] for r in recs) / n,
            "median_tokens": sorted(r["n_tokens"] for r in recs)[n // 2],
            "no_box_finished": sum((r["pred"] is None) and not r["cut"] for r in recs),
            "acc_among_finished": (sum(r["ok"] for r in recs if not r["cut"]) / max(1, sum(not r["cut"] for r in recs))),
            "correct_although_cut": sum(r["ok"] and r["cut"] for r in recs)}


def main(base_p, edit_p, logs_dir, out_p):
    res = {}
    for name, p in (("base", base_p), ("edited", edit_p)):
        recs = [json.loads(l) for l in open(p)]
        for r in recs:
            r["pred"], r["ok"], r["ok_num"] = grade(r)
        res[name] = {r["id"]: r for r in recs}
    ids = sorted(set(res["base"]) & set(res["edited"]), key=lambda s: int(s.split("_")[-1]))
    B = [res["base"][i] for i in ids]
    E = [res["edited"][i] for i in ids]
    rescues = sum((not b["ok"]) and e["ok"] for b, e in zip(B, E))
    losses = sum(b["ok"] and (not e["ok"]) for b, e in zip(B, E))
    diffs = [int(e["ok"]) - int(b["ok"]) for b, e in zip(B, E)]
    md = sum(diffs) / len(diffs)
    sd = math.sqrt(sum((d - md) ** 2 for d in diffs) / (len(diffs) - 1))
    # the authors' own per-problem results (2 samples, 32k) on the same problems
    authors = {}
    for run in ("unedited", "p50c"):
        pp = {}
        for f in glob.glob(f"{logs_dir}/{run}_pp_*.json"):
            for r in json.load(open(f)):
                pp[r["id"]] = r["correct"]
        got = [pp[i] for i in ids if i in pp]
        if not got:  # no per-problem logs for this benchmark (e.g. MATH-500)
            continue
        authors[run] = {"problems": len(got), "acc_all_samples": sum(map(sum, got)) / sum(map(len, got)),
                        "acc_first_sample": sum(g[0] for g in got) / len(got)}
    out = {"problems": len(ids), "base": summarize(B), "edited": summarize(E),
           "paired": {"rescues": rescues, "losses": losses, "gain": md, "se": sd / math.sqrt(len(diffs)),
                      "mcnemar_p": mcnemar_exact(rescues, losses),
                      "grader_agreement_mathverify_vs_numeric": sum((r["ok"] == r["ok_num"]) for r in B + E) / (2 * len(ids))},
           "authors_same_problems": authors,
           "per_problem": [{"id": i, "gold": b["answer"], "base_pred": b["pred"], "base_ok": b["ok"], "base_cut": b["cut"],
                            "base_tokens": b["n_tokens"], "edited_pred": e["pred"], "edited_ok": e["ok"],
                            "edited_cut": e["cut"], "edited_tokens": e["n_tokens"]} for i, b, e in zip(ids, B, E)]}
    json.dump(out, open(out_p, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_problem"}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
