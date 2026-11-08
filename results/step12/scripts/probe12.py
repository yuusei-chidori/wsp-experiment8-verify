"""Step 12 Track A: same-problem right-vs-wrong directions (see runs12/step12_PROTOCOL.md section 2).

python probe12.py            -> runs12/acts12.pt, out12/probe12.json, out12/candidates.json, data/check12.jsonl
Reads runs12/hard_s{0..3}.jsonl (original model, 4 samples of 200 Level-5 MATH train problems).
"""
import json, math, random, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, "scripts")
from grade import grade

MD, INSTR = "dl/r1d15", "Please reason step by step, and put your final answer within \\boxed{}."
EARLY, WHOLE = 1024, 4096
probs = {json.loads(l)["id"]: json.loads(l) for l in open("data/hard200.jsonl")}
S = []  # samples
for s in range(4):
    for l in open(f"runs12/hard_s{s}.jsonl"):
        r = json.loads(l); S.append({"id": r["id"], "seed": s, "text": r["text"], "n": r["n_tokens"], "cut": bool(r["cut"]),
                                     "ok": bool(grade(r)[1])})
by = {}
for i, x in enumerate(S):
    by.setdefault(x["id"], []).append(i)
nright = {k: sum(S[i]["ok"] for i in v) for k, v in by.items()}
mixed = sorted(k for k, v in nright.items() if 1 <= v <= 3)
rng = random.Random(0); m2 = mixed[:]; rng.shuffle(m2)
nf = round(0.6 * len(m2)); find, check = sorted(m2[:nf]), sorted(m2[nf:])
pairs = {"find": [], "check": []}
for part, ids in (("find", find), ("check", check)):
    for k in ids:
        R = [i for i in by[k] if S[i]["ok"]]; W = [i for i in by[k] if not S[i]["ok"] and not S[i]["cut"]]
        for w in W:
            pairs[part].append((w, min(R, key=lambda j: abs(math.log(S[j]["n"]) - math.log(S[w]["n"])))))
summ = {"samples": len(S), "acc_per_seed": [sum(x["ok"] for x in S if x["seed"] == s) for s in range(4)],
        "problems_by_right": {str(c): sum(1 for v in nright.values() if v == c) for c in range(5)},
        "mixed": len(mixed), "find_problems": len(find), "check_problems": len(check),
        "find_pairs": len(pairs["find"]), "check_pairs": len(pairs["check"]),
        "cut_samples": sum(x["cut"] for x in S)}
print(json.dumps(summ))
with open("data/check12.jsonl", "w") as f:
    for k in check: f.write(json.dumps(probs[k]) + "\n")

need = sorted({i for p in pairs.values() for ab in p for i in ab})
tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForCausalLM.from_pretrained(MD, torch_dtype=torch.bfloat16).to("mps").eval()
names = [f"model.layers.{i}" for i in range(28)] + [f"model.layers.{i}.mlp.down_proj" for i in range(28)] \
      + [f"model.layers.{i}.self_attn.o_proj" for i in range(28)]
mods = dict(model.named_modules()); cur = {}
def hook(n):
    def f(m, a, o):
        cur[n] = (o[0] if isinstance(o, tuple) else o)[0].float()
    return f
hs = [mods[n].register_forward_hook(hook(n)) for n in names]
acts = {p: {n: torch.zeros(len(S), 1536) for n in names} for p in ("early", "whole")}
with torch.no_grad():
    for c, i in enumerate(need):
        x = S[i]; p = probs[x["id"]]
        pt = tok.encode(tok.apply_chat_template([{"role": "user", "content": p["problem"] + "\n" + INSTR}],
                                                add_generation_prompt=True, tokenize=False), add_special_tokens=False)
        gt = tok.encode(x["text"], add_special_tokens=False)[:WHOLE]
        model.model(input_ids=torch.tensor([pt + gt], device="mps"))
        P = len(pt)
        for n in names:
            acts["early"][n][i] = cur[n][P:P + EARLY].mean(0).cpu()
            acts["whole"][n][i] = cur[n][P:P + WHOLE].mean(0).cpu()
        if c % 50 == 0: print(c, len(need), flush=True)
for h in hs: h.remove()
torch.save({"acts": acts, "samples": S, "pairs": pairs, "find": find, "check": check}, "runs12/acts12.pt")

ll = torch.tensor([math.log(x["n"]) for x in S])
F = torch.tensor(pairs["find"]); C = torch.tensor(pairs["check"])
g = torch.Generator().manual_seed(0)
def unit(v): return v / v.norm()
rows = []
for pool in ("early", "whole"):
    for n in names:
        X = acts[pool][n]
        d = unit((X[F[:, 0]] - X[F[:, 1]]).mean(0))
        gap = (X[C[:, 0]] - X[C[:, 1]]) @ d
        auc = (gap > 0).float().mean().item()
        lg = ll[C[:, 0]] - ll[C[:, 1]]
        corr = torch.corrcoef(torch.stack([gap, lg]))[0, 1].item()
        sh = []
        for _ in range(200):
            sf = (torch.randint(0, 2, (len(F),), generator=g) * 2 - 1).float()[:, None]
            sc = (torch.randint(0, 2, (len(C),), generator=g) * 2 - 1).float()
            ds = unit(((X[F[:, 0]] - X[F[:, 1]]) * sf).mean(0))
            sh.append((((X[C[:, 0]] - X[C[:, 1]]) @ ds) * sc > 0).float().mean().item())
        sh.sort()
        fgap = (X[F[:, 0]] - X[F[:, 1]]) @ d
        rows.append({"site": n, "pool": pool, "check_auc": auc, "check_dz": (gap.mean() / gap.std()).item(),
                     "find_dz": (fgap.mean() / fgap.std()).item(), "shuffled_p95": sh[189], "shuffled_mean": sum(sh) / 200,
                     "corr_len_gap": corr, "editable": not n.endswith(tuple(f"layers.{i}" for i in range(28)))})
rows.sort(key=lambda r: -r["check_auc"])
for r in rows[:15]:
    print(f"{r['site']:34} {r['pool']:5} check AUC {r['check_auc']:.3f} (shuffled 95% {r['shuffled_p95']:.3f}) "
          f"dz {r['check_dz']:+.2f} find dz {r['find_dz']:+.2f} len-corr {r['corr_len_gap']:+.2f}")
cands, seen = [], set()
for r in rows:
    if r["editable"] and r["site"] not in seen:
        cands.append(r); seen.add(r["site"])
    if len(cands) == 2: break
for r in cands:
    X = acts[r["pool"]][r["site"]]
    r["direction"] = unit((X[F[:, 0]] - X[F[:, 1]]).mean(0)).tolist()
json.dump({"summary": summ, "rows": rows}, open("out12/probe12.json", "w"), indent=1)
json.dump(cands, open("out12/candidates.json", "w"))
print("CANDIDATES", [(r["site"], r["pool"], round(r["check_auc"], 3)) for r in cands])
