"""Step 11: are the harness's top sites real? Held-out d' (direction from one half, scored on the other, 20 random
splits stratified by set and label), within-set d', and correlation of the projection with answer length / set."""
import json, torch, wsh
z = torch.load("runs11/probe_acts.pt"); A, ok, kind = z["acts"], torch.tensor(z["ok"]), z["kind"]
ntok = torch.tensor([json.loads(l)["n_tokens"] for l in open("runs10/probe_base.jsonl")]).float()
ism = torch.tensor([k == "math" for k in kind])
div = json.load(open("out11/divergence.json"))
g = torch.Generator().manual_seed(0)
def dprime(a, b, d):
    pa, pb = a @ d, b @ d
    return ((pa.mean() - pb.mean()) / torch.sqrt(0.5 * (pa.var(unbiased=False) + pb.var(unbiased=False)))).item()
rows = []
for r in div[:12] + [x for x in div if x["layer"] in ("model.layers.15", "model.layers.19")]:
    X = A[r["layer"]]
    ho = []
    for _ in range(20):
        tr = torch.zeros(len(ok), dtype=torch.bool)
        for m in (ok & ism, ok & ~ism, ~ok & ism, ~ok & ~ism):
            idx = torch.nonzero(m).flatten(); idx = idx[torch.randperm(len(idx), generator=g)]
            tr[idx[: len(idx) // 2]] = True
        d = wsh.direction(X[tr & ~ok], X[tr & ok])
        ho.append(dprime(X[~tr & ~ok], X[~tr & ok], d))
    d = wsh.direction(X[~ok], X[ok]); p = X @ d
    within = [dprime(X[~ok & m], X[ok & m], d) for m in (~ism, ism)]
    c_len = torch.corrcoef(torch.stack([p, ntok.log()]))[0, 1].item()
    c_set = torch.corrcoef(torch.stack([p, ism.float()]))[0, 1].item()
    # same direction, length only: d' between long and short RIGHT answers (is it a 'long answer' detector?)
    med = ntok[ok].median(); len_d = dprime(X[ok & (ntok > med)], X[ok & (ntok <= med)], d)
    rows.append({"layer": r["layer"], "insample": r["separation"], "heldout_mean": sum(ho) / len(ho), "heldout_min": min(ho),
                 "within_gsm": within[0], "within_math": within[1], "corr_loglen": c_len, "corr_math": c_set, "long_vs_short_right": len_d})
    print(f"{r['layer'][:34]:34} in {r['separation']:.2f} held-out {sum(ho)/len(ho):.2f} (min {min(ho):.2f}) | within gsm {within[0]:.2f} math {within[1]:.2f} | corr len {c_len:+.2f} math {c_set:+.2f} | long-vs-short(right) {len_d:+.2f}")
json.dump(rows, open("out11/probe_check.json", "w"), indent=1)
