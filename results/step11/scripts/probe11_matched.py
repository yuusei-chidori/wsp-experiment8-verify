"""Step 11: wrong vs RIGHT answers matched on set and length (each wrong answer paired with the right answer of the
same set whose log-length is closest, without reuse). Is there an error direction once length is taken out?
Held-out d' over 50 random half-splits of the pairs; also a shuffled-label baseline (same pairs, labels swapped
at random) to show what overfitting alone gives."""
import json, torch, wsh
z = torch.load("runs11/probe_acts.pt"); A, ok, kind = z["acts"], torch.tensor(z["ok"]), z["kind"]
ntok = torch.tensor([json.loads(l)["n_tokens"] for l in open("runs10/probe_base.jsonl")]).float().log()
ism = torch.tensor([k == "math" for k in kind])
wrong = torch.nonzero(~ok).flatten().tolist(); used = set(); pairs = []
for w in sorted(wrong, key=lambda i: -ntok[i].item()):
    cand = [i for i in torch.nonzero(ok & (ism == ism[w])).flatten().tolist() if i not in used]
    j = min(cand, key=lambda i: abs(ntok[i] - ntok[w]).item()); used.add(j); pairs.append((w, j))
W = torch.tensor([p[0] for p in pairs]); R = torch.tensor([p[1] for p in pairs])
print(f"{len(pairs)} pairs; mean |log-length gap| {(ntok[W]-ntok[R]).abs().mean():.3f}; wrong mean len {ntok[W].exp().mean():.0f}, matched right {ntok[R].exp().mean():.0f}")
def dp(a, b, d):
    pa, pb = a @ d, b @ d
    return ((pa.mean() - pb.mean()) / torch.sqrt(0.5 * (pa.var(unbiased=False) + pb.var(unbiased=False)))).item()
g = torch.Generator().manual_seed(0); out = []
sites = [r["layer"] for r in json.load(open("out11/divergence.json"))]
for s in sites:
    X = A[s]; ho, sh = [], []
    for _ in range(50):
        perm = torch.randperm(len(pairs), generator=g); tr, te = perm[: len(pairs) // 2], perm[len(pairs) // 2:]
        d = wsh.direction(X[W[tr]], X[R[tr]]); ho.append(dp(X[W[te]], X[R[te]], d))
        flip = torch.rand(len(pairs), generator=g) < 0.5
        Wf, Rf = torch.where(flip, R, W), torch.where(flip, W, R)
        d = wsh.direction(X[Wf[tr]], X[Rf[tr]]); sh.append(dp(X[Wf[te]], X[Rf[te]], d))
    out.append({"site": s, "heldout": sum(ho) / 50, "shuffled": sum(sh) / 50, "insample": dp(X[W], X[R], wsh.direction(X[W], X[R]))})
out.sort(key=lambda r: -r["heldout"])
for r in out[:12]:
    print(f"{r['site'][:36]:36} matched in-sample {r['insample']:.2f}  held-out {r['heldout']:+.2f}  shuffled-label held-out {r['shuffled']:+.2f}")
print("median held-out over all 84 sites", sorted(r["heldout"] for r in out)[42])
json.dump({"pairs": pairs, "sites": out}, open("out11/probe_matched.json", "w"), indent=1)
