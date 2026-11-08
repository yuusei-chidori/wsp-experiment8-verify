"""Step 10: candidate directions from the probe-set activations (TRAIN problems only).

python directions.py runs10/acts_probe_base.npz out/step10_directions.json

1. "Wait" direction: residual at "\n\n" positions followed by "Wait" minus those followed by anything else, per layer.
   For each layer: how much of that difference this layer's own MLP writes (share of the residual difference along u).
2. Wrong-vs-right direction: mean residual over the answer, wrong minus right, computed within each set (GSM8K / MATH)
   and averaged so the difficulty gap between sets does not drive it. Separation: Cohen's d of the projection.
"""
import json
import sys

import numpy as np


def main(npz, out):
    z = np.load(npz)
    L = z["prewait_resid"].shape[0]
    rep = {"n_prewait": int(z["n_prewait"]), "n_preother": int(z["n_preother"]), "wait": [], "wrong": []}
    for l in range(L):
        d = z["prewait_resid"][l] - z["preother_resid"][l]
        u = d / np.linalg.norm(d)
        dm = z["prewait_mlp"][l] - z["preother_mlp"][l]
        rep["wait"].append({"layer": l, "diff_norm": float(np.linalg.norm(d)),
                            "rel_diff": float(np.linalg.norm(d) / np.linalg.norm(z["preother_resid"][l])),
                            "mlp_share_along_u": float(dm @ u / np.linalg.norm(d))})
    R, ok, kind = z["resid"], z["correct"], z["kind"]
    for l in range(L):
        ds = []
        for k in ("gsm8k", "math"):
            m = kind == k
            if (m & ok).sum() and (m & ~ok).sum():
                ds.append(R[m & ~ok, l].mean(0) - R[m & ok, l].mean(0))
        d = np.mean(ds, axis=0)
        u = d / np.linalg.norm(d)
        p = R[:, l] @ u
        cd = []
        for k in ("gsm8k", "math"):
            m = kind == k
            a, b = p[m & ~ok], p[m & ok]
            cd.append((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))
        rep["wrong"].append({"layer": l, "cohen_d_gsm8k": float(cd[0]), "cohen_d_math": float(cd[1]),
                             "rel_diff": float(np.linalg.norm(d) / np.linalg.norm(R[:, l].mean(0)))})
    json.dump(rep, open(out, "w"), indent=1)
    print("prewait", rep["n_prewait"], "preother", rep["n_preother"])
    for w, r in zip(rep["wait"], rep["wrong"]):
        print(f"L{w['layer']:2d} wait rel {w['rel_diff']:.3f} mlp_share {w['mlp_share_along_u']:+.3f} | "
              f"wrong d gsm {r['cohen_d_gsm8k']:+.2f} math {r['cohen_d_math']:+.2f}")


def unit(npz, kind, layer):
    """The unit vector of a candidate direction (used when writing an edit spec)."""
    z = np.load(npz)
    if kind == "wait":
        d = z["prewait_resid"][layer] - z["preother_resid"][layer]
    else:
        R, ok, k = z["resid"], z["correct"], z["kind"]
        d = np.mean([R[(k == s) & ~ok, layer].mean(0) - R[(k == s) & ok, layer].mean(0) for s in ("gsm8k", "math")], axis=0)
    return d / np.linalg.norm(d)


if __name__ == "__main__":
    main(*sys.argv[1:3])
