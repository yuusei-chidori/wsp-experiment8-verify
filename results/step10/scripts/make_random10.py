"""Step 10: random-direction control for the frozen edit (same layers, same change size).

python -I make_random10.py <frozen_spec.json> <base_dir> <out_spec.json> [--seed 0]
For each edit (layer, u, alpha): r = Gaussian unit vector orthogonal to u (numpy default_rng(seed));
alpha_r chosen so ||alpha_r * r (r^T W)||_F == ||alpha * u (u^T W)||_F on that layer's down_proj.
Then build the model with make_edit.py from the written spec.
"""
import json
import struct
import sys

import numpy as np


def load_down(base_dir, layer):
    p = f"{base_dir}/model.safetensors"
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        name = f"model.layers.{layer}.mlp.down_proj.weight"
        a, b = h[name]["data_offsets"]
        f.seek(8 + n + a)
        bits = np.frombuffer(f.read(b - a), dtype=np.uint16).reshape(h[name]["shape"])
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def main(spec_p, base_dir, out_p, seed=0):
    spec = json.load(open(spec_p))
    rng = np.random.default_rng(seed)
    out = {"edits": [], "seed": seed, "from": spec_p}
    for e in spec["edits"]:
        W = load_down(base_dir, e["layer"])
        u = np.asarray(e["u"]); u /= np.linalg.norm(u)
        r = rng.standard_normal(W.shape[0]); r -= (r @ u) * u; r /= np.linalg.norm(r)
        size_u = abs(e["alpha"]) * np.linalg.norm(u @ W)   # ||outer(u, u^T W)||_F = ||u^T W||
        alpha_r = np.sign(e["alpha"]) * size_u / np.linalg.norm(r @ W)
        out["edits"].append({"layer": e["layer"], "u": r.tolist(), "alpha": float(alpha_r),
                             "alpha_frozen": e["alpha"], "cos_r_u": float(abs(r @ u))})
        print(f"layer {e['layer']}: frozen alpha {e['alpha']} -> random alpha {alpha_r:.3f}")
    json.dump(out, open(out_p, "w"))


if __name__ == "__main__":
    a = sys.argv[1:]
    s = 0
    if "--seed" in a:
        i = a.index("--seed"); s = int(a[i + 1]); del a[i:i + 2]
    main(*a, seed=s)
