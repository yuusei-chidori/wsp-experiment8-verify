"""Step 3: build a norm-matched random-direction control of the published Qwen3.5-0.8B edit.

python -I make_random_control.py <base_dir> <edited_dir> <out_dir> <out.json> [--seed 0] [--alpha A]

For each edited tensor (layers 15 and 19 mlp.down_proj, shape 1024 x 3584):
  - D = edited - base on the 64 changed columns; u = its top left singular vector (the published "loop" direction)
  - r = a Gaussian random unit vector, made orthogonal to u
  - control: W' = W - a * outer(r, r @ W[:, cols]) on the same 64 columns, with a chosen so that
    ||W' - W||_F equals ||D||_F (the size of the published change), as the authors' control did
  - with --alpha A: a = A instead (e.g. 6, the published strength; a much smaller change than norm matching)
The output directory is a copy of the base model with only these two tensors' bytes replaced (BF16, round to nearest even).
"""
import hashlib
import json
import os
import shutil
import struct
import sys

import numpy as np


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def read_bf16(path, h, off, name):
    a, b = h[name]["data_offsets"]
    with open(path, "rb") as f:
        f.seek(off + a)
        bits = np.frombuffer(f.read(b - a), dtype=np.uint16).reshape(h[name]["shape"])
    return bits


def to_f64(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def to_bf16(x):
    f = x.astype(np.float32).view(np.uint32)
    return ((f + 0x7FFF + ((f >> 16) & 1)) >> 16).astype(np.uint16)


def main(base_dir, edit_dir, out_dir, out_json, seed=0, alpha=None):
    st = "model.safetensors-00001-of-00001.safetensors"
    bp, ep, op = (os.path.join(d, st) for d in (base_dir, edit_dir, out_dir))
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(base_dir):
        if fn != st and os.path.isfile(os.path.join(base_dir, fn)):
            shutil.copy2(os.path.join(base_dir, fn), os.path.join(out_dir, fn))
    shutil.copyfile(bp, op)
    hb, ob = header(bp)
    he, oe = header(ep)
    rng = np.random.default_rng(seed)
    names = [n for n in sorted(hb) if n.endswith(("layers.15.mlp.down_proj.weight", "layers.19.mlp.down_proj.weight"))]
    assert len(names) == 2, names
    report = {"seed": seed, "alpha_mode": "norm_matched" if alpha is None else f"fixed {alpha}", "tensors": {}}
    for name in names:
        Bb, Eb = read_bf16(bp, hb, ob, name), read_bf16(ep, he, oe, name)
        Wb, We = to_f64(Bb), to_f64(Eb)
        cols = np.where((Bb != Eb).any(axis=0))[0]
        D = (We - Wb)[:, cols]
        u = np.linalg.svd(D, full_matrices=False)[0][:, 0]
        r = rng.standard_normal(Wb.shape[0])
        r -= (r @ u) * u
        r /= np.linalg.norm(r)
        P = np.outer(r, r @ Wb[:, cols])
        a = np.linalg.norm(D) / np.linalg.norm(P) if alpha is None else float(alpha)
        Wc = Wb.copy()
        Wc[:, cols] -= a * P
        Cb = to_bf16(Wc)
        Cb[:, np.setdiff1d(np.arange(Wb.shape[1]), cols)] = Bb[:, np.setdiff1d(np.arange(Wb.shape[1]), cols)]
        dC = to_f64(Cb) - Wb
        # the published edit's own strength, for comparison: D ~ -alpha * outer(u, u @ W)
        Pu = np.outer(u, u @ Wb[:, cols])
        alpha_pub = float(-(D * Pu).sum() / (Pu * Pu).sum())
        x, y = hb[name]["data_offsets"]
        with open(op, "r+b") as f:
            f.seek(ob + x)
            f.write(Cb.tobytes())
        report["tensors"][name] = {
            "columns": len(cols), "cos_r_u": float(abs(r @ u)),
            "published_alpha": alpha_pub, "control_alpha": float(a),
            "published_change_norm": float(np.linalg.norm(D)), "control_change_norm": float(np.linalg.norm(dC)),
            "norm_ratio_after_bf16": float(np.linalg.norm(dC) / np.linalg.norm(D)),
            "u_proj_mean_abs_cols": float(np.abs(u @ Wb[:, cols]).mean()),
            "r_proj_mean_abs_cols": float(np.abs(r @ Wb[:, cols]).mean()),
            "changed_entries": int((Cb != Bb).sum()),
            "changed_columns": int((Cb != Bb).any(axis=0).sum()),
            "sha256_control": hashlib.sha256(Cb.tobytes()).hexdigest()}
    # sanity: every other tensor is byte-identical to the base
    with open(bp, "rb") as f1, open(op, "rb") as f2:
        same_header = f1.read(ob) == f2.read(ob)
    report["header_identical_to_base"] = same_header
    json.dump(report, open(out_json, "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    args = sys.argv[1:]
    seed, alpha = 0, None
    if "--seed" in args:
        i = args.index("--seed"); seed = int(args[i + 1]); del args[i:i + 2]
    if "--alpha" in args:
        i = args.index("--alpha"); alpha = float(args[i + 1]); del args[i:i + 2]
    main(*args, seed=seed, alpha=alpha)
