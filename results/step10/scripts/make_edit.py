"""Step 10: write a rank-1 edit into a single-file safetensors model (the paper's edit form).

python -I make_edit.py <base_dir> <out_dir> <spec.json>

spec.json: {"edits": [{"layer": 14, "u": [... hidden-size floats ...], "alpha": 1.0, "cols": null}, ...]}
For each edit, on model.layers.<layer>.mlp.down_proj.weight (hidden x intermediate):
    W' = W - alpha * outer(u, u @ W)      (u normalised; on all columns, or only `cols`)
so the MLP's output along u is multiplied by (1 - alpha): alpha 1 removes it, 2 reverses it, < 0 strengthens it.
Computed in float64 and written back as bf16 (round to nearest even). All other bytes stay identical to the base.
Writes <out_dir>/edit_report.json with each tensor's SHA-256 and change size.
"""
import hashlib
import json
import os
import shutil
import struct
import sys

import numpy as np

ST = "model.safetensors"


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def to_f64(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def to_bf16(x):
    f = x.astype(np.float32).view(np.uint32)
    return ((f + 0x7FFF + ((f >> 16) & 1)) >> 16).astype(np.uint16)


def main(base_dir, out_dir, spec_p):
    spec = json.load(open(spec_p))
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(base_dir):
        if os.path.isfile(os.path.join(base_dir, fn)) and fn != ST:
            shutil.copy2(os.path.join(base_dir, fn), os.path.join(out_dir, fn))
    bp, op = os.path.join(base_dir, ST), os.path.join(out_dir, ST)
    shutil.copyfile(bp, op)
    h, off = header(bp)
    assert all(h[n]["dtype"] == "BF16" for n in h if n.endswith("down_proj.weight"))
    rep = {"spec": spec_p, "tensors": {}}
    by_layer = {}
    for e in spec["edits"]:
        by_layer.setdefault(e["layer"], []).append(e)
    for layer, edits in sorted(by_layer.items()):
        name = f"model.layers.{layer}.mlp.down_proj.weight"
        a, b = h[name]["data_offsets"]
        with open(bp, "rb") as f:
            f.seek(off + a)
            B = np.frombuffer(f.read(b - a), dtype=np.uint16).reshape(h[name]["shape"]).copy()
        W0 = to_f64(B)
        W = W0.copy()
        for e in edits:
            u = np.asarray(e["u"], dtype=np.float64)
            u /= np.linalg.norm(u)
            cols = np.arange(W.shape[1]) if e.get("cols") is None else np.asarray(e["cols"])
            W[:, cols] -= e["alpha"] * np.outer(u, u @ W[:, cols])
        C = to_bf16(W)
        d = to_f64(C) - W0
        with open(op, "r+b") as f:
            f.seek(off + a)
            f.write(C.tobytes())
        rep["tensors"][name] = {"edits": [{"alpha": e["alpha"], "cols": None if e.get("cols") is None else len(e["cols"])}
                                          for e in edits],
                                "change_fro": float(np.linalg.norm(d)), "weight_fro": float(np.linalg.norm(W0)),
                                "changed_entries": int((C != B).sum()), "entries": int(C.size),
                                "sha256": hashlib.sha256(C.tobytes()).hexdigest()}
    with open(op, "rb") as f:
        rep["sha256_file"] = hashlib.sha256(f.read()).hexdigest()
    json.dump(rep, open(os.path.join(out_dir, "edit_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
