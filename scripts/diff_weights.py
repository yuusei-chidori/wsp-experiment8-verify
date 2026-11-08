"""Step 1: independent weight diff of Qwen3.5-0.8B base vs the published 'weight-space-programming' edit.

Usage: python -I diff_weights.py <base.safetensors> <edited.safetensors> <edit.npz> <manifest.json> <out.json>
Reads safetensors by hand (header + raw bytes) so BF16 is compared bit for bit with no library conversion.
"""
import hashlib
import json
import struct
import sys

import numpy as np


def open_st(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    return header, mm, 8 + n


def raw(st, name):
    header, mm, off = st
    h = header[name]
    a, b = h["data_offsets"]
    return mm[off + a: off + b], h


def bf16_bits(st, name):
    buf, h = raw(st, name)
    assert h["dtype"] == "BF16"
    return np.frombuffer(buf, dtype=np.uint16).reshape(h["shape"])


def bf16_to_f64(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def main(base_p, edit_p, npz_p, man_p, out_p):
    base, edit = open_st(base_p), open_st(edit_p)
    out = {}
    nb, ne = set(base[0]), set(edit[0])
    out["tensor_count"] = {"base": len(nb), "edited": len(ne), "only_base": sorted(nb - ne), "only_edited": sorted(ne - nb)}
    total_params, differing = 0, []
    for name in sorted(nb & ne):
        bb, hb = raw(base, name)
        eb, he = raw(edit, name)
        assert hb["dtype"] == he["dtype"] and hb["shape"] == he["shape"], name
        total_params += int(np.prod(hb["shape"]))
        if not np.array_equal(bb, eb):
            differing.append(name)
    out["total_params"] = total_params
    out["differing_tensors"] = differing

    man = json.load(open(man_p))
    z = np.load(npz_p, allow_pickle=False)
    per = {}
    dirs = {}
    for i, t in enumerate(man["tensors"]):
        name = t["name"]
        B, E = bf16_bits(base, name), bf16_bits(edit, name)
        r = {"shape": list(B.shape)}
        r["sha256_base"] = hashlib.sha256(B.tobytes()).hexdigest()
        r["sha256_edited"] = hashlib.sha256(E.tobytes()).hexdigest()
        r["sha_base_matches_manifest"] = r["sha256_base"] == t["base_sha256"]
        r["sha_edited_matches_manifest"] = r["sha256_edited"] == t["edited_sha256"]
        neq = B != E
        r["changed_entries"] = int(neq.sum())
        cols = np.where(neq.any(axis=0))[0]
        r["changed_columns"] = int(len(cols))
        r["changed_rows"] = int(neq.any(axis=1).sum())
        r["columns"] = cols.tolist()
        r["touched_weights"] = int(B.shape[0] * len(cols))
        # rebuild from the published diff with my own code
        e = B.copy().reshape(-1)
        e[z[f"t{i}_idx"]] = z[f"t{i}_val"]
        r["rebuild_from_npz_bit_exact"] = bool(np.array_equal(e.reshape(B.shape), E))
        # structure of the change
        Wb, We = bf16_to_f64(B), bf16_to_f64(E)
        D = (We - Wb)[:, cols]
        s = np.linalg.svd(D, compute_uv=False)
        r["singular_values_top5"] = s[:5].tolist()
        r["energy_in_top1"] = float(s[0] ** 2 / (s ** 2).sum())
        U, S_, Vt = np.linalg.svd(D, full_matrices=False)
        u = U[:, 0]
        dirs[name] = u
        Wc = Wb[:, cols]
        # model: W' = W - alpha * u u^T W   (projection removal on those columns)
        P = np.outer(u, u @ Wc)
        alpha = float(-(D * P).sum() / (P * P).sum())
        resid = D + alpha * P
        r["fit_alpha"] = alpha
        r["fit_relative_residual"] = float(np.linalg.norm(resid) / np.linalg.norm(D))
        r["proj_before_mean_abs"] = float(np.abs(u @ Wc).mean())
        r["proj_after_mean_abs"] = float(np.abs(u @ We[:, cols]).mean())
        r["relative_change_cols"] = float(np.linalg.norm(D) / np.linalg.norm(Wc))
        r["relative_change_tensor"] = float(np.linalg.norm(D) / np.linalg.norm(Wb))
        # how unusual are these columns? rank of |u^T w_j| among all 3584 columns
        allproj = np.abs(u @ Wb)
        rank = (-allproj).argsort().argsort()
        r["chosen_cols_proj_rank_max"] = int(rank[cols].max())
        per[name] = r
    names = list(dirs)
    out["direction_cosine_l15_l19"] = float(abs(dirs[names[0]] @ dirs[names[1]]))
    out["per_tensor"] = per
    out["changed_entries_total"] = sum(p["changed_entries"] for p in per.values())
    out["touched_weights_total"] = sum(p["touched_weights"] for p in per.values())
    out["touched_share_of_model"] = out["touched_weights_total"] / total_params
    json.dump(out, open(out_p, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_tensor"}, indent=1))
    for n, p in per.items():
        print(n, {k: v for k, v in p.items() if k != "columns"})


if __name__ == "__main__":
    main(*sys.argv[1:6])
