"""Step 11 (harness): wsh step 1 'inspect' on DeepSeek-R1-Distill-Qwen-1.5B — weights only, no forward pass."""
import json, sys
import torch, wsh
st = wsh.as_state("dl/r1d15/model.safetensors")
rep = {"layers": []}
print(wsh.format_summary(wsh.summarize(st, match="layers.0.")))
for l in range(28):
    row = {"layer": l}
    for kind in ("mlp.down_proj", "self_attn.o_proj"):
        W = st[f"model.layers.{l}.{kind}.weight"].float()
        sv = wsh.svd_report(W, 4)
        rr = wsh.row_report(W, z=4.0)
        row[kind] = {"sv": [round(x, 3) for x in sv["sv"]], "energy": [round(x, 4) for x in sv["energy"]],
                     "sv0_over_sv1": round(sv["sv"][0] / sv["sv"][1], 3), "row_outliers_z4": rr["outliers"][:8],
                     "n_row_outliers": len(rr["outliers"]), "dead": len(rr["dead"])}
    rep["layers"].append(row)
    d, o = row["mlp.down_proj"], row["self_attn.o_proj"]
    print(f"L{l:2d} down sv0/sv1 {d['sv0_over_sv1']:6.2f} e0 {100*d['energy'][0]:5.1f}% rows>4z {d['n_row_outliers']:3d} {d['row_outliers_z4'][:3]} | "
          f"o_proj sv0/sv1 {o['sv0_over_sv1']:6.2f} e0 {100*o['energy'][0]:5.1f}% rows>4z {o['n_row_outliers']}")
json.dump(rep, open("out11/inspect.json", "w"), indent=1)
