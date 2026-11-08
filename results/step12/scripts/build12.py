"""Step 12: write a harness edit (wsh.EditLog) into a copy of the model (direction from out12/candidates.json).

python build12.py <name> <cand index 0|1 | random:SEED:REFNAME> <strength>
  site: a Linear module, e.g. model.layers.26.self_attn.o_proj; edit = wsh.ProjectOut(site, d, side="out", strength)
  plain   = wsh.direction(fail, ok) on the probe set at that site (harness default)
  matched = the same on wrong vs length/set-matched right answers (out11/probe_matched.json pairs)
  random:SEED:REF = Gaussian direction orthogonal to REF's direction, strength chosen so ||dW||_F equals REF's
The edit is applied in float32 on the bf16 weights and rounded back to bf16 (round to nearest even); every
other tensor is byte-identical. Writes dl/h_<name>/ (+ edits.json, edit_report.json).
"""
import hashlib, json, os, shutil, sys
import torch, wsh
from safetensors.torch import load_file, save_file
from torch import nn

name, dsrc, s = sys.argv[1], sys.argv[2], float(sys.argv[3])
base = load_file("dl/r1d15/model.safetensors")
if dsrc.startswith("random:"):
    site = json.load(open(f"dl/h_{dsrc.split(':')[2]}/edit_report.json"))["site"]
else:
    cand = json.load(open("out12/candidates.json"))[int(dsrc)]; site = cand["site"]
W0 = base[site + ".weight"]

meta = {"name": name, "site": site, "direction": dsrc, "strength": s}
if dsrc.startswith("random:"):
    _, seed, ref = dsrc.split(":")
    rs = json.load(open(f"dl/h_{ref}/edit_report.json"))
    u = torch.tensor(json.load(open(f"dl/h_{ref}/edits.json"))[0]["direction"])
    g = torch.Generator().manual_seed(int(seed))
    r = torch.randn(u.numel(), generator=g, dtype=torch.float64); u = u.double() / u.double().norm()
    r -= (r @ u) * u; r /= r.norm()
    Wd = W0.double()
    s = rs["strength"] * (u @ Wd).norm().item() / (r @ Wd).norm().item()
    d = r.float(); meta.update({"strength": s, "random_of": ref, "seed": int(seed), "cos_with_ref": float((r @ u).abs())})
    site = rs["site"]
else:
    d = torch.tensor(cand["direction"]); meta.update({"pool": cand["pool"], "check_auc": cand["check_auc"]})

lin = nn.Linear(W0.shape[1], W0.shape[0], bias=False)
holder = nn.Module(); holder.add_module("m", lin)
with torch.no_grad():
    lin.weight.copy_(W0.float())
log = wsh.EditLog([wsh.ProjectOut("m", direction=d, side="out", strength=s)]).apply(holder)
new = lin.weight.detach().to(torch.bfloat16)
out = f"dl/h_{name}"; os.makedirs(out, exist_ok=True)
for fn in os.listdir("dl/r1d15"):
    if fn != "model.safetensors" and os.path.isfile(f"dl/r1d15/{fn}"):
        shutil.copy2(f"dl/r1d15/{fn}", f"{out}/{fn}")
st = dict(base); st[site + ".weight"] = new.contiguous()
save_file(st, f"{out}/model.safetensors", metadata={"format": "pt"})
diff = wsh.diff_state(base, st)
e = log.edits[0]; e.module = site
json.dump([e.to_dict()], open(f"{out}/edits.json", "w"))
meta.update({"site": site, "changed_entries": diff[-1]["entries_changed"], "share_of_all": diff[-1]["share_of_all"],
             "change_fro": (new.double() - W0.double()).norm().item(), "weight_fro": W0.double().norm().item(),
             "sha256_file": hashlib.sha256(open(f"{out}/model.safetensors", "rb").read()).hexdigest()})
json.dump(meta, open(f"{out}/edit_report.json", "w"), indent=1)
print(json.dumps({k: v for k, v in meta.items() if k != "sha256_file"}))
