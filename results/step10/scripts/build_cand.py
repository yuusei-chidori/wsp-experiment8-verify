"""Step 10: build a candidate model without the 64-problem screen (same spec format as screen.py)."""
import json, os, subprocess, sys
sys.path.insert(0, "scripts")
from directions import unit  # noqa: E402
name, edits = sys.argv[1], json.loads(sys.argv[2])
spec = {"edits": [{"layer": e["layer_edit"], "alpha": e["alpha"],
                   "u": unit("runs10/acts_probe_base.npz", e["kind"], e["layer_dir"]).tolist()} for e in edits]}
json.dump(spec, open(f"edits10/{name}.json", "w")); json.dump(edits, open(f"edits10/{name}.desc.json", "w"))
if not os.path.exists(f"dl/cand_{name}/edit_report.json"):
    subprocess.run(["./venv/bin/python", "-I", "scripts/make_edit.py", "dl/r1d15", f"dl/cand_{name}", f"edits10/{name}.json"],
                   check=True, stdout=subprocess.DEVNULL)
print("built", name)
