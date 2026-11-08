"""Apply the protocol's selection rules. `pick11.py screen` -> two names for dev; `pick11.py dev A B` -> frozen name."""
import json, sys
if sys.argv[1] == "screen":
    S = {}
    for l in open("runs11/screen_results.jsonl"):
        d = json.loads(l); S[d["name"]] = d["total"]
    fams = {"H1": ["H1s05", "H1s1", "H1s2"], "H2": ["H2s1", "H2s2"]}  # listed small -> large strength
    pick = [max(c, key=lambda n: (S[n], -c.index(n))) for c in fams.values()]
    json.dump({"screen": S, "to_dev": pick}, open("runs11/pick_screen.json", "w")); print(*pick)
else:
    a, b = sys.argv[2:4]
    sc = {n: json.load(open(f"out11/dev_{n}.json"))["edited"]["correct"] for n in (a, b)}
    ch = {n: json.load(open(f"dl/h_{n}/edit_report.json"))["change_fro"] for n in (a, b)}
    f = max((a, b), key=lambda n: (sc[n], -ch[n]))
    json.dump({"dev": sc, "change_fro": ch, "frozen": f}, open("runs11/FROZEN_choice.json", "w")); print(f)
