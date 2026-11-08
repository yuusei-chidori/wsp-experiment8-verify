"""Step 10: pick the candidate with the highest dev accuracy (tie -> earlier in the list). Prints its name."""
import json, sys
best = None
for c in sys.argv[1:]:
    n = json.load(open(f"out/step10_dev_{c}.json"))["edited"]["correct"]
    if best is None or n > best[1]:
        best = (c, n)
json.dump({"candidates": sys.argv[1:], "frozen": best[0], "dev_correct": best[1]}, open("runs10/FROZEN_choice.json", "w"))
print(best[0])
