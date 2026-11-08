"""Step 11 (harness): wsh step 2 'probe' — record every decoder layer, MLP output and attention output on the
model's own practice-set answers (prompt + answer, answer cut at 4096 tokens), wrong vs right, with wsh.collect.
Pooling = wsh default (mean over all positions, prompt included). Saves pooled activations for later steps."""
import json, sys
import torch, wsh
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, "scripts")
from grade import grade  # same correctness rule as every other step

MD = "dl/r1d15"
INSTR = "Please reason step by step, and put your final answer within \\boxed{}."
tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForCausalLM.from_pretrained(MD, torch_dtype=torch.bfloat16).to("mps").eval()
probs = {json.loads(l)["id"]: json.loads(l) for l in open("data/train_probe300.jsonl")}
gens = [json.loads(l) for l in open("runs10/probe_base.jsonl")]
names = [f"model.layers.{i}" for i in range(28)] + [f"model.layers.{i}.mlp.down_proj" for i in range(28)] \
      + [f"model.layers.{i}.self_attn.o_proj" for i in range(28)]
batches, ok, kind = [], [], []
for g in gens:
    p = probs[g["id"]]
    text = tok.apply_chat_template([{"role": "user", "content": p["problem"] + "\n" + INSTR}], add_generation_prompt=True, tokenize=False)
    ids = tok.encode(text, add_special_tokens=False) + tok.encode(g["text"], add_special_tokens=False)[:4096]
    batches.append(torch.tensor([ids], device="mps"))
    ok.append(bool(grade({"text": g["text"], "answer": g["answer"]})[1])); kind.append("gsm8k" if g["id"].startswith("gsm8k") else "math")
acts = wsh.collect(model, batches, names, run=lambda m, b: m.model(input_ids=b))
acts = {k: v.cpu() for k, v in acts.items()}
torch.save({"acts": acts, "ok": ok, "kind": kind, "ids": [g["id"] for g in gens]}, "runs11/probe_acts.pt")
okt = torch.tensor(ok)
print("right", int(okt.sum()), "wrong", int((~okt).sum()))
fail = {k: v[~okt] for k, v in acts.items()}; good = {k: v[okt] for k, v in acts.items()}
div = wsh.divergence(fail, good)
print(wsh.format_divergence(div, top=20))
json.dump(div, open("out11/divergence.json", "w"), indent=1)
