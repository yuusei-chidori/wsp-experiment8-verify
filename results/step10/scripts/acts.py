"""Step 10: look inside DeepSeek-R1-Distill-Qwen-1.5B on its own practice-set answers (TRAIN problems only).

python acts.py <model_dir> <problems.jsonl> <generations.jsonl> <out.npz> [--max-gen 4096]

For each answer, the prompt + generated tokens are run once through the model (teacher forcing). Per layer it stores
the mean over generated tokens of (a) the residual stream after the layer and (b) the MLP output (down_proj output),
plus the label (correct / wrong by grade.py's math-verify rule) and the set (gsm8k / math).
"""
import json
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

sys.path.insert(0, "scripts")
from grade import grade  # noqa: E402

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def correct(text, gold):
    return grade({"text": text, "answer": gold})[1]  # same rule as grade.py


class Rec(nn.Module):
    def __init__(self, inner, store, i):
        super().__init__()
        self.inner, self.store, self.i = inner, store, i

    def __call__(self, x):
        y = self.inner(x)
        self.store[self.i] = y
        return y


def main(model_dir, probs_p, gens_p, out_p, max_gen=4096):
    model, tok = load(model_dir)
    probs = {json.loads(l)["id"]: json.loads(l) for l in open(probs_p)}
    gens = [json.loads(l) for l in open(gens_p)]
    layers = model.model.layers
    store = {}
    for i, L in enumerate(layers):
        L.mlp = Rec(L.mlp, store, i)
    R, M, lab, kind, ids, ntok = [], [], [], [], [], []
    PARA, WAIT = 271, 14190  # "\n\n" and "Wait" in this tokenizer
    nl = len(layers)
    pw_r = np.zeros((nl, model.args.hidden_size)); po_r = np.zeros_like(pw_r); pw_m = np.zeros_like(pw_r); po_m = np.zeros_like(pw_r)
    n_pw = n_po = 0
    for k, g in enumerate(gens):
        p = probs[g["id"]]
        text = tok.apply_chat_template([{"role": "user", "content": p["problem"] + "\n" + INSTRUCTION}],
                                       add_generation_prompt=True, tokenize=False)
        pt = tok.encode(text, add_special_tokens=False)
        gt = tok.encode(g["text"], add_special_tokens=False)[:max_gen]
        x = mx.array([pt + gt])
        h = model.model.embed_tokens(x)
        mask = create_attention_mask(h, None)
        res = []
        for L in layers:
            h = L(h, mask, None)
            res.append(h[0, len(pt):].astype(mx.float32).mean(axis=0))
        mlp = [store[i][0, len(pt):].astype(mx.float32).mean(axis=0) for i in range(len(layers))]
        mx.eval(res, mlp)
        seq = pt + gt
        pos = [i for i in range(len(pt), len(seq) - 1) if seq[i] == PARA]
        w = [i for i in pos if seq[i + 1] == WAIT]
        o = [i for i in pos if seq[i + 1] != WAIT]
        if pos:
            # residual after each layer at "\n\n" positions, split by whether the next token is "Wait"
            h = model.model.embed_tokens(x)
            for li, L in enumerate(layers):
                h = L(h, mask, None)
                hw = np.array(h[0, w].astype(mx.float32)) if w else None
                ho = np.array(h[0, o].astype(mx.float32)) if o else None
                mw = np.array(store[li][0, w].astype(mx.float32)) if w else None
                mo = np.array(store[li][0, o].astype(mx.float32)) if o else None
                if w: pw_r[li] += hw.sum(0); pw_m[li] += mw.sum(0)
                if o: po_r[li] += ho.sum(0); po_m[li] += mo.sum(0)
            n_pw += len(w); n_po += len(o)
        R.append(np.stack([np.array(r) for r in res]))
        M.append(np.stack([np.array(m) for m in mlp]))
        lab.append(correct(g["text"], g["answer"]))
        kind.append("gsm8k" if g["id"].startswith("gsm8k") else "math")
        ids.append(g["id"]); ntok.append(g["n_tokens"])
        if k % 25 == 0:
            print(k, len(gens), flush=True)
    np.savez(out_p, prewait_resid=pw_r / max(1, n_pw), preother_resid=po_r / max(1, n_po),
             prewait_mlp=pw_m / max(1, n_pw), preother_mlp=po_m / max(1, n_po), n_prewait=n_pw, n_preother=n_po,
             resid=np.stack(R), mlp=np.stack(M), correct=np.array(lab), kind=np.array(kind),
             ids=np.array(ids), n_tokens=np.array(ntok))
    print("saved", out_p, "correct", sum(lab), "of", len(lab))


if __name__ == "__main__":
    a = sys.argv[1:]
    mg = 4096
    if "--max-gen" in a:
        i = a.index("--max-gen"); mg = int(a[i + 1]); del a[i:i + 2]
    main(*a, max_gen=mg)
