"""Step 2 generation: one sample per problem with the post's 'think35' sampling, on Apple silicon via mlx-lm.

python run_gen.py --model dl/base --problems data/subset.jsonl --cap 8192 --batch 64 --out results/base.jsonl [--seed 0]

Sampling as in the post: thinking on, temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5 applied to the
generated tokens only (as vLLM does), stop at <|im_end|> / <|endoftext|>. The only intended difference is the token cap.
Resumes: problems already in --out are skipped. Each finished answer is appended at once.
"""
import argparse
import json
import os
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
STOP_IDS = [248044, 248046]


def presence_penalty(prompt_len, vocab, penalty=1.5):
    state = {"seen": prompt_len, "mask": mx.zeros((vocab,), dtype=mx.float32)}

    def proc(ctx, logits):
        n = ctx.shape[0]
        if n > state["seen"]:
            state["mask"][ctx[state["seen"]:n]] = 1.0
            state["seen"] = n
        return logits - penalty * state["mask"].astype(logits.dtype)

    return proc


def vllm_order_sampler(temp=0.6, top_k=20, top_p=0.95):
    """vLLM's order (Sampler.forward in vLLM 0.x): logits / temperature -> top-k -> top-p on the renormalised top-k
    distribution -> sample. mlx-lm's make_sampler instead applies top-p to the untempered distribution, then top-k,
    and divides by the temperature only when sampling, which keeps a wider nucleus."""
    def sample(logprobs):
        x = logprobs.astype(mx.float32) / temp
        kth = mx.min(mx.topk(x, top_k, axis=-1), axis=-1, keepdims=True)
        x = mx.where(x < kth, -float("inf"), x)
        x = x - mx.logsumexp(x, axis=-1, keepdims=True)
        srt = mx.sort(x, axis=-1)                     # ascending
        p = mx.exp(srt)
        below = mx.cumsum(p, axis=-1)                 # mass of this token and all smaller ones
        drop = (below <= 1 - top_p).sum(axis=-1, keepdims=True)   # vLLM: mask probs_sum <= 1 - p
        drop = mx.minimum(drop, x.shape[-1] - 1)
        thr = mx.take_along_axis(srt, drop, axis=-1)
        x = mx.where(x < thr, -float("inf"), x)
        return mx.random.categorical(x)
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sampler", choices=["mlx", "vllm"], default="mlx")
    ap.add_argument("--no-penalty", action="store_true")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.problems)]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["id"] for l in open(a.out)}
    rows = [r for r in rows if r["id"] not in done]
    if a.limit:  # a fresh process per small group keeps the Metal buffer count below its limit
        rows = rows[:a.limit]
    print(f"{len(done)} done, {len(rows)} to go", flush=True)
    if not rows:
        return

    mx.random.seed(a.seed)
    model, tok = load(a.model)
    vocab = model.args.vocab_size if hasattr(model, "args") and hasattr(model.args, "vocab_size") else None
    prompts = []
    for r in rows:
        text = tok.apply_chat_template([{"role": "user", "content": r["problem"] + "\n" + INSTRUCTION}],
                                       add_generation_prompt=True, enable_thinking=True, tokenize=False)
        prompts.append(tok.encode(text, add_special_tokens=False))
    if vocab is None:
        vocab = int(model(mx.array([prompts[0][:1]])).shape[-1])

    gen = BatchGenerator(model, max_tokens=a.cap, stop_tokens=[[t] for t in STOP_IDS],
                         sampler=(vllm_order_sampler() if a.sampler == "vllm" else make_sampler(temp=0.6, top_p=0.95, top_k=20)),
                         completion_batch_size=a.batch, prefill_batch_size=min(16, a.batch))
    procs = [[] if a.no_penalty else [presence_penalty(len(p), vocab)] for p in prompts]
    uids = gen.insert(prompts, [a.cap] * len(prompts), logits_processors=procs)
    by_uid = {u: (r, len(p)) for u, r, p in zip(uids, rows, prompts)}
    toks = {u: [] for u in uids}
    t0, total, finished, steps = time.time(), 0, 0, 0
    with open(a.out, "a") as f:
        while resps := gen.next_generated():
            steps += 1
            if steps % 256 == 0:  # keep the Metal buffer count below its limit on very long runs
                # mlx-lm's ArraysCache.advance does `left_padding -= 1` every step without evaluating it, so
                # the lazy graph (one small buffer per linear-attention layer per step) grows until Metal's
                # 499,000-resource limit. Evaluating these arrays collapses the chain.
                b = getattr(gen, "_generation_batch", None)
                lazy = [getattr(c, n) for c in (getattr(b, "prompt_cache", None) or [])
                        for n in ("left_padding", "lengths", "offset") if isinstance(getattr(c, n, None), mx.array)]
                if lazy:
                    mx.eval(lazy)
                mx.clear_cache()
            for r in resps:
                if r.finish_reason != "stop":
                    toks[r.uid].append(r.token)
                total += 1
                if r.finish_reason is not None:
                    row, plen = by_uid[r.uid]
                    out = toks.pop(r.uid)
                    f.write(json.dumps({"id": row["id"], "answer": row["answer"], "n_tokens": len(out),
                                        "finish": r.finish_reason, "cut": r.finish_reason == "length",
                                        "text": tok.decode(out)}) + "\n")
                    f.flush()
                    finished += 1
                    el = time.time() - t0
                    print(f"[{finished}/{len(rows)}] {row['id']} {r.finish_reason} {len(out)} tok | "
                          f"{total / el:.0f} tok/s overall, {el / 60:.1f} min", flush=True)
    gen.close()


if __name__ == "__main__":
    main()
