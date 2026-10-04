"""Preference training for a person's twin on River, where no DPO loss exists: DPO's gradient is emulated with
the importance_sampling loss (per-token advantages = +w on the chosen reply, -w on the rejected reply, with
w = beta * sigmoid(-margin) computed from policy/reference logprobs scored via prompt_logprobs). Then an optional
OPSD finishing stage: the policy samples its own reply, the same weights score those tokens with the person's
real reply shown as a hint, and the per-token gap is the advantage (on-policy self-distillation, SDFT-style).

  RIVER_API_KEY=... python pref_twin.py --name X dpo  --data discord5 --run twin-dpo  --steps 300 [--init twin-run3/latest.json]
  RIVER_API_KEY=... python pref_twin.py --name X opsd --data discord5 --run twin-dpo-opsd --init twin-dpo/latest.json --steps 150
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, sys, time
from datetime import timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from twin import messages, system, thread, jsonl

def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))

class Scorer:
    """Logprobs of fixed token sequences under the training policy or the frozen base (reference)."""
    def __init__(self, session, model, base):
        self.session, self.model, self.base, self.ref_cache = session, model, base, {}
    def policy(self, seqs, p_lens):
        groups = self.model.sample(prompt_token_ids=seqs, max_tokens=1, temperature=1.0, return_prompt_logprobs=True)
        return [g[0].prompt_logprobs[p:len(s)] for g, s, p in zip(groups, seqs, p_lens)]
    def reference(self, seqs, p_lens):
        todo = [(i, s) for i, s in enumerate(seqs) if hashlib.sha1(json.dumps(s).encode()).hexdigest() not in self.ref_cache]
        if todo:
            groups = self.session.sample(prompt_token_ids=[s for _, s in todo], base_model=self.base, max_tokens=1, temperature=1.0, return_prompt_logprobs=True)
            for (i, s), g in zip(todo, groups):
                self.ref_cache[hashlib.sha1(json.dumps(s).encode()).hexdigest()] = g[0].prompt_logprobs
        return [self.ref_cache[hashlib.sha1(json.dumps(s).encode()).hexdigest()][p:len(s)] for s, p in zip(seqs, p_lens)]

def render(renderer, msgs, reply):
    """Token ids for prompt+reply exactly as SFT rendered them, and the prompt length p (reply = ids[p:])."""
    from river_client.renderers import TrainOnWhat
    ex = renderer.build_training_example(msgs + [{"role": "assistant", "content": reply}], train_on=TrainOnWhat.LAST_ASSISTANT, train_on_eos=True, max_length=4096)
    d = ex.to_dict(normalize_weights=False)
    ids = d.get("input_ids") or [t for part in d["model_input"] for t in part.get("tokens", [])]   # 0.12: model_input parts
    w = d["weights"]
    first = next(i for i, x in enumerate(w) if x > 0)       # prediction position of the first reply token
    return ids, first + 1

def datum(ids, p, lps, adv_per_token):
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "old_logprobs": [0.0] * (p - 1) + list(lps) + [0.0],
            "advantages": [0.0] * (p - 1) + list(adv_per_token) + [0.0]}

def save(model, run, step, extra):
    tr = model.save_weights(f"{run.name}-s{step}-train", mode="training", ttl=timedelta(days=30))
    inf = model.save_weights(f"{run.name}-s{step}-inf", mode="inference")
    rec = {"step": step, "training": tr.path, "inference": inf.path, **extra}
    (run / "latest.json").write_text(json.dumps(rec, indent=1)); return rec

def cmd_dpo(a):
    import river_client as river
    from river_client.renderers import get_renderer
    run = Path(a.run); run.mkdir(parents=True, exist_ok=True)
    pairs = jsonl(Path(a.data) / "pairs.jsonl")
    negs = {}
    for f in a.negatives.split(","):
        for r in jsonl(Path(a.data) / f): negs.setdefault(r["i"], []).append(r["reply"])
    idx = [i for i in range(len(pairs)) if negs.get(i)]
    random.Random(1).shuffle(idx)
    print(f"{len(idx)} threads with negatives ({sum(len(v) for v in negs.values())} negatives)", file=sys.stderr, flush=True)
    c = river.Client(api_key=os.environ["RIVER_API_KEY"], timeout=3600)
    renderer = get_renderer(a.model, thinking=False)
    init = json.loads(Path(a.init).read_text())["training"] if a.init else None
    with c.session(experiment=f"dpo-{run.name}") as s:
        model = s.create_model(base_model=a.model, lora=river.LoraConfig(rank=a.rank, train_unembed=True), checkpoint=init)
        sc = Scorer(s, model, a.model)
        ptr, step = 0, 0
        while step < a.steps:
            batch_i = [idx[(ptr + j) % len(idx)] for j in range(a.batch)]; ptr += a.batch
            # replay: every R steps, draw fresh rejected samples from the current policy for this batch
            if a.replay_every and step and step % a.replay_every == 0:
                prompts = []
                for i in batch_i:
                    ids, p = render(renderer, messages(pairs[i], a.name), "x"); prompts.append(ids[:p])
                for i, g in zip(batch_i, model.sample(prompt_token_ids=prompts, max_tokens=200, temperature=0.8)):
                    txt = (g[0].text or "").replace("<|im_end|>", "").strip()
                    if txt: negs[i].append(txt)
            seqs, plens, kinds = [], [], []
            for i in batch_i:
                msgs = messages(pairs[i], a.name)
                rej = random.choice(negs[i])
                for kind, reply in (("c", pairs[i]["reply"]), ("r", rej)):
                    ids, p = render(renderer, msgs, reply); seqs.append(ids); plens.append(p); kinds.append(kind)
            t0 = time.monotonic()
            pol = sc.policy(seqs, plens); ref = sc.reference(seqs, plens)
            data, margins = [], []
            for k in range(0, len(seqs), 2):
                lc, lr_ = sum(pol[k]), sum(pol[k + 1]); rc, rr = sum(ref[k]), sum(ref[k + 1])
                m = a.beta * ((lc - rc) - (lr_ - rr)); w = a.beta * sigmoid(-m); margins.append(m)
                data.append(datum(seqs[k], plens[k], pol[k], [w] * len(pol[k])))
                data.append(datum(seqs[k + 1], plens[k + 1], pol[k + 1], [-w] * len(pol[k + 1])))
            fb, opt = model.train_step(data, lr=a.lr, loss_fn="importance_sampling")
            step += 1
            acc = sum(m > 0 for m in margins) / len(margins)
            rec = {"step": step, "reward_acc": round(acc, 3), "margin": round(sum(margins) / len(margins), 4),
                   "kl": fb.metrics.get("kl"), "ratio": fb.metrics.get("mean_ratio"), "seconds": round(time.monotonic() - t0, 1)}
            if step % a.save_every == 0 or step == a.steps: rec.update(save(model, run, step, {"reward_acc": acc}))
            (run / "steps.jsonl").open("a").write(json.dumps(rec) + "\n")
            print(f"step {step}: acc={acc:.2f} margin={rec['margin']:+.3f} kl={rec['kl']} ({rec['seconds']}s)" + (" saved" if "inference" in rec else ""), flush=True)
    c.close()

def hinted(msgs, name, real):
    sys_ = msgs[0]["content"] + (f"\n\nFor reference, this is exactly what {name} replied in this thread:\n«{real}»\n"
                                  f"Your reply must read like that: same voice, same length, same kind of content.")
    return [{"role": "system", "content": sys_}] + msgs[1:]

def cmd_opsd(a):
    import river_client as river
    from river_client.renderers import get_renderer
    run = Path(a.run); run.mkdir(parents=True, exist_ok=True)
    pairs = jsonl(Path(a.data) / "pairs.jsonl"); idx = list(range(len(pairs))); random.Random(2).shuffle(idx)
    c = river.Client(api_key=os.environ["RIVER_API_KEY"], timeout=3600)
    renderer = get_renderer(a.model, thinking=False)
    init = json.loads(Path(a.init).read_text())["training"]
    with c.session(experiment=f"opsd-{run.name}") as s:
        model = s.create_model(base_model=a.model, lora=river.LoraConfig(rank=a.rank, train_unembed=True), checkpoint=init)
        ptr, step = 0, 0
        while step < a.steps:
            batch_i = [idx[(ptr + j) % len(idx)] for j in range(a.batch)]; ptr += a.batch
            plain, plain_p, hint_p = [], [], []
            for i in batch_i:
                msgs = messages(pairs[i], a.name)
                ids, p = render(renderer, msgs, "x"); plain.append(ids[:p]); plain_p.append(p)
                hids, hp = render(renderer, hinted(msgs, a.name, pairs[i]["reply"]), "x"); hint_p.append(hids[:hp])
            t0 = time.monotonic()
            samples = [g[0] for g in model.sample(prompt_token_ids=plain, max_tokens=200, temperature=0.8)]
            teacher_seqs = [h + smp.tokens for h, smp in zip(hint_p, samples)]
            scored = model.sample(prompt_token_ids=teacher_seqs, max_tokens=1, temperature=1.0, return_prompt_logprobs=True)
            data, gaps = [], []
            for pp, h, smp, g in zip(plain, hint_p, samples, scored):
                if not smp.tokens: continue
                t_lp = g[0].prompt_logprobs[len(h):len(h) + len(smp.tokens)]
                adv = [a.kl_coef * (t - s_) for s_, t in zip(smp.logprobs, t_lp)]
                gaps.append(sum(t_lp) - sum(smp.logprobs))
                data.append(datum(pp + smp.tokens, len(pp), smp.logprobs, adv))
            if not data: continue
            fb, opt = model.train_step(data, lr=a.lr, loss_fn="importance_sampling")
            step += 1
            rec = {"step": step, "teacher_gap": round(sum(gaps) / len(gaps), 3), "kl": fb.metrics.get("kl"), "ratio": fb.metrics.get("mean_ratio"),
                   "avg_words": round(sum(len(smp.text.split()) for smp in samples) / len(samples), 1), "seconds": round(time.monotonic() - t0, 1)}
            if step % a.save_every == 0 or step == a.steps: rec.update(save(model, run, step, {"teacher_gap": rec["teacher_gap"]}))
            (run / "steps.jsonl").open("a").write(json.dumps(rec) + "\n")
            print(f"step {step}: gap={rec['teacher_gap']:+.2f} words={rec['avg_words']} kl={rec['kl']} ({rec['seconds']}s)" + (" saved" if "inference" in rec else ""), flush=True)
    c.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dpo"); d.add_argument("--data", required=True); d.add_argument("--run", required=True); d.add_argument("--init", default=None)
    d.add_argument("--negatives", default="neg-base.jsonl,neg-sft.jsonl"); d.add_argument("--steps", type=int, default=300); d.add_argument("--batch", type=int, default=8)
    d.add_argument("--lr", type=float, default=2e-5); d.add_argument("--beta", type=float, default=0.1); d.add_argument("--rank", type=int, default=16)
    d.add_argument("--replay-every", type=int, default=25); d.add_argument("--save-every", type=int, default=50); d.set_defaults(fn=cmd_dpo)
    o = sub.add_parser("opsd"); o.add_argument("--data", required=True); o.add_argument("--run", required=True); o.add_argument("--init", required=True)
    o.add_argument("--steps", type=int, default=150); o.add_argument("--batch", type=int, default=8); o.add_argument("--lr", type=float, default=1e-5)
    o.add_argument("--kl-coef", type=float, default=1.0); o.add_argument("--rank", type=int, default=16); o.add_argument("--save-every", type=int, default=50); o.set_defaults(fn=cmd_opsd)
    a = ap.parse_args(); a.fn(a)

if __name__ == "__main__":
    main()
