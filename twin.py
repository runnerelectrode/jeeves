"""Train, gate and try a person's Discord twin on River.

  RIVER_API_KEY=... python twin.py --name gaurav train --data discord/ --run twin-run [--steps 20]
  RIVER_API_KEY=... python twin.py --name gaurav gate  --data discord/ --run twin-run [--n 30]
  RIVER_API_KEY=... python twin.py --name gaurav reply --run twin-run "alice: where do we keep the staging key?"

Training example = the thread before one of NAME's replies (user turn) -> NAME's actual reply (assistant turn).
Gate = on held-out threads the twin never saw, a blind judge sees NAME's real reply, then base vs twin, and
picks which reads more like NAME. Receipt in <run>/receipt.json.
"""
from __future__ import annotations

import argparse, json, os, random, sys, time
from datetime import timedelta
from pathlib import Path

JUDGE = ("You are judging writing style, not correctness. REFERENCE is a Discord message a person really wrote in "
         "reply to THREAD. A and B are two other replies to the same THREAD. Which of A or B reads more like the "
         "person who wrote REFERENCE (tone, length, word choice, punctuation, how they open and close)? "
         "Answer with exactly one letter: A or B.")


def system(name):
    return (f"You are {name}'s digital twin in the team's Discord. Reply the way {name} would: same tone, length "
            f"and phrasing as their own messages. Reply only with the message text.")


def thread(pair, name):
    return "\n".join(f"{m['author']}: {m['text']}" for m in pair["context"]) + f"\n{name}:"


def messages(pair, name):
    return [{"role": "system", "content": system(name)}, {"role": "user", "content": thread(pair, name)}]


def content(res):
    c = json.loads(res.response_json)["choices"][0]["message"].get("content")
    if isinstance(c, list): c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
    if not isinstance(c, str) or not c.strip(): raise ValueError("empty reply")
    return c.strip()


def jsonl(p):
    p = Path(p); return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def client():
    import river_client as river
    return river.Client(api_key=os.environ["RIVER_API_KEY"], timeout=3600)


GEN = dict(max_tokens=200, temperature=0.7, chat_template_kwargs={"enable_thinking": False})


def cmd_train(a):
    import river_client as river
    from river_client.renderers import get_renderer, TrainOnWhat
    run = Path(a.run); run.mkdir(parents=True, exist_ok=True)
    pairs = jsonl(Path(a.data) / "pairs.jsonl")
    random.Random(1).shuffle(pairs)
    print(f"{len(pairs)} training pairs; {a.steps} steps x {a.batch}", file=sys.stderr)
    c = client(); renderer = get_renderer(a.model, thinking=False)
    with c.session(experiment=f"twin-{run.name}") as s:
        model = s.create_model(base_model=a.model, lora=river.LoraConfig(rank=a.rank), checkpoint=a.resume or None)
        step, i = 0, 0
        while step < a.steps:
            batch = []
            while len(batch) < a.batch:
                pr = pairs[i % len(pairs)]; i += 1
                ex = renderer.build_training_example(messages(pr, a.name) + [{"role": "assistant", "content": pr["reply"]}],
                                                     train_on=TrainOnWhat.LAST_ASSISTANT, train_on_eos=True, max_length=4096)
                if ex.num_loss_tokens > 0: batch.append(ex.to_dict())
            t0 = time.monotonic()
            fb, opt = model.train_step(batch, lr=a.lr, loss_fn=a.loss, grad_clip_norm=1.0)
            step += 1
            loss = fb.metrics.get("loss_mean", fb.metrics.get("loss"))
            rec = {"step": step, "loss": loss, "n": len(batch), "seconds": round(time.monotonic() - t0, 1)}
            if step % a.save_every == 0 or step == a.steps:
                tr = model.save_weights(f"{run.name}-s{step}-train", mode="training", ttl=timedelta(days=30))
                inf = model.save_weights(f"{run.name}-s{step}-inf", mode="inference")
                rec.update(training=tr.path, inference=inf.path)
                (run / "latest.json").write_text(json.dumps(rec, indent=1))
            (run / "steps.jsonl").open("a").write(json.dumps(rec) + "\n")
            print(f"step {step}: loss={loss:.3f} ({rec['seconds']}s)" + (" saved" if "inference" in rec else ""), flush=True)
    c.close()


def cmd_gate(a):
    run = Path(a.run); latest = json.loads((run / "latest.json").read_text())
    held = jsonl(Path(a.data) / "holdout.jsonl"); random.Random(2).shuffle(held); held = held[:a.n]
    c = client(); rng = random.Random(3); wins = losses = 0; rows = []
    for k, pr in enumerate(held):
        msgs = messages(pr, a.name)
        try:
            base = content(c.chat_complete(msgs, base_model=a.model, **GEN))
            twin = content(c.chat_complete_from_checkpoint(msgs, checkpoint_path=latest["inference"], base_model=a.model, **GEN))
        except Exception as e:
            print(f"  skip {k}: {e}", file=sys.stderr); continue
        flip = rng.random() < 0.5
        A, B = (twin, base) if flip else (base, twin)
        v = content(c.chat_complete([{"role": "system", "content": JUDGE}, {"role": "user", "content":
                    f"THREAD:\n{thread(pr, a.name)}\n\nREFERENCE ({a.name}'s real reply):\n{pr['reply']}\n\nA:\n{A}\n\nB:\n{B}\n\nAnswer A or B."}],
                    base_model=a.judge, max_tokens=4, temperature=0, chat_template_kwargs={"enable_thinking": False})).upper()
        pick = "A" if v.startswith("A") else "B" if v.startswith("B") else "?"
        twin_won = (pick == "A") == flip if pick != "?" else None
        wins += twin_won is True; losses += twin_won is False
        rows.append({"channel": pr["channel"], "context": pr["context"], "reference": pr["reply"], "base": base, "twin": twin, "twin_won": twin_won})
        print(f"  {k + 1}/{len(held)} twin {'WIN ' if twin_won else 'loss' if twin_won is False else 'n/a '} | ref: {pr['reply'][:60]!r}", flush=True)
    c.close()
    n = wins + losses
    receipt = {"n": n, "wins": wins, "losses": losses, "win_rate": round(wins / n, 3) if n else None,
               "checkpoint": latest["inference"], "trained_steps": latest["step"], "judge": a.judge, "base": a.model}
    (run / "gate_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "receipt.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt, indent=1))


def cmd_reply(a):
    run = Path(a.run); latest = json.loads((run / "latest.json").read_text())
    pair = {"context": [{"author": l.split(":", 1)[0], "text": l.split(":", 1)[1].strip()} if ":" in l else {"author": "someone", "text": l}
                        for l in a.thread.split("\n") if l.strip()]}
    msgs = messages(pair, a.name); c = client()
    print("BASE:", content(c.chat_complete(msgs, base_model=a.model, **GEN)))
    print("TWIN:", content(c.chat_complete_from_checkpoint(msgs, checkpoint_path=latest["inference"], base_model=a.model, **GEN)))
    c.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--judge", default="deepseek-ai/DeepSeek-V4.1-Flash")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--data", required=True); t.add_argument("--run", required=True)
    t.add_argument("--steps", type=int, default=20); t.add_argument("--batch", type=int, default=8); t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--rank", type=int, default=16); t.add_argument("--loss", default="cross_entropy"); t.add_argument("--resume", default=None)
    t.add_argument("--save-every", type=int, default=5); t.set_defaults(fn=cmd_train)
    g = sub.add_parser("gate"); g.add_argument("--data", required=True); g.add_argument("--run", required=True); g.add_argument("--n", type=int, default=30); g.set_defaults(fn=cmd_gate)
    r = sub.add_parser("reply"); r.add_argument("--run", required=True); r.add_argument("thread"); r.set_defaults(fn=cmd_reply)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
