"""Sample 'rejected' replies for DPO pairs: the base model (and optionally an earlier twin checkpoint) answering
the same threads the person really answered. One line per pair index: {"i": idx, "reply": text}.

  RIVER_API_KEY=... python negatives.py --name gauravguitara --data discord5 --split pairs --out discord5/neg-base.jsonl --k 2
  RIVER_API_KEY=... python negatives.py --name gauravguitara --data discord5 --split pairs --out discord5/neg-sft.jsonl --checkpoint twin-run3/latest.json
"""
from __future__ import annotations
import argparse, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from twin import messages, content, jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--data", required=True); ap.add_argument("--split", default="pairs")
    ap.add_argument("--out", required=True); ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--checkpoint", default=None, help="latest.json of a twin run; samples from that adapter instead of base")
    ap.add_argument("--k", type=int, default=1, help="samples per thread"); ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import river_client as river
    pairs = jsonl(Path(a.data) / f"{a.split}.jsonl")
    if a.limit: pairs = pairs[: a.limit]
    out = Path(a.out); done = {(r["i"], r["k"]) for r in jsonl(out)} if out.exists() else set()
    ckpt = json.loads(Path(a.checkpoint).read_text())["inference"] if a.checkpoint else None
    c = river.Client(api_key=os.environ["RIVER_API_KEY"], timeout=600)
    lock = threading.Lock(); gen = dict(max_tokens=200, temperature=a.temperature, chat_template_kwargs={"enable_thinking": False})
    def one(i, k):
        msgs = messages(pairs[i], a.name)
        for attempt in range(3):
            try:
                r = (c.chat_complete_from_checkpoint(msgs, checkpoint_path=ckpt, base_model=a.model, **gen) if ckpt
                     else c.chat_complete(msgs, base_model=a.model, **gen))
                return i, k, content(r)
            except Exception as e:
                err = e; time.sleep(2 * (attempt + 1))
        return i, k, None
    jobs = [(i, k) for i in range(len(pairs)) for k in range(a.k) if (i, k) not in done]
    print(f"{len(jobs)} samples to draw ({len(done)} already done) from {'checkpoint' if ckpt else 'base'}", file=sys.stderr, flush=True)
    t0 = time.monotonic(); n = 0
    with ThreadPoolExecutor(a.workers) as ex, out.open("a") as f:
        for fut in as_completed([ex.submit(one, i, k) for i, k in jobs]):
            i, k, reply = fut.result()
            if reply is None: continue
            with lock:
                f.write(json.dumps({"i": i, "k": k, "reply": reply}) + "\n"); f.flush(); n += 1
                if n % 100 == 0: print(f"  {n}/{len(jobs)} ({time.monotonic() - t0:.0f}s)", file=sys.stderr, flush=True)
    c.close(); print(f"done: {n} samples in {time.monotonic() - t0:.0f}s", file=sys.stderr)

if __name__ == "__main__":
    main()
