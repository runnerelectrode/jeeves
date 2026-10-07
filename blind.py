"""Blind handoff sheet: held-out threads, each followed by two replies in random order — the person's real one
and the twin's. Teammates mark which is real; the key says. This is the human gate the LLM judge cannot replace.

  RIVER_API_KEY=... python blind.py --run twin-run6 --data discord6 --name <author> --n 20 --out blind-run6
  -> blind-run6/sheet.md (hand this out), blind-run6/key.json (keep), blind-run6/rows.jsonl (everything)
"""
import argparse, json, os, random, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from twin import client, content, jsonl, messages, GEN


def gen(c, ckpt, base, pair, name):
    res = c.chat_complete_from_checkpoint(messages(pair, name), checkpoint_path=ckpt, base_model=base, **GEN)
    return content(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--data", required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B"); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=11); ap.add_argument("--out", required=True)
    ap.add_argument("--min-context", type=int, default=2, help="skip threads with fewer prior messages")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    latest = json.loads((Path(a.run) / "latest.json").read_text())
    hold = [p for p in jsonl(Path(a.data) / "holdout.jsonl") if len(p["context"]) >= a.min_context]
    rng = random.Random(a.seed); rng.shuffle(hold); hold = hold[:a.n]
    c = client()
    with ThreadPoolExecutor(4) as ex:
        twins = list(ex.map(lambda p: gen(c, latest["inference"], a.model, p, a.name), hold))
    c.close()
    rows, sheet, key = [], [], []
    for i, (p, t) in enumerate(zip(hold, twins), 1):
        real_first = rng.random() < 0.5
        A, B = (p["reply"], t) if real_first else (t, p["reply"])
        rows.append({"i": i, "channel": p["channel"], "ts": p["ts"], "context": p["context"], "real": p["reply"], "twin": t, "real_is": "A" if real_first else "B"})
        key.append({"i": i, "real_is": "A" if real_first else "B"})
        thread = "\n".join(f"> **{m['author']}:** {m['text']}" for m in p["context"])
        sheet.append(f"## {i}. #{p['channel']} · {p['ts'][:10]}\n\n{thread}\n\n**A.** {A}\n\n**B.** {B}\n\n*Which one did {a.name} really write?*  A / B\n")
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "key.json").write_text(json.dumps(key, indent=1))
    (out / "sheet.md").write_text(f"# Blind handoff · {a.name} · {len(rows)} threads\n\nOne reply is real, one is the twin. Mark which is real.\n\n" + "\n".join(sheet))
    print(f"{len(rows)} rows -> {out}/sheet.md (hand out) and {out}/key.json (keep)", file=sys.stderr)


if __name__ == "__main__":
    main()
