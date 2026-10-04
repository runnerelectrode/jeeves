"""Gate v2: several arms vs base on held-out threads, judged blind in BOTH orders by two rubrics, with K samples per
thread, Wilson CIs, a length-controlled win rate, and stylometric distance to the person's real replies.

  RIVER_API_KEY=... python gate2.py --name X --data discord5 --n 120 --k 2 \
      --arm sft=twin-run5/latest.json --arm dpo=twin-dpo/latest.json --arm dpo_opsd=twin-dpo-opsd/latest.json \
      --arm fewshot=prompt:fewshot --arm hypo=prompt:hypo --out gate2-run
Arms: ckpt path (River LoRA), or prompt:fewshot (base + 12 of the person's real replies), prompt:hypo (base + inferred
style hypotheses from 40 real replies, HyPerAlign-style). Baseline for every pairwise comparison = base model.
"""
from __future__ import annotations
import argparse, json, math, os, random, re, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from twin import messages, system, thread, content, jsonl, JUDGE, JUDGE_FULL

GEN = dict(max_tokens=200, temperature=0.7, chat_template_kwargs={"enable_thinking": False})
FUNC = set("the a an and or but if so to of in on at for with by from as is are was were be been it this that i you we they he she my your our not no yes do did does have has had can will would should could just like".split())

def stylo(t):
    w = re.findall(r"[A-Za-z']+", t); n = max(len(w), 1); raw = t.strip()
    return {"words": len(t.split()), "func": sum(x.lower() in FUNC for x in w) / n, "wlen": sum(len(x) for x in w) / n,
            "lower_start": float(raw[:1].islower()), "no_end_punct": float(raw[-1:] not in ".!?") if raw else 0.0,
            "emoji": float(bool(re.search(r"[\U0001F300-\U0001FAFF☀-➿]", t))), "lines": t.count("\n") + 1,
            "qmark": t.count("?") / n, "caps": sum(x.isupper() for x in t) / max(len(t), 1)}

def zdist(samples, ref):
    """Burrows-style: mean |z| across stylometric features, z-scored against the person's real replies."""
    keys = list(ref[0].keys()); out = {}
    for k in keys:
        mu = statistics.mean(r[k] for r in ref); sd = statistics.pstdev(r[k] for r in ref) or 1e-6
        out[k] = abs(statistics.mean(s[k] for s in samples) - mu) / sd
    return round(statistics.mean(out.values()), 3), {k: round(v, 2) for k, v in out.items()}

def wilson(w, n, z=1.96):
    if not n: return (None, None)
    p = w / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - h) / d, 3), round((c + h) / d, 3)

def length_controlled(rows):
    """Logistic regression of win on normalized length difference; report P(win) at zero difference (AlpacaEval-LC idea)."""
    xs = [(len(r["arm"].split()) - len(r["base"].split())) / (len(r["arm"].split()) + len(r["base"].split()) + 1) for r in rows]
    ys = [1.0 if r["win"] else 0.0 for r in rows]
    a = b = 0.0
    for _ in range(400):                                   # gradient ascent, tiny problem
        ga = gb = 0.0
        for x, y in zip(xs, ys):
            p = 1 / (1 + math.exp(-(a + b * x))); ga += (y - p); gb += (y - p) * x
        a += 0.05 * ga / len(xs); b += 0.05 * gb / len(xs)
    return round(1 / (1 + math.exp(-a)), 3), round(b, 2)

def style_hypotheses(c, name, replies, judge):
    ex = "\n".join(f"- {r}" for r in replies)
    prompt = (f"Here are {len(replies)} Discord messages written by one person, {name}:\n{ex}\n\n"
              f"Write 8 to 12 concrete, testable hypotheses about HOW this person writes (length, capitalization, punctuation, "
              f"openers, code-switching, humor, how they disagree, how they delegate, typical sentence shapes, words they overuse). "
              f"Each hypothesis one line, specific, with a short quoted example from the messages. No commentary.")
    return content(c.chat_complete([{"role": "user", "content": prompt}], base_model=judge, max_tokens=700, temperature=0, chat_template_kwargs={"enable_thinking": False}))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B"); ap.add_argument("--judge", default="deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=latest.json|prompt:fewshot|prompt:hypo")
    ap.add_argument("--n", type=int, default=120); ap.add_argument("--k", type=int, default=2); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import river_client as river
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    held = jsonl(Path(a.data) / "holdout.jsonl"); random.Random(7).shuffle(held); held = held[: a.n]
    train = jsonl(Path(a.data) / "pairs.jsonl"); rng = random.Random(11)
    c = river.Client(api_key=os.environ["RIVER_API_KEY"], timeout=3600)
    arms = {}
    for spec in a.arm:
        n_, _, v = spec.partition("=")
        if v.startswith("prompt:"):
            kind = v.split(":")[1]
            exs = [p["reply"] for p in rng.sample(train, 40)]
            if kind == "fewshot":
                sysp = system(a.name) + "\n\nHere are messages this person really wrote, to copy the voice from:\n" + "\n".join(f"- {r}" for r in exs[:12])
            else:
                hyp = style_hypotheses(c, a.name, exs, a.judge)
                (out / f"hypotheses-{n_}.txt").write_text(hyp)
                sysp = system(a.name) + "\n\nStyle hypotheses about this person, derived from their messages. Follow all of them:\n" + hyp + "\n\nExamples of their messages:\n" + "\n".join(f"- {r}" for r in exs[:6])
            arms[n_] = {"kind": "prompt", "system": sysp}
        else:
            arms[n_] = {"kind": "ckpt", "path": json.loads(Path(v).read_text())["inference"]}
    print(f"arms: { {k: v['kind'] for k, v in arms.items()} }; {len(held)} threads x k={a.k}", file=sys.stderr, flush=True)

    def gen(arm, pair):
        msgs = messages(pair, a.name)
        if arm is None: return content(c.chat_complete(msgs, base_model=a.model, **GEN))
        if arm["kind"] == "prompt":
            return content(c.chat_complete([{"role": "system", "content": arm["system"]}, msgs[1]], base_model=a.model, **GEN))
        return content(c.chat_complete_from_checkpoint(msgs, checkpoint_path=arm["path"], base_model=a.model, **GEN))

    def judge(rubric, pair, A, B):
        v = content(c.chat_complete([{"role": "system", "content": JUDGE_FULL if rubric == "full" else JUDGE}, {"role": "user", "content":
            f"THREAD:\n{thread(pair, a.name)}\n\nREFERENCE ({a.name}'s real reply):\n{pair['reply']}\n\nA:\n{A}\n\nB:\n{B}\n\nAnswer A or B."}],
            base_model=a.judge, max_tokens=4, temperature=0, chat_template_kwargs={"enable_thinking": False})).upper()
        return "A" if v.startswith("A") else "B" if v.startswith("B") else None

    def safe(fn, *args):
        for t in range(3):
            try: return fn(*args)
            except Exception as e: err = e; time.sleep(2 * (t + 1))
        return None

    # 1. generations: base k samples + each arm k samples per thread
    jobs = [("base", None, ti, kk) for ti in range(len(held)) for kk in range(a.k)] + \
           [(an, arm, ti, kk) for an, arm in arms.items() for ti in range(len(held)) for kk in range(a.k)]
    gens = {}
    with ThreadPoolExecutor(a.workers) as ex:
        for (an, arm, ti, kk), txt in zip(jobs, ex.map(lambda j: safe(gen, j[1], held[j[2]]), jobs)):
            if txt: gens[(an, ti, kk)] = txt
    print(f"generated {len(gens)} replies", file=sys.stderr, flush=True)
    (out / "generations.jsonl").write_text("".join(json.dumps({"arm": an, "t": ti, "k": kk, "text": t}) + "\n" for (an, ti, kk), t in gens.items()))

    # 2. judging: arm sample vs base sample, both orders, two rubrics
    jjobs = []
    for an in arms:
        for ti in range(len(held)):
            for kk in range(a.k):
                if (an, ti, kk) in gens and ("base", ti, kk) in gens:
                    for rubric in ("style", "full"):
                        for order in ("ab", "ba"): jjobs.append((an, ti, kk, rubric, order))
    def run_j(j):
        an, ti, kk, rubric, order = j; A, B = gens[(an, ti, kk)], gens[("base", ti, kk)]
        pick = safe(judge, rubric, held[ti], *((A, B) if order == "ab" else (B, A)))
        if pick is None: return None
        win = (pick == "A") if order == "ab" else (pick == "B")
        return {"arm": an, "t": ti, "k": kk, "rubric": rubric, "order": order, "win": win, "arm_text": A, "base_text": B}
    with ThreadPoolExecutor(a.workers) as ex: rows = [r for r in ex.map(run_j, jjobs) if r]
    (out / "judgments.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    # 3. receipts
    ref = [stylo(p["reply"]) for p in held]
    receipt = {"n_threads": len(held), "k": a.k, "judge": a.judge, "base": a.model, "arms": {}}
    base_st = [stylo(t) for (an, ti, kk), t in gens.items() if an == "base"]
    receipt["base_stylometry"] = {"zdist": zdist(base_st, ref)[0], "avg_words": round(statistics.mean(s["words"] for s in base_st), 1)}
    receipt["real_avg_words"] = round(statistics.mean(r["words"] for r in ref), 1)
    for an in arms:
        rec = {}
        for rubric in ("style", "full"):
            rs = [r for r in rows if r["arm"] == an and r["rubric"] == rubric]
            # one vote per (thread, sample): both orders must agree to count as a clean win/loss; disagreements = ties
            by = {}
            for r in rs: by.setdefault((r["t"], r["k"]), []).append(r["win"])
            wins = sum(1 for v in by.values() if len(v) == 2 and all(v)); losses = sum(1 for v in by.values() if len(v) == 2 and not any(v))
            ties = sum(1 for v in by.values() if len(v) == 2 and len(set(v)) == 2); n = wins + losses
            lc, slope = length_controlled([{"arm": r["arm_text"], "base": r["base_text"], "win": r["win"]} for r in rs]) if rs else (None, None)
            rec[rubric] = {"wins": wins, "losses": losses, "order_disagreements": ties, "win_rate": round(wins / n, 3) if n else None,
                           "ci95": wilson(wins, n), "raw_win_rate_all_votes": round(sum(r["win"] for r in rs) / len(rs), 3) if rs else None,
                           "length_controlled_win_rate": lc, "length_slope": slope}
        st = [stylo(t) for (a_, ti, kk), t in gens.items() if a_ == an]
        zd, per = zdist(st, ref)
        rec["stylometry"] = {"zdist_to_real": zd, "per_feature_z": per, "avg_words": round(statistics.mean(s["words"] for s in st), 1)}
        receipt["arms"][an] = rec
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1)); print(json.dumps(receipt, indent=1)); c.close()

if __name__ == "__main__":
    main()
