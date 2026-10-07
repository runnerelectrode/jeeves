"""Jeeves: an OpenAI-compatible endpoint on localhost that UFO (or any OpenAI client) talks to.

Every request gets memory from GBrain (keyword search over the imported Discord/Notion pages) added to the
system prompt, then is answered by either the River base model (mode "base") or the person's trained twin
adapter (mode "twin"). The mode is read from the file `mode` next to this script on every request, so the demo
can flip base -> twin live without restarting UFO.

  RIVER_API_KEY=... python proxy.py --twin ../twin-run/latest.json --twin-base Qwen/Qwen3.5-9B --name gaurav --gbrain
  # UFO side:  OPENAI_BASE_URL=http://127.0.0.1:8711/v1  UFO_OPENAI_API_KEY=anything

Base mode forwards the request to River as-is (model rewritten, unsupported fields stripped) and streams
River's bytes straight back. Twin mode calls River's checkpoint sampler (non-streaming) and replays the
answer as SSE chunks. Tools are dropped in twin mode: the twin answers in text.
"""
from __future__ import annotations

import argparse, urllib.parse, json, os, re, subprocess, sys, time, urllib.error, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
RIVER = "https://api.river.ai/v1"
STRIP = {"reasoning_effort", "parallel_tool_calls", "stream_options", "store", "metadata", "service_tier"}
ARGS = None
LOG = HERE / "proxy.log"
TWINS = {}          # name -> {"run": Path, "data": Path|None}
UFO_MODEL_MAP = {}  # OpenAI model id UFO sends -> twin name ("base" for the base model)


def log(rec):
    rec["t"] = time.strftime("%H:%M:%S")
    LOG.open("a").write(json.dumps(rec) + "\n")
    print(json.dumps(rec), file=sys.stderr, flush=True)


def mode():
    f = HERE / "mode"
    return f.read_text().strip() if f.exists() else ("twin" if TWINS else "base")


def pick_twin(req, query):
    """Which twin answers: ?twin=<name> on the URL, else the OpenAI model id UFO sends, else the mode file."""
    for part in query.split("&"):
        if part.startswith("twin="):
            return part[5:] if part[5:] in TWINS else None
    mapped = UFO_MODEL_MAP.get(req.get("model", ""))
    if mapped == "base": return None
    if mapped in TWINS: return mapped
    if "mode=base" in query: return None
    if "mode=twin" in query or mode() == "twin":
        return next(iter(TWINS), None)
    return None


def last_user_text(messages):
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            return strip_context(c)[:500]
    return ""


STOP = {"the","and","for","what","how","did","does","when","where","who","our","are","you","can","get","time","with","this","that",
        "have","has","was","were","will","would","should","could","about","from","into","any","all","some","there","here","just","like",
        "please","guys","hey","sort","out","need","want","know","think","going","reaching","doing"}


def brain_dir(who):
    """The org brain for the base model; each twin's own brain (built from their messages only) for that twin."""
    if who and (TWINS.get(who) or {}).get("brain"):
        return TWINS[who]["brain"]
    return ARGS.gbrain if not who else None


def memory(query, who=None):
    cwd = brain_dir(who)
    if not cwd or not query.strip():
        return "", []
    kw = [w for w in re.findall(r"[a-z0-9]{3,}", query.lower()) if w not in STOP]
    fts = " ".join(kw[:8]) or query
    try:
        out = subprocess.run(["gbrain", "search", fts, "--limit", str(ARGS.memory_k), "--json"],
                             capture_output=True, text=True, timeout=20, env={**os.environ, "GBRAIN_HOME": str(cwd)})
        hits = json.loads(out.stdout) if out.stdout.strip() else []
    except Exception as e:  # memory is best-effort; the answer must still come
        log({"memory_error": str(e)[:200]}); return "", []
    if isinstance(hits, dict):
        hits = hits.get("results") or hits.get("hits") or []
    qw = {w for w in re.findall(r"[a-z0-9]{3,}", query.lower())} - STOP
    facts = []
    for h in hits:
        txt = h.get("chunk_text") or h.get("content") or ""
        for l in txt.splitlines():
            if l.strip().startswith("- "):                     # one bullet = one dated message
                f = l.strip()[2:]
                facts.append((len(qw & set(re.findall(r"[a-z0-9]{3,}", f.lower()))), f))
        if not any(l.strip().startswith("- ") for l in txt.splitlines()):
            facts.append((0, txt[:400]))
    facts.sort(key=lambda x: -x[0])                            # the lines that share words with the question first
    lines = [f for _, f in facts[:12]]
    return "\n".join(f"- {l}" for l in lines), [h.get("slug") or h.get("title") for h in hits]


def base_messages(messages, mem):
    """Plain assistant prompt + memory + the stripped conversation. UFO's harness prompt assumes tools the base
    model doesn't get here, so it is dropped (set --keep-harness-prompt to pass it through)."""
    if ARGS.keep_harness_prompt:
        return with_memory(messages, mem)
    sys_prompt = (f"You are Jeeves, the team's assistant. Answer briefly and directly, in one to three sentences.")
    conv = []
    for m in messages:
        if m.get("role") == "user":
            t = strip_context(m.get("content"))
            if t: conv.append({"role": "user", "content": t})
        elif m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"].strip():
            conv.append({"role": "assistant", "content": m["content"].strip()})
    return with_memory([{"role": "system", "content": sys_prompt}] + conv[-8:], mem)


def with_memory(messages, mem):
    if not mem:
        return messages
    note = (f"\n\nMEMORY — things {ARGS.name} actually said or wrote, retrieved for this question. Use them as facts; "
            f"if the answer is not in them, say you don't know rather than guessing:\n{mem}")
    msgs = [dict(m) for m in messages]
    for m in msgs:
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            m["content"] += note; return msgs
    return [{"role": "system", "content": note.strip()}] + msgs


def strip_context(c):
    if isinstance(c, list):
        c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    c = c or ""
    if "</context>" in c:
        c = c.split("</context>", 1)[1]
    return c.strip()


def prompt_system(kind, name):
    """System prompt for a prompting arm: fewshot = 12 of the person's real replies; hypo = inferred style hypotheses + 6 examples."""
    import random
    from twin import system as twin_sys
    person = ARGS.name if ARGS.name != "the user" else name
    data = ARGS.prompt_data
    pairs = [json.loads(l) for l in Path(data, "pairs.jsonl").read_text().splitlines() if l.strip()]
    exs = [p["reply"] for p in random.Random(11).sample(pairs, 40)]
    if kind == "fewshot":
        return twin_sys(person) + "\n\nHere are messages this person really wrote, to copy the voice from:\n" + "\n".join(f"- {r}" for r in exs[:12])
    hyp = Path(ARGS.hypotheses).read_text() if ARGS.hypotheses else ""
    return twin_sys(person) + "\n\nStyle hypotheses about this person, derived from their messages. Follow all of them:\n" + hyp + "\n\nExamples of their messages:\n" + "\n".join(f"- {r}" for r in exs[:6])


def grounded_draft(messages, mem, name):
    """Base model answers the question from NAME's own memory, honestly saying when there is no record.
    The twin then replies from that draft, so facts come from retrieval and only the voice comes from the LoRA."""
    q = last_user_text(messages)
    sys_prompt = (f"You answer on behalf of {name} using ONLY the memory lines below (things {name} actually wrote, with dates). "
                  f"Reply in one to three plain sentences quoting the concrete facts (dates, times, names, numbers, places). "
                  f"If some lines are relevant but do not fully answer, say what they do say (e.g. 'On 2023-11-28 {name} wrote that ...'). "
                  f"Only if NO line is relevant to the question, reply exactly: NO_RECORD\n\nMEMORY:\n{mem or '(none)'}")
    body = json.dumps({"model": ARGS.base, "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": q}],
                       "max_tokens": 160, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = urllib.request.Request(RIVER + "/chat/completions", data=body,
                               headers={"Authorization": "Bearer " + os.environ["RIVER_API_KEY"], "Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=120) as up:
        d = json.loads(up.read())
    t = d["choices"][0]["message"].get("content") or ""
    if isinstance(t, list): t = "".join(p.get("text", "") for p in t if isinstance(p, dict))
    return t.strip()


AUTHOR_LINE = re.compile(r"^([A-Za-z0-9_.#\- ]{1,24}): (.+)$")


def thread_lines(text):
    """A pasted thread keeps its `author: text` lines (the shape the adapter was trained on);
    anything else is one message from `someone`."""
    ls = [l for l in text.splitlines() if l.strip()]
    if any(AUTHOR_LINE.match(l) for l in ls):
        out, cur = [], None
        for l in ls:
            if AUTHOR_LINE.match(l): out.append(l.strip()); cur = len(out) - 1
            elif cur is not None: out[cur] += "\n" + l.strip()      # continuation line of the same message
            else: out.append(f"someone: {l.strip()}")
        return out
    return [f"someone: {text.strip()}"]


RECENT = []   # messages.jsonl rows, if --messages was given: lets the page load a real recent thread per channel


def twin_messages(messages, mem, name, draft=None):
    """Shape the conversation exactly like the training threads: `someone: ...` lines, ending in the person's name.
    UFO's harness prompt is dropped; only the twin prompt (and, if asked, the memory note) remain."""
    sys_prompt = (f"You are {name}'s digital twin in the team's Discord. Reply the way {name} would: same tone, "
                  f"length and phrasing as their own messages. Reply only with the message text.")
    lines = []
    if draft:
        lines.append(f"{name} (earlier): {draft[:600]}")
    elif mem:
        if ARGS.memory_in_thread:
            # the adapter was trained on threads, so facts go into the thread as earlier messages it can quote
            for h in mem.split("\n"):
                if h.strip(): lines.append(f"{ARGS.name} (earlier): {h.strip().lstrip('- ')[:400]}")
        else:
            sys_prompt += (f"\n\nMEMORY — things {ARGS.name} actually said, retrieved for this thread. Use them as facts; if the "
                           f"answer is not in them, say you don't know:\n{mem}")
    for m in messages:
        if m.get("role") == "user":
            t = strip_context(m.get("content"))
            if t: lines.extend(thread_lines(t))
        elif m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"].strip():
            lines.append(f"{ARGS.name}: {m['content'].strip()}")
    lines = lines[-14:]
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "\n".join(lines) + f"\n{ARGS.name}:"}]


DANGLING = {"at","in","to","the","a","an","and","but","or","of","for","with","on","is","are","was","were","we","i","that",
            "which","so","because","if","then","from","by","as","about","into","like","have","has","had","be","will","can","would"}


def dangling(t):
    """True when a message reads unfinished: ends on a function word, a comma/colon/dash, or is a 1-2 word stub."""
    w = t.rstrip().split()
    if not w: return False
    last = w[-1].lower().strip("*_")
    return t.rstrip()[-1] in ",:-—" or last in DANGLING or (len(w) <= 2 and t.rstrip()[-1] not in ".!?")


class H(BaseHTTPRequestHandler):
    _mem_slugs = []

    def log_message(self, *a):  # quiet
        pass

    def _mem_header(self):
        if self._mem_slugs:
            self.send_header("X-Jeeves-Memory", ",".join(self._mem_slugs)[:900])
        self.send_header("Access-Control-Expose-Headers", "X-Jeeves-Memory")

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self._mem_header()
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (HERE / "demo.html").read_bytes()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); return self.wfile.write(body)
        if self.path == "/twins":
            out = []
            for name, t in TWINS.items():
                rc = (t["run"] / "receipt.json") if t["run"] else None; lt = (t["run"] / "latest.json") if t["run"] else None
                out.append({"name": name, "kind": t["kind"], "receipt": json.loads(rc.read_text()) if rc and rc.exists() else None,
                            "steps": json.loads(lt.read_text()).get("step") if lt and lt.exists() else t["kind"]})
            return self._json(200, {"base": ARGS.base, "twins": out, "ufo_models": UFO_MODEL_MAP})
        if self.path == "/channels":
            from collections import Counter
            c = Counter(r["channel"] for r in RECENT if r.get("ts", "") >= "2025-01")
            return self._json(200, {"channels": [{"name": k, "n": v} for k, v in c.most_common()]})
        if self.path.startswith("/recent"):
            qs = dict(part.split("=", 1) for part in self.path.partition("?")[2].split("&") if "=" in part)
            ch = urllib.parse.unquote(qs.get("channel", "")); n = int(qs.get("n", 12))
            rows = [r for r in RECENT if r["channel"] == ch][-n:]
            return self._json(200, {"channel": ch, "messages": [{"author": r["author"], "text": r["text"], "ts": r["ts"][:16]} for r in rows]})
        if self.path.startswith("/gate_rows"):
            name = self.path.partition("twin=")[2].partition("&")[0]
            t = TWINS.get(name)
            rows_f = t and t["run"] / "gate_rows.jsonl"
            if not rows_f or not rows_f.exists(): return self._json(404, {"error": "no gate rows"})
            rows = [json.loads(l) for l in rows_f.read_text().splitlines() if l.strip()]
            if t.get("data"):   # older gates did not record the thread; recover it from the holdout by the real reply
                hold = {p["reply"]: p["context"] for p in (json.loads(l) for l in (t["data"] / "holdout.jsonl").read_text().splitlines() if l.strip())}
                for r in rows: r.setdefault("context", hold.get(r["reference"], []))
            return self._json(200, {"twin": name, "rows": rows[:40]})
        if self.path.rstrip("/").endswith("/models"):
            return self._json(200, {"object": "list", "data": [{"id": "jeeves", "object": "model", "owned_by": "jeeves"}]})
        log({"unhandled": self.path}); self._json(404, {"error": "not found"})

    def do_POST(self):
        path, _, query = self.path.partition("?")
        if not path.rstrip("/").endswith("/chat/completions"):
            log({"unhandled": self.path}); return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        t0 = time.monotonic()
        q = last_user_text(req.get("messages", []))
        who = pick_twin(req, query)
        mem, slugs = memory(q, who)      # base: the org brain; twin: that person's own brain, if one was given
        self._mem_slugs = sorted(set(slugs))
        m = f"twin:{who}" if who else "base"
        try:
            if who:
                draft = None
                if "grounded=1" in query or (ARGS.grounded_default and "grounded=0" not in query and "\n" not in q.strip()):
                    draft = grounded_draft(req.get("messages", []), mem, who)
                    log({"grounded_draft": draft[:120], "mem_lines": mem.count("\n") + (1 if mem else 0)})
                    if draft.startswith("NO_RECORD") or not mem:
                        # nothing in this person's messages: say so instead of letting the LoRA invent a fact in their voice
                        self._mem_slugs = []
                        return self._plain(req, f"(nothing in {who}'s messages about this)")
                req["messages"] = twin_messages(req.get("messages", []), mem, who, draft)
                self._twin(req, who, receipt=draft)
            else:
                req["messages"] = base_messages(req.get("messages", []), mem)
                self._forward(req)
            log({"mode": m, "q": q[:80], "memory": slugs, "ms": int((time.monotonic() - t0) * 1000)})
        except Exception as e:
            log({"mode": m, "error": repr(e)[:300]})
            try: self._json(502, {"error": {"message": repr(e)[:300], "type": "jeeves_upstream"}})
            except Exception: pass

    # -- base: stream River straight through
    def _forward(self, req):
        req["model"] = ARGS.base
        for k in STRIP | {"tools", "tool_choice"}: req.pop(k, None)
        if "max_completion_tokens" in req: req["max_tokens"] = req.pop("max_completion_tokens")
        stream = bool(req.get("stream"))
        if stream: req["stream_options"] = {"include_usage": True}
        req.setdefault("chat_template_kwargs", {})["enable_thinking"] = False   # Qwen: no <think> in the answer
        r = urllib.request.Request(RIVER + "/chat/completions", data=json.dumps(req).encode(),
                                   headers={"Authorization": "Bearer " + os.environ["RIVER_API_KEY"], "Content-Type": "application/json"})
        try:
            up = urllib.request.urlopen(r, timeout=300)
        except urllib.error.HTTPError as e:
            body = e.read(); self.send_response(e.code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            log({"upstream": e.code, "body": body[:300].decode(errors="ignore")}); return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
        self.send_header("Cache-Control", "no-cache"); self._mem_header(); self.end_headers()
        while True:
            chunk = up.read(1024)
            if not chunk: break
            self.wfile.write(chunk); self.wfile.flush()

    def _plain(self, req, text):
        cid, now = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if not req.get("stream"):
            return self._json(200, {"id": cid, "object": "chat.completion", "created": now, "model": "jeeves",
                                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], "usage": usage})
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache"); self._mem_header(); self.end_headers()
        for delta, fin in (({"role": "assistant", "content": text}, None), ({}, "stop")):
            self.wfile.write(f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': now, 'model': 'jeeves', 'choices': [{'index': 0, 'delta': delta, 'finish_reason': fin}]})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()

    # -- twin: River checkpoint sampler, replayed as SSE
    def _twin(self, req, who, receipt=None):
        import river_client as river
        t = TWINS[who]
        c = river.Client(api_key=os.environ["RIVER_API_KEY"], timeout=600)
        msgs = req["messages"]
        gen_kw = dict(max_tokens=min(int(req.get("max_completion_tokens") or req.get("max_tokens") or 512), 1024),
                      temperature=req.get("temperature", ARGS.twin_temperature), chat_template_kwargs={"enable_thinking": False})
        if t["kind"] == "prompt":
            msgs = [{"role": "system", "content": t["system"]}] + msgs[1:]
            body = json.loads(c.chat_complete(msgs, base_model=ARGS.twin_base, **gen_kw).response_json)
            c.close(); return self._emit(req, body, who)
        if t["kind"] == "restyle":
            from restyle_twin import style_messages
            raw = json.loads(c.chat_complete(msgs, base_model=ARGS.twin_base, **gen_kw).response_json)["choices"][0]["message"].get("content") or ""
            latest = json.loads((t["run"] / "latest.json").read_text())
            body = json.loads(c.chat_complete_from_checkpoint(style_messages(raw.strip()), checkpoint_path=latest["inference"], base_model=ARGS.twin_base,
                                                               max_tokens=gen_kw["max_tokens"], temperature=0.3, chat_template_kwargs={"enable_thinking": False}).response_json)
            c.close(); return self._emit(req, body, who)
        latest = json.loads((t["run"] / "latest.json").read_text())
        gen = dict(checkpoint_path=latest["inference"], base_model=ARGS.twin_base,
                   max_tokens=min(int(req.get("max_completion_tokens") or req.get("max_tokens") or 512), 1024),
                   temperature=req.get("temperature", ARGS.twin_temperature), chat_template_kwargs={"enable_thinking": False})
        parts, body = [], None
        for _ in range(ARGS.max_messages):        # Discord style: a fragment, then the next message; stop at punctuation
            res = c.chat_complete_from_checkpoint(msgs, **gen)
            body = json.loads(res.response_json)
            t = body["choices"][0]["message"].get("content") or ""
            if isinstance(t, list): t = "".join(p.get("text", "") for p in t if isinstance(p, dict))
            t = t.strip()
            if not t or t in parts: break
            parts.append(t)
            if "<link>" in t or "@someone" in t: break         # a new topic, not a continuation
            if not dangling(t): break
            msgs = msgs[:-1] + [{"role": "user", "content": msgs[-1]["content"] + " " + t + f"\n{who}:"}]
        c.close()
        text = "\n".join(parts)
        if receipt:
            text += f"\n\n— from {who}'s messages: {receipt}"
        return self._emit(req, body, who, text)

    def _emit(self, req, body, who, text=None):
        if text is None:
            text = body["choices"][0]["message"].get("content") or ""
            if isinstance(text, list): text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
            text = text.strip()
        cid, now = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
        text = text.replace("<link>", "(link)").replace("@someone", "@you")
        usage = body.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if not req.get("stream"):
            return self._json(200, {"id": cid, "object": "chat.completion", "created": now, "model": "jeeves",
                                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], "usage": usage})
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache"); self._mem_header(); self.end_headers()
        def send(delta, finish=None, extra=None):
            ch = {"id": cid, "object": "chat.completion.chunk", "created": now, "model": "jeeves",
                  "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if extra: ch.update(extra)
            self.wfile.write(f"data: {json.dumps(ch)}\n\n".encode()); self.wfile.flush()
        send({"role": "assistant", "content": ""})
        for i in range(0, len(text), 24):
            send({"content": text[i:i + 24]})
        send({}, "stop")
        self.wfile.write(f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': now, 'model': 'jeeves', 'choices': [], 'usage': usage})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8711)
    ap.add_argument("--base", default="Qwen/Qwen3.8-27B-FP8", help="River model for base mode")
    ap.add_argument("--twin", action="append", default=[], metavar="NAME=RUN_DIR[:DATA_DIR[:BRAIN_DIR]]",
                    help="a trained twin: its name (as used in training), the run dir with latest.json, optionally its data dir and its own GBrain dir")
    ap.add_argument("--twin-base", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--name", default="the user", help="(legacy) name for a single --twin given as a bare path")
    ap.add_argument("--gbrain", default=None, help="the org brain: a GBRAIN_HOME directory (gbrain keeps one brain per home, not per cwd)")
    ap.add_argument("--memory-k", type=int, default=8)
    ap.add_argument("--twin-temperature", type=float, default=0.4)
    ap.add_argument("--grounded-default", action="store_true", help="single-line questions: base drafts from the twin's memory, twin replies from the draft")
    ap.add_argument("--prompt-data", default="discord5", help="pairs dir for prompting arms' examples")
    ap.add_argument("--hypotheses", default=None, help="style hypotheses file for the hypo prompting arm")
    ap.add_argument("--max-messages", type=int, default=3, help="twin mode: continue an unfinished fragment with up to N more messages")
    ap.add_argument("--keep-harness-prompt", action="store_true")
    ap.add_argument("--messages", default=None, help="messages.jsonl from pull_discord: lets the demo page load a channel's last messages as the thread")
    ap.add_argument("--memory-in-thread", action="store_true", help="twin mode: put memory hits in the thread instead of the system prompt")
    ARGS = ap.parse_args()
    if ARGS.messages:
        RECENT.extend(json.loads(l) for l in Path(ARGS.messages).read_text().splitlines() if l.strip())
        RECENT[:] = [r for r in RECENT if not r.get("bot") and r.get("text")]
    if "RIVER_API_KEY" not in os.environ: sys.exit("RIVER_API_KEY not set")
    for spec in ARGS.twin:
        if "=" in spec:
            name, _, rest = spec.partition("=")
        else:
            name, rest = ARGS.name, spec
        kind = "ckpt"
        if rest.startswith("restyle:"): kind, rest = "restyle", rest[len("restyle:"):]
        elif rest.startswith("prompt:"): kind, rest = "prompt", rest[len("prompt:"):]
        run, _, data = rest.partition(":"); data, _, brain = data.partition(":")
        entry = {"kind": kind, "data": Path(data).resolve() if data else None, "brain": Path(brain).resolve() if brain else None}
        if kind == "prompt":
            d = Path(f"twin-{name}"); entry["run"] = d.resolve() if d.exists() else None; entry["system"] = prompt_system(run, name)
        else:
            run = Path(run); run = run.parent if run.name == "latest.json" else run; entry["run"] = run.resolve()
        TWINS[name] = entry
    # UFO's closed model list -> bots: `ufo --model gpt-5.4 "..."` talks to the first twin, etc.
    for mid, who in zip(["gpt-5.4", "gpt-5.5", "gpt-5.4-nano"], TWINS):
        UFO_MODEL_MAP[mid] = who
    UFO_MODEL_MAP["gpt-5.4-mini"] = "base"
    print(f"jeeves on http://127.0.0.1:{ARGS.port}/v1  base={ARGS.base} twins={list(TWINS)} ufo_models={UFO_MODEL_MAP} memory={'gbrain' if ARGS.gbrain else 'off'}", file=sys.stderr)
    ThreadingHTTPServer(("127.0.0.1", ARGS.port), H).serve_forever()


if __name__ == "__main__":
    main()
