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

import argparse, json, os, subprocess, sys, time, urllib.error, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
RIVER = "https://api.river.ai/v1"
STRIP = {"reasoning_effort", "parallel_tool_calls", "stream_options", "store", "metadata", "service_tier"}
ARGS = None
LOG = HERE / "proxy.log"


def log(rec):
    rec["t"] = time.strftime("%H:%M:%S")
    LOG.open("a").write(json.dumps(rec) + "\n")
    print(json.dumps(rec), file=sys.stderr, flush=True)


def mode():
    f = HERE / "mode"
    return f.read_text().strip() if f.exists() else ("twin" if ARGS.twin else "base")


def last_user_text(messages):
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            return strip_context(c)[:500]
    return ""


def memory(query):
    if not ARGS.gbrain or not query.strip():
        return "", []
    try:
        out = subprocess.run(["gbrain", "search", query, "--limit", str(ARGS.memory_k), "--json"],
                             capture_output=True, text=True, timeout=20, cwd=ARGS.gbrain)
        hits = json.loads(out.stdout) if out.stdout.strip() else []
    except Exception as e:  # memory is best-effort; the answer must still come
        log({"memory_error": str(e)[:200]}); return "", []
    if isinstance(hits, dict):
        hits = hits.get("results") or hits.get("hits") or []
    lines = []
    for h in hits:
        txt = h.get("chunk_text") or h.get("content") or ""
        facts = [l.strip()[2:] for l in txt.splitlines() if l.strip().startswith("- ")]   # one bullet = one dated message
        lines.extend(facts[:6] if facts else [txt[:400]])
    return "\n".join(f"- {l}" for l in lines[:18]), [h.get("slug") or h.get("title") for h in hits]


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


def twin_messages(messages, mem):
    """Shape the conversation exactly like the training threads: `someone: ...` lines, ending in the person's name.
    UFO's harness prompt is dropped; only the twin prompt and the memory note remain."""
    sys_prompt = (f"You are {ARGS.name}'s digital twin in the team's Discord. Reply the way {ARGS.name} would: same tone, "
                  f"length and phrasing as their own messages. Reply only with the message text.")
    lines = []
    if mem:
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
            if t: lines.append(f"someone: {t}")
        elif m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"].strip():
            lines.append(f"{ARGS.name}: {m['content'].strip()}")
    lines = lines[-14:]
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "\n".join(lines) + f"\n{ARGS.name}:"}]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (HERE / "demo.html").read_bytes()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); return self.wfile.write(body)
        if self.path == "/receipt":
            for r in sorted(HERE.glob("twin-run*/receipt.json"), key=lambda f: f.stat().st_mtime, reverse=True):
                return self._json(200, json.loads(r.read_text()))
            return self._json(404, {"error": "no receipt yet"})
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
        mem, slugs = memory(q)
        m = mode()
        if "mode=twin" in query: m = "twin"          # the demo page asks both modes side by side
        elif "mode=base" in query: m = "base"
        try:
            if m == "twin" and ARGS.twin:
                req["messages"] = twin_messages(req.get("messages", []), mem)
                self._twin(req)
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
        self.send_header("Cache-Control", "no-cache"); self.end_headers()
        while True:
            chunk = up.read(1024)
            if not chunk: break
            self.wfile.write(chunk); self.wfile.flush()

    # -- twin: River checkpoint sampler, replayed as SSE
    def _twin(self, req):
        import river_client as river
        latest = json.loads(Path(ARGS.twin).read_text())
        c = river.Client(api_key=os.environ["RIVER_API_KEY"], timeout=600)
        msgs = req["messages"]
        res = c.chat_complete_from_checkpoint(msgs, checkpoint_path=latest["inference"], base_model=ARGS.twin_base,
                                              max_tokens=min(int(req.get("max_completion_tokens") or req.get("max_tokens") or 512), 1024),
                                              temperature=req.get("temperature", 0.7), chat_template_kwargs={"enable_thinking": False})
        c.close()
        body = json.loads(res.response_json)
        text = body["choices"][0]["message"].get("content") or ""
        if isinstance(text, list): text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        cid, now = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
        usage = body.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if not req.get("stream"):
            return self._json(200, {"id": cid, "object": "chat.completion", "created": now, "model": "jeeves",
                                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], "usage": usage})
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache"); self.end_headers()
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
    ap.add_argument("--twin", default=None, help="path to the trainer's latest.json (inference checkpoint)")
    ap.add_argument("--twin-base", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--name", default="the user")
    ap.add_argument("--gbrain", default=None, help="directory holding the GBrain brain (cwd for `gbrain search`)")
    ap.add_argument("--memory-k", type=int, default=5)
    ap.add_argument("--keep-harness-prompt", action="store_true")
    ap.add_argument("--memory-in-thread", action="store_true", help="twin mode: put memory hits in the thread instead of the system prompt")
    ARGS = ap.parse_args()
    if "RIVER_API_KEY" not in os.environ: sys.exit("RIVER_API_KEY not set")
    print(f"jeeves on http://127.0.0.1:{ARGS.port}/v1  base={ARGS.base} twin={'yes' if ARGS.twin else 'no'} mode={mode()} memory={'gbrain' if ARGS.gbrain else 'off'}", file=sys.stderr)
    ThreadingHTTPServer(("127.0.0.1", ARGS.port), H).serve_forever()


if __name__ == "__main__":
    main()
