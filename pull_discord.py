"""Pull a Discord server's channel history with a bot token and build the twin's training data.

  DISCORD_BOT_TOKEN=... python pull_discord.py --guild "My Server" --author gaurav --out discord/

Writes
  discord/messages.jsonl        every readable message (kept local, never committed)
  discord/brain/<channel>.md    the AUTHOR's own messages as markdown pages for `gbrain import`
  discord/pairs.jsonl           {"channel","context":[{"author","text"}],"reply"} — author's replies with the thread before them
  discord/holdout.jsonl         the most recent 15% of pairs, frozen for the gate

Bot needs: Message Content intent on, invited with "Read Message History" + "View Channels".
Stdlib only.
"""
from __future__ import annotations

import argparse, json, os, re, sys, time, urllib.error, urllib.parse, urllib.request
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

API = "https://discord.com/api/v10"


def get(path, token, params=None):
    url = API + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={"Authorization": f"Bot {token}", "User-Agent": "jeeves-puller/0.1"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = float(json.loads(e.read().decode()).get("retry_after", 1.0))
                time.sleep(wait + 0.2); continue
            if e.code in (403, 404):
                return None
            raise
    raise RuntimeError(f"gave up on {path}")


def pull_channel(cid, token, limit):
    out, before = [], None
    while len(out) < limit:
        params = {"limit": 100}
        if before: params["before"] = before
        page = get(f"/channels/{cid}/messages", token, params)
        if not page: break
        out.extend(page); before = page[-1]["id"]
        if len(page) < 100: break
        time.sleep(0.25)
    return out


def clean(text):
    text = re.sub(r"<@!?\d+>", "@someone", text)
    text = re.sub(r"<#\d+>", "#channel", text)
    text = re.sub(r"<a?:\w+:\d+>", "", text)
    text = re.sub(r"https?://\S+", "<link>", text)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guild", required=True, help="server name or id")
    ap.add_argument("--author", required=True, help="username, global name or user id of the person being cloned")
    ap.add_argument("--out", default="discord")
    ap.add_argument("--per-channel", type=int, default=2000)
    ap.add_argument("--context", type=int, default=6, help="messages of thread before each reply")
    ap.add_argument("--window-min", type=int, default=180, help="context must be within this many minutes")
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--holdout-mode", choices=["newest", "recent-random"], default="newest",
                    help="newest: hold out the last N%% by time; recent-random: hold out a random N%% of pairs since --recent-since, train on everything else")
    ap.add_argument("--recent-since", default="2025-01", help="recent-random: pairs at/after this ISO prefix are the holdout pool")
    ap.add_argument("--merge-min", type=int, default=5, help="merge the author's consecutive messages within this many minutes into one reply")
    ap.add_argument("--min-words", type=int, default=4)
    ap.add_argument("--cached", action="store_true", help="rebuild pages and pairs from <out>/messages.jsonl without pulling")
    a = ap.parse_args()
    out = Path(a.out); (out / "brain").mkdir(parents=True, exist_ok=True)
    if a.cached:
        rows = [json.loads(l) for l in (out / "messages.jsonl").read_text().splitlines() if l.strip()]
        return build(a, rows, out)
    token = os.environ["DISCORD_BOT_TOKEN"]

    guilds = get("/users/@me/guilds", token) or []
    g = next((x for x in guilds if x["id"] == a.guild or x["name"].lower() == a.guild.lower()), None)
    if not g:
        sys.exit(f"bot is not in a guild called {a.guild!r}; it is in: {[x['name'] for x in guilds]}")
    channels = [c for c in (get(f"/guilds/{g['id']}/channels", token) or []) if c["type"] in (0, 5, 11, 12, 15)]
    print(f"guild {g['name']}: {len(channels)} text channels", file=sys.stderr)

    rows = []
    for ch in channels:
        msgs = pull_channel(ch["id"], token, a.per_channel)
        if not msgs: continue
        for m in msgs:
            if m.get("type") not in (0, 19) or not m.get("content"): continue
            au = m["author"]
            rows.append({"channel": ch["name"], "id": m["id"], "ts": m["timestamp"],
                         "author": au.get("global_name") or au["username"], "username": au["username"],
                         "author_id": au["id"], "bot": au.get("bot", False), "text": clean(m["content"])})
        print(f"  #{ch['name']}: {len(msgs)} messages", file=sys.stderr)
    rows.sort(key=lambda r: r["ts"])
    (out / "messages.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    build(a, rows, out)


def words(text):
    return [w for w in text.split() if w not in ("<link>", "@someone", "#channel")]


def build(a, rows, out):
    def is_author(r):
        return a.author.lower() in (r["author"].lower(), r["username"].lower(), r["author_id"])
    mine = [r for r in rows if is_author(r)]
    print(f"{len(rows)} messages, {len(mine)} by {a.author}", file=sys.stderr)
    if not mine:
        sys.exit(f"no messages by {a.author!r}; authors seen: {sorted({r['author'] for r in rows})[:30]}")

    # brain pages: author's own messages, per channel, dated — facts the twin can recall with provenance
    by_ch = defaultdict(list)
    for r in mine: by_ch[r["channel"]].append(r)
    for ch, rs in by_ch.items():
        body = "\n\n".join(f"- {r['ts'][:10]} in #{ch}: {r['text']}" for r in rs if len(r["text"]) > 15)
        (out / "brain" / f"{re.sub(r'[^a-z0-9-]+', '-', ch.lower())}.md").write_text(
            f"# {a.author} in #{ch}\n\nThings {a.author} said in the #{ch} channel, newest last.\n\n{body}\n")

    # pairs: each author reply (consecutive author messages merged) with the thread before it
    pairs = []
    by_channel = defaultdict(list)
    for r in rows: by_channel[r["channel"]].append(r)
    for ch, rs in by_channel.items():
        i = 0
        while i < len(rs):
            if not is_author(rs[i]): i += 1; continue
            j = i
            while j + 1 < len(rs) and is_author(rs[j + 1]) and _dt(rs[j + 1]) - _dt(rs[j]) < timedelta(minutes=a.merge_min): j += 1
            reply = "\n".join(rs[k]["text"] for k in range(i, j + 1))
            ctx = [r for r in rs[max(0, i - a.context):i] if _dt(rs[i]) - _dt(r) < timedelta(minutes=a.window_min)]
            if ctx and len(words(reply)) >= a.min_words and reply.count("<link>") <= 1:   # no link-only replies
                pairs.append({"channel": ch, "ts": rs[i]["ts"], "context": [{"author": r["author"], "text": r["text"]} for r in ctx], "reply": reply})
            i = j + 1
    pairs.sort(key=lambda p: p["ts"])
    if a.holdout_mode == "recent-random":
        import random
        recent = [i for i, p in enumerate(pairs) if p["ts"] >= a.recent_since]
        hold = set(random.Random(7).sample(recent, int(len(recent) * a.holdout)))
        train = [p for i, p in enumerate(pairs) if i not in hold]; held = [p for i, p in enumerate(pairs) if i in hold]
    else:
        cut = int(len(pairs) * (1 - a.holdout)); train, held = pairs[:cut], pairs[cut:]
    (out / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in train))
    (out / "holdout.jsonl").write_text("".join(json.dumps(p) + "\n" for p in held))
    recent_n = sum(p["ts"] >= a.recent_since for p in train)
    print(f"pairs: {len(train)} train ({recent_n} since {a.recent_since}), {len(held)} holdout ({a.holdout_mode}); brain pages: {len(by_ch)}", file=sys.stderr)


def _dt(r):
    return datetime.fromisoformat(r["ts"].replace("Z", "+00:00"))


if __name__ == "__main__":
    main()
