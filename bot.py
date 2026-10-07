"""Jeeves drafting bot: the twin writes the owner's next message in watched channels; the owner approves, edits or
skips from a DM card; every decision is a training row.

  DISCORD_BOT_TOKEN=... python bot.py --owner-id <id> --channels founderchat standup --twin sft6
  options: --auto <channel>...   post straight into these channels without asking (the blind handoff)
           --proxy http://localhost:8711   --context 12   --min-gap 45   --ledger ledger.jsonl

What it writes (ledger.jsonl, one row per card):
  {"ts","channel","thread":[{"author","text"}],"draft","action":"send|edit|skip|auto","final"}
  send/auto -> a positive pair (thread -> final)      edit -> a preference pair (final over draft)      skip -> a negative
Message content comes from the REST history endpoint, so the bot works without the privileged message-content intent.
"""
import argparse, asyncio, json, os, time
from pathlib import Path

import discord
import urllib.request

ARGS = None
LAST_CARD = {}   # channel id -> monotonic time of the last draft, to avoid a card per message in a burst


def log(row):
    row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    Path(ARGS.ledger).open("a").write(json.dumps(row, ensure_ascii=False) + "\n")


def draft(thread):
    text = "\n".join(f"{m['author']}: {m['text']}" for m in thread)
    req = urllib.request.Request(f"{ARGS.proxy}/v1/chat/completions?twin={ARGS.twin}",
                                 data=json.dumps({"model": "jeeves", "max_tokens": 200, "messages": [{"role": "user", "content": text}]}).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=180))
    return (d["choices"][0]["message"].get("content") or "").strip()


class Card(discord.ui.View):
    def __init__(self, channel, thread, text):
        super().__init__(timeout=6 * 3600)
        self.channel, self.thread, self.text = channel, thread, text

    async def finish(self, inter, action, final):
        if final:
            await self.channel.send(final)
        log({"channel": self.channel.name, "thread": self.thread, "draft": self.text, "action": action, "final": final})
        for c in self.children: c.disabled = True
        await inter.response.edit_message(content=f"**{action}** → #{self.channel.name}\n{final or self.text}", view=self)
        self.stop()

    @discord.ui.button(label="Send", style=discord.ButtonStyle.success)
    async def send(self, inter, _):
        await self.finish(inter, "send", self.text)

    @discord.ui.button(label="Edit", style=discord.ButtonStyle.primary)
    async def edit(self, inter, _):
        card = self

        class M(discord.ui.Modal, title=f"Edit for #{self.channel.name}"):
            body = discord.ui.TextInput(label="your version", style=discord.TextStyle.paragraph, default=card.text, max_length=1900)

            async def on_submit(self, i):
                await card.finish(i, "edit", str(self.body.value).strip())

        await inter.response.send_modal(M())

    @discord.ui.button(label="Skip", style=discord.ButtonStyle.secondary)
    async def skip(self, inter, _):
        await self.finish(inter, "skip", "")


class Bot(discord.Client):
    async def on_ready(self):
        self.owner = await self.fetch_user(ARGS.owner_id)
        self.watch = {}
        for g in self.guilds:
            for ch in g.text_channels:
                if ch.name in ARGS.channels or ch.name in ARGS.auto: self.watch[ch.id] = ch
        print(f"ready as {self.user} · watching {[c.name for c in self.watch.values()]} · auto {ARGS.auto} · owner DM {self.owner}", flush=True)

    async def on_message(self, msg):
        ch = self.watch.get(msg.channel.id)
        if not ch or msg.author.id in (self.user.id, ARGS.owner_id) or msg.author.bot: return
        now = time.monotonic()
        if now - LAST_CARD.get(ch.id, 0) < ARGS.min_gap: return
        LAST_CARD[ch.id] = now
        await asyncio.sleep(ARGS.settle)                    # let a burst finish before reading the thread
        hist = [m async for m in ch.history(limit=ARGS.context)]
        thread = [{"author": m.author.display_name, "text": m.content} for m in reversed(hist) if m.content and not m.author.bot]
        if not thread: return
        try:
            text = await asyncio.get_event_loop().run_in_executor(None, draft, thread)
        except Exception as e:
            print(f"draft failed: {e!r}", flush=True); return
        if not text: return
        if ch.name in ARGS.auto:
            await ch.send(text)
            log({"channel": ch.name, "thread": thread, "draft": text, "action": "auto", "final": text})
            return
        tail = "\n".join(f"> **{m['author']}:** {m['text'][:200]}" for m in thread[-3:])
        await self.owner.send(f"**#{ch.name}** · the twin would say:\n{tail}\n\n{text}", view=Card(ch, thread, text))


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner-id", type=int, required=True)
    ap.add_argument("--channels", nargs="*", default=[], help="draft + ask the owner by DM")
    ap.add_argument("--auto", nargs="*", default=[], help="post without asking (demo mode); still logged")
    ap.add_argument("--twin", default="sft6"); ap.add_argument("--proxy", default="http://localhost:8711")
    ap.add_argument("--context", type=int, default=12); ap.add_argument("--min-gap", type=int, default=45)
    ap.add_argument("--settle", type=int, default=20, help="seconds to wait after a message before drafting")
    ap.add_argument("--ledger", default="ledger.jsonl")
    ARGS = ap.parse_args()
    intents = discord.Intents.default()
    intents.message_content = True      # if the portal has it off, history() still returns content via REST
    try:
        Bot(intents=intents).run(os.environ["DISCORD_BOT_TOKEN"], log_handler=None)
    except discord.PrivilegedIntentsRequired:
        print("message-content intent is off in the developer portal; running without it (REST history still has content)", flush=True)
        Bot(intents=discord.Intents.default()).run(os.environ["DISCORD_BOT_TOKEN"], log_handler=None)


if __name__ == "__main__":
    main()
