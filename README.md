# Jeeves

A digital twin of a teammate, built from their Discord history in an afternoon.

- **What the twin knows** lives in [GBrain](https://github.com/garrytan/gbrain): every message the person wrote,
  one page per channel, retrieved by keyword per question with the channel and date as provenance.
- **How the twin sounds** lives in a LoRA trained on [River](https://docs.river.ai): the thread before each of the
  person's replies is the prompt, their actual reply is the target. A blind judge on held-out threads is the receipt.
- **Where you talk to it** is [UFO](https://github.com/ufo-ai/ufo-core), unmodified. UFO thinks it is talking to
  OpenAI; it is talking to `proxy.py`, which adds memory to every request and answers from the base model or the twin.

```
  UFO (terminal client + server)
        │  OPENAI_BASE_URL=http://127.0.0.1:8711/v1
        ▼
  proxy.py ── gbrain search ──► memory (facts, provenance) ──┐
        │                                                    ▼
        ├─ mode=base ─► River /v1  Qwen3.8-27B          (right facts, chatbot voice)
        └─ mode=twin ─► River checkpoint sampler, LoRA  (right facts, the person's voice)
```

The `mode` file is read on every request, so the demo flips base → twin live without restarting anything.

## Run it

Needs: Python 3.12 + `uv`, Rust (`cargo`) for UFO's client, Bun ≥ 1.3.11 for GBrain, a River API key, a Discord bot
token for a server you admin (Message Content intent on, "Read Message History" permission).

```bash
git clone https://github.com/ufo-ai/ufo-core ufo && (cd ufo && make install && make build)
git clone https://github.com/garrytan/gbrain && (cd gbrain && bun install && bun link)
uv venv .venv && uv pip install --python .venv river-client transformers
export RIVER_API_KEY=... DISCORD_BOT_TOKEN=...   # keep these in a .env you never commit

# 1. Discord -> messages, brain pages, reply pairs (last 15% frozen as holdout)
python pull_discord.py --guild "<server>" --author <username> --out discord

# 2. memory
(cd gb && gbrain init --pglite --no-embedding --non-interactive && gbrain import ../discord/brain --no-embed --allow-noncanonical-root)

# 3. voice: train on River (Qwen3.5-9B, LoRA rank 16, ~3 s/step), then gate blind
python twin.py --name <username> train --data discord --run twin-run --steps 40
python twin.py --name <username> gate  --data discord --run twin-run --n 30     # -> twin-run/receipt.json

# 4. serve
echo base > mode
python proxy.py --name <username> --twin twin-run/latest.json --twin-base Qwen/Qwen3.5-9B --gbrain "$PWD/gb" &
cp ufo.toml ufo/ufo.toml && (cd ufo && UFO_OPENAI_API_KEY=proxy-holds-the-key uv run ufoctl init --email you@x.com --model gpt-5.4-mini --reasoning low --member-model-provider openai)
(cd ufo && OPENAI_BASE_URL=http://127.0.0.1:8711/v1 uv run ufoctl serve) &
mkdir -p ~/.ufo && install -m 600 ~/.ufoctl/token ~/.ufo/credentials && echo http://localhost:8710 > ~/.ufo/workspace
./ufo/client/target/debug/ufo "where do we keep the staging key?"

# 5. flip
echo twin > mode
```

## The demo

1. Ask UFO something only this person knows. Memory finds the message; the answer is right, the voice is a chatbot's.
2. `echo twin > mode`. Same question, same facts, now typed the way they would have typed it.
3. `twin-run/receipt.json`: on threads the twin never saw, a blind judge picks twin over base N times out of M.
4. Onboarding: a new hire asks "how do we do X here?" and gets the team's own answer, with channel and date.

## Results

Three teammates from one Discord server (31k messages), one LoRA each on Qwen3.5-9B via River. Gate = 30 held-out threads the twin never saw; a blind judge (DeepSeek-V4.1-Flash) sees the person's real reply, then base and twin in random order, and picks the one that reads more like them. Base alone would score 0.5.

| twin | training messages | held-out threads | config | twin wins | losses | win rate |
|---|---|---|---|---|---|---|
| A | 3162 | 559 | 300 steps, batch 16, rank 32 | 19 | 11 | 0.633 |
| A (40 steps) | 3162 | 559 | 40 steps, batch 8, rank 16 | 18 | 12 | 0.6 |
| A (merged replies) | 2645 | 467 | 120 steps, batch 16, rank 32; replies merged over 15 min | 16 | 14 | 0.533 |
| B | 1818 | 321 | 120 steps, batch 16, rank 32 | 20 | 10 | 0.667 |
| C | 1264 | 224 | 120 steps, batch 16, rank 32 | 17 | 13 | 0.567 |

### Length is a data question, not a step count

Same person, same judge model, two data cuts. "Completeness" is a second rubric that asks the blind judge for a reply that both sounds like the person and answers the thread fully.

| twin | data cut | style judge | completeness judge | twin avg words (real ≈ 40, base ≈ 15–20) |
|---|---|---|---|---|
| A | consecutive messages merged over 5 min, ≥4 words | 19/30 | 24/40 | 9 |
| A | merged over 20 min, ≥15 words, 600 steps | 24/40 | 25/40 | 34–38 |

Cutting the pairs to full bursts moved the twin from 9 to 38 words with no change in either win rate. With 40 threads the interval is about ±15 points, so the next step is the gate (both orders, 115+ threads, length control), not more training.

### Weights vs prompting vs preference training (gate v2, one person)

120 held-out threads, 2 samples each, every arm judged against base in both orders; wins/losses count only pairs where the judge agreed with itself. Real replies average 41 words, base 18.

| arm | style wins/losses | rate | 95% CI | length-controlled | completeness rate | words | stylometric distance to real |
|---|---|---|---|---|---|---|---|
| SFT, 600 steps | 114/36 | 0.76 | 0.69–0.82 | 0.66 | 0.71 | 37 | 0.18 |
| DITTO (SFT step 200, then DPO 150) | 132/44 | 0.75 | 0.68–0.81 | 0.68 | 0.68 | 42 | 0.20 |
| River `style_chat` recipe (restyle) | 67/32 | 0.68 | 0.58–0.76 | 0.57 | 0.59 | 17 | 0.34 |
| DPO from base, 150 steps | 82/59 | 0.58 | 0.50–0.66 | 0.56 | 0.40 | 27 | 0.39 |
| prompting: inferred style hypotheses | 87/76 | 0.53 | 0.46–0.61 | 0.56 | 0.53 | 76 | 0.84 |
| prompting: 12 real replies few-shot | 72/64 | 0.53 | 0.45–0.61 | 0.53 | 0.53 | 24 | 0.36 |
| DPO then OPSD (real reply as hint) | 33/168 | 0.16 | 0.12–0.22 | 0.35 | 0.09 | 117 | 0.88 |

Findings: per-person weights beat prompting by ~20 points here; DITTO ties SFT (fewer judge order-flips, best length-controlled rate); DPO from base alone is worse than SFT and passes through a repetition-collapse phase (steps ~50–100) that on-policy replay later repairs; OPSD as a finishing stage lengthened replies to 117 words and lost badly; River's restyle recipe returns the base draft nearly unchanged (17 words) and the judge flipped on 141/240 of its pairs. DPO on River has no native loss and is emulated with `importance_sampling` advantages (`pref_twin.py`). `receipts/twin-a-gate2.json` has the numbers.

Training cost for all five runs together was under a dollar of River credit. `receipts/` has the JSON; per-thread rows are not committed because they quote messages.

## How UFO was pointed at River without touching it

UFO builds its OpenAI client with no base URL, so the OpenAI SDK honours `OPENAI_BASE_URL`. Its model ids are a
closed list, so `ufo.toml` pins a chat-surface id (`gpt-5.4-mini`) and the proxy rewrites it. Tools are dropped at
the proxy because River has no tool-call parser for Qwen. UFO's own memory extension asks for embeddings River does
not serve; it logs `embed_query_failed` and carries on. Jeeves memory comes from GBrain instead.

## Files

| file | what |
|---|---|
| `pull_discord.py` | bot-token REST puller; brain pages per channel; pairs = thread before a reply → the reply; holdout = newest 15% |
| `twin.py` | `train` / `gate` / `reply` on River. Cross-entropy by default; River also takes `--loss opsd` |
| `proxy.py` | OpenAI-compatible endpoint: GBrain memory injection, base passthrough (streamed), twin via checkpoint sampler (replayed as SSE) |
| `ufo.toml` | the UFO config used (SQLite, filesystem blobs, `gpt-5.4-mini` as the id UFO sends) |
| `mode` | `base` or `twin` |

Not committed: the Discord export, the brain, logs, keys. `.gitignore` covers them.
