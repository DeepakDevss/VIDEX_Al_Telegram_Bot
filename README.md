# Avenor | AI Companion

A Telegram AI bot written in Python. It holds conversations, understands media, remembers users across sessions, and can reply with text or voice.

- Bot: https://t.me/AvenorAlBot
- Repository: https://github.com/bydenvik/avenor-ai-telegram-bot
- Developer: https://github.com/bydenvik

---

## Overview

Avenor runs as a single Python process and talks to Telegram through the Bot API. Google Gemini is the main language model. Groq is used as a fallback chat provider and handles the speech-to-text side. User and chat state lives in a local JSON file managed by `memory.py`.

The bot is meant to run as a long-lived service. It should work on any platform that can run a Python process and set environment variables.

---

## What it can do

- Replies in whatever language the user writes in
- Remembers users and chats between restarts
- Tells apart who said what in group conversations
- Reads photos, documents, voice notes, and audio
- Understands plain requests with no command — say "make me a report" and it attaches the file, or "draw a logo" and it makes the image
- Sets reminders and delivers them on time ("remind me in 2h to call mom"), with no external service
- Reads a link you paste and answers about the page — news, blogs and social links
- Reads social **posts**, not just their preview card, using public endpoints only (no API
  key, no login, no app registration): X/Twitter, Reddit (post + top comments), Mastodon,
  Bluesky (post + replies), TikTok, and public Telegram channels. Instagram, Facebook and
  LinkedIn fall back to their Open Graph data, which is all they publish
- `/skills` prints exactly what it can and cannot do — generated from the same table the
  persona prompt uses, so the answer can never drift from the code
- YouTube links too: title, channel, description and public captions — no browser, no key
- Fetches politely and resiliently: per-host pacing, backoff + jitter on transient errors, and a per-host circuit breaker, so a site never blocks the bot
- Reads RSS/Atom feeds — send a feed URL, or `/feed <url>`, and get its latest items
- `/papers` (arXiv), `/define` (dictionary), `/holiday` (public holidays) — all keyless
- Prices come from several sources including a jsDelivr CDN rate table (FX + crypto + gold/silver in one file), so they keep working even when finance APIs are blocked
- Looks things up live (prices, weather, news) with no API key — gold, silver, USD/INR, and crypto (Binance), each with several fallback sources
- Remembers the links you share, so you can ask about them later
- Does exact arithmetic on request
- Sends voice replies using text-to-speech
- Picks a fitting emoji reaction for each message
- Uses Telegram's own formatting properly: bold only on the phrase that matters, bullet
  lists for 3+ items, `code` for commands and paths, `>` quotes, and `>>` for an
  **expandable** quote that stays collapsed behind "show more"

---

## Requirements

- Python 3.10 or newer (tested on 3.12; nothing in the code needs newer than 3.9)
- A Telegram bot token from BotFather
- A Google AI Studio API key
- A Groq API key

---

## Setup

Clone the repo and install the dependencies:

```bash
git clone https://github.com/bydenvik/avenor-ai-telegram-bot.git
cd avenor-ai-telegram-bot

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Create a local environment file from the template:

```bash
cp env.example.txt .env
```

The bot loads `.env` itself (via `python-dotenv`), so `python bot.py` picks it up.
Open `.env` and fill in the values you need. `env.example.txt` lists every supported variable — credentials, model selection, the text-to-speech voice, group behaviour, and memory limits.

Don't commit `.env`. In production, set the same values through your host's secret or environment settings.

---

## Running

```bash
python bot.py
```

The process starts and begins polling Telegram. The same command works under any process manager or hosting platform.

---

## Configuration

All runtime settings come from environment variables. See `env.example.txt` for the full list, the default values, and a short note on each one.

---

## Commands

`/start` `/help` `/skills` `/about` `/id` `/info` — getting started and what it can do

`/ai` `/research` `/image` — ask, look up the live web, make a picture

`/voice on|off` `/voicemessage` `/setvoice` `/voicepreview` — voice replies

`/remind 30m <task>` `/reminders` `/delremind <id>` — reminders

`/feed <url>` `/papers <topic>` `/define <word>` `/holiday [cc]` — keyless lookups

`/memory` (private chat only) `/forget` `/privacy` — your data

`/rules` `/setup` `/welcome` `/stats` `/members` — group tools (admin-only where marked)

`/diag` `/voicecreate` `/voicelist` `/voicefind` — owner only

Everything else works without a command: send a photo, a voice note, a document, a link,
or a plain sentence like "remind me in 2 hours to call mom".

---

## Project structure

```
avenor-ai-telegram-bot/
├── bot.py              Main application logic (kept as one file on purpose)
├── memory.py           Persistent JSON store for users and chats
├── requirements.txt    Python dependencies
├── env.example.txt     Environment variable template
├── privacy-policy.md   Privacy policy
├── welcome.png         Optional group-welcome image (NOT included - the bot falls back
│                       to a text card when it is absent)
├── LICENSE.txt         MIT
└── README.md
```

---

## Memory

Avenor keeps user and chat records in a JSON file (`avenor_memory.json` by default), and the
path is set through an environment variable. A file written under an earlier name of this bot
(`videx_memory.json`, `linnet_memory.json`) is read once and migrated automatically. Writes happen atomically, so an interrupted write won't corrupt the file. The exact record layout is defined in `memory.py`.

Memory is kept per user, keyed by their Telegram ID, and only ever injected into that person's own conversation — one person's facts are never shown to another. When enough notes accumulate, a background job merges and de-duplicates them into one short, clean profile. A user can see and wipe their own memory with `/memory` and `/forget`.

**Important — the host filesystem is ephemeral.** On most free hosts (Render's free plan included) any change to the local filesystem is lost on every **redeploy, restart, and spin-down**. A free service also sleeps after a period of inactivity, and sleeping wipes the filesystem too. On Render's free plan you **cannot attach a persistent disk** — disks need a paid service. So on free hosting:

- Keep the service awake (a keep-alive ping to `/health` every few minutes) so it does not sleep. This preserves memory between deploys.
- Accept that a redeploy still clears memory. For real durability, attach a persistent disk on a paid plan and point `MEMORY_FILE` at it.

---

## Deployment (Render free tier)

- Build command: `pip install -r requirements.txt`
- Start command: `python bot.py`  (not gunicorn/wsgi)
- Delete the `GEMINI_MODELS` env var if it is set — a stale value overrides the tuned built-in chain.
- Set `GROQ_API_KEY`, or Gemini rate limits leave the bot replying "busy".
- Run exactly **one** instance. Two instances cause `telegram.error.Conflict` and dropped messages.
- Free services sleep after ~15 minutes idle. Ping `/health` every 5 minutes from an uptime monitor to keep it awake (this also keeps memory alive).

---

## Failure modes

- **LLM timeout / rate limit** — the model chain falls through to the next provider; if all fail the bot replies with a short "busy" message.
- **Media download fails** — the message is still answered, with the attachment noted as unavailable.
- **TTS fails** — the text reply is already sent, so the answer is not lost.
- **Memory unavailable** — the bot still runs; it just has no long-term recall (see Memory above).
- **Reminder delivery fails** — the reminder is retried with exponential backoff (5 attempts,
  up to 30 minutes apart) and only deleted after it is actually delivered; if every attempt
  fails the user gets an explicit "could not be delivered" message instead of silence.

---

## Known limitations

These are known and accepted, not oversights. Each one is here so a future maintainer does
not rediscover it the hard way.

- **DNS rebinding window.** Every outbound fetch validates the URL structurally, resolves the
  host and rejects any private/loopback/link-local answer, and re-validates every redirect hop.
  The HTTP client then resolves the name again, so a hostile DNS server with a very short TTL
  could in principle answer public for our check and private for the connection. Pinning the
  connection to the validated IP was considered and rejected: it bypasses the egress proxy that
  many hosts (and most corporate networks) require, which would break every fetch. The impact
  is limited because the bot reaches no internal services.
- **Single-instance only.** State lives in one JSON file, so two polling processes would
  overwrite each other and Telegram would return `Conflict`. One instance per token.
- **The JSON store does not scale indefinitely.** It is rewritten as a whole file, which is
  fine for a community bot and wrong for thousands of very active chats. The migration path is
  SQLite; the accessor functions in `memory.py` are the seam to swap.
- **Social reading is best-effort.** The X, Reddit, Mastodon, Bluesky, TikTok and Telegram
  endpoints are public and undocumented for this use, so any of them can rate-limit, change
  shape, or block a datacenter IP without notice. Every one of them falls back to the page's
  Open Graph data, and a failed read never fails the reply.
- **Model-emitted action tags are trusted.** `[remember:]`, `[remind:]` and `[file:]` in the
  model's own output trigger real actions. User text cannot inject them (they are stripped from
  the transcript, and stored memory is sanitised on read), but a page the model *reads* could in
  theory steer one. Every action is rate-limited per chat, and the stored text is sanitised.
- **`bot.py` is one large file on purpose** (see Contributing). It is reviewed and tested as a
  unit; splitting it would need the shared prompt, model chain and delivery path to become
  explicit interfaces first.

---

## Data handling

- All user and chat records are stored locally in the memory file. There's no external database.
- Message content goes to Google Gemini and, when fallback kicks in, to Groq. If you run Avenor in a regulated environment, review the provider policies that apply to you.
- Credentials come from environment variables and are not part of the repository.

If you run Avenor as a public service, you are the data controller for the records it stores. See `privacy-policy.md`.

---

## Contributing

Issues and pull requests are welcome.

- Don't commit secrets or `.env`.
- If you add a config variable, document it in `env.example.txt`.
- Keep `bot.py` as a single file. That's on purpose.

---

## Related projects

- **Ongoing Links** — A Telegram community bot with a force-join gate, welcome messages, referral tracking, and AI chat.
  Repository: https://github.com/bydenvik/Ongoing_Links_Bot
  Bot: https://t.me/ANIME_WORLD_8BOT

---

## Author

Deepak Danodia

- GitHub: https://github.com/bydenvik
- X: https://x.com/bydenvik
- Reddit: https://www.reddit.com/u/bydenvik

---

## License

MIT — see `LICENSE.txt`. Every source file carries an `SPDX-License-Identifier: MIT` header.

Third-party dependencies keep their own licenses — notably **python-telegram-bot** (LGPL-3.0),
**google-genai** (Apache-2.0) and **httpx** (BSD-3-Clause). See `requirements.txt` and each
package for the full text.

## Privacy

See `privacy-policy.md`, or send `/privacy` in the bot. In short: memory and reminders are
stored **per user, keyed by Telegram ID**; message content is sent to Google Gemini (and Groq
as a fallback) to produce replies; media is processed but not kept; `/memory` shows what is
held and `/forget` deletes it.

Group commands: `/rules`, `/setup`, `/welcome`, `/stats`, `/members`.
