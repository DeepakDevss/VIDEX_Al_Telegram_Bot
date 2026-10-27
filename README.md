# VIDEX | AI Companion

A Telegram AI bot written in Python. It holds conversations, understands media, remembers users across sessions, and can reply with text or voice.

- Bot: https://t.me/Linnet_Agents_Bot  (the handle keeps its old name; the bot itself is VIDEX)
- Repository: https://github.com/DeepakDevss/VIDEX_Al_Telegram_Bot

---

## Overview

VIDEX runs as a single Python process and talks to Telegram through the Bot API. Google Gemini is the main language model. Groq is used as a fallback chat provider and handles the speech-to-text side. User and chat state lives in a local JSON file managed by `memory.py`.

The bot is meant to run as a long-lived service. It should work on any platform that can run a Python process and set environment variables.

---

## What it can do

- Replies in whatever language the user writes in
- Remembers users and chats between restarts
- Tells apart who said what in group conversations
- Reads photos, documents, voice notes, and audio
- Understands plain requests with no command — say "make me a report" and it attaches the file, or "draw a logo" and it makes the image
- Sets reminders and delivers them on time ("remind me in 2h to call mom"), with no external service
- Reads a link you paste and answers about the page — news, blogs, and social links (Instagram, X, Reddit, Facebook all expose Open Graph data, which is read directly)
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

---

## Requirements

- Python 3.10 or newer
- A Telegram bot token from BotFather
- A Google AI Studio API key
- A Groq API key

---

## Setup

Clone the repo and install the dependencies:

```bash
git clone https://github.com/DeepakDevss/VIDEX_Al_Telegram_Bot.git
cd VIDEX_Al_Telegram_Bot

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Create a local environment file from the template:

```bash
cp env.example.txt .env
```

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

Command handlers are registered in `bot.py`. Check that file for the full list of commands and exactly what each one does.

---

## Project structure

```
VIDEX_Al_Telegram_Bot/
├── bot.py              Main application logic (kept as one file on purpose)
├── memory.py           Persistent JSON store for users and chats
├── requirements.txt    Python dependencies
├── env.example.txt     Environment variable template
├── privacy-policy.md   Privacy policy
├── welcome.png         Image used in the group welcome message
├── LICENSE.txt         MIT
└── README.md
```

---

## Memory

VIDEX keeps user and chat records in a JSON file, and the path is set through an environment variable. Writes happen atomically, so an interrupted write won't corrupt the file. The exact record layout is defined in `memory.py`.

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

---

## Data handling

- All user and chat records are stored locally in the memory file. There's no external database.
- Message content goes to Google Gemini and, when fallback kicks in, to Groq. If you run VIDEX in a regulated environment, review the provider policies that apply to you.
- Credentials come from environment variables and are not part of the repository.

If you run VIDEX as a public service, you are the data controller for the records it stores. See `privacy-policy.md`.

---

## Contributing

Issues and pull requests are welcome.

- Don't commit secrets or `.env`.
- If you add a config variable, document it in `env.example.txt`.
- Keep `bot.py` as a single file. That's on purpose.

---

## Related projects

- **Ongoing Links** — A Telegram community bot with a force-join gate, welcome messages, referral tracking, and AI chat.
  Repository: https://github.com/DeepakDevss/Ongoing_Links_Bot
  Bot: https://t.me/ANIME_WORLD_8BOT

---

## Author

Deepak Devs

- GitHub: https://github.com/DeepakDevss
- X: https://x.com/Deepak_Devs
- Reddit: https://www.reddit.com/u/Deepak_Devs

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
