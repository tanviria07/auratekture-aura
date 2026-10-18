# Aura: the voice concierge for Auratekture

Aura is the talking face of [Auratekture](https://auratekture.base44.app). Auratekture turns one photo of a room into a styled, ready-to-buy plan. It works out what the room is missing, finds real products from real stores, shows them in an AI image of your own room, and lets you pay with one signed approval in the Visa sandbox.

With Aura you can do all of it by voice, from the home page to checkout. Two steps stay with your own tap: choosing the photo, and signing the payment approval.

This repo is the Aura worker. The app itself is built on Base44 ([tanviria07/auratekture](https://github.com/tanviria07/auratekture)).

## How it works

```
Browser (Auratekture on Base44)              This worker (agent.py)
  mic, avatar video, the page                  Silero VAD -> speech-to-text
        |                                       -> Gemini Flash with page tools
        |  LiveKit Cloud room                   -> text-to-speech
        +--------------------------------------  -> Synthesia avatar (video + lip sync)
  "context" topic: what's on screen  ------>
  "ui" topic: actions to perform     <------
```

1. **Join:** "Talk to Aura" calls the app's `livekitToken` function, which creates a room token and dispatches this worker by name (`aura`).
2. **Read the page:** the page sends a short summary of what's on screen on the `context` topic. It covers the current step, the room facts, the numbered products with store and price, the total against the budget, and any warnings.
3. **Act on the page:** Aura sends actions on the `ui` topic, and the page runs them with the same code its buttons use.
4. **Speak up:** when the page reaches a new step, Aura speaks without being asked. The steps are: photo analyzed, products ready, picture ready, image updated, payment approved, payment blocked.

## What Aura can do

| Tool | What it does |
|---|---|
| `get_screen` | Reads what the user currently sees |
| `new_room` / `open_room` | Starts a new room, or opens a saved one by name |
| `request_photo` | Makes the upload button glow (browsers don't let a page open the file picker) |
| `set_room` | Fills in purpose, ceiling height, people, budget, and the needed-by date |
| `mark_already_here` | Marks things the room already has, so they're never bought |
| `design_room` | Runs "Design my room": plan, real products, room image |
| `focus_product` | Opens a product's card and highlights its number on the image |
| `swap_product` | Swaps a product for one of its alternatives |
| `update_image` | Redraws the room image after a swap |
| `open_checkout` | Opens checkout, where the user signs the approval themselves |

## Rules Aura never breaks

- **No paying.** Aura can't approve, sign, or pay. The Ed25519 approval is signed in the browser, by the user's own tap, and this worker never has the key.
- **No invented facts.** Aura only mentions products, prices, stores, and checks that the page reports.
- **No guessing the purpose.** Aura asks what the room is for instead of guessing from the photo.
- **The budget is a limit.** Aura treats it as a cap, not a target.

## Models

Everything runs through LiveKit Inference, so this worker needs no extra model keys.

| Part | Default | Change with |
|---|---|---|
| Speech-to-text | `deepgram/nova-3` | `AURA_STT` |
| Language model | `google/gemini-3.5-flash` | `AURA_LLM` |
| Text-to-speech | `deepgram/aura-2`, voice `odysseus` | `AURA_TTS`, `AURA_VOICE` |
| Avatar | Synthesia Interactive Avatar | `SYNTHESIA_AVATAR_ID` |
| Voice activity | Silero VAD | |

If the avatar can't start, Aura keeps working voice-only, and the page shows an animated orb instead of video.

## Run it

You need Python 3.10 or newer, a LiveKit Cloud project, and a Synthesia API key with Interactive Avatars access.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
copy .env.example .env          # Windows (cp on macOS / Linux), then fill it in
python agent.py dev
```

When the worker is ready it logs `registered worker {"agent_name": "aura", ...}`. Open the app and click **Talk to Aura**.

`.env` holds the following:

| Variable | What it is |
|---|---|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | Your LiveKit Cloud project. All three must come from the same project. |
| `SYNTHESIA_API_KEY` | Your Synthesia API key |
| `SYNTHESIA_AVATAR_ID` | The avatar's Interactive ID |

Keys live only in `.env`, which is git-ignored.

On startup, `python agent.py dev` prints a note suggesting `lk agent dev`. You can ignore it; the worker runs normally.

Built solo for LA Hacks.
