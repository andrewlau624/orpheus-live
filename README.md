# Orpheus Live

A fully local, voice-to-voice conversational AI on Apple Silicon, aiming to simulate
human conversational cognition — not just turn-taking, but hesitation, silence,
barge-in, and thinking-while-listening.

**Pipeline (all on-device):**
`mic → silero-vad → mlx-whisper (STT) → Ollama llama3.2:3b → Orpheus-TTS via mlx-audio → speakers`

Every launch without a preset picks a **random voice** (one of Orpheus's eight base
voices plus randomized delivery params) so you can audition voices in real conversation
and save the ones you like as named presets.

## Setup

```bash
uv sync
make pull-models   # ollama pull llama3.2:3b + llama3.2:1b (cognition)
```

## Run

```bash
make run           # random voice
make run Emma      # load the saved preset "Emma"
make voices        # list saved presets
ollama serve &      # if not already running
```

- Like the current voice? **Type a name + Enter** (e.g. `Emma`) — it saves to
  `saved_voices/Emma.json` and `make run Emma` brings it back exactly.
- **Talk over it** — the AI keeps talking through short "yeah"/"mm-hm" adlibs (they're
  classified as backchannels), and stops mid-sentence only once it's confident you're
  actually taking the turn; sustained speech always stops it.
- Go quiet for a while and it may decide, out loud, to break the silence (its inner
  monologue prints dim in the terminal either way).

> First launch asks for **microphone permission** — approve it. The Orpheus model
> (~1.9 GB, `mlx-community/orpheus-3b-0.1-ft-4bit`) downloads to the Hugging Face
> cache on first run.

**Headphones recommended** — without them the AI hears itself through the mic and can
false-trigger barge-in.

## Human-cognition features

- **Inner-monologue memory (no transcript)** — the bot keeps *no* verbatim chat history.
  Its only memory is a running first-person *inner monologue*: after each exchange it writes
  itself a short impression ("sounds like they just got a puppy and it's chewing everything")
  and that — not the words spoken — is all it carries forward. Replies are conditioned on
  those impressions plus whatever was just heard. Like a real person, it remembers the gist
  and forgets specifics (ask it your dog's name later and it may only recall "a new puppy").
  Each note prints dim in the console as it forms.
- **Silence cognition** — during lulls, a dedicated small model (`llama3.2:1b`) is
  consulted on a jittered schedule for a structured decision (`speak`/`wait` + urgency +
  a first-person `thought`, always printed dim). Escalation is prompt-driven, so
  breaking the silence reads as fading hesitation rather than a timer.
- **LLM-driven turn-taking (VAD is just a gate)** — VAD no longer decides when your turn
  ends; it only detects speech vs. a pause. On every pause the 1b model sees the rough
  transcript so far and whether the AI is talking, and picks one move: **speak** (you seem
  done — respond), **wait** (mid-thought — keep listening), **backchannel** (drop a "yeah"
  without taking the turn), or **interrupt** (you clearly cut in — stop and yield). The mic
  is captured 24/7, so interruptions are heard as real turns, not dropped. A long-silence
  safety net forces a response so a turn can never hang. Its `thought` prints dim.
- **Thinking-while-listening** — while you're mid-sentence, partial transcripts drive a
  speculative reply generation. The reply is streamed token-by-token, and the moment its
  first sentence is complete its audio starts synthesizing (into a buffer) — *before you
  even stop talking*. If your final words match what it bet on, that buffered audio starts
  playing the instant the model decides to speak (`(reply was ready)` in the console).

All tunables live in `config.py` as pydantic settings, overridable via
`ORPHEUS_LIVE_*` env vars.

## Remote TTS on an NVIDIA GPU (optional)

TTS is the latency bottleneck on Apple Silicon (Orpheus generates slower than realtime).
To hit vLLM-class latency, offload *only synthesis* to an Orpheus server on an NVIDIA GPU
while STT + the LLM stay local — flip one env var, no code change:

```bash
# On the GPU box (H100/H200/etc.):
pip install "orpheus-live[server] @ ."
python -m server.orpheus_server            # serves POST /tts on :8000

# On the Mac:
ORPHEUS_LIVE_TTS_BACKEND=remote \
ORPHEUS_LIVE_TTS_REMOTE_URL=http://<gpu-host>:8000 make run
```

### Don't have a GPU? Use a free Google Colab one

Open **`colab/orpheus_tts_server.ipynb`** in Colab (`File → Upload notebook`, or push this
repo to GitHub and open it from there), set the runtime to **GPU**, and run every cell. It
installs the server deps, downloads Orpheus, starts `server/orpheus_server.py`, and opens a
free public **Cloudflare tunnel** — the last cell prints the exact command to paste on your
Mac, e.g.:

```bash
ORPHEUS_LIVE_TTS_BACKEND=remote \
ORPHEUS_LIVE_TTS_REMOTE_URL=https://<random>.trycloudflare.com make run
```

Keep the notebook tab open for the whole conversation. The tunnel URL changes each session
(rerun the last cell to get a fresh one). A T4 is ~realtime; on it you can add
`ORPHEUS_LIVE_TTS_STREAMING=1` on the Mac for lower first-audio latency (leave it off if you
hear chop). Only synthesis leaves your machine — mic, STT, and the LLM stay local.

The client (`RemoteOrpheusVoice`) streams the server's float32 PCM chunks straight into the
same persistent output stream, so remote audio plays (and barges-in) exactly like local.
`server/orpheus_server.py` is a **reference** built on `transformers` + `snac` — for
production latency swap its generate loop for a vLLM engine (canopyai/Orpheus-TTS); the
HTTP contract is unchanged. The default backend stays `mlx` (fully local); `remote` is
opt-in and never affects Apple-Silicon installs.

## Development

```bash
make lint     # ruff check
make format   # ruff format
make test     # pytest (fakes only — no model weights needed)
```

CI (`.github/workflows/ci.yml`) runs lint + tests against fakes; it deliberately never
installs `torch`/`mlx-audio`/`mlx-whisper` (Apple-Silicon-only) or downloads model
weights.

## Project layout

```
src/orpheus_live/
  config.py          # pydantic Settings — every tunable, env-overridable
  models/            # pydantic domain models + StrEnums (voice, conversation, cognition)
  console.py         # terminal colors + log()
  audio/
    capture.py       # AudioIn: mic capture, VAD segmentation, barge-in detection
    playback.py      # AudioSink (persistent output stream) + SpeechPlayer (streamed, cancellable)
    vad.py           # silero-vad wrapper (per-frame speech probability)
  engines/
    base.py          # TtsBackend Protocol — the interchangeable TTS interface
    factory.py       # load_voice(): pick the local (mlx) or remote (GPU) backend
    stt.py           # mlx-whisper wrapper
    llm.py           # Ollama Brain: replies, silence-breakers, interjections
    tts.py           # Orpheus via mlx-audio: seamless streaming decode, presets, save/load
    tts_remote.py    # RemoteOrpheusVoice: stream synthesis from a GPU server over HTTP
    sanitize.py      # clean_for_tts() / strip_markers() — Orpheus input hygiene
  core/
    conversation.py  # Conversation: orchestrates everything into the live loop
    cognition.py     # SilenceCognition + InterruptMonitor + overlap classifier
    speculation.py   # Speculator: reply generation from partial transcripts
tests/               # unit tests against fakes (see tests/conftest.py)
server/              # reference Orpheus GPU server for the remote backend (deploy-only)
saved_voices/        # named voice presets (<Name>.json)
```

## How the voice works

- Orpheus has eight base voices (`tara leah jess leo dan mia zac zoe`); a **preset** is
  one base voice plus delivery params (`temperature`, `top_p`, `repetition_penalty`,
  which must be ≥ 1.1 for stable output).
- A random launch rolls all of those fresh and keeps them fixed for the session, so the
  greeting and every reply match; saving writes them to JSON, reproducible exactly.
- Replies may include Orpheus emotion tags — `<laugh> <chuckle> <sigh> <gasp> <groan>
  <yawn> <cough> <sniffle>` — which the LLM sprinkles in sparingly.
- Replies play as complete sentences through a single persistent output stream
  (pipelined: sentence N+1 is synthesized while N plays, so the only wait is the first).
  Each sentence is decoded and buffered fully before it plays — Orpheus on Apple Silicon
  generates slower than realtime, so playing chunks as they arrive underruns into choppy
  silence between chunks. Speculation (below) is what keeps the first sentence's latency
  low: its audio is synthesized during your speech, so its buffer is already full when you
  stop. A barge-in still cuts playback instantly (the sink is cleared mid-sentence).
- A **corrected incremental decode** is available opt-in (`ORPHEUS_LIVE_TTS_STREAMING=1`)
  for faster-than-realtime hardware: it lowers first-audio latency and, unlike mlx-audio
  0.4.4's own streaming, its chunk seams are click-free (it trims exactly the overlap-save
  context by the *measured* samples-per-frame, instead of a fixed window that mismatches
  what it prepends — the cause of the "h-h-h-hi" repeat-a-beat glitch).
- Latency reality check: the Orpheus README's ~200ms figures are vLLM on datacenter
  GPUs (150–180ms TTFA on an H100, per canopyai/Orpheus-TTS#61). Local MLX on an M4
  is ~1.6s to first audio and slower-than-realtime generation — the speculation fast
  path and sentence pipelining are what keep the conversation feeling responsive.
# orpheus-live
