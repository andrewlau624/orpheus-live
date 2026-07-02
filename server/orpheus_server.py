"""Reference Orpheus TTS server for the remote backend — run this on your NVIDIA GPU box.

    pip install "orpheus-live[server] @ ."     # torch, transformers, snac, fastapi, uvicorn
    python -m server.orpheus_server            # serves on :8000

Then point the Mac app at it:

    ORPHEUS_LIVE_TTS_BACKEND=remote ORPHEUS_LIVE_TTS_REMOTE_URL=http://<gpu-host>:8000 make run

Wire contract (what `engines/tts_remote.RemoteOrpheusVoice` expects):

    POST /tts   {"text","voice","temperature","top_p","repetition_penalty"}
    -> 200, streaming body of raw little-endian float32 PCM @ 24kHz mono
       (chunks may split mid-sample; the client reassembles on 4-byte boundaries)

This reference uses transformers + the `snac` package and streams SNAC frames as they
decode. It mirrors the local MLX path's token handling (7 codes/frame, offset 128266,
SOA 128257, EOS 128258) so both backends sound the same. It is CANNOT be exercised on
Apple Silicon and was NOT run in CI — verify it on the GPU box. For production latency,
swap the transformers generate loop for a vLLM engine (see canopyai/Orpheus-TTS); the
HTTP contract above stays identical.
"""

from __future__ import annotations

import os
import struct

CODES_PER_FRAME = 7
SOA_TOKEN = 128257  # start-of-audio
EOS_TOKEN = 128258  # end-of-speech
CODE_OFFSET = 128266  # audio tokens are codes + this offset
SAMPLE_RATE = 24000
CONTEXT_FRAMES = 4  # left-context frames for seamless streaming decode
CHUNK_FRAMES = 12  # SNAC frames decoded per streamed chunk after the first
FIRST_CHUNK_FRAMES = 3  # tiny first chunk -> low time-to-first-byte

# Default HF repos, env-overridable. canopylabs/orpheus-3b-0.1-ft is GATED (accept the
# license on its HF page + authenticate with HF_TOKEN); unsloth/orpheus-3b-0.1-ft mirrors
# the same weights ungated if you'd rather skip that.
ORPHEUS_MODEL = os.environ.get("ORPHEUS_MODEL", "canopylabs/orpheus-3b-0.1-ft")
SNAC_MODEL = os.environ.get("SNAC_MODEL", "hubertsiuzdak/snac_24khz")


def _build() -> object:
    """Construct the FastAPI app. Imported lazily so this module documents even without deps."""
    import numpy as np
    import torch
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    from snac import SNAC
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # fp16 everywhere CUDA: bf16 needs Ampere+ (Colab's free T4 is Turing and silently
    # produces garbage with it); fp16 works on every CUDA GPU Orpheus fits on.
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(ORPHEUS_MODEL)
    # transformers v5 renamed `torch_dtype=` to `dtype=` (Colab ships v5 now);
    # try the new name first and fall back for v4 installs.
    try:
        model = AutoModelForCausalLM.from_pretrained(ORPHEUS_MODEL, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(ORPHEUS_MODEL, torch_dtype=dtype)
    model = model.to(device).eval()
    snac = SNAC.from_pretrained(SNAC_MODEL).to(device).eval()

    def sample_token(
        logits: torch.Tensor,
        recent: list[int],
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> int:
        """Temperature + top-p + repetition-penalty sampling (one token).

        Orpheus NEEDS this: greedy argmax (and rep penalty < ~1.1) makes it loop and
        repeat phrases -- the same stability floor the local MLX path enforces.
        """
        logits = logits[0].float()
        if repetition_penalty != 1.0 and recent:
            idx = torch.tensor(sorted(set(recent)), device=logits.device)
            picked = logits[idx]
            logits[idx] = torch.where(
                picked > 0, picked / repetition_penalty, picked * repetition_penalty
            )
        if temperature <= 0:
            return int(torch.argmax(logits))
        probs = torch.softmax(logits / temperature, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        sorted_probs[cum - sorted_probs > top_p] = 0.0
        sorted_probs /= sorted_probs.sum()
        return int(sorted_idx[torch.multinomial(sorted_probs, 1)])

    def codes_to_layers(flat: list[int]):
        l1, l2, l3 = [], [], []
        for i in range(len(flat) // CODES_PER_FRAME):
            b = flat[CODES_PER_FRAME * i : CODES_PER_FRAME * i + CODES_PER_FRAME]
            l1.append(b[0])
            l2.append(b[1] - 4096)
            l3.append(b[2] - 2 * 4096)
            l3.append(b[3] - 3 * 4096)
            l2.append(b[4] - 4 * 4096)
            l3.append(b[5] - 5 * 4096)
            l3.append(b[6] - 6 * 4096)
        t = lambda xs: torch.tensor([xs], device=device)  # noqa: E731
        return [t(l1), t(l2), t(l3)]

    def decode(flat: list[int]) -> np.ndarray:
        with torch.inference_mode():
            audio = snac.decode(codes_to_layers(flat)).squeeze().float().cpu().numpy()
        return audio.reshape(-1)

    def prompt_ids(text: str, voice: str):
        ids = tokenizer(f"{voice}: {text}", return_tensors="pt").input_ids
        start = torch.tensor([[128259]], dtype=torch.long)
        end = torch.tensor([[128009, 128260]], dtype=torch.long)
        return torch.cat([start, ids, end], dim=1).to(device)

    def audio_codes(generated: list[int]) -> list[int]:
        """Flat SNAC codes from generated tokens (crop after last SOA, drop EOS, de-offset)."""
        if SOA_TOKEN in generated:
            generated = generated[len(generated) - 1 - generated[::-1].index(SOA_TOKEN) + 1 :]
        codes = [t - CODE_OFFSET for t in generated if t != EOS_TOKEN]
        return codes[: (len(codes) // CODES_PER_FRAME) * CODES_PER_FRAME]

    class Req(BaseModel):
        text: str
        voice: str = "tara"
        temperature: float = 0.6
        top_p: float = 0.9
        repetition_penalty: float = 1.2

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True, "device": device, "dtype": str(dtype)}

    @app.post("/tts")
    def tts(req: Req):
        def gen_pcm():
            input_ids = prompt_ids(req.text, req.voice)
            past = None
            generated: list[int] = []
            emitted = 0  # frames already sent
            spf: int | None = None
            cur = input_ids
            for _ in range(1200):  # max tokens
                with torch.inference_mode():
                    out = model(cur, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :]
                nxt = sample_token(
                    logits,
                    generated[-64:],  # penalize a sliding window of recent tokens
                    req.temperature,
                    req.top_p,
                    req.repetition_penalty,
                )
                generated.append(nxt)
                cur = torch.tensor([[nxt]], device=device)
                if nxt == EOS_TOKEN:
                    break
                if len(generated) % CODES_PER_FRAME:
                    continue
                flat = audio_codes(generated)
                total = len(flat) // CODES_PER_FRAME
                target = FIRST_CHUNK_FRAMES if emitted == 0 else CHUNK_FRAMES
                while total - emitted >= target:
                    start = max(0, emitted - CONTEXT_FRAMES)
                    win = flat[start * CODES_PER_FRAME : (emitted + target) * CODES_PER_FRAME]
                    audio = decode(win)
                    if spf is None:
                        spf = audio.shape[0] // (len(win) // CODES_PER_FRAME)
                    trim = (emitted - start) * (spf or 0)
                    chunk = audio[trim:]
                    if chunk.size:
                        yield struct.pack(f"<{chunk.size}f", *chunk.tolist())
                    emitted += target
                    target = CHUNK_FRAMES
            # flush remaining frames
            flat = audio_codes(generated)
            total = len(flat) // CODES_PER_FRAME
            if total > emitted:
                start = max(0, emitted - CONTEXT_FRAMES)
                win = flat[start * CODES_PER_FRAME :]
                audio = decode(win)
                trim = (emitted - start) * (spf or 0)
                chunk = audio[trim:]
                if chunk.size:
                    yield struct.pack(f"<{chunk.size}f", *chunk.tolist())

        return StreamingResponse(gen_pcm(), media_type="application/octet-stream")

    return app


app = None  # populated on first run; kept lazy so the module imports without GPU deps


def main() -> None:
    import os

    import uvicorn

    global app
    app = _build()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
