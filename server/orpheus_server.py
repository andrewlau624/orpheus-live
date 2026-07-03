"""Reference Orpheus TTS server for the remote backend — run this on your NVIDIA GPU box.

    pip install "orpheus-live[server] @ ."     # vllm, transformers, snac, fastapi, uvicorn
    python -m server.orpheus_server            # serves on :8000

Then point the Mac app at it:

    ORPHEUS_LIVE_TTS_BACKEND=remote ORPHEUS_LIVE_TTS_REMOTE_URL=http://<gpu-host>:8000 make run

Wire contract (what `engines/tts_remote.RemoteOrpheusVoice` expects):

    POST /tts   {"text","voice","temperature","top_p","repetition_penalty"}
    -> 200, streaming body of raw little-endian float32 PCM @ 24kHz mono
       (chunks may split mid-sample; the client reassembles on 4-byte boundaries)

Token generation runs on **vLLM** (paged attention + continuous batching + CUDA graphs),
which is what lets a 3B model synthesize *faster than realtime* on a T4 — the naive
`transformers` generate loop runs at ~0.1x realtime, so streamed audio underruns into
choppy stop-and-go no matter how it's chunked. SNAC decoding stays local to this process.

Streaming cadence mirrors canopyai/Orpheus-TTS: one SNAC frame is emitted per 7 new tokens
using an overlap-save window (decode a few frames of left context, keep only the newest
frame's samples) so chunk seams are seamless. Token handling matches the local MLX path
(7 codes/frame, offset 128266, SOA 128257, EOS 128258) so both backends sound the same.

CANNOT be exercised on Apple Silicon and is NOT run in CI — verify it on the GPU box.
"""

import itertools
import os
import struct

# SNAC's snake activation is TorchScript-fused at runtime, which invokes nvrtc; on
# hosts with a mixed CUDA toolchain (e.g. Colab: cu12 system toolkit, cu13 pip torch)
# nvrtc can't find libnvrtc-builtins and every decode crashes. Eager mode costs
# nothing measurable for SNAC's small decoder. Must be set before torch imports.
os.environ.setdefault("PYTORCH_JIT", "0")

from pydantic import BaseModel

CODES_PER_FRAME = 7
SOA_TOKEN = 128257  # start-of-audio
EOS_TOKEN = 128258  # end-of-speech
CODE_OFFSET = 128266  # audio tokens are codes + this offset
SAMPLE_RATE = 24000
CONTEXT_FRAMES = 4  # left-context frames decoded for a seamless (overlap-save) seam
FIRST_CHUNK_FRAMES = 3  # small first chunk -> low time-to-first-byte, but not tiny
CHUNK_FRAMES = 12  # frames per streamed chunk (~0.7s @24kHz); big enough to absorb jitter
MAX_TOKENS = 1200  # generation cap (~ a long sentence of audio)

# Default HF repos, env-overridable. canopylabs/orpheus-3b-0.1-ft is GATED (accept the
# license on its HF page + authenticate with HF_TOKEN); unsloth/orpheus-3b-0.1-ft mirrors
# the same weights ungated if you'd rather skip that.
ORPHEUS_MODEL = os.environ.get("ORPHEUS_MODEL", "canopylabs/orpheus-3b-0.1-ft")
SNAC_MODEL = os.environ.get("SNAC_MODEL", "hubertsiuzdak/snac_24khz")

# vLLM engine knobs (env-overridable so a bigger GPU can lift them).
MAX_MODEL_LEN = int(os.environ.get("ORPHEUS_MAX_MODEL_LEN", "2048"))
GPU_MEM_UTIL = float(os.environ.get("ORPHEUS_GPU_MEM_UTIL", "0.90"))


class Req(BaseModel):
    # Defined at MODULE scope on purpose: FastAPI resolves the `tts(req: Req)` annotation
    # by name at introspection time. Nested inside _build() it can't be found, so FastAPI
    # falls back to treating `req` as a query param -> every POST 422s with
    # {"loc": ["query", "req"], "msg": "Field required"}. Keep this here.
    text: str
    voice: str = "tara"
    temperature: float = 0.6
    top_p: float = 0.9
    repetition_penalty: float = 1.2


def _build() -> object:
    """Construct the FastAPI app. Imported lazily so this module documents even without deps."""
    import numpy as np
    import torch
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from snac import SNAC
    from transformers import AutoTokenizer
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # fp16 everywhere CUDA: bf16 needs Ampere+ (Colab's free T4 is Turing and silently
    # produces garbage with it); fp16 works on every CUDA GPU Orpheus fits on.
    dtype = "float16" if device == "cuda" else "float32"

    # vLLM drives token generation. enforce_eager avoids CUDA-graph capture, which keeps
    # startup fast and memory low on a 15GB T4 (the graphs buy little for a single stream).
    engine = AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(
            model=ORPHEUS_MODEL,
            dtype=dtype,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEM_UTIL,
            enforce_eager=True,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(ORPHEUS_MODEL)
    snac = SNAC.from_pretrained(SNAC_MODEL).to(device).eval()
    req_ids = itertools.count()

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

    def prompt_ids(text: str, voice: str) -> list[int]:
        ids = tokenizer(f"{voice}: {text}", return_tensors="pt").input_ids[0].tolist()
        return [128259, *ids, 128009, 128260]

    def audio_codes(generated: list[int]) -> list[int]:
        """Flat SNAC codes from generated tokens (crop after last SOA, drop EOS, de-offset)."""
        if SOA_TOKEN in generated:
            generated = generated[len(generated) - 1 - generated[::-1].index(SOA_TOKEN) + 1 :]
        codes = [t - CODE_OFFSET for t in generated if t != EOS_TOKEN]
        return codes[: (len(codes) // CODES_PER_FRAME) * CODES_PER_FRAME]

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True, "device": device, "dtype": dtype, "engine": "vllm"}

    @app.post("/tts")
    async def tts(req: Req):
        async def gen_pcm():
            sampling = SamplingParams(
                temperature=req.temperature,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                max_tokens=MAX_TOKENS,
                stop_token_ids=[EOS_TOKEN],
            )
            prompt = {"prompt_token_ids": prompt_ids(req.text, req.voice)}
            request_id = f"tts-{next(req_ids)}"

            emitted = 0  # frames already sent
            spf: int | None = None  # samples per SNAC frame (learned on first decode)

            def emit(flat: list[int], final: bool):
                """Emit whole new frames using overlap-save; returns packed PCM bytes list."""
                nonlocal emitted, spf
                out: list[bytes] = []
                total = len(flat) // CODES_PER_FRAME
                target = FIRST_CHUNK_FRAMES if emitted == 0 else CHUNK_FRAMES
                while total - emitted >= target or (final and total > emitted):
                    take = min(target, total - emitted)
                    if take <= 0:
                        break
                    start = max(0, emitted - CONTEXT_FRAMES)
                    win = flat[start * CODES_PER_FRAME : (emitted + take) * CODES_PER_FRAME]
                    audio = decode(win)
                    if spf is None:
                        spf = audio.shape[0] // (len(win) // CODES_PER_FRAME)
                    chunk = audio[(emitted - start) * (spf or 0) :]
                    if chunk.size:
                        out.append(struct.pack(f"<{chunk.size}f", *chunk.tolist()))
                    emitted += take
                    target = CHUNK_FRAMES
                return out

            # vLLM yields the CUMULATIVE token list each step; recompute flat codes and
            # drain whole new frames as they land, so audio streams out during generation.
            generated: list[int] = []
            async for out in engine.generate(prompt, sampling, request_id):
                generated = list(out.outputs[0].token_ids)
                for pcm in emit(audio_codes(generated), final=False):
                    yield pcm
            # flush the trailing partial-window frames once generation is done.
            for pcm in emit(audio_codes(generated), final=True):
                yield pcm

        return StreamingResponse(gen_pcm(), media_type="application/octet-stream")

    return app


app = None  # populated on first run; kept lazy so the module imports without GPU deps


def main() -> None:
    import uvicorn

    global app
    app = _build()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
