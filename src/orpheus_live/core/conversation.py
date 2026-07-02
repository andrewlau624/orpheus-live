"""Top-level orchestration: wires config, engines, and audio into the live loop."""

import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..audio import AudioIn, AudioSink, PreSynthStream, SpeechPlayer, Vad, split_sentences
from ..config import Settings, configure_hf_token
from ..config import settings as default_settings
from ..console import AI, DIM, SYS, YOU, log
from ..engines import GREETING, Brain, Transcriber, load_voice, resolve_backend, start_save_listener
from ..engines.sanitize import clean_for_tts, strip_markers
from ..engines.tts import random_preset
from ..models import CognitionAction, ConversationState, OverlapVerdict, VoicePreset
from .cognition import (
    SilenceCognition,
    consult,
    decide_turn,
    is_stop_command,
    quick_overlap_verdict,
)
from .speculation import Speculator

_WARMUP_LINE = "Give me just a second to warm up my voice."


def _first_complete_sentence(text: str) -> str | None:
    """The first fully-terminated sentence in partial reply `text`, or None.

    Used to trigger speculative pre-synth as soon as one sentence is done streaming.
    Requires a clear boundary (a following sentence, or terminal punctuation) so the
    pre-synth target matches what the full reply will split into.
    """
    parts = split_sentences(text)
    if not parts:
        return None
    if len(parts) >= 2:
        return parts[0]
    if parts[0].rstrip().endswith((".", "!", "?", "…")):
        return parts[0]
    return None


class Conversation:
    """One live voice conversation: mic -> STT -> LLM -> TTS -> speakers.

    Owns the shared state (speaking/user_speaking events, conversation state)
    and the three background loops: frame processing (inside AudioIn), silence
    cognition, and the listening tick (speculation + optional AI interrupts).
    """

    def __init__(self, settings: Settings, preset: VoicePreset | None = None):
        self.settings = settings
        self.state = ConversationState.IDLE

        self.voice = load_voice(settings, preset or random_preset(settings))
        self.transcriber = Transcriber(settings)
        self.brain = Brain(settings.ollama_model)

        self.speaking = threading.Event()  # set while the AI is talking
        self.user_speaking = threading.Event()  # set during the user's VAD-triggered speech
        self.speak_done_at = [0.0]
        self._speak_lock = threading.Lock()  # one speaker at a time, ever
        # Lag-aware pickup: set when the AI commits to a reply, cleared when its first audio
        # is audible. While set, AudioIn ignores the mic so the think-gap picks up no noise.
        self.muted = threading.Event()

        self.sink = AudioSink(settings)
        self.player = SpeechPlayer(
            self._stream_sentence,
            self.sink,
            buffer_whole=not settings.tts_streaming,
            on_first_audio=self._on_first_audio,
        )
        vad = Vad(settings.vad_threshold, settings.mic_sample_rate)
        self.audio_in = AudioIn(
            settings,
            vad,
            self.speaking,
            self.speak_done_at,
            self.user_speaking,
            on_pause=self._on_pause,
            muted=self.muted,
        )
        self._deciding = threading.Lock()  # one turn-decision in flight at a time
        self._decision_missed_final = False  # final=True was dropped while deciding

        # Every Orpheus generation (playback streaming + speculative pre-synth)
        # holds this lock: MLX runs one generation at a time, which keeps memory
        # bounded and avoids concurrent Metal command buffers.
        self._synth_lock = threading.Lock()
        self._tts_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-presynth")
        self._pre_synth: PreSynthStream | None = None  # in-flight speculative first sentence
        self._pre_synth_lock = threading.Lock()

        self.speculator = Speculator(
            self.brain.generate_stream,
            on_first_sentence=self._pre_synthesize,
            first_sentence=_first_complete_sentence,
        )
        self.cognition = SilenceCognition(
            settings, consult=lambda s, c: consult(settings.cognition_model, s, c)
        )

    # -- synthesis -----------------------------------------------------------

    def _run_pre_synth(self, ps: PreSynthStream) -> None:
        """Stream a speculative sentence into `ps` while the user is still talking."""
        try:
            with self._synth_lock:
                for chunk in self.voice.stream(clean_for_tts(ps.text)):
                    if ps.cancelled:
                        break
                    ps.add(chunk)
        except Exception:
            pass
        finally:
            ps.finish()

    def _stream_sentence(self, text: str) -> Iterator[np.ndarray]:
        """Yield audio for one sentence, reusing a matching speculative pre-synth if present.

        In the default (non-streaming) mode every sentence is fully buffered before it plays
        -- Orpheus is slower than realtime, so playing chunks as they arrive underruns into
        choppy silence. A speculation hit is still fast because its pre-synth ran during the
        user's speech, so its buffer is already full (or nearly) by the time we drain it.
        With `tts_streaming` on (faster-than-realtime HW), chunks play as they arrive for
        lower TTFB.
        """
        streaming = self.settings.tts_streaming
        ps = self._take_pre_synth(text)
        if ps is not None:
            try:
                if streaming:
                    yield from ps.iter_chunks()
                else:
                    chunks = list(ps.iter_chunks())  # wait for the full sentence -> smooth
                    if chunks:
                        yield np.concatenate(chunks)
            finally:
                ps.cancel()  # stop the background fill if playback was cut short
            return
        with self._synth_lock:
            if streaming:
                yield from self.voice.stream(clean_for_tts(text))
            else:
                yield self.voice.synthesize(clean_for_tts(text))

    def _take_pre_synth(self, text: str) -> PreSynthStream | None:
        """Consume the pre-synth stream if it was for `text`; cancel a stale one."""
        with self._pre_synth_lock:
            ps, self._pre_synth = self._pre_synth, None
        if ps is None or ps.text != text:
            if ps is not None:
                ps.cancel()
            return None
        return ps

    def _pre_synthesize(self, sentence: str) -> None:
        """Start streaming a speculative first sentence's audio (fired once per generation).

        Called by the Speculator the moment the reply's first complete sentence is known,
        while the user may still be talking -- so its head is buffered by the time they stop.
        """
        ps = PreSynthStream(sentence)
        with self._pre_synth_lock:
            old, self._pre_synth = self._pre_synth, ps
        if old is not None:
            old.cancel()
        self._tts_pool.submit(self._run_pre_synth, ps)

    def _cancel_pre_synth(self) -> None:
        """Cancel and drop any in-flight speculative pre-synth (wrong bet / turn taken)."""
        with self._pre_synth_lock:
            ps, self._pre_synth = self._pre_synth, None
        if ps is not None:
            ps.cancel()

    # -- turn-taking (the LLM decides, VAD only detects the pause) -----------

    def _cancel_playback(self) -> None:
        log("  (you cut in — stopping)", DIM)
        self.state = ConversationState.OVERLAP
        self.player.cancel()  # bumps the gen-id and clears the sink -> instant mid-sentence stop

    def _on_pause(self, audio: np.ndarray, final: bool) -> None:
        """Fired by AudioIn when the talker pauses; hand off to a worker (single-flight)."""
        if not self._deciding.acquire(blocking=False):
            # The safety-net final=True was dropped because Ollama is still thinking about
            # the previous pause. Flag the in-flight decision so it treats itself as final
            # (and never WAITs into a deadlock).
            if final:
                self._decision_missed_final = True
            return
        self._decision_missed_final = False
        ai_speaking = self.speaking.is_set()
        threading.Thread(target=self._decide, args=(audio, final, ai_speaking), daemon=True).start()

    def _decide(self, audio: np.ndarray, final: bool, ai_speaking: bool) -> None:
        """Transcribe the turn-so-far, ask cognition what to do, then act on it."""
        try:
            text = ""
            try:
                text = self.transcriber.transcribe(audio)
            except Exception:
                text = ""
            action = self._choose(text, ai_speaking, final)
        finally:
            # Release BEFORE acting so an interrupt can be judged while we speak.
            self._deciding.release()
        self._act(action, text, ai_speaking)
        # The long-silence safety net (final=True) can fire while we're mid-decision and
        # get dropped by the single-flight gate. If we didn't take the turn, honor it now
        # with the freshest audio, so a slow cognition model can never hang the turn.
        if self._decision_missed_final and not self.speaking.is_set():
            self._decision_missed_final = False
            fresh = self.audio_in.turn_audio()
            if fresh is not None:
                self._on_pause(fresh, final=True)

    def _choose(self, text: str, ai_speaking: bool, final: bool) -> CognitionAction:
        """Pick wait / speak / interrupt / backchannel for this pause."""
        words = text.split()
        if not words:
            # Nothing intelligible. When idle, end the empty turn on a long silence.
            return CognitionAction.SPEAK if (final and not ai_speaking) else CognitionAction.WAIT
        # Cheap heuristic tier first: obvious tiny acknowledgements are backchannels.
        verdict = quick_overlap_verdict(text)
        if verdict == OverlapVerdict.BACKCHANNEL and not final:
            return CognitionAction.BACKCHANNEL if ai_speaking else CognitionAction.WAIT
        if ai_speaking and is_stop_command(text):
            return CognitionAction.INTERRUPT  # explicit "stop"/"hold on" -> yield now, no consult
        if not ai_speaking and final:
            return CognitionAction.SPEAK  # safety net: never hang a finished turn
        # Everything else -- including an obvious barge-in while we're speaking -- goes to the
        # model. When speaking, it chooses to YIELD (interrupt) or HOLD the floor and talk over
        # (wait); giving it that agency is the point, so we don't shortcut the obvious case.
        try:
            decision = decide_turn(self.settings.cognition_model, text, ai_speaking)
            log(f"  (…{decision.thought})", DIM)
            return decision.action
        except Exception:
            # On a flaky model, don't cut the user off and don't hang: wait, unless the
            # long-silence safety net already fired (then respond).
            return CognitionAction.SPEAK if (final and not ai_speaking) else CognitionAction.WAIT

    def _act(self, action: CognitionAction, text: str, ai_speaking: bool) -> None:
        if ai_speaking:
            if action == CognitionAction.INTERRUPT:
                self._cancel_playback()
                self._respond(text)
            else:  # wait / backchannel while we're talking -> keep going, drop the overlap
                self.audio_in.reset_turn()
            return
        if action == CognitionAction.SPEAK:
            self._respond(text)
            return
        # WAIT or BACKCHANNEL while idle: the user may still be going. Don't drop the turn
        # -- AudioIn keeps accumulating. The long-silence safety net (final=True) will
        # force SPEAK on a quiet pause.

    def _on_first_audio(self) -> None:
        """The reply's first audio is now audible -> lift the lag-aware mic mute.

        From here the AI is really speaking, so normal barge-in resumes: the user can talk
        over it or cut in. (During the preceding generate+synth gap the mic was muted so
        room noise couldn't spawn extra cognition.)
        """
        self.muted.clear()

    def _respond(self, user_text: str) -> None:
        """Take the turn: this utterance is answered, so clear it and reply."""
        if self.settings.lag_aware:
            self.muted.set()  # mute the mic through generate+synth; _on_first_audio lifts it
        self.audio_in.reset_turn()
        self.state = ConversationState.PREPARING
        log(f"You: {user_text}", YOU)
        s = self.settings
        reply = None
        if s.speculate:
            reply = self.speculator.take(user_text, timeout=s.speculation_take_timeout_s)
            if reply:
                log("  (reply was ready)", DIM)
        if reply:
            # Speculation hit: the full reply (and its first sentence's audio) already
            # exists, so speak it as a whole. Memory is written after it's spoken.
            log(f"Voice: {strip_markers(reply)}", AI)
            self.speak(reply)
        else:
            # Miss: think and speak IN PARALLEL -- stream the reply sentence-by-sentence
            # into TTS, so the first sentence plays while the rest is still generating.
            self._cancel_pre_synth()  # wrong bet -> drop the speculative audio
            reply = self._speak_streaming(user_text)
            log(f"Voice: {strip_markers(reply)}", AI)
        self._remember(user_text, reply)

    def _speak_streaming(self, user_text: str) -> str:
        """Generate and speak concurrently; returns the full reply text once done."""
        self._speak_lock.acquire()
        try:
            self.state = ConversationState.SPEAKING
            self.speaking.set()
            self.speculator.reset()  # the turn is ours now; any bet on user text is void
            return self.player.speak_stream(self.brain.generate_stream(user_text))
        finally:
            self.muted.clear()  # safety: never leave the mic muted if synth produced no audio
            self.speak_done_at[0] = time.time()
            self.speaking.clear()
            self.state = ConversationState.IDLE
            self._speak_lock.release()

    def speak(self, text: str, *, skip_if_busy: bool = False) -> None:
        """Speak `text`; a confirmed interrupt during it stops playback early.

        The lock serializes every speech source (turn replies, silence cognition):
        two speakers at once would fight over the sink. Opportunistic speakers pass
        skip_if_busy so a queued-up silence-breaker doesn't play right after a reply.
        """
        if skip_if_busy:
            if not self._speak_lock.acquire(blocking=False):
                return
        else:
            self._speak_lock.acquire()
        try:
            self.state = ConversationState.SPEAKING
            self.speaking.set()
            self.speculator.reset()  # the turn is ours now; any bet on user text is void
            self.player.speak(text)
        finally:
            self.muted.clear()  # safety: never leave the mic muted if synth produced no audio
            self.speak_done_at[0] = time.time()
            self.speaking.clear()
            self.state = ConversationState.IDLE
            self._speak_lock.release()

    # -- background loops ------------------------------------------------------

    def _cognition_loop(self) -> None:
        """Silence self-questioning: 'should I talk? nah... maybe I should'.

        Only GENUINE mutual silence counts. The AI talking while the user listens is not
        awkward silence, and neither is:
          - the think-gap where a reply is being generated/synthesized (`muted` is set), nor
          - the beat right after the AI stops, where the user is expected to respond
            (a grace window after `speak_done_at`).
        Treating any of those as a lull to fill is exactly the "it talks to itself" bug.
        """
        while True:
            time.sleep(self.settings.cognition_tick_s)
            since_ai_spoke = time.time() - self.speak_done_at[0]
            if (
                self.speaking.is_set()
                or self.user_speaking.is_set()
                or self.muted.is_set()
                or since_ai_spoke < self.settings.cognition_post_ai_grace_s
            ):
                self.cognition.reset()
                continue
            self.cognition.note_silence_start()
            try:
                decision = self.cognition.tick()
            except Exception:
                continue
            if decision is not None and decision.action == CognitionAction.SPEAK:
                reply = self.brain.break_silence()
                log(f"Voice: {strip_markers(reply)}", AI)
                self.speak(reply, skip_if_busy=True)

    def _listening_tick_loop(self) -> None:
        """While the user talks, speculatively pre-generate the reply from partial transcripts.

        Runs whenever the user is speaking — even over the AI's own speech — so if their
        overlap turns into a real interrupt, the reply is already forming when we yield.
        """
        s = self.settings
        while True:
            time.sleep(s.speculation_interval_s)
            if not self.user_speaking.is_set():
                continue
            if not self.speaking.is_set():
                self.state = ConversationState.LISTENING
            snapshot = self.audio_in.turn_audio()
            if snapshot is None or snapshot.size < s.mic_sample_rate // 2:
                continue  # not enough audio yet to be worth a transcription
            try:
                partial = self.transcriber.transcribe(snapshot)
            except Exception:
                continue
            if partial and len(partial) >= 2:
                self.speculator.on_partial(partial)

    # -- main loop -------------------------------------------------------------

    def _remember(self, user_text: str, reply: str) -> None:
        """Write the inner-monologue impression of the exchange (the bot's only memory)."""
        try:
            note = self.brain.reflect(user_text, reply)
            log(f"  (…{note})", DIM)
        except Exception:
            pass

    def _warm_up(self) -> None:
        """Warm every engine; the (slow, separate-process) Ollama load runs in parallel."""
        log("  · warming up voice + transcriber...", DIM)
        brain_warm = threading.Thread(target=self.brain.warm_up, daemon=True)
        brain_warm.start()
        self.speak(_WARMUP_LINE)  # warms Orpheus + the SNAC decoder end-to-end
        self.transcriber.warm_up()
        brain_warm.join()
        log("Ready.\n", SYS)

    def _print_banner(self) -> None:
        p = self.voice.preset
        log("=" * 60, SYS)
        if p.name == "random":
            log(
                f"  Orpheus Live  ·  random voice: {p.voice} (t={p.temperature}, "
                f"p={p.top_p}, rp={p.repetition_penalty})",
                AI,
            )
            log("  Like this voice? Type a name + Enter to save it as a preset.", SYS)
            log("  Restart for a new random voice, or: make run <name>. Ctrl+C quits.", DIM)
        else:
            log(f"  Orpheus Live  ·  preset '{p.name}' ({p.voice})", AI)
            log("  Type a new name + Enter anytime to re-save this voice.", DIM)
        backend = resolve_backend(self.settings)
        mode = "streaming (low-latency)" if self.settings.tts_streaming else "buffered (smooth)"
        log(f"  TTS: {backend} backend · {mode} playback", DIM)
        log("=" * 60 + "\n", SYS)

    def run(self) -> None:
        try:
            self._warm_up()
            self._print_banner()

            # Listen for a preset name on the terminal alongside the mic.
            start_save_listener(self.voice)

            # Greet first (speaker output needs no permission), then open the mic.
            log(f"Voice: {strip_markers(GREETING)}", AI)
            self.speak(GREETING)

            log("  (opening microphone — approve the mic prompt if macOS asks)", DIM)
            self.audio_in.start()
            threading.Thread(target=self._cognition_loop, daemon=True).start()
            threading.Thread(target=self._listening_tick_loop, daemon=True).start()

            # Everything is event-driven now (AudioIn.on_pause -> _decide). Idle here until
            # Ctrl+C; the mic thread, speculation tick, and silence cognition run the show.
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            log("\nBye! 👋", SYS)
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        """Best-effort teardown; each step guarded so a second Ctrl+C can't crash exit."""
        for step in (self.player.cancel, self.audio_in.stop, self.sink.close):
            try:
                step()
            except (KeyboardInterrupt, Exception):
                pass


def run(settings: Settings | None = None, preset: VoicePreset | None = None) -> None:
    settings = settings or default_settings
    configure_hf_token()

    log("Loading models (first run downloads a few GB — grab a coffee)...", SYS)
    Conversation(settings, preset).run()
