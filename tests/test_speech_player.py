"""Tests for SpeechPlayer's sentence-by-sentence streaming and mid-sentence cancellation."""

from orpheus_live.audio.playback import SpeechPlayer, split_sentences


class FakeSink:
    """Records begin/write/flush/clear instead of touching sounddevice.

    Mirrors the epoch contract: writes tagged with a stale epoch (after clear())
    are dropped, exactly like the real AudioSink. `on_audible` fires on the first
    accepted write of the epoch (the real sink fires it when playback arms).
    """

    def __init__(self):
        self.written: list[tuple[int, object]] = []
        self.begins: list[int] = []
        self.flushes: list[int] = []
        self.paces: list[float] = []
        self.clears = 0
        self._epoch = -1
        self._on_audible = None

    def begin(self, epoch: int, on_audible=None) -> None:
        self.begins.append(epoch)
        self._epoch = epoch
        self._on_audible = on_audible

    def pace(self, expected_s: float) -> None:
        self.paces.append(expected_s)

    def write(self, chunk, epoch: int) -> None:
        if epoch == self._epoch:
            self.written.append((epoch, chunk))
            cb, self._on_audible = self._on_audible, None
            if cb is not None:
                cb()

    def flush(self, epoch: int) -> None:
        self.flushes.append(epoch)

    def clear(self) -> None:
        self.clears += 1
        self._epoch += 1
        self._on_audible = None

    @property
    def chunks(self) -> list:
        return [c for _, c in self.written]


def one_chunk_per_sentence(sentence: str):
    """A stream() stand-in: emit each sentence as a single chunk (its own text)."""
    yield sentence


def test_split_sentences_on_punctuation():
    text = "Hello there my friend. How are you doing today? I'm doing great!"
    assert split_sentences(text) == [
        "Hello there my friend.",
        "How are you doing today?",
        "I'm doing great!",
    ]


def test_split_sentences_keeps_emotion_tag_with_its_sentence():
    text = "That's absolutely hilarious. <laugh> Anyway, what's new with you?"
    assert split_sentences(text) == [
        "That's absolutely hilarious. <laugh>",
        "Anyway, what's new with you?",
    ]


def test_short_fragments_merge_with_the_next_sentence():
    # Orpheus glitches/repeats on tiny prompts, so "Wow." must not synth alone.
    text = "Wow. That is genuinely impressive work."
    assert split_sentences(text) == ["Wow. That is genuinely impressive work."]


def test_short_trailing_fragment_merges_backward():
    text = "That is genuinely impressive work. Wow."
    assert split_sentences(text) == ["That is genuinely impressive work. Wow."]


def test_tag_only_fragment_never_stands_alone():
    text = "<laugh> Okay okay, you got me there my friend."
    assert split_sentences(text) == ["<laugh> Okay okay, you got me there my friend."]


def test_speaks_all_sentences_in_order_when_uncancelled():
    sink = FakeSink()
    player = SpeechPlayer(one_chunk_per_sentence, sink)

    player.speak("One is the loneliest number. Two can be as bad as one. Three is company.")

    assert sink.chunks == [
        "One is the loneliest number.",
        "Two can be as bad as one.",
        "Three is company.",
    ]
    assert sink.clears == 0
    assert sink.begins == [0] and sink.flushes == [0]


def test_cancel_between_sentences_stops_remaining():
    sink = FakeSink()
    stream = _make_cancelling_stream(lambda: player, after="One is the loneliest number.")
    player = SpeechPlayer(stream, sink)

    player.speak("One is the loneliest number. Two can be as bad as one. Three is company.")

    assert sink.chunks == ["One is the loneliest number."]
    assert sink.clears == 1


def test_cancel_mid_sentence_stops_further_chunks():
    sink = FakeSink()

    def multi_chunk_stream(sentence):
        for i in range(5):
            yield f"{sentence}:{i}"
            if i == 1:
                player.cancel()  # barge-in fires after the 2nd chunk

    player = SpeechPlayer(multi_chunk_stream, sink)
    player.speak("A long winding sentence that streams in several chunks.")

    # chunk :0 and :1 written, cancel fires, :2.. never reach the sink
    assert sink.chunks == [
        "A long winding sentence that streams in several chunks.:0",
        "A long winding sentence that streams in several chunks.:1",
    ]
    assert sink.clears == 1


def test_cancel_clears_the_sink_and_bumps_generation():
    sink = FakeSink()
    player = SpeechPlayer(one_chunk_per_sentence, sink)
    gen_before = player._gen_id

    player.cancel()

    assert sink.clears == 1
    assert player._gen_id == gen_before + 1


def test_on_first_audio_fires_once_on_the_first_chunk():
    sink = FakeSink()
    calls = []
    player = SpeechPlayer(
        one_chunk_per_sentence, sink, on_first_audio=lambda: calls.append(len(sink.written))
    )

    player.speak("One is the loneliest number. Two can be as bad as one. Three is company.")

    # Fired exactly once, on the first write that reached the sink (audibility).
    assert calls == [1]


def test_on_first_audio_not_fired_when_nothing_synthesized():
    sink = FakeSink()
    calls = []

    def empty_stream(sentence):
        return
        yield  # unreachable: a stream that produces no audio

    player = SpeechPlayer(empty_stream, sink, on_first_audio=lambda: calls.append(1))
    player.speak("First sentence here. Second sentence here.")

    assert calls == []  # no audio -> callback never fires (safety-clear handles the mute)


def _make_cancelling_stream(get_player, *, after: str):
    def stream(sentence):
        yield sentence
        if sentence == after:
            get_player().cancel()

    return stream


# -- buffered mode (buffer_whole=True): synthesize everything, THEN play ------


def test_buffered_mode_synthesizes_all_before_playing_any():
    """No sentence is written to the sink until every sentence has been synthesized.

    This is what keeps playback smooth on sub-realtime hardware: nothing generates
    while audio plays, so the real-time callback is never starved.
    """
    sink = FakeSink()
    synth_order: list[str] = []

    def stream(sentence):
        synth_order.append(sentence)
        # If any playback had started, the sink would already hold a chunk.
        assert sink.written == [], "playback started before synthesis finished"
        yield sentence

    player = SpeechPlayer(stream, sink, buffer_whole=True)
    player.speak("First sentence here. Second sentence here. Third sentence here.")

    assert synth_order == [
        "First sentence here.",
        "Second sentence here.",
        "Third sentence here.",
    ]
    assert sink.chunks == synth_order  # all played, in order, after synthesis


def test_buffered_mode_cancel_during_synth_plays_nothing():
    sink = FakeSink()

    def stream(sentence):
        if sentence == "Second sentence here.":
            player.cancel()  # barge-in while still synthesizing the reply
        yield sentence

    player = SpeechPlayer(stream, sink, buffer_whole=True)
    player.speak("First sentence here. Second sentence here. Third sentence here.")

    # Cancelled before the write phase -> nothing reaches the sink.
    assert sink.chunks == []
    assert sink.clears == 1


# -- streamed replies (thinking in parallel with speaking) ---------------------


def _chunks(*parts: str):
    """A fake LLM token stream."""
    yield from parts


def test_iter_stream_sentences_yields_as_boundaries_form():
    from orpheus_live.audio.playback import iter_stream_sentences

    out = list(
        iter_stream_sentences(
            _chunks("Hello there my ", "friend. How are ", "you doing today? I'm doing great!")
        )
    )
    assert out == [
        "Hello there my friend.",
        "How are you doing today?",
        "I'm doing great!",
    ]


def test_iter_stream_sentences_first_sentence_available_before_stream_ends():
    """The first sentence must be yielded while the LLM is still 'thinking'."""
    import threading

    from orpheus_live.audio.playback import iter_stream_sentences

    gate = threading.Event()  # holds the fake LLM open until the test saw sentence 1

    def slow_llm():
        yield "First sentence is done here. Sec"
        gate.wait(5)  # the rest of the reply is still generating...
        yield "ond sentence lands later."

    it = iter_stream_sentences(slow_llm())
    first = next(it)  # must arrive without waiting for the gate
    assert first == "First sentence is done here."
    gate.set()
    assert list(it) == ["Second sentence lands later."]


def test_iter_stream_sentences_merges_short_fragments():
    from orpheus_live.audio.playback import iter_stream_sentences

    out = list(iter_stream_sentences(_chunks("Wow. That is genuinely ", "impressive work.")))
    assert out == ["Wow. That is genuinely impressive work."]


def test_speak_stream_speaks_sentences_and_returns_full_text():
    from orpheus_live.audio.playback import SpeechPlayer

    sink = FakeSink()
    player = SpeechPlayer(one_chunk_per_sentence, sink)

    reply = player.speak_stream(
        _chunks("One is the loneliest number. ", "Two can be as bad as one.")
    )

    assert sink.chunks == ["One is the loneliest number.", "Two can be as bad as one."]
    assert reply == "One is the loneliest number. Two can be as bad as one."


def test_speak_stream_cancel_stops_audio_but_returns_text():
    from orpheus_live.audio.playback import SpeechPlayer

    sink = FakeSink()
    stream = _make_cancelling_stream(lambda: player, after="One is the loneliest number.")
    player = SpeechPlayer(stream, sink)

    reply = player.speak_stream(
        _chunks("One is the loneliest number. ", "Two can be as bad as one. ", "Three is company.")
    )

    assert sink.chunks == ["One is the loneliest number."]  # audio stopped at the barge-in
    assert "Three is company." in reply  # but the full text still comes back for memory
