"""Tests for Orpheus voice presets: randomization ranges and save/load round-trip."""

from orpheus_live.config import Settings
from orpheus_live.engines.tts import list_presets, load_preset, random_preset, save_preset
from orpheus_live.models import ORPHEUS_VOICES


def test_random_preset_within_configured_ranges():
    settings = Settings()
    for _ in range(200):
        p = random_preset(settings)
        assert p.voice in ORPHEUS_VOICES
        assert settings.rand_temp[0] <= p.temperature <= settings.rand_temp[1]
        assert settings.rand_top_p[0] <= p.top_p <= settings.rand_top_p[1]
        assert settings.rand_rep_penalty[0] <= p.repetition_penalty <= settings.rand_rep_penalty[1]


def test_random_preset_respects_custom_ranges():
    settings = Settings(rand_temp=(0.5, 0.5), rand_top_p=(0.9, 0.9), rand_rep_penalty=(1.2, 1.2))
    p = random_preset(settings)
    assert p.temperature == 0.5
    assert p.top_p == 0.9
    assert p.repetition_penalty == 1.2


def test_save_and_load_preset_round_trip(tmp_path):
    settings = Settings(saved_voices_dir=str(tmp_path))
    preset = random_preset(settings).model_copy(update={"name": "Emma"})

    save_preset(settings, preset)

    assert list_presets(settings) == ["Emma"]
    loaded = load_preset(settings, "emma")  # case-insensitive lookup
    assert loaded == preset


def test_load_missing_preset_exits_with_available_list(tmp_path):
    import pytest

    settings = Settings(saved_voices_dir=str(tmp_path))
    save_preset(settings, random_preset(settings).model_copy(update={"name": "Emma"}))

    with pytest.raises(SystemExit, match="Emma"):
        load_preset(settings, "nosuchvoice")
