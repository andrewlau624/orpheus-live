"""Entry point: `python -m orpheus_live [voice]` or the `orpheus-live` console script."""

import argparse
import os

os.environ.setdefault("TQDM_DISABLE", "1")  # silence per-token progress bars

from .config import settings  # noqa: E402
from .core.conversation import run  # noqa: E402
from .engines.tts import list_presets, load_preset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="orpheus-live",
        description="Local voice-to-voice conversational AI (Orpheus TTS).",
    )
    parser.add_argument(
        "voice",
        nargs="?",
        default=None,
        help="saved voice preset to load (from saved_voices/); omit for a random voice",
    )
    parser.add_argument("--list", action="store_true", help="list saved voice presets and exit")
    args = parser.parse_args()

    if args.list:
        presets = list_presets(settings)
        print("\n".join(presets) if presets else "(no saved voices yet)")
        return

    preset = load_preset(settings, args.voice) if args.voice else None
    run(settings, preset)


if __name__ == "__main__":
    main()
