"""Reject known incomplete MOSS chunks without altering tokens or sampling."""

from __future__ import annotations

import logging
import unicodedata


log = logging.getLogger(__name__)


def minimum_chunk_frames(text: str) -> int:
    """Detect only impossible long-text collapses, not pronunciation quality.

    One native frame is 80 ms. Eight letters/digits cannot form a complete
    ordinary utterance in that time. Thai vowel and tone marks are part of
    the written utterance but are not counted by ``isalnum``. Count those
    marks only within the Thai Unicode block; other languages retain their
    existing behavior. Keep short interjections and punctuation-only input.
    """
    written = sum(
        character.isalnum()
        or ("\u0e00" <= character <= "\u0e7f"
            and unicodedata.category(character).startswith("M"))
        for character in text
    )
    return 2 if written >= 8 else 1


def generate_with_early_eos_guard(generate, request_rows, *, minimum_frames,
                                  on_frame=None):
    """Retry an empty/one-frame collapse once, before publishing any of it.

    Only the first frame is held for long text. On the next frame, the original
    callbacks are replayed in order, unchanged. No retry after audio was sent,
    no RNG reset or sampler change, and no whole-sentence buffering.
    """
    for attempt in range(2):
        pending = []

        def observe(frames, step, frame):
            if len(frames) < minimum_frames:
                # The original list grows in place; preserve its prefix as it
                # was at this callback, not the list after generation finishes.
                pending.append((list(frames), step, frame))
                return
            if on_frame is not None:
                for args in pending:
                    on_frame(*args)
                on_frame(frames, step, frame)
            pending.clear()

        frames = generate(request_rows, on_frame=observe if on_frame else None)
        if len(frames) >= minimum_frames:
            return frames
        reason = "empty_generation" if not frames else "early_eos"
        if attempt == 0:
            log.warning("[moss] retrying %s before audio delivery frames=%d minimum=%d",
                        reason, len(frames), minimum_frames)
    raise IncompleteSynthesisError(reason, frames=len(frames), limit=None)


class IncompleteSynthesisError(RuntimeError):
    def __init__(self, reason: str, *, frames: int, limit=None):
        self.reason = reason
        self.frames = frames
        self.limit = limit
        super().__init__(
            f"MOSS incomplete synthesis: {reason} (frames={frames}, limit={limit})"
        )


def validate_chunk_result(result: dict, generation_defaults: dict, *, text: str = "") -> None:
    """An EOS is not proof of transcription accuracy; only catch certain failures.

    The original generator checks EOS before appending each audio frame. When
    it returns exactly max_new_frames, the loop exhausted its budget without
    observing EOS. Do not accept the decoded prefix as a complete utterance.
    """
    count = len(result["generated_frames"])
    limit = generation_defaults.get("max_new_frames")
    if count == 0:
        raise IncompleteSynthesisError("empty_generation", frames=count, limit=limit)
    if count < minimum_chunk_frames(text):
        raise IncompleteSynthesisError("early_eos", frames=count, limit=limit)
    if limit is not None and count >= int(limit):
        raise IncompleteSynthesisError("frame_limit", frames=count, limit=limit)
    samples = result.get("audio_samples")
    if samples is None and result.get("waveform") is not None:
        samples = len(result["waveform"])
    if samples is not None and int(samples) <= 0:
        raise IncompleteSynthesisError("empty_audio", frames=count, limit=limit)
