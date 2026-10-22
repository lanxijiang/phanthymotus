"""Reject known incomplete MOSS chunks without altering tokens or sampling."""

from __future__ import annotations


class IncompleteSynthesisError(RuntimeError):
    def __init__(self, reason: str, *, frames: int, limit=None):
        self.reason = reason
        self.frames = frames
        self.limit = limit
        super().__init__(
            f"MOSS incomplete synthesis: {reason} (frames={frames}, limit={limit})"
        )


def validate_chunk_result(result: dict, generation_defaults: dict) -> None:
    """An EOS is not proof of transcription accuracy; only catch certain failures.

    The original generator checks EOS before appending each audio frame. When
    it returns exactly max_new_frames, the loop exhausted its budget without
    observing EOS. Do not accept the decoded prefix as a complete utterance.
    """
    count = len(result["generated_frames"])
    limit = generation_defaults.get("max_new_frames")
    if count == 0:
        raise IncompleteSynthesisError("empty_generation", frames=count, limit=limit)
    if limit is not None and count >= int(limit):
        raise IncompleteSynthesisError("frame_limit", frames=count, limit=limit)
    samples = result.get("audio_samples")
    if samples is None and result.get("waveform") is not None:
        samples = len(result["waveform"])
    if samples is not None and int(samples) <= 0:
        raise IncompleteSynthesisError("empty_audio", frames=count, limit=limit)
