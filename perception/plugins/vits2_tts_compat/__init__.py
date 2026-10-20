"""Standard TTS calls with optional explicit conversation language turns."""

__all__ = ["TTSPlugin"]


def __getattr__(name):
    if name == "TTSPlugin":
        from .plugin import TTSPlugin

        return TTSPlugin
    raise AttributeError(name)
