"""Optional session language policy around the existing VITS2 TensorRT plugin."""

__all__ = ["TTSPlugin"]


def __getattr__(name):
    if name == "TTSPlugin":
        from .plugin import TTSPlugin

        return TTSPlugin
    raise AttributeError(name)
