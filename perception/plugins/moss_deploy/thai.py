"""Dictionary word boundaries for long Thai runs; no text rewriting.

libthai's TIS-620 offsets map one-to-one to Python character offsets for a
Thai run. The deployment installs libthai0 and its dictionary. The dictionary
is opened only when a long Thai run needs splitting, never for other scripts.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
import threading
import unicodedata


_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _library():
    try:
        library = ctypes.CDLL("libthai.so.0")
    except OSError as exc:
        raise RuntimeError("Long Thai text requires libthai0 and libthai-data") from exc
    library.th_brk_new.argtypes = [ctypes.c_char_p]
    library.th_brk_new.restype = ctypes.c_void_p
    library.th_brk_find_breaks.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.c_size_t,
    ]
    library.th_brk_find_breaks.restype = ctypes.c_int
    library.th_brk_delete.argtypes = [ctypes.c_void_p]
    library.th_brk_delete.restype = None
    return library


def thai_word_breaks(text: str) -> list[int]:
    """Return dictionary cuts inside one Thai run, without inserting spaces.

    No character-based fallback is used if a dictionary is missing. Invalid
    or unrepresentable Thai input is rejected rather than encoded lossily.
    Creation/destruction is serialized as required by libthai's API.
    """
    if not text:
        return []
    if any(not "\u0e00" <= char <= "\u0e7f" for char in text):
        raise ValueError("Thai word breaking expects one Thai-only run")
    try:
        encoded = text.encode("tis-620")
    except UnicodeEncodeError as exc:
        raise ValueError("Thai text contains characters unsupported by the word dictionary") from exc
    if len(encoded) != len(text):
        raise AssertionError("Thai dictionary offsets do not match source characters")
    library = _library()
    positions = (ctypes.c_int * len(text))()
    with _LOCK:
        breaker = library.th_brk_new(None)
        if not breaker:
            raise RuntimeError("libthai dictionary is unavailable; install libthai-data")
        try:
            count = library.th_brk_find_breaks(breaker, encoded, positions, len(positions))
        finally:
            library.th_brk_delete(breaker)
    if not 0 <= count <= len(positions):
        raise RuntimeError("libthai returned an invalid word-boundary count")
    # Keep combining marks and dependent vowels attached even if a future
    # dictionary version proposes a boundary there. Never cut after a Thai
    # leading vowel, whose consonant follows it in Unicode order.
    cuts = []
    for cut in positions[:count]:
        if not 0 < cut < len(text):
            continue
        right, left = text[cut], text[cut - 1]
        if (unicodedata.category(right).startswith("M")
                or right in "ะาำๅ"
                or "\u0e40" <= left <= "\u0e44"):
            continue
        cuts.append(cut)
    return sorted(set(cuts))
