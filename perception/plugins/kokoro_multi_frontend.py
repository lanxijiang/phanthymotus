"""Resident multilingual G2P for the Kokoro Multi checkpoint.

Notation and business pronunciations use the release's external lexicon.  Thai
tones and the Russian/German phone substitutions preserve the audition baseline;
they are approximations and still need native-speaker review.
"""
from itertools import groupby
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
from unicodedata import category

from .kokoro_multi_notation import configure_lexicon, normalize_notation, protected_spans

LANGUAGES = ("zh", "en", "ja", "ru", "de", "fr", "es", "th")
_LATIN = re.compile(r"([A-Za-z][A-Za-z0-9]*(?:[ \t]+[A-Za-z][A-Za-z0-9]*)*)")
_SYMBOL_MAP = {"ru": {"ɭ": "l", "ɵ": "o", "ʑ": "zʲ"}, "de": {"ʏ": "y"}}
_STOPS = re.compile(r'[.!?…;:,—][)\]”"]*\s*')


def phoneme_chunks(phones, limit=220, preferred=180, first_limit=12):
    """Split only at whole-word/punctuation boundaries, retaining every phone.

    A short first word lets the non-autoregressive graph publish its first PCM
    before synthesizing the remaining phrases. An unusually long first word is
    kept whole, even if that costs first-audio latency. No phone is truncated to
    make a TensorRT profile fit. Callers may retry with a smaller ``limit`` when
    the predicted duration exceeds the vocoder profile.
    """
    if not 1 <= preferred <= limit <= 510 or not 0 <= first_limit <= limit:
        raise ValueError("Invalid Kokoro Multi phoneme chunk limits")
    remaining, chunks = phones.strip(), []
    if not remaining:
        return chunks
    while remaining:
        first = not chunks and first_limit > 0
        if len(remaining) <= (first_limit if first else limit):
            chunks.append(remaining)
            break
        stops = [m.end() for m in _STOPS.finditer(remaining[:limit])]
        words = [m.end() for m in re.finditer(r"\s+", remaining[:limit])]
        # A leading quote or parenthesis is not a speakable first fragment.
        boundaries = sorted({end for end in stops + words
                             if any(c.isalpha() for c in remaining[:end])})
        if first:
            early = [end for end in boundaries if end <= first_limit]
            phrase = [end for end in stops if first_limit // 2 <= end <= first_limit]
            end = max(phrase or early) if (phrase or early) else (boundaries[0] if boundaries else None)
        elif len(remaining) <= limit:
            chunks.append(remaining)
            break
        else:
            natural = [end for end in stops if preferred // 2 <= end <= limit]
            choices = natural or boundaries
            end = min(choices, key=lambda value: abs(value - preferred)) if choices else None
        if end is None:
            raise ValueError(f"No whole-word Kokoro Multi boundary within {limit} phonemes")
        chunks.append(remaining[:end].strip())
        remaining = remaining[end:].lstrip()
    if "".join("".join(chunk.split()) for chunk in chunks) != "".join(phones.split()):
        raise RuntimeError("Kokoro Multi chunking changed phoneme content")
    return chunks


class Frontend:
    """Load dictionaries once; select language from this utterance alone."""

    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        # The release vendors only FastThai's pure Python code and offline data.
        # Compiled ARM dependencies come from the image's pinned requirements.
        extra = model_dir / "frontend_data"
        if extra.is_dir():
            sys.path.insert(0, str(extra))
        cache = Path(tempfile.gettempdir()) / "kokoro-multi-frontend"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PYTHAINLP_DATA", str(cache / "pythainlp"))
        os.environ.setdefault("PYTHAINLP_OFFLINE", "1")
        import jieba
        jieba.dt.cache_file = str(cache / "jieba.cache")
        jieba.initialize()
        from langid.langid import LanguageIdentifier, model
        from misaki.cutlet import Cutlet
        from misaki.en import G2P as EnglishG2P
        from misaki import espeak
        from misaki.tone_sandhi import ToneSandhi
        from fastthaig2p import G2P as ThaiG2P
        from pythainlp.tokenize import Tokenizer
        from phonemizer.backend import EspeakBackend

        if importlib.util.find_spec("en_core_web_sm") is None:
            raise RuntimeError("Kokoro Multi requires the pinned offline en_core_web_sm model")
        self._lock = threading.RLock()
        self._identifier = LanguageIdentifier.from_modelstring(model, norm_probs=False)
        self._identifier.set_languages(LANGUAGES)
        self._lexicon_path = model_dir / "frontend_lexicon.json"
        lexicon = json.loads(self._lexicon_path.read_text(encoding="utf-8"))
        self._literal_symbol_words = lexicon["normalization"].get("literal_symbol_words", {})
        configure_lexicon(self._lexicon_path)
        self._english = EnglishG2P(british=False, fallback=espeak.EspeakFallback(british=False))
        for word, overrides in lexicon.get("pos_overrides", {}).items():
            self._english.lexicon.golds[word].update(overrides)
        self._japanese, self._sandhi = Cutlet(), ToneSandhi()
        available = EspeakBackend.supported_languages()
        french = next(code for code in ("fr-fr", "fr") if code in available)
        self._espeak = {lang: espeak.EspeakG2P(language=french if lang == "fr" else lang)
                        for lang in ("ru", "de", "fr", "es")}
        self._thai = ThaiG2P()
        overrides = lexicon.get("thai_ipa_overrides", {})
        self._thai._ipa.update(overrides)
        self._thai._tokenizer._dict = self._thai._tokenizer.vocab | set(overrides)
        self._thai._tokenizer._engine = Tokenizer(
            custom_dict=self._thai._tokenizer.vocab, engine="newmm")
        alternatives, self._entries = [], {}
        for index, entry in enumerate(e for e in lexicon["entries"] if e["enabled"]):
            for form in [entry["term"], *entry["aliases"]]:
                name = f"w{index}_{len(alternatives)}"
                body = re.escape(form).replace(r"\ ", r"[ \t]+")
                body = body if entry["case_sensitive"] else "(?i:" + body + ")"
                alternatives.append((len(form), f"(?P<{name}>{body})"))
                self._entries[name] = self._dictionary_phonemes(entry)
        alternatives.sort(reverse=True)
        self._terms = re.compile(r"(?<![A-Za-z0-9_])(?:" +
                                 "|".join(body for _, body in alternatives) +
                                 r")(?![A-Za-z0-9_])")
        # Load lazy corpora before the first queued utterance, without running a
        # neural model or downloading resources during requests.
        for language, text in zip(LANGUAGES, (
                "你好", "Hello", "こんにちは", "Здравствуйте", "Guten Tag",
                "Bonjour", "Buenos días", "สวัสดี")):
            self.convert(text, language)

    def detect_language(self, text):
        """Use strong script cues, then restricted eight-language identification."""
        if re.search(r"[\u0e00-\u0e7f]", text):
            return "th"
        if re.search(r"[\u3040-\u30ff]", text):
            return "ja"
        if re.search(r"[\u0400-\u04ff]", text):
            return "ru"
        if re.search(r"[\u4e00-\u9fff]", text):
            # Pure Han text without kana is normally Chinese in this deployment;
            # callers can explicitly select Japanese for ambiguous short names.
            return "zh"
        ordinary = self._terms.sub(" ", text)
        if not re.search(r"[^\W\d_]", ordinary, re.UNICODE):
            return "en"
        with self._lock:
            return self._identifier.classify(ordinary)[0]

    def _dictionary_phonemes(self, entry):
        if entry.get("phonemes"):
            return entry["phonemes"]
        parts = []
        for letters, words in groupby(entry["spoken_english"].split(),
                key=lambda word: bool(re.fullmatch(r"[A-Z]", word))):
            text = "".join(words) if letters else " ".join(words).lower()
            parts.append(self._english.lexicon.get_NNP(text)[0]
                         if letters else self._english(text)[0])
        return " ".join(parts)

    def _chinese(self, text):
        import cn2an
        import jieba
        from misaki.zh import ZHG2P
        from pypinyin import lazy_pinyin, Style
        text = ZHG2P.map_punctuation(cn2an.transform(text, "an2cn"))
        parts = []
        for span in re.findall(r"[\u4e00-\u9fff]+|[^\u4e00-\u9fff]+", text):
            if not re.fullmatch(r"[\u4e00-\u9fff]+", span):
                parts.append(span)
                continue
            rendered = []
            for word in jieba.lcut(span, cut_all=False):
                phones = lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)
                if len(word) == 2 and word[0] != word[1]:
                    phones = self._sandhi._three_sandhi(word, phones)
                rendered.append("".join(ZHG2P.py2ipa(phone) for phone in phones))
            parts.append(" ".join(rendered))
        return "".join(parts).replace(chr(815), "")

    def _thai_phones(self, text):
        from fastthaig2p import normalize, ipa_to_kokoro
        from fastthaig2p.fallback import fallback_g2p
        parts = []
        for token in self._thai._tokenizer.tokenize(normalize(text)):
            if not token.strip():
                continue
            parts.append(token if all(category(c).startswith("P") for c in token)
                         else self._thai._ipa.get(token) or fallback_g2p(token))
        ipa = " ".join(parts)
        phones = re.sub(r"/[^/]+/", lambda match: ipa_to_kokoro(match[0]).replace(".", ""), ipa)
        return phones.translate(str.maketrans({"→": "", "↑": "→"}))

    def _native(self, text, language):
        if language == "en":
            return self._english(text)[0]
        if language == "zh":
            return self._chinese(text)
        if language == "ja":
            return self._japanese(text)[0].translate(str.maketrans({"«": "(", "»": ")"}))
        if language == "th":
            return self._thai_phones(text)
        return self._espeak[language](text)[0].translate(str.maketrans(_SYMBOL_MAP.get(language, {})))

    def _literal_spans(self, text):
        spans = protected_spans(text)
        # The notation parser's Unicode word boundary misses "report_v2.txtを".
        # Reuse that parser on ASCII runs, preserving offsets and still requiring
        # a recognized literal; a bare a_b remains ordinary frontend input.
        for match in re.finditer(r"[!-~]*_[!-~]*", text):
            spans.extend((match.start() + start, match.start() + end)
                         for start, end in protected_spans(match[0]))
        return spans

    def _expand_literal_symbols(self, text, language):
        words = self._literal_symbol_words
        if not any(symbol in text for symbol in words):
            return text
        spans = self._literal_spans(text)
        return "".join(
            " " + words[char][language] + " "
            if char in words and any(start <= index < end for start, end in spans)
            else char for index, char in enumerate(text))

    def convert(self, text, language="auto"):
        if language == "auto":
            language = self.detect_language(text)
        if language not in LANGUAGES:
            raise ValueError(f"Unsupported Kokoro Multi language: {language}")
        with self._lock:
            spoken, _ = normalize_notation(text, language)
            protected = self._literal_spans(spoken)
            matches = [match for match in self._terms.finditer(spoken)
                       if not any(match.start() < end and match.end() > start
                                  for start, end in protected)]
            cursor, parts = 0, []
            if language == "en":
                for match in matches:
                    parts.extend((spoken[cursor:match.start()],
                                  "[" + match[0] + "](/" + self._entries[match.lastgroup] + "/)"))
                    cursor = match.end()
                parts.append(spoken[cursor:])
                return self._english(self._expand_literal_symbols("".join(parts), language))[0]

            def plain(value):
                # Lexical matches were already selected against the intact
                # literal above; expansion cannot expose aliases inside a file.
                value = self._expand_literal_symbols(value, language)
                pieces = _LATIN.split(value) if language in ("zh", "ja", "ru", "th") else [value]
                for piece in pieces:
                    if piece.strip():
                        lang = "en" if language in ("zh", "ja", "ru", "th") and _LATIN.fullmatch(piece) else language
                        parts.append(self._native(piece, lang))

            for match in matches:
                plain(spoken[cursor:match.start()])
                parts.append(self._entries[match.lastgroup])
                cursor = match.end()
            plain(spoken[cursor:])
            return " ".join(parts)
