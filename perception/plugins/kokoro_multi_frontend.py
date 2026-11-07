"""Resident Kokoro Multi text normalization, pronunciation dictionaries and G2P.

All language resources remain in the verified external release. Thai tones and
Russian/German substitutions preserve the audition baseline; native review is
still required. No TTSV4 source or acoustic weights are imported here.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from functools import lru_cache
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


_LANGUAGES = {"zh", "en", "ja", "ru", "de", "fr", "es", "th"}
_PHONE_HINT = re.compile(r"\[[^\]\n]+\]\(/[^/\n]+/\)")
_ADDRESS = re.compile(
    r"`[^`\n]+`|(?<!\w)--[A-Za-z][\w-]*(?:=[^\s;,]+)?|"
    r"(?<!\w)-[A-Za-z]\s+[+\-]?\d+(?:\.\d+)?|https?://[^\s<>`]+|www\.[^\s<>`]+|"
    r"[\w.!#$%&'*+/=?^`{|}~-]+@[\w-]+(?:\.[\w-]+)+|"
    r"(?<!\w)(?:[A-Za-z]:[\\/]|\.?\.?/)[\w./\\+@~%-]+|"
    r"(?<![\w/])[\w.-]+(?:[/\\][\w.+@~-]+)*[/\\][\w.-]+\.[A-Za-z]{1,8}(?!\w)|"
    r"(?<![\w.])[\w+-]+(?:\.[\w+-]+)*\.(?:txt|csv|jsonl?|onnx|pt|pth|wav|mp3|py|sh|zip|pdf|md|yaml|yml|log)(?!\w)",
    re.IGNORECASE,
)
_VERSION = re.compile(
    r"(?:(?P<label>\b(?:version|ver\.?|versión|версия)\s*|版本\s*|バージョン\s*|เวอร์ชัน\s*|รุ่น\s*)(?P<v1>[vV]?\d+(?:\.\d+)+)|"
    r"(?<![A-Za-z0-9_.])(?P<v2>[vV]\d+(?:\.\d+)+))(?![A-Za-z0-9_]|\.[A-Za-z0-9_])", re.IGNORECASE,
)
_DATE = re.compile(r"(?<![A-Za-z0-9_.\-])\d{4}-\d{2}-\d{2}(?![A-Za-z0-9_\-]|\.[A-Za-z0-9_])")
_DATE_LIKE = re.compile(r"(?<![A-Za-z0-9_])\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:[-/.]\d+)*(?![A-Za-z0-9_])")
_AMBIGUOUS_PAIR = re.compile(r"(?<![A-Za-z0-9_.])\d+(?:-\d+|\s*/\s*(?:\d+|[A-Za-z]{1,3}))(?![A-Za-z0-9_.])")
_DOTTED = re.compile(r"(?<![A-Za-z0-9_.])\d{1,4}(?:\.\d{1,4}){2,}(?![A-Za-z0-9_.])")
_CLOCK = re.compile(r"(?<![A-Za-z0-9_:])\d{1,3}:\d{2}(?::\d{2})?(?![A-Za-z0-9_:])")
_NUM = r"(?:\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)"
_EN_GROUPED_NUM = r"[1-9]\d{0,2}(?:,\d{3})+(?:\.\d+)?"
_EN_GROUPED = re.compile(r"(?<![A-Za-z0-9_.,])(?P<n>[+\-−]?" +
                         _EN_GROUPED_NUM + r")(?![A-Za-z0-9_]|[.,]\d)")
_EN_IPV4 = re.compile(r"(?<![A-Za-z0-9_.])(?:\d{1,3}\.){3}\d{1,3}(?![A-Za-z0-9_]|\.\d)")
_EN_MONEY = re.compile(r"(?<![A-Za-z0-9_])(?P<left>[+\-−]?)(?P<c>[$£€])"
                       r"(?P<right>[+\-−]?)(?P<n>" + _EN_GROUPED_NUM + "|" +
                       _NUM + r")(?![A-Za-z0-9_]|[.,]\d)")
_EN_HARDWARE = re.compile(r"(?<![A-Za-z0-9_])(?P<family>RTX|GTX)[ \t]*"
                          r"(?P<n>[1-9]\d{3})(?:[ \t]*(?P<suffix>Ti|SUPER))?"
                          r"(?![A-Za-z0-9_])", re.I)
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])" + _NUM + r"(?![A-Za-z0-9_]|\.[A-Za-z0-9_])")
_FORMULA_LIKE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[+\-−][ \t]*)?[A-Za-z\u0370-\u03ff0-9_().]+"
    r"(?:[ \t]*[+\-−‑×·*/÷^²³=<>≤≥≠≈][ \t]*[+\-−]?[ \t]*[A-Za-z\u0370-\u03ff0-9_().]*)+"
    r"|(?<![A-Za-z0-9_])[A-Za-z]+[ \t]*\([A-Za-z\u0370-\u03ff0-9_().+\-−‑×·*/÷^²³ \t]+\)"
)
_TOKEN = re.compile(_NUM + r"|[A-Za-z]+|π|[()+\-−×·*÷/^²³=<>≤≥≠≈]")
_COMMA_LANGUAGES = {"de", "fr", "es", "ru"}
_RU_UNIT_ALIASES = {"ГБ": "GB", "мс": "ms", "ms": "ms", "кВт": "kW"}
_COMMA_NUM = _NUM.replace(r"\.", "[.,]")
_COMMA_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])(?<!\d,)" + _COMMA_NUM + r"(?![A-Za-z0-9_]|\.[A-Za-z0-9_])(?!,\d)")
_COMMA_TOKEN = re.compile(_TOKEN.pattern.replace(_NUM, _COMMA_NUM))
_COMMA_FORMULA_LIKE = re.compile(_FORMULA_LIKE.pattern.replace("_().", "_().,"))
_RELATIONS = {"=": "equals", "<": "less", ">": "greater", "≤": "le", "≥": "ge", "≠": "ne", "≈": "approx"}
_IDENTIFIER_CUES = {
    "zh": r"验证码", "ja": r"認証コード|確認コード", "en": r"verification code",
    "ru": r"проверочный код|номер заказа", "de": r"Bestätigungscode",
    "fr": r"code de vérification(?: fictif)?|numéro de commande",
    "es": r"código de verificación(?: ficticio)?|número de pedido",
    "th": r"รหัสยืนยัน(?:สมมติ)?|หมายเลขคำสั่งซื้อ",
}


_lexicon_path = None


def configure_lexicon(path):
    global _lexicon_path
    _lexicon_path = Path(path)
    _config.cache_clear()
    _unit_pattern.cache_clear()


@lru_cache(maxsize=1)
def _config():
    if _lexicon_path is None:
        raise RuntimeError("Kokoro Multi lexicon is not configured")
    cfg = json.loads(_lexicon_path.read_text(encoding="utf-8"))["normalization"]
    cfg["units"].setdefault("%", {
        "zh": ["百分之"], "en": ["percent"], "ja": ["パーセント"],
        "ru": ["процент", "процента", "процентов"], "de": ["Prozent"],
        "fr": ["pour cent"], "es": ["por ciento"], "th": ["ร้อยละ"]})
    cfg["unit_symbol_aliases"].setdefault("％", "%")
    templates = cfg.setdefault("quantity_templates", {}).setdefault("%", {})
    for lang, template in (("zh", "百分之{number}"), ("th", "ร้อยละ{number}")):
        templates.setdefault(lang, template)
    return cfg


@lru_cache(maxsize=8)
def _unit_pattern(language=""):
    cfg = _config()
    symbols = (set(cfg["units"]) - {"hour", "minute", "second"}) | set(cfg["unit_symbol_aliases"])
    if language == "ru":
        symbols |= _RU_UNIT_ALIASES.keys()
    names = "|".join(re.escape(symbol) for symbol in sorted(symbols, key=len, reverse=True))
    number = r"(?:" + _EN_GROUPED_NUM + "|" + _NUM + ")" if language == "en" else _NUM
    if language in _COMMA_LANGUAGES:
        # Preserve the existing separate-number reading of comma lists.
        number = r"(?:(?<!\d,)" + _COMMA_NUM + r"|(?<=\d,)" + _NUM + r")"
    return re.compile(r"(?<![A-Za-z0-9_.\-−])(?P<n>[+\-−]?" + number +
        r")\s*(?P<u>" + names + r")(?![A-Za-z0-9_]|[.\-][A-Za-z0-9_])")


def protected_spans(text: str) -> list[tuple[int, int]]:
    """Original-text offsets for addresses, paths, filenames, code and versions.

    Protect these when matching lexical aliases; nothing is removed or replaced.
    The normalizer separately handles explicit versions as typed numeric spans.
    """
    spans = [(m.start(), m.end()) for m in _address_matches(text)]
    spans += [(m.start(), m.end()) for m in _VERSION.finditer(text)]
    return sorted(spans)


def _address_matches(text):
    hints = [m.span() for m in _PHONE_HINT.finditer(text)]
    for match in _ADDRESS.finditer(text):
        if any(match.start() < end and match.end() > start for start, end in hints):
            continue
        # The sentence following a URL/path is not part of that literal.
        end = match.start() + len(match.group().rstrip(".,!?;，。！？；"))
        if end < match.end():
            match = _ADDRESS.match(text, match.start(), end)
            if match is None:
                continue
        # A single /x after a numeric operand may be division, not a root path.
        if re.fullmatch(r"/[A-Za-z]{1,3}", match.group()) and re.search(r"\d\s*$", text[:match.start()]):
            continue
        yield match


def _join(parts, language):
    return ("" if language in {"zh", "ja", "th"} else " ").join(parts)


def _context(text, start, end):
    """Do not borrow a semantic cue from a neighbouring sentence/clause."""
    left = re.split(r"[;；。!?？！\n]", text[max(0, start-64):start])[-1]
    right = re.split(r"[;；。!?？！\n]", text[end:end+40])[0]
    return (left + text[start:end] + right).lower()


def _ja_integer(value):
    if not 0 <= value < 10**16:
        raise ValueError("Japanese cardinal outside supported range")
    if value == 0:
        return "零"
    digits = "零一二三四五六七八九"
    groups = []
    for group_unit in ("", "万", "億", "兆"):
        group, value = value % 10000, value // 10000
        if group:
            words = ""
            for power, unit in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
                digit, group = divmod(group, power)
                if digit:
                    words += ("" if digit == 1 and power > 1 else digits[digit]) + unit
            groups.append(words + group_unit)
    return "".join(reversed(groups))


def _integer(value, language, **kwargs):
    if language == "zh":
        from cn2an import an2cn
        return an2cn(str(value))
    if language == "ja":
        return _ja_integer(value)
    from num2words import num2words
    return num2words(value, lang=language, **kwargs)


def _digits(value, language, cfg):
    return _join([cfg["digits"][language][int(c)] for c in value], language)


def _number(value, language, cfg):
    value = str(value).replace("−", "-")
    if language == "en" and re.fullmatch(r"[+\-]?" + _EN_GROUPED_NUM, value):
        value = value.replace(",", "")
    if language in _COMMA_LANGUAGES:
        value = value.replace(",", ".")
    if value[:1] in {"-", "+"}:
        key = "negative" if value[0] == "-" else "positive"
        return _join([cfg["operator_words"][language][key], _number(value[1:], language, cfg)], language)
    if "e" in value.lower():
        mantissa, exponent = re.split("[eE]", value)
        return _join([_number(mantissa, language, cfg), cfg["operator_words"][language]["times"],
                      _power(_integer(10, language), _number(str(int(exponent)), language, cfg), language, cfg)], language)
    if "." in value:
        if language == "ru":
            return _integer(value, language)
        whole, fraction = value.split(".")
        return _join([_integer(int(whole), language), cfg["operator_words"][language]["point"],
                      _digits(fraction, language, cfg)], language)
    return _integer(int(value), language)


def _power(base, exponent, language, cfg):
    pattern = cfg["operator_words"][language]["power"]
    return _join([base, pattern.replace("{n}", exponent)], language)


def _unit_form(forms, value, language):
    raw = value.replace("−", "-")
    if language == "en" and re.fullmatch(r"[+\-]?" + _EN_GROUPED_NUM, raw):
        raw = raw.replace(",", "")
    number = Decimal(raw.replace(",", ".") if language in _COMMA_LANGUAGES else raw)
    if language == "ru":
        if number != int(number) or "." in value or "," in value:
            return forms[1]
        n = abs(int(number))
        return forms[0] if n % 10 == 1 and n % 100 != 11 else forms[1] if n % 10 in {2, 3, 4} and n % 100 not in {12, 13, 14} else forms[2]
    return forms[0] if abs(number) == 1 else forms[-1]


def _quantity(value, unit, language, cfg):
    words = _number(value, language, cfg)
    if language == "de" and re.fullmatch(r"[+\-−]?1", value):
        words = re.sub(r"eins$", "eine" if unit in {"hour", "minute", "second"} else "ein", words)
    if language == "ru" and unit in {"minute", "second"} and re.fullmatch(r"\d+", value):
        words = _integer(int(value), language, gender="feminine")
    if language == "ru" and unit == "ms" and re.fullmatch(r"[+\-−]?\d+", value):
        words = _integer(int(value.replace("−", "-")), language, gender="feminine")
        if value.startswith("+"):
            words = _join([cfg["operator_words"][language]["positive"], words], language)
    if language == "fr" and unit in {"hour", "minute", "second"} and re.fullmatch(r"[+\-−]?\d+", value):
        words = re.sub(r"\bun$", "une", words)
    if language == "es" and re.fullmatch(r"[+\-−]?\d+", value):
        if unit == "hour":
            words = re.sub(r"uno$", "una", words)
        else:
            words = re.sub(r"veintiuno$", "veintiún", words)
            words = re.sub(r"uno$", "un", words)
    forms = (["милли" + form for form in cfg["units"]["second"]["ru"]]
             if language == "ru" and unit == "ms" else cfg["units"][unit][language])
    unit_words = _unit_form(forms, value, language)
    template = cfg.get("quantity_templates", {}).get(unit, {}).get(language)
    return template.format(number=words, unit=unit_words) if template else _join([words, unit_words], language)


def _date(raw, language, cfg):
    year, month, day = map(int, raw.split("-"))
    date(year, month, day)  # Gregorian validation, including leap years.
    if language == "zh":
        return f"{_digits(str(year), language, cfg)}年{_integer(month, language)}月{_integer(day, language)}日"
    if language == "ja":
        return f"{_integer(year, language)}年{_integer(month, language)}月{_integer(day, language)}日"
    if language == "ru":
        d = _integer(day, language, to="ordinal", case="genitive")
        y = _integer(year, language, to="ordinal", case="genitive")
        return f"{d} {cfg['months'][language][month-1]} {y} года"
    d = _integer(day, language, to="ordinal") if language in {"en", "de"} else _integer(day, language)
    if language == "de":
        d += "r"
    if language == "fr" and day == 1:
        d = "premier"
    return cfg["date_templates"][language].format(day=d, month=cfg["months"][language][month-1], year=_integer(year, language))


def _clock(raw, duration, language, cfg):
    values = list(map(int, raw.split(":")))
    if any(v > 59 for v in values[1:]) or (not duration and values[0] > 23):
        raise ValueError("invalid clock/duration fields")
    if duration:
        units = ["minute", "second"] if len(values) == 2 else ["hour", "minute", "second"]
        return _join([_quantity(str(v), u, language, cfg) for v, u in zip(values, units) if v or not any(values)], language)
    hour, minute = values[:2]
    if language == "en":
        result = _integer(hour, language)
        result += " " + (("oh " if minute < 10 else "") + _integer(minute, language) if minute else "o'clock")
    elif language in {"zh", "ja"}:
        result = _integer(hour, language) + "点" if language == "zh" else cfg["clock_hour_readings"]["ja"][hour]
        result += (("零" if language == "zh" and 0 < minute < 10 else "") + _integer(minute, language) + "分") if minute or len(values) == 3 else ("整" if language == "zh" else "ちょうど")
    elif language == "de":
        result = ("ein" if hour == 1 else _integer(hour, language)) + " Uhr" + (" " + _integer(minute, language) if minute else "")
    elif language == "th":
        result = _integer(hour, language) + "นาฬิกา" + (_integer(minute, language) + "นาที" if minute else "")
    else:
        result = _join([_quantity(str(hour), "hour", language, cfg), _quantity(str(minute), "minute", language, cfg)], language)
    if len(values) == 3:
        result = _join([result, _quantity(str(values[2]), "second", language, cfg)], language)
    return result


class _MathParser:
    """Small expression AST with retained parentheses; no eval or Python code."""
    def __init__(self, tokens):
        self.tokens, self.pos = tokens, 0

    def expression(self, minimum=0):
        if self.pos >= len(self.tokens):
            raise ValueError("missing operand")
        token = self.tokens[self.pos]
        self.pos += 1
        if token in {"+", "-", "−"}:
            node = ("unary", token, self.expression(30))
        elif token == "(":
            node = ("group", self.expression())
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ")":
                raise ValueError("unbalanced parentheses")
            self.pos += 1
        elif re.fullmatch(_NUM, token) or re.fullmatch("[A-Za-z]{1,3}|π", token):
            node = ("atom", token)
        else:
            raise ValueError("unsupported operand")
        while self.pos < len(self.tokens):
            op = self.tokens[self.pos]
            # Only an explicit numeric coefficient or pi may imply a product
            # with one variable.  Parsing the right operand above multiply
            # precedence keeps 2x² as 2 * (x²), not (2x)².
            implicit = node[0] == "atom" and (re.fullmatch(_NUM, node[1]) or node[1] in {"π", "pi"}) and re.fullmatch("[A-Za-z]|π", op)
            precedence = 20 if implicit else {"+": 10, "-": 10, "−": 10, "*": 20, "×": 20, "·": 20, "/": 20, "÷": 20, "^": 40, "²": 40, "³": 40}.get(op, -1)
            if precedence < minimum:
                break
            if implicit:
                node = ("binary", "*", node, self.expression(precedence + 1))
                continue
            self.pos += 1
            right = ("atom", "2" if op == "²" else "3") if op in {"²", "³"} else self.expression(precedence if op == "^" else precedence + 1)
            node = ("binary", "^" if op in {"²", "³"} else op, node, right)
        return node

    def parse(self):
        left = self.expression()
        if self.pos < len(self.tokens) and self.tokens[self.pos] in _RELATIONS:
            relation = self.tokens[self.pos]
            self.pos += 1
            left = ("relation", relation, left, self.expression())
        if self.pos != len(self.tokens):
            raise ValueError("unsupported expression")
        return left


def _speak_math(node, language, cfg):
    words = cfg["operator_words"][language]
    kind = node[0]
    if kind == "atom":
        atom = node[1]
        if atom in {"π", "pi"}:
            return cfg["constant_words"]["π"][language]
        letters = cfg["letters"][language]
        return _number(atom, language, cfg) if atom[0].isdigit() else _join([letters.get(c.lower(), c) for c in atom], language)
    if kind == "group":
        return _join([words["left"], _speak_math(node[1], language, cfg), words["right"]], language)
    if kind == "unary":
        operand = _speak_math(node[2], language, cfg)
        if node[2][0] in {"binary", "relation"}:
            operand = _join([words["left"], operand, words["right"]], language)
        return _join([words["positive" if node[1] == "+" else "negative"], operand], language)
    op, left, right = node[1], _speak_math(node[2], language, cfg), _speak_math(node[3], language, cfg)
    if op == "^":
        if node[3] in {("atom", "2"), ("atom", "3")}:
            return _join([left, words["square" if node[3][1] == "2" else "cube"]], language)
        if node[3][0] == "binary":
            right = _join([words["left"], right, words["right"]], language)
        return _power(left, right, language, cfg)
    key = _RELATIONS[op] if kind == "relation" else {"+": "plus", "-": "minus", "−": "minus", "*": "times", "×": "times", "·": "times", "/": "divide", "÷": "divide"}[op]
    return _join([left, words[key], right], language)


def normalize_notation(text: str, language: str) -> tuple[str, list[str]]:
    """Return spoken input plus decision notes; preserve uncertain source spans."""
    language = language.lower().split("-")[0].split("_")[0]
    notes, spans = [], []
    if language not in _LANGUAGES:
        return text, [f"notation:unsupported_language:{language}"]
    try:
        cfg = _config()
    except (OSError, KeyError, ValueError) as exc:
        return text, [f"notation:configuration_unavailable:{type(exc).__name__}"]
    number_pattern, token_pattern, formula_pattern = ((_COMMA_NUMBER, _COMMA_TOKEN, _COMMA_FORMULA_LIKE)
        if language in _COMMA_LANGUAGES else (_NUMBER, _TOKEN, _FORMULA_LIKE))
    cues = [re.escape(cue) for cue in cfg.get("unit_identifier_cues", {}).get(language, [])]
    cues.append(_IDENTIFIER_CUES[language])
    bridge = {"en": "is|are", "de": "lautet|ist", "fr": "est", "es": "es"}.get(language)
    bridge = r"(?:[ \t]+(?:" + bridge + r"))?" if bridge else ""
    identifier_prefix = re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(cues) +
                                   r")" + bridge + r"[ \t:：#=\"'「]*$", re.IGNORECASE)

    def overlaps(start, end):
        return any(start < b and end > a for a, b, _ in spans)

    def add(start, end, kind, render):
        if overlaps(start, end):
            return
        try:
            spoken = render()
        except (ImportError, ValueError, TypeError, NotImplementedError, OverflowError, KeyError) as exc:
            spoken = text[start:end]
            notes.append(f"notation:preserved:{kind}:{type(exc).__name__}:{text[start:end]}")
        else:
            notes.append(f"notation:{kind}:{text[start:end]}")
        spans.append((start, end, spoken))

    spans.extend((m.start(), m.end(), m.group()) for m in _PHONE_HINT.finditer(text))
    for match in _address_matches(text):
        spans.append((match.start(), match.end(), match.group()))
        notes.append(f"notation:protected_literal:{match.group()}")
    if language == "en":
        for match in _EN_MONEY.finditer(text):
            def money(m=match):
                if m["left"] and m["right"]:
                    raise ValueError("ambiguous currency sign")
                sign = m["left"] or m["right"]
                word = "negative" if sign in {"-", "−"} else "positive"
                prefix = cfg["operator_words"]["en"][word] + " " if sign else ""
                if m["c"] == "£":
                    amount = Decimal(m["n"].replace(",", ""))
                    if amount * 100 != (amount * 100).to_integral_value():
                        return prefix + _number(m["n"], "en", cfg) + " pounds sterling"
                    words = _integer(amount, "en", to="currency", currency="GBP")
                    if amount == int(amount):
                        words = words.removesuffix(", zero pence")
                    elif amount < 1:
                        words = words.removeprefix("zero pounds sterling, ")
                    return prefix + words
                return prefix + m["c"] + m["n"]
            add(match.start(), match.end(), "english_currency", money)
        for match in _EN_HARDWARE.finditer(text):
            def hardware(m=match):
                suffix = "T I" if (m["suffix"] or "").lower() == "ti" else (m["suffix"] or "")
                return (m["family"].upper() + m["n"] + " " + suffix).strip()
            add(match.start(), match.end(), "english_hardware_model", hardware)
    for match in _VERSION.finditer(text):
        value = match.group("v1") or match.group("v2")
        start = match.start("v1") if match.group("v1") else match.start("v2")
        def version(value=value):
            parts = value.lstrip("vV").split(".")
            def component(part):
                return _digits(part, language, cfg) if len(part) >= 4 or (len(part) > 1 and part.startswith("0")) else _number(part, language, cfg)
            pieces = [component(parts[0])]
            for part in parts[1:]:
                pieces.extend([cfg["operator_words"][language]["dot"], component(part)])
            spoken = _join(pieces, language)
            if value[:1].lower() == "v":
                prefix = cfg["version_prefix_words"][language]["lower" if value[0] == "v" else "upper"]
                return _join([prefix, spoken], language)
            return spoken
        add(start, start + len(value), "version", version)
    for match in _DATE.finditer(text):
        local = _context(text, match.start(), match.end())
        if any(cue in local for cue in ("พ.ศ.", "佛历", "佛曆", "buddhist")):
            add(match.start(), match.end(), "calendar_requires_context", lambda m=match: m.group())
        else:
            add(match.start(), match.end(), "iso_date", lambda m=match: _date(m.group(), language, cfg))
    for match in _DATE_LIKE.finditer(text):
        add(match.start(), match.end(), "ambiguous_date_format", lambda m=match: m.group())
    if language == "en":
        for match in _EN_IPV4.finditer(text):
            prefix = text[max(0, match.start()-48):match.start()]
            valid = all(int(part) <= 255 for part in match.group().split("."))
            cue = re.search(r"\b(?:ipv4|ip|address)(?:[ \t]+(?:address|is|at))?"
                            r"[ \t:=]*$", prefix, re.I)
            if valid and cue:
                add(match.start(), match.end(), "ipv4",
                    lambda m=match: " dot ".join(_digits(part, "en", cfg) for part in m.group().split(".")))
    for match in _DOTTED.finditer(text):
        if not overlaps(match.start(), match.end()):
            add(match.start(), match.end(), "ambiguous_dotted_identifier", lambda m=match: m.group())
    for match in _CLOCK.finditer(text):
        context = _context(text, match.start(), match.end())
        duration = any(re.search(r"(?<!\w)"+re.escape(cue)+r"(?!\w)", context) if cue.isascii() else cue in context for cue in cfg["duration_cues"][language])
        if not duration and any(cue in context for cue in cfg["ambiguous_clock_cues"][language]):
            add(match.start(), match.end(), "ambiguous_clock_or_offset", lambda m=match: m.group())
            continue
        end = match.end()
        if language == "th" and not duration:
            suffix = re.match(r"[ \t]*นาฬิกา", text[end:])
            if suffix:
                end += suffix.end()
        add(match.start(), end, "duration" if duration else "clock", lambda m=match, d=duration: _clock(m.group(), d, language, cfg))
    formula_ranges = [(match.start(), match.end()) for match in formula_pattern.finditer(text)]
    for match in _unit_pattern(language).finditer(text):
        unit = cfg["unit_symbol_aliases"].get(match["u"], match["u"])
        if language == "ru":
            unit = _RU_UNIT_ALIASES.get(unit, unit)
        # Keep single-letter operands in algebra on the existing math path.
        if len(unit) == 1 and any(start <= match.start() and match.end() <= end for start, end in formula_ranges):
            continue
        compound = re.match(r"[/^²³][A-Za-z0-9µμ°℃²³^/().+\-−]*", text[match.end():])
        if compound:
            end = match.end() + compound.end()
            # A suffix such as /s may already be protected as a path. Merge
            # only unchanged literals into this larger literal span.
            spans[:] = [s for s in spans if not (match.start() <= s[0] and s[1] <= end and s[2] == text[s[0]:s[1]])]
            add(match.start(), end, "unsupported_compound_unit", lambda m=match, e=end: text[m.start():e])
            continue
        prefix = text[max(0, match.start()-48):match.start()]
        if any(re.search(r"(?<![A-Za-z0-9_])" + re.escape(cue) + r"[ \t:：#=\"'「]*$", prefix, re.I)
               for cue in cfg.get("unit_identifier_cues", {}).get(language, [])):
            add(match.start(), match.end(), "quantity_or_identifier_requires_context", lambda m=match: m.group())
            continue
        context = _context(text, match.start(), match.end())
        if match["u"] in {"KJ", "KM"}:
            cues = cfg["unit_alias_context"][unit][language]
            if not any(cue in context for cue in cues):
                add(match.start(), match.end(), "ambiguous_nonstandard_unit", lambda m=match: m.group())
                continue
        add(match.start(), match.end(), "quantity", lambda m=match, u=unit: _quantity(m["n"], u, language, cfg))

    if language == "en":
        for match in _EN_GROUPED.finditer(text):
            prefix = text[max(0, match.start()-64):match.start()]
            add(match.start(), match.end(), "english_grouped_number",
                lambda m=match, p=prefix: _digits(m["n"].replace(",", ""), "en", cfg) if m["n"][0].isdigit() and identifier_prefix.search(p) else _number(m["n"], "en", cfg))

    for match in _AMBIGUOUS_PAIR.finditer(text):
        context = _context(text, match.start(), match.end())
        if not any(cue in context for cue in cfg["math_cues"][language]) and not any(op in context for op in "=<>≤≥≠≈"):
            add(match.start(), match.end(), "ambiguous_fraction_or_identifier", lambda m=match: m.group())

    # Claim a complete equation/power before looking for smaller expressions.
    # Otherwise an unsupported lhs/body could leave a perfectly parseable tail
    # (e.g. -3 or r²) that would silently change only part of the source formula.
    for match in formula_pattern.finditer(text):
        start, end = match.start(), match.end()
        while end > start and text[end-1] in ("., \t" if language in _COMMA_LANGUAGES else ". \t"):
            end -= 1
        if overlaps(start, end):
            continue
        raw = text[start:end]
        mathematical = any(op in raw for op in "=<>≤≥≠≈^²³×·*/÷") or any(
            cue in _context(text, start, end) for cue in cfg["math_cues"][language])
        parsed = raw.replace("‑", "-") if mathematical else raw
        matches = list(token_pattern.finditer(parsed))
        values = [m.group().replace(",", ".") if language in _COMMA_LANGUAGES else m.group() for m in matches]
        covered = token_pattern.sub("", parsed).strip()
        strong = any(v in _RELATIONS or v in {"^", "²", "³", "×", "·", "*", "÷"} for v in values)
        strong |= bool(re.fullmatch(r"[A-Za-z][ \t]*\+[ \t]*[A-Za-z]", parsed))
        strong |= any(v.isdigit() or (language in _COMMA_LANGUAGES and re.fullmatch(_NUM, v)) for v in values) and any(v in {"+", "-", "−"} for v in values)
        strong |= any(cue in _context(text, start, end) for cue in cfg["math_cues"][language])
        try:
            if covered or not strong:
                raise ValueError("unclassified expression")
            node = _MathParser(values).parse()
        except ValueError:
            # Read an explicit plus without claiming the unknown expression parsed.
            add(start, end, "unsupported_formula", lambda raw=raw: raw.replace(
                "+", " " + cfg["operator_words"][language]["plus"] + " "))
        else:
            add(start, end, "math", lambda node=node: _speak_math(node, language, cfg))

    # Search only bounded token runs.  Only explicit numeric/pi coefficients
    # may imply multiplication; group nodes retain the source formula scope.
    tokens = list(token_pattern.finditer(text))
    for first, token in enumerate(tokens):
        if overlaps(token.start(), token.end()):
            continue
        best = None
        for last in range(first + 1, min(first + 40, len(tokens))):
            end_token = tokens[last]
            if text[tokens[last-1].end():end_token.start()].strip() or overlaps(end_token.start(), end_token.end()):
                break
            values = [t.group().replace(",", ".") if language in _COMMA_LANGUAGES else t.group() for t in tokens[first:last+1]]
            if not any(v in _RELATIONS or v in {"^", "²", "³", "+", "-", "−", "×", "·", "*", "÷", "/"} for v in values):
                continue
            # A bare a-b or 3/4 is also a label/date/fraction: require context.
            strong = any(v in _RELATIONS or v in {"^", "²", "³", "×", "·", "*", "÷"} for v in values)
            strong |= any(v.isdigit() or (language in _COMMA_LANGUAGES and re.fullmatch(_NUM, v)) for v in values) and any(v in {"+", "-", "−"} for v in values)
            strong |= any(cue in _context(text, token.start(), end_token.end()) for cue in cfg["math_cues"][language])
            if not strong:
                continue
            try:
                node = _MathParser(values).parse()
            except ValueError:
                continue
            # A final period is punctuation; .digits/.name still continues an
            # identifier.  Never accept a prefix before a remaining exponent,
            # bracket or operator (in particular the c before c² at sentence end).
            start, end = token.start(), end_token.end()
            if (start and re.match(r"[A-Za-z0-9_.(]", text[start-1])) or re.match(r"[A-Za-z0-9_)]|\.[A-Za-z0-9_]|[ \t]*[+\-−‑×·*/÷^²³=<>≤≥≠≈]", text[end:]):
                continue
            best = (end, node)
        if best:
            end, node = best
            add(token.start(), end, "math", lambda node=node: _speak_math(node, language, cfg))
    numbers = list(number_pattern.finditer(text))
    if language in _COMMA_LANGUAGES:
        # Two commas identify an ambiguous list, not a single decimal. Retain
        # the existing independent-number readings, including leading-zero rules.
        lists = [match.span() for match in re.finditer(r"\d+(?:,\d+){2,}", text)]
        numbers.extend(match for match in _NUMBER.finditer(text)
                       if any(start <= match.start() and match.end() <= end for start, end in lists))
        numbers.sort(key=lambda match: match.start())
    for match in numbers:
        prefix = text[max(0, match.start()-64):match.start()]
        phone = language == "zh" and re.fullmatch(r"\d{7,15}", match.group()) and re.search(
            r"(?:联系电话|电话号码|手机号码|手机号|电话)(?:为|是)?[ \t:：]*$", prefix)
        if language not in _COMMA_LANGUAGES and re.search(r"\d,$", prefix) and text[match.end():].startswith("."):
            # Preserve an unsupported comma-number fragment rather than just its tail.
            add(match.start(), match.end(), "ambiguous_comma_fragment", lambda m=match: m.group())
        elif match.group().isdigit() and (phone or identifier_prefix.search(prefix)):
            add(match.start(), match.end(), "identifier_digits", lambda m=match: _digits(m.group(), language, cfg))
        elif len(match.group()) > 1 and match.group().startswith("0") and match.group()[1].isdigit():
            add(match.start(), match.end(), "ambiguous_leading_zero_number", lambda m=match: m.group())
        else:
            def cardinal(m=match):
                if language == "ru" and m.group().isdigit() and re.search(r"%[ \t]+от[ \t]+$", text[:m.start()]):
                    return _integer(int(m.group()), language, case="g")
                return _number(m.group(), language, cfg)
            add(match.start(), match.end(), "number", cardinal)
    # These operators can occur outside an equation, e.g. AI Agent＋平台 or
    # ±0.5℃. Protected addresses and already-rendered expressions stay intact.
    for match in re.finditer(r"[＋±×÷−]|(?<![A-Za-z0-9_])\+(?![A-Za-z0-9_])", text):
        key = {"±": "pm", "×": "times", "÷": "divide", "−": "minus"}.get(match.group(), "plus")
        add(match.start(), match.end(), "operator",
            lambda key=key: " " + cfg["operator_words"][language][key] + " ")
    result, cursor = [], 0
    for start, end, spoken in sorted(spans):
        result.extend((text[cursor:start], spoken))
        cursor = end
    result.append(text[cursor:])
    return "".join(result), notes


LANGUAGES = ("zh", "en", "ja", "ru", "de", "fr", "es", "th")
_LATIN_LETTERS = r"A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u024f\u1e00-\u1eff"
_LATIN_WORD = (rf"[{_LATIN_LETTERS}][{_LATIN_LETTERS}0-9]*"
               rf"(?:[-'&][{_LATIN_LETTERS}0-9]+)*")
_LATIN = re.compile(r"(" + _LATIN_WORD + r"(?:[ \t]+(?:&[ \t]+)?" + _LATIN_WORD + r")*)")
_SYMBOL_MAP = {"ru": {"ɭ": "l", "ɵ": "o", "ʑ": "zʲ"}, "de": {"ʏ": "y"}}
_STOPS = re.compile(r'[.!?…;:,—][)\]”"]*\s*')


def phoneme_chunks(phones, limit=220, preferred=180, first_limit=0):
    """Split only at whole-word/punctuation boundaries, retaining every phone.

    Keep the opening phrase together by default. No phone is truncated to make
    a TensorRT profile fit. Callers may retry with a smaller ``limit`` when the
    predicted duration exceeds the vocoder profile.
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


class ReadingSession:
    """One node's language state; update only after complete PCM delivery."""

    def __init__(self):
        self.previous_language = None
        self.pending_language = None

    def resolve(self, frontend, text, preference):
        preference = str(preference or "auto").lower().replace("_", "-").split("-")[0]
        if preference not in ("auto", *LANGUAGES):
            raise ValueError(f"Unsupported reading language: {preference}")
        evidence = (frontend.language_evidence(text) if preference == "auto"
                    else preference)
        self.pending_language = evidence
        return evidence or self.previous_language or "zh"

    def commit(self):
        if self.pending_language is not None:
            self.previous_language = self.pending_language
        self.pending_language = None


class Frontend:
    """Load dictionaries once; expose stateless per-utterance language evidence."""

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
        cfg = lexicon["normalization"]
        self._literal_digits = cfg["digits"]["en"]
        symbols = {"_": "underscore", ".": "dot", "-": "dash", "/": "slash",
                   "\\": "backslash", ":": "colon", "+": "plus"}
        self._literal_symbol_words = {
            symbol: {lang: cfg["operator_words"][lang][key] for lang in LANGUAGES}
            for symbol, key in symbols.items()}
        self._literal_symbol_words["@"] = {
            "zh": "at", "en": "at", "ja": "アットマーク", "ru": "собака",
            "de": "ät", "fr": "arobase", "es": "arroba", "th": "แอท"}
        for symbol, words in cfg.get("literal_symbol_words", {}).items():
            self._literal_symbol_words.setdefault(symbol, {}).update(words)
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
        return self.language_evidence(text) or "en"

    def language_evidence(self, text):
        """Return this request's language, or None for script-neutral input."""
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
        spans = protected_spans(ordinary)
        spans += [m.span() for m in _unit_pattern().finditer(ordinary)]
        spans += [m.span() for m in _EN_HARDWARE.finditer(ordinary)]
        masked = list(ordinary)
        for start, end in spans:
            masked[start:end] = " " * (end - start)
        ordinary = "".join(masked)
        if not re.search(r"[^\W\d_]", ordinary, re.UNICODE):
            return None
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
        # A transliterated name's middle dot is a word boundary, not a phone.
        text = re.sub(r"(?<=[\u4e00-\u9fff])·(?=[\u4e00-\u9fff])", " ", text)
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
            return self._japanese(text.replace("/", " スラッシュ "))[0].translate(
                str.maketrans({"«": "(", "»": ")"}))
        if language == "es":
            text = text.replace("¿", "").replace("¡", "")
        if language == "th":
            return self._thai_phones(text)
        return self._espeak[language](text)[0].translate(str.maketrans(_SYMBOL_MAP.get(language, {})))

    def _literal_spans(self, text):
        spans = protected_spans(text)
        # The notation parser's Unicode word boundary misses "report-v2.txtを".
        # Reuse that parser on ASCII runs, preserving offsets and still requiring
        # a recognized literal; a bare a_b remains ordinary frontend input.
        for match in re.finditer(r"[!-~]*[_.][!-~]*", text):
            spans.extend((match.start() + start, match.start() + end)
                         for start, end in protected_spans(match[0]))
        return spans

    def _expand_literal_symbols(self, text, language):
        words = self._literal_symbol_words
        if not any(symbol in text for symbol in words):
            return text
        spans = self._literal_spans(text)
        digits = {}
        if language == "en":
            for match in re.finditer(r"(?<!\d)0\d+(?!\d)", text):
                if any(start <= match.start() and match.end() <= end for start, end in spans):
                    digits.update({i: self._literal_digits[int(text[i])] for i in range(match.start(), match.end())})
        return "".join(
            " " + digits[index] + " " if index in digits else
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
                    parts.extend((self._expand_literal_symbols(spoken[cursor:match.start()], language),
                                  "[" + match[0] + "](/" + self._entries[match.lastgroup] + "/)"))
                    cursor = match.end()
                parts.append(self._expand_literal_symbols(spoken[cursor:], language))
                return self._english("".join(parts))[0]

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
