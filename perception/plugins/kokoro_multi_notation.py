"""Conservative typed notation expansion before language G2P.

This module owns no pronunciation dictionary and imports no TTSV4 code.  Unknown
or ambiguous spans survive verbatim.  Notes describe decisions, not certification.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from functools import lru_cache
import json
from pathlib import Path
import re


_LANGUAGES = {"zh", "en", "ja", "ru", "de", "fr", "es", "th"}
_ADDRESS = re.compile(
    r"`[^`\n]+`|(?<!\w)--[A-Za-z][\w-]*(?:=[^\s;,]+)?|"
    r"(?<!\w)-[A-Za-z]\s+[+\-]?\d+(?:\.\d+)?|https?://[^\s<>`]+|www\.[^\s<>`]+|"
    r"[\w.!#$%&'*+/=?^`{|}~-]+@[\w-]+(?:\.[\w-]+)+|"
    r"(?<!\w)(?:[A-Za-z]:[\\/]|\.?\.?/)[\w./\\+@~%-]+|"
    r"(?<![\w/])[\w.-]+(?:[/\\][\w.+@~-]+)*[/\\][\w.-]+\.[A-Za-z]{1,8}(?!\w)|"
    r"(?<![\w.])[\w-]+(?:\.[\w-]+)*\.(?:txt|csv|jsonl?|onnx|pt|pth|wav|mp3|py|sh|zip|pdf|md|yaml|yml|log)(?!\w)",
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
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])" + _NUM + r"(?![A-Za-z0-9_.])")
_FORMULA_LIKE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[+\-−][ \t]*)?[A-Za-z\u0370-\u03ff0-9_().]+"
    r"(?:[ \t]*[+\-−×*/÷^²³=<>≤≥≠≈][ \t]*[+\-−]?[ \t]*[A-Za-z\u0370-\u03ff0-9_().]*)+"
    r"|(?<![A-Za-z0-9_])[A-Za-z]+[ \t]*\([A-Za-z\u0370-\u03ff0-9_().+\-−×*/÷^²³ \t]+\)"
)
_TOKEN = re.compile(_NUM + r"|[A-Za-z]+|π|[()+\-−×*÷/^²³=<>≤≥≠≈]")
_RELATIONS = {"=": "equals", "<": "less", ">": "greater", "≤": "le", "≥": "ge", "≠": "ne", "≈": "approx"}


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
    return json.loads(_lexicon_path.read_text(encoding="utf-8"))["normalization"]


@lru_cache(maxsize=1)
def _unit_pattern():
    cfg = _config()
    symbols = (set(cfg["units"]) - {"hour", "minute", "second"}) | set(cfg["unit_symbol_aliases"])
    names = "|".join(re.escape(symbol) for symbol in sorted(symbols, key=len, reverse=True))
    return re.compile(r"(?<![A-Za-z0-9_.\-−])(?P<n>[+\-−]?" + _NUM +
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
    for match in _ADDRESS.finditer(text):
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
    if value[:1] in {"-", "+"}:
        key = "negative" if value[0] == "-" else "positive"
        return _join([cfg["operator_words"][language][key], _number(value[1:], language, cfg)], language)
    if "e" in value.lower():
        mantissa, exponent = re.split("[eE]", value)
        return _join([_number(mantissa, language, cfg), cfg["operator_words"][language]["times"],
                      _power(_integer(10, language), _number(str(int(exponent)), language, cfg), language, cfg)], language)
    if "." in value:
        whole, fraction = value.split(".")
        return _join([_integer(int(whole), language), cfg["operator_words"][language]["point"],
                      _digits(fraction, language, cfg)], language)
    return _integer(int(value), language)


def _power(base, exponent, language, cfg):
    pattern = cfg["operator_words"][language]["power"]
    return _join([base, pattern.replace("{n}", exponent)], language)


def _unit_form(forms, value, language):
    number = Decimal(value.replace("−", "-"))
    if language == "ru":
        if number != int(number):
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
    if language == "fr" and unit in {"hour", "minute", "second"} and re.fullmatch(r"[+\-−]?\d+", value):
        words = re.sub(r"\bun$", "une", words)
    if language == "es" and re.fullmatch(r"[+\-−]?\d+", value):
        if unit == "hour":
            words = re.sub(r"uno$", "una", words)
        else:
            words = re.sub(r"veintiuno$", "veintiún", words)
            words = re.sub(r"uno$", "un", words)
    unit_words = _unit_form(cfg["units"][unit][language], value, language)
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
            precedence = 20 if implicit else {"+": 10, "-": 10, "−": 10, "*": 20, "×": 20, "/": 20, "÷": 20, "^": 40, "²": 40, "³": 40}.get(op, -1)
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
        letters=cfg["letters"][language]
        trial=cfg.get("math_letter_trial", {})
        if trial.get("enabled", False):letters=dict(letters,**trial.get("letters", {}).get(language, {}))
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
    key = _RELATIONS[op] if kind == "relation" else {"+": "plus", "-": "minus", "−": "minus", "*": "times", "×": "times", "/": "divide", "÷": "divide"}[op]
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

    for match in _address_matches(text):
        spans.append((match.start(), match.end(), match.group()))
        notes.append(f"notation:protected_literal:{match.group()}")
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
    for match in _DOTTED.finditer(text):
        if not overlaps(match.start(), match.end()):
            add(match.start(), match.end(), "ambiguous_dotted_identifier", lambda m=match: m.group())
    for match in _CLOCK.finditer(text):
        context = _context(text, match.start(), match.end())
        duration = any(re.search(r"(?<!\w)"+re.escape(cue)+r"(?!\w)", context) if cue.isascii() else cue in context for cue in cfg["duration_cues"][language])
        if not duration and any(cue in context for cue in cfg["ambiguous_clock_cues"][language]):
            add(match.start(), match.end(), "ambiguous_clock_or_offset", lambda m=match: m.group())
            continue
        add(match.start(), match.end(), "duration" if duration else "clock", lambda m=match, d=duration: _clock(m.group(), d, language, cfg))
    formula_ranges = [(match.start(), match.end()) for match in _FORMULA_LIKE.finditer(text)]
    for match in _unit_pattern().finditer(text):
        unit = cfg["unit_symbol_aliases"].get(match["u"], match["u"])
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

    for match in _AMBIGUOUS_PAIR.finditer(text):
        context = _context(text, match.start(), match.end())
        if not any(cue in context for cue in cfg["math_cues"][language]) and not any(op in context for op in "=<>≤≥≠≈"):
            add(match.start(), match.end(), "ambiguous_fraction_or_identifier", lambda m=match: m.group())

    # Claim a complete equation/power before looking for smaller expressions.
    # Otherwise an unsupported lhs/body could leave a perfectly parseable tail
    # (e.g. -3 or r²) that would silently change only part of the source formula.
    for match in _FORMULA_LIKE.finditer(text):
        start, end = match.start(), match.end()
        while end > start and text[end-1] in ". \t":
            end -= 1
        if overlaps(start, end):
            continue
        raw = text[start:end]
        matches = list(_TOKEN.finditer(raw))
        values = [m.group() for m in matches]
        covered = _TOKEN.sub("", raw).strip()
        strong = any(v in _RELATIONS or v in {"^", "²", "³", "×", "*", "÷"} for v in values)
        strong |= any(v.isdigit() for v in values) and any(v in {"+", "-", "−"} for v in values)
        strong |= any(cue in _context(text, start, end) for cue in cfg["math_cues"][language])
        try:
            if covered or not strong:
                raise ValueError("unclassified expression")
            node = _MathParser(values).parse()
        except ValueError:
            add(start, end, "unsupported_formula", lambda raw=raw: raw)
        else:
            add(start, end, "math", lambda node=node: _speak_math(node, language, cfg))

    # Search only bounded token runs.  Only explicit numeric/pi coefficients
    # may imply multiplication; group nodes retain the source formula scope.
    tokens = list(_TOKEN.finditer(text))
    for first, token in enumerate(tokens):
        if overlaps(token.start(), token.end()):
            continue
        best = None
        for last in range(first + 1, min(first + 40, len(tokens))):
            end_token = tokens[last]
            if text[tokens[last-1].end():end_token.start()].strip() or overlaps(end_token.start(), end_token.end()):
                break
            values = [t.group() for t in tokens[first:last+1]]
            if not any(v in _RELATIONS or v in {"^", "²", "³", "+", "-", "−", "×", "*", "÷", "/"} for v in values):
                continue
            # A bare a-b or 3/4 is also a label/date/fraction: require context.
            strong = any(v in _RELATIONS or v in {"^", "²", "³", "×", "*", "÷"} for v in values)
            strong |= any(v.isdigit() for v in values) and any(v in {"+", "-", "−"} for v in values)
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
            if (start and re.match(r"[A-Za-z0-9_.(]", text[start-1])) or re.match(r"[A-Za-z0-9_)]|\.[A-Za-z0-9_]|[ \t]*[+\-−×*/÷^²³=<>≤≥≠≈]", text[end:]):
                continue
            best = (end, node)
        if best:
            end, node = best
            add(token.start(), end, "math", lambda node=node: _speak_math(node, language, cfg))
    for match in _NUMBER.finditer(text):
        if len(match.group()) > 1 and match.group().startswith("0") and match.group()[1].isdigit():
            add(match.start(), match.end(), "ambiguous_leading_zero_number", lambda m=match: m.group())
        else:
            add(match.start(), match.end(), "number", lambda m=match: _number(m.group(), language, cfg))
    result, cursor = [], 0
    for start, end, spoken in sorted(spans):
        result.extend((text[cursor:start], spoken))
        cursor = end
    result.append(text[cursor:])
    return "".join(result), notes
