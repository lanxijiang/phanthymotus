"""Request-local notation guards around the author's WeText frontend."""

from __future__ import annotations

import re
from collections.abc import Callable
from types import FunctionType


# Only a numeric quantity followed by the lowercase SI spelling m/s is
# rewritten. In particular, x=m/s and uppercase M/s are left to the author
# normalizer rather than guessing whether they denote a unit or an expression.
_METERS_PER_SECOND = re.compile(
    r"(?<![A-Za-z0-9_.])(?P<number>[+-]?(?:\d+(?:[.,]\d+)?|\.\d+))[ \t\u00a0]*m[ \t\u00a0]*/[ \t\u00a0]*s(?![A-Za-z0-9/])"
)
_STABLE_PHRASE = "meters each second"
_SPOKEN_PHRASE = "meters per second"

# Arithmetic requires explicit evidence; a plain hyphen/slash is ambiguous
# with a date, telephone, model identifier or filename. Scientific notation
# is one numeric token, including its exponent sign.
_NUMBER = r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(?:[eE][+\-−]?\d+)?"
_POWER = r"(?:[²³]|[ \t]*\^[ \t]*[+\-−]?(?:\d+|[A-Za-z]))"


def _math_pattern(number: str) -> re.Pattern[str]:
    atom = rf"(?:{number}(?:[A-Za-z])?|[A-Za-z]+)(?:{_POWER})?"
    term = rf"[+\-−‑–]?[ \t]*{atom}"
    simple = rf"{term}(?:[ \t]*[+\-−‑–*/×÷±][ \t]*{term})*"
    factor = rf"(?:{term}|[+\-−‑–]?[ \t]*(?:\({simple}\)|\[{simple}\])(?:{_POWER})?)"
    expression = rf"{factor}(?:[ \t]*[+\-−‑–*/×÷±][ \t]*{factor})*"
    return re.compile(rf"(?<![A-Za-z0-9_.]){expression}(?:[ \t]*(?:<=|>=|!=|==|[=<>≤≥≠≈])[ \t]*{expression})*(?![A-Za-z0-9_]|[.,][A-Za-z0-9_])")


_MATH = _math_pattern(_NUMBER)
_COMMA_NUMBER = r"(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:[eE][+\-−]?\d+)?"
_COMMA_MATH = _math_pattern(_COMMA_NUMBER)
_MATH_EVIDENCE = re.compile(r"[=<>≤≥≠≈²³^×÷±]|\S[ \t]*\+[ \t]*\S|\d[eE][+\-−]?\d")
_PROTECTED = re.compile(
    r"(?P<address>(?:[A-Za-z][A-Za-z0-9+.-]*://|www\.)[^\s，。！？；<>\"“”]+"
    r"|(?<![A-Za-z0-9_.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z0-9_.-]))"
    r"|(?P<measure>(?<![A-Za-z0-9_.])\d+(?:\.\d+)?[ \t]*(?:mm|cm|km|m)[²³](?![A-Za-z0-9_]))"
    r"|(?P<abbreviation>(?<![A-Za-z0-9_])(?:[A-Za-z]\.)+[A-Za-z]\.?(?:[ \t]*/[ \t]*(?:[A-Za-z]\.)+[A-Za-z]\.?)*(?![A-Za-z0-9_.]))"
    r"|(?P<file>(?<![A-Za-z0-9_])(?:[A-Za-z]:)?[\\/]?(?:[\w+²³.-]+[\\/])*"
    r"[\w+²³.-]+\.[A-Za-z][A-Za-z0-9]{0,9}(?![A-Za-z0-9_]|\.[A-Za-z0-9_]))"
    r"|(?P<version>(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_-]*\d+(?:\.\d+)+(?![A-Za-z0-9_]|\.[A-Za-z0-9_]))"
    r"|(?P<technical>(?<![A-Za-z0-9_])(?:C\+\+|C#)(?:\d+(?:\.\d+)*)?(?![A-Za-z0-9_]))"
    r"|(?P<structured>(?<![A-Za-z0-9_])(?:\d{4}[-./]\d{1,2}[-./]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?"
    r"|\d{1,2}:\d{2}(?::\d{2})?|\d{1,2}/\d{1,2}[ \t]*[–−-][ \t]*\d{1,2}/\d{1,2}"
    r"|\d{2,4}-\d{6,8}|\d{2,4}-\d{2,4}-\d{3,4})(?![A-Za-z0-9_]))",
    re.IGNORECASE,
)
_DIGITS = {"zh": tuple("零一二三四五六七八九"),
           "en": "zero one two three four five six seven eight nine".split(),
           "ja": "ゼロ いち に さん よん ご ろく なな はち きゅう".split(),
           "ru": "ноль один два три четыре пять шесть семь восемь девять".split(),
           "de": "null eins zwei drei vier fünf sechs sieben acht neun".split(),
           "fr": "zéro un deux trois quatre cinq six sept huit neuf".split(),
           "es": "cero uno dos tres cuatro cinco seis siete ocho nueve".split()}

# Grammar vocabulary, not business-name pronunciation overrides. One shared
# expression grammar selects these words using the request's resolved language.
_WORDS = {
    "zh": dict(equals="等于", plus="加", minus="减", negative="负", positive="正", times="乘以", divide="除以",
               square="的平方", cube="的立方", power="的 {n} 次方", left="左括号", right="右括号",
               less="小于", greater="大于", le="小于等于", ge="大于等于", ne="不等于", approx="约等于", pm="正负",
               point="点", dot="点", underscore="下划线", dash="连字符", slash="斜杠", backslash="反斜杠", colon="冒号"),
    "en": dict(equals="equals", plus="plus", minus="minus", negative="negative", positive="positive", times="times", divide="divided by",
               square="squared", cube="cubed", power="to the power of {n}", left="left parenthesis", right="right parenthesis",
               less="less than", greater="greater than", le="less than or equal to", ge="greater than or equal to", ne="not equal to", approx="approximately equals", pm="plus or minus",
               point="point", dot="dot", underscore="underscore", dash="dash", slash="slash", backslash="backslash", colon="colon"),
    "ja": dict(equals="イコール", plus="プラス", minus="マイナス", negative="マイナス", positive="プラス", times="かける", divide="割る",
               square="のにじょう", cube="のさんじょう", power="の {n} じょう", left="かっこ", right="かっことじ",
               less="しょうなり", greater="だいなり", le="しょうなりイコール", ge="だいなりイコール", ne="ノットイコール", approx="ほぼひとしい", pm="プラスマイナス",
               point="てん", dot="ドット", underscore="アンダースコア", dash="ハイフン", slash="スラッシュ", backslash="バックスラッシュ", colon="コロン"),
    "ru": dict(equals="равно", plus="плюс", minus="минус", negative="минус", positive="плюс", times="умножить на", divide="разделить на",
               square="в квадрате", cube="в кубе", power="в степени {n}", left="скобка открывается", right="скобка закрывается",
               less="меньше", greater="больше", le="меньше или равно", ge="больше или равно", ne="не равно", approx="приблизительно равно", pm="плюс минус",
               point="запятая", dot="точка", underscore="нижнее подчёркивание", dash="дефис", slash="косая черта", backslash="обратная косая черта", colon="двоеточие"),
    "de": dict(equals="gleich", plus="plus", minus="minus", negative="minus", positive="plus", times="mal", divide="geteilt durch",
               square="hoch zwei", cube="hoch drei", power="hoch {n}", left="Klammer auf", right="Klammer zu",
               less="kleiner als", greater="größer als", le="kleiner oder gleich", ge="größer oder gleich", ne="ungleich", approx="ungefähr gleich", pm="plus minus",
               point="Komma", dot="Punkt", underscore="Unterstrich", dash="Bindestrich", slash="Schrägstrich", backslash="umgekehrter Schrägstrich", colon="Doppelpunkt"),
    "fr": dict(equals="égale", plus="plus", minus="moins", negative="moins", positive="plus", times="fois", divide="divisé par",
               square="au carré", cube="au cube", power="à la puissance {n}", left="parenthèse ouvrante", right="parenthèse fermante",
               less="inférieur à", greater="supérieur à", le="inférieur ou égal à", ge="supérieur ou égal à", ne="différent de", approx="environ égal à", pm="plus ou moins",
               point="virgule", dot="point", underscore="tiret bas", dash="tiret", slash="barre oblique", backslash="barre oblique inverse", colon="deux-points"),
    "es": dict(equals="igual a", plus="más", minus="menos", negative="menos", positive="más", times="por", divide="dividido por",
               square="al cuadrado", cube="al cubo", power="elevado a {n}", left="abre paréntesis", right="cierra paréntesis",
               less="menor que", greater="mayor que", le="menor o igual a", ge="mayor o igual a", ne="distinto de", approx="aproximadamente igual a", pm="más o menos",
               point="coma", dot="punto", underscore="guion bajo", dash="guion", slash="barra", backslash="barra invertida", colon="dos puntos"),
}
_SYMBOLS = {"=": "equals", "+": "plus", "-": "minus", "−": "minus", "*": "times", "×": "times",
            "/": "divide", "÷": "divide", "²": "square", "³": "cube", "(": "left", ")": "right", "[": "left", "]": "right",
            "<": "less", ">": "greater", "≤": "le", "≥": "ge", "≠": "ne", "≈": "approx", "±": "pm"}
_FILE_SYMBOLS = {".": "dot", "_": "underscore", "-": "dash", "/": "slash", "\\": "backslash", ":": "colon", "+": "plus"}
_LETTER_NAMES = {
    "en": "ay|bee|see|dee|ee|ef|gee|aitch|eye|jay|kay|el|em|en|oh|pee|cue|ar|ess|tee|you|vee|double you|ex|why|zee",
    "ja": "エー|ビー|シー|ディー|イー|エフ|ジー|エイチ|アイ|ジェー|ケー|エル|エム|エヌ|オー|ピー|キュー|アール|エス|ティー|ユー|ブイ|ダブリュー|エックス|ワイ|ゼット",
    "ru": "а|бэ|цэ|дэ|е|эф|гэ|аш|и|йот|ка|эль|эм|эн|о|пэ|ку|эр|эс|тэ|у|вэ|дубль вэ|икс|игрек|зет",
    "de": "ah|be|ze|de|eh|ef|ge|ha|ih|jot|ka|el|em|en|oh|pe|ku|er|es|te|uh|fau|we|iks|ypsilon|zett",
    "fr": "a|bé|cé|dé|e|effe|gé|ache|i|ji|ka|elle|emme|enne|o|pé|ku|erre|esse|té|u|vé|double vé|iks|i grec|zède",
    "es": "a|be|ce|de|e|efe|ge|hache|i|jota|ka|ele|eme|ene|o|pe|cu|erre|ese|te|u|uve|uve doble|equis|i griega|zeta",
}
_WIDTH = str.maketrans({chr(ord(c) + 0xFEE0): c for c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz=+-*/()[]^<>"})
_TEENS = {
    "ru": "десять одиннадцать двенадцать тринадцать четырнадцать пятнадцать шестнадцать семнадцать восемнадцать девятнадцать".split(),
    "de": "zehn elf zwölf dreizehn vierzehn fünfzehn sechzehn siebzehn achtzehn neunzehn".split(),
    "fr": "dix onze douze treize quatorze quinze seize dix-sept dix-huit dix-neuf".split(),
    "es": "diez once doce trece catorce quince dieciséis diecisiete dieciocho diecinueve".split(),
}
_TENS = {
    "ru": "двадцать тридцать сорок пятьдесят шестьдесят семьдесят восемьдесят девяносто".split(),
    "de": "zwanzig dreißig vierzig fünfzig sechzig siebzig achtzig neunzig".split(),
    "fr": "vingt trente quarante cinquante soixante".split(),
    "es": "veinte treinta cuarenta cincuenta sesenta setenta ochenta noventa".split(),
}


def _small_cardinal(number: int, language: str) -> str:
    """Unambiguous small formula operands where no WeText grammar is used."""
    if number < 10:
        return _DIGITS[language][number]
    tens, ones = divmod(number, 10)
    if number < 20:
        return _TEENS[language][number - 10]
    if language == "fr" and number >= 70:
        if number < 80:
            return "soixante" + (" et " if number == 71 else "-") + _small_cardinal(number - 60, language)
        return "quatre-vingts" if number == 80 else "quatre-vingt-" + _small_cardinal(number - 80, language)
    prefix = _TENS[language][tens - 2]
    if not ones:
        return prefix
    digit = _DIGITS[language][ones]
    if language == "de":
        return ("ein" if ones == 1 else digit) + "und" + prefix
    if language == "fr":
        return prefix + (" et " if ones == 1 else "-") + digit
    if language == "es":
        return ("veinti" + {2: "dós", 3: "trés", 6: "séis"}.get(ones, digit)
                if tens == 2 else prefix + " y " + digit)
    return prefix + " " + digit


def _speak_number(value: str, language: str, *, digits: bool = False) -> str:
    words = _WORDS[language]
    if value[:1] in {"+", "-", "−"}:
        sign = words["positive" if value[0] == "+" else "negative"]
        return sign + " " + _speak_number(value[1:], language, digits=digits)
    if "e" in value.lower():
        base, exponent = re.split("[eE]", value)
        return " ".join((_speak_number(base, language), words["times"], _speak_number("10", language),
                         words["power"].format(n=_speak_number(exponent, language))))
    if digits:
        return " ".join(_DIGITS[language][int(d)] for d in value)
    if language in {"zh", "en", "ja"}:
        value = value.replace(",", "")
    if "." in value or "," in value:
        whole, fraction = re.split("[.,]", value)
        return " ".join((_speak_number(whole or "0", language), words["point"],
                         _speak_number(fraction, language, digits=True)))
    if len(value) == 1:
        return _DIGITS[language][int(value)]
    if language in {"ru", "de", "fr", "es"} and int(value) < 100:
        return _small_cardinal(int(value), language)
    if language == "ja" and len(value) <= 16:
        # Ordinary Japanese cardinal numbers, not digit-by-digit IDs.
        groups, number = [], int(value)
        for large in ("", "まん", "おく", "ちょう"):
            group, number = number % 10000, number // 10000
            small = ""
            for divisor, unit in ((1000, "せん"), (100, "ひゃく"), (10, "じゅう"), (1, "")):
                digit, group = divmod(group, divisor)
                if digit:
                    special = {(1000, 3): "さんぜん", (1000, 8): "はっせん",
                               (100, 3): "さんびゃく", (100, 6): "ろっぴゃく", (100, 8): "はっぴゃく"}
                    small += special.get((divisor, digit),
                        ("" if digit == 1 and divisor > 1 else _DIGITS["ja"][digit]) + unit)
            if small:
                if large == "ちょう":
                    small = re.sub(r"(?:いち|はち|じゅう)$", lambda m: {"いち":"いっ", "はち":"はっ", "じゅう":"じゅっ"}[m.group()], small)
                groups.insert(0, small + large)
        return "".join(groups) or _DIGITS["ja"][0]
    return value


def _speak_identifier(value: str, language: str) -> str:
    # Read individual variables, while keeping words and technical names on
    # the normal pronunciation path instead of spelling every short word.
    if len(value) != 1:
        return value
    if language == "zh":
        # Retain the existing mixed Chinese/Latin pronunciation path.
        return value
    return _LETTER_NAMES[language].split("|")[ord(value.lower()) - ord("a")]


def _speak_math(match: re.Match[str], language: str) -> str:
    value = match.group()
    if not _MATH_EVIDENCE.search(value):
        return value
    # Do not rewrite a valid-looking suffix of unsupported syntax such as
    # function calls or nested groups. Whitespace does not break an operator.
    before, after = match.string[:match.start()].rstrip(), match.string[match.end():].lstrip()
    if (before and before[-1] in "=<>≤≥≠≈+−-‑–*/×÷±^([{") or (after and after[0] in "=<>≤≥≠≈+−-‑–*/×÷±^)]}"):
        return value
    if (value[:1].isdecimal() and before.endswith((".", ","))
            and before[:-1].endswith(tuple("0123456789"))):
        return value
    if value.lstrip()[0] in "+-−‑–" and before and (before[-1].isalnum() or before[-1] in ")²³"):
        return value
    value = value.replace("‑", "-").replace("–", "-")
    value = value.replace("<=", "≤").replace(">=", "≥").replace("!=", "≠").replace("==", "=")
    number = _COMMA_NUMBER if language in {"ru", "de", "fr", "es"} else _NUMBER
    tokens = re.findall(rf"{_POWER}|{number}|[A-Za-z]+|[^\s]", value)
    words = _WORDS[language]
    result, previous = [], None
    for token in tokens:
        token = token.strip()
        if re.fullmatch(r"[A-Za-z]+", token) and previous and re.fullmatch(number, previous):
            result.append(words["times"])
        if token.startswith("^"):
            exponent = token[1:].strip()
            if exponent in {"2", "3"}:
                result.append(words["square" if exponent == "2" else "cube"])
            else:
                unsigned = exponent.lstrip("+-−")
                if unsigned.isalpha():
                    spoken = _speak_identifier(unsigned, language)
                    if unsigned != exponent:
                        spoken = words["positive" if exponent[0] == "+" else "negative"] + " " + spoken
                else:
                    spoken = _speak_number(exponent, language)
                result.append(words["power"].format(n=spoken))
        elif token in {"-", "−", "+"} and (previous is None or previous in "([=<>≤≥≠≈+−-*/×÷±"):
            result.append(words["positive" if token == "+" else "negative"])
        elif re.fullmatch(number, token):
            result.append(_speak_number(token, language))
        elif token.isascii() and token.isalpha():
            result.append(_speak_identifier(token, language))
        else:
            result.append(words[_SYMBOLS[token]] if token in _SYMBOLS else token)
        previous = token
    leading = value[:len(value) - len(value.lstrip())]
    trailing = value[len(value.rstrip()):]
    return leading + " ".join(result) + trailing


def _rewrite_notation(text: str, language: str) -> str:
    """Reconstruct spans directly; never run math rules inside addresses/files."""
    result, cursor = [], 0
    words = _WORDS[language]
    pattern = _COMMA_MATH if language in {"ru", "de", "fr", "es"} else _MATH

    def plain(value):
        # Full-width ASCII is common in Japanese. Do not use NFKC: it would
        # flatten superscript ² into an ordinary digit before we can read it.
        value = value.translate(_WIDTH)
        value = pattern.sub(lambda m: _speak_math(m, language), value)
        # Unambiguous standalone symbols, e.g. an isolated equals sign. ASCII
        # hyphens and slashes remain punctuation unless part of a formula.
        return re.sub(r"(?<!\S)[=−×÷²³±≤≥≠≈](?=\s|$|[。,.!?！？、，；;])",
                      lambda m: words[_SYMBOLS[m.group()]], value)

    for match in _PROTECTED.finditer(text):
        result.append(plain(text[cursor:match.start()]))
        value = match.group()
        if match.lastgroup == "file":
            parts = []
            extension_start = value.rfind(".") + 1
            for token_match in re.finditer(r"\d+|[^\W\d_]+|[^\s]", value):
                token = token_match.group()
                if token in _FILE_SYMBOLS:
                    parts.append(words[_FILE_SYMBOLS[token]])
                elif token.isdecimal():
                    parts.append(_speak_number(token, language, digits=True))
                elif token_match.start() == extension_start and token.isascii() and token.isalpha():
                    # Extension spelling is structural, not a list of special
                    # filenames: ONNX, TXT and future extensions use one path.
                    parts.append(" ".join(_speak_identifier(c.upper(), language) for c in token))
                else:
                    parts.append(_speak_identifier(token, language) if len(token) == 1 and token.isascii() and token.isalpha() else token)
            value = " ".join(parts)
        elif match.lastgroup == "version":
            prefix, number = re.fullmatch(r"(.*?)(\d+(?:\.\d+)+)", value).groups()
            parts = number.split(".")
            spoken = [parts[0], *(" ".join(_DIGITS[language][int(d)] for d in part) for part in parts[1:])]
            value = prefix + " " + (" " + words["dot"] + " ").join(spoken)
        result.append(value)
        cursor = match.end()
    result.append(plain(text[cursor:]))
    return "".join(result)


def resolve_request_language(assets, text, voice, language, *, normalize=True):
    """Preserve the author's voice-based choice for language-neutral input."""
    if language != "auto":
        return language
    raw = str(text or "").strip().rstrip("。.!?！？")
    if (_MATH.fullmatch(raw) and _MATH_EVIDENCE.search(raw) and not _PROTECTED.search(raw)
            and all(len(name) == 1 for name in re.findall(r"[A-Za-z]+", raw))):
        from text_normalization_pipeline import resolve_text_normalization_language

        return resolve_text_normalization_language(text="", voice=voice)
    detected = assets.detect_language(text)
    if detected == "en" and not re.search(r"[A-Za-z\u3400-\u9fff]", str(text or "")):
        from text_normalization_pipeline import normalize_tts_text, resolve_text_normalization_language

        raw = str(text or "")
        value = normalize_tts_text(raw) if normalize else raw
        return resolve_text_normalization_language(text=value, voice=voice)
    return detected


def _rewrite_mps(text: str) -> tuple[str, int]:
    count = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f"{match.group('number')} {_STABLE_PHRASE}"

    return _METERS_PER_SECOND.sub(replace, text), count


def _restore_mps(text: str, count: int) -> str:
    if count and text.count(_STABLE_PHRASE) != count:
        raise ValueError(
            "WeText meters-per-second restoration is ambiguous; "
            "the protected phrase was changed or already occurs in the input"
        )
    return text.replace(_STABLE_PHRASE, _SPOKEN_PHRASE) if count else text


def prepare_with_si_unit_guard(
    base_prepare: Callable[..., dict],
    *,
    text: str,
    voice: str = "",
    prompt_text: str = "",
    enable_wetext: bool,
    enable_normalize_tts_text: bool = True,
    normalization_language: str | None = None,
) -> dict:
    """Run the original normalizer after narrow notation and SI guards.

    Explicit arithmetic and file separators become ordinary spoken text.
    Addresses and unsupported expressions retain their original path.
    WeText reads ``3 m/s`` as ``three minutes per second``. Feeding it the
    literal words ``3 meters per second`` also corrupts ``per second`` in the
    installed grammar. ``meters each second`` survives WeText intact; we
    restore the equivalent conventional wording in the final model text.
    """
    original_text = str(text or "")
    original_prompt = str(prompt_text or "")
    normalized_branch = normalization_language or ""
    if enable_wetext:
        # These functions are from the unchanged author source, imported when
        # ModelAssets loads that source. Match its robust-pre language choice.
        from text_normalization_pipeline import normalize_tts_text, resolve_text_normalization_language

        branch_input = normalize_tts_text(original_text) if enable_normalize_tts_text else original_text
        normalized_branch = normalization_language or resolve_text_normalization_language(text=branch_input, voice=voice)
    # Thai keeps the author's input: symbol expansion regressed GPU speech.
    if (normalized_branch in _WORDS
            and (enable_wetext or enable_normalize_tts_text)):
        guarded_text = _rewrite_notation(original_text, normalized_branch)
        guarded_prompt = _rewrite_notation(original_prompt, normalized_branch)
    else:
        guarded_text, guarded_prompt = original_text, original_prompt
    if normalized_branch == "en":
        input_text, text_count = _rewrite_mps(guarded_text)
        input_prompt, prompt_count = _rewrite_mps(guarded_prompt)
    else:
        input_text, input_prompt = guarded_text, guarded_prompt
        text_count = prompt_count = 0

    prepared = base_prepare(
        text=input_text,
        voice=voice,
        prompt_text=input_prompt,
        enable_wetext=enable_wetext,
        enable_normalize_tts_text=enable_normalize_tts_text,
    )
    if text_count or prompt_count:
        for key in ("text", "normalized_text"):
            prepared[key] = _restore_mps(str(prepared[key]), text_count)
        for key in ("prompt_text", "normalized_prompt_text"):
            prepared[key] = _restore_mps(str(prepared[key]), prompt_count)
        prepared["normalization_method"] = str(prepared["normalization_method"]) + "+en_si_mps_guard"
        prepared["si_unit_guard"] = {
            "original_text": original_text,
            "pre_wetext_text": input_text,
            "original_prompt_text": original_prompt,
            "pre_wetext_prompt_text": input_prompt,
            "text_replacements": text_count,
            "prompt_replacements": prompt_count,
            "final_text": prepared["text"],
        }
    if guarded_text != original_text or guarded_prompt != original_prompt:
        prepared["normalization_method"] = str(prepared["normalization_method"]) + "+notation_guard"
    return prepared


def prepare_with_request_language(runtime, base_prepare, *, normalization_language=None, **kwargs):
    """Keep WeText's choice local to this request without changing author globals."""
    if kwargs.get("enable_wetext") and normalization_language in {"zh", "en"}:
        prepare_method = base_prepare.__func__
        author_pipeline = prepare_method.__globals__["prepare_tts_request_texts"]
        pipeline_globals = dict(author_pipeline.__globals__)
        pipeline_globals["resolve_text_normalization_language"] = lambda **_kwargs: normalization_language
        request_pipeline = FunctionType(author_pipeline.__code__, pipeline_globals,
                                        author_pipeline.__name__, author_pipeline.__defaults__,
                                        author_pipeline.__closure__)
        request_pipeline.__kwdefaults__ = author_pipeline.__kwdefaults__
        manager = runtime._ensure_text_normalizer(True)

        def prepare(**request):
            return request_pipeline(**request, text_normalizer_manager=manager)
    else:
        prepare = base_prepare
    return prepare_with_si_unit_guard(prepare, normalization_language=normalization_language, **kwargs)
