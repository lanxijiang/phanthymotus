"""Spoken forms for explicit math/units; ordinary words and numbers stay intact."""

import re
from decimal import Decimal


# key: Chinese, singular English, plural English. Long units match first.
_UNITS = {
    "kg/m³": ("千克每立方米", "kilogram per cubic meter", "kilograms per cubic meter"),
    "km/h": ("千米每小时", "kilometer per hour", "kilometers per hour"),
    "m/s": ("米每秒", "meter per second", "meters per second"),
    "m/s²": ("米每二次方秒", "meter per second squared", "meters per second squared"),
    "m²": ("平方米", "square meter", "square meters"),
    "cm²": ("平方厘米", "square centimeter", "square centimeters"),
    "m³": ("立方米", "cubic meter", "cubic meters"),
    "kg": ("千克", "kilogram", "kilograms"),
    "g": ("克", "gram", "grams"),
    "mg": ("毫克", "milligram", "milligrams"),
    "km": ("千米", "kilometer", "kilometers"),
    "m": ("米", "meter", "meters"),
    "cm": ("厘米", "centimeter", "centimeters"),
    "mm": ("毫米", "millimeter", "millimeters"),
    "°C": ("摄氏度", "degree Celsius", "degrees Celsius"),
    "°F": ("华氏度", "degree Fahrenheit", "degrees Fahrenheit"),
    "%": ("百分比", "percent", "percent"),
    "GB": ("G B", "gigabyte", "gigabytes"),
    "MB": ("M B", "megabyte", "megabytes"),
    "KB": ("K B", "kilobyte", "kilobytes"),
}
_NUMBER = r"\d+(?:\.\d+)?"
_UNIT_FORMS = sorted(_UNITS, key=len, reverse=True)
_UNIT_PATTERN = "|".join(re.escape(unit).replace(r"/", r"\s*/\s*") for unit in _UNIT_FORMS)
_MEASURE = re.compile(
    rf"(?<![A-Za-z0-9_.])(?P<number>[+-]?{_NUMBER})\s*"
    rf"(?P<unit>(?:{_UNIT_PATTERN})|℃|℉|(?i:degrees?\s+(?:Celsius|Fahrenheit)))"
    r"(?![A-Za-z0-9_/²³^])"
)
_ATOM = rf"(?<![A-Za-z0-9_.])(?:{_NUMBER}|[A-Za-z])(?![A-Za-z0-9_])|π"
_TOKEN = rf"(?:{_ATOM}|<=|>=|!=|[+\-*×/÷^=<>≤≥≠²³()])"
_CANDIDATE = re.compile(rf"{_TOKEN}(?:\s*{_TOKEN})*")
_TOKENIZER = re.compile(_TOKEN)
_PRESERVE = re.compile(
    r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|"
    r"\b\d[\d-]*-[A-Za-z]{2}[A-Za-z0-9.-]*"
)
_OPERATORS = {
    "+": ("加", "plus"), "-": ("减", "minus"),
    "*": ("乘以", "times"), "×": ("乘以", "times"),
    "/": ("除以", "divided by"), "÷": ("除以", "divided by"),
    "=": ("等于", "equals"), "<": ("小于", "less than"),
    ">": ("大于", "greater than"), "<=": ("小于等于", "less than or equal to"),
    ">=": ("大于等于", "greater than or equal to"),
    "≤": ("小于等于", "less than or equal to"),
    "≥": ("大于等于", "greater than or equal to"),
    "≠": ("不等于", "not equal to"), "!=": ("不等于", "not equal to"),
    "(": ("左括号", "open parenthesis"), ")": ("右括号", "close parenthesis"),
    "π": ("派", "pi"), "²": ("的平方", "squared"), "³": ("的立方", "cubed"),
}


def _valid_math(tokens):
    """Validate token order, returning implicit multiplication positions."""
    expect_operand, depth, explicit = True, 0, False
    implicit = set()
    for index, token in enumerate(tokens):
        value = token.group()
        atom = bool(re.fullmatch(rf"{_NUMBER}|[A-Za-zπ]", value))
        if expect_operand:
            if value == "(":
                depth += 1
            elif value in {"+", "-"}:
                pass  # Unary signs, spoken distinctly from binary subtraction.
            elif atom:
                expect_operand = False
            else:
                return None
        elif value == ")":
            depth -= 1
            if depth < 0:
                return None
        elif value in {"²", "³"}:
            explicit = True
        elif value in _OPERATORS and value not in {"(", "π"} or value == "^":
            explicit, expect_operand = True, True
        elif value == "(" or (
            atom and index and tokens[index - 1].end() == token.start()
            and (tokens[index - 1].group() == "π" or value == "π")
        ):
            implicit.add(index)
            explicit = True
            if value == "(":
                depth += 1
                expect_operand = True
        else:
            return None
    return implicit if explicit and not expect_operand and depth == 0 else None


def _speak_math(tokens, implicit, language, verbalize_number):
    zh = language.lower().startswith("zh")
    words, index, expect_operand = [], 0, True
    while index < len(tokens):
        value = tokens[index].group()
        if index in implicit:
            words.append("乘以" if zh else "times")
        if value == "^":
            exponent = tokens[index + 1].group()
            if exponent in {"2", "3"}:
                words.append(_OPERATORS["²" if exponent == "2" else "³"][not zh])
                index += 2
                continue
            end = index + 2
            if exponent == "(":
                depth = 1
                while depth:
                    depth += (tokens[end].group() == "(") - (tokens[end].group() == ")")
                    end += 1
            elif exponent in {"+", "-"}:
                end += 1
            part = tokens[index + 1:end]
            spoken = _speak_math(part, set(), language, verbalize_number)
            words.append(f"的 {spoken} 次方" if zh else f"to the power of {spoken}")
            index = end
            continue
        if re.fullmatch(_NUMBER, value):
            words.append(verbalize_number(value, language))
        elif value in {"+", "-"} and expect_operand:
            words.append(("正" if value == "+" else "负") if zh else
                         ("positive" if value == "+" else "negative"))
        else:
            words.append(_OPERATORS[value][not zh] if value in _OPERATORS else value)
        expect_operand = value in _OPERATORS and value not in {"π", "²", "³", ")"}
        index += 1
    return " ".join(words)


def find_spans(text, language, verbalize_number):
    """Return nonoverlapping ``(start, end, spoken)`` spans; URLs are excluded upstream.

    ``verbalize_number(value, language)`` receives the original numeric spelling,
    including decimal zeroes. Language can be ``ZH``/``EN`` or ``zh``/``en``.
    """
    zh = language.lower().startswith("zh")
    spans = []
    preserved = [(match.start(), match.end()) for match in _PRESERVE.finditer(text)]
    for match in _MEASURE.finditer(text):
        if any(match.start() < end and match.end() > start for start, end in preserved):
            continue
        value = match["number"]
        unit = re.sub(r"\s+", "", match["unit"])
        if unit in {"℃", "℉"} or "celsius" in unit.lower() or "fahrenheit" in unit.lower():
            unit = "°C" if unit == "℃" or "celsius" in unit.lower() else "°F"
        number = verbalize_number(value, language)
        label = _UNITS[unit][0 if zh else (1 if abs(Decimal(value)) == 1 else 2)]
        spoken = f"百分之 {number}" if zh and unit == "%" else f"{number} {label}"
        spans.append((match.start(), match.end(), spoken))
    for candidate in _CANDIDATE.finditer(text):
        if any(candidate.start() < end and candidate.end() > start for start, end in preserved):
            continue
        if re.fullmatch(r"[A-Za-z]\s*/\s*[A-Za-z]", candidate.group().strip()):
            continue  # I/O and similar notation are not unambiguous division.
        tokens = list(_TOKENIZER.finditer(text, candidate.start(), candidate.end()))
        # Leave unusually long expressions to the existing frontend. Bounding
        # the parser prevents quadratic/cubic prefix searches on long input.
        if len(tokens) > 64:
            continue
        start = 0
        while start < len(tokens):
            for end in range(len(tokens), start + 1, -1):
                part = tokens[start:end]
                left, right = part[0].start(), part[-1].end()
                if any(left < b and right > a for a, b, _ in spans):
                    continue
                implicit = _valid_math(part)
                if implicit is not None:
                    spans.append((left, right, _speak_math(part, implicit, language, verbalize_number)))
                    start = end
                    break
            else:
                start += 1
    return sorted(spans)
