"""Read explicit addresses, formulas and quantities before lexical TN.

The delivered lexicon is still used for every ordinary text span. Special spans
are spoken separately so a product-name rule cannot change an address, decimal
or exponent. This is deterministic text processing, not model inference.
"""
import re

from .math_units import find_spans as find_math_unit_spans

_EMAIL = r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.[A-Za-z]{2,63}"
_URL = r"(?:https?://|www\.)[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+"
_ADDRESSES = re.compile(r"(?<![A-Za-z0-9_])(?:" + _URL + "|" + _EMAIL + ")", re.I)
_CJK = re.compile(r"[\u4e00-\u9fff]")
# An explicit count classifier disambiguates a cardinal from the released
# FST's telephone-like reading (111台 -> 幺幺幺台). Keep the set small and do
# not pick up numeric suffixes in identifiers, decimals or formatted numbers.
_ZH_COUNT = re.compile(
    r"(?<![A-Za-z0-9_.,:/#%+\-])(?P<number>[0-9]+)[ \t]*"
    r"(?P<classifier>台|个|件|人|份|套)"
)
_DIGITS = {"ZH": tuple("零一二三四五六七八九"),
           "EN": "zero one two three four five six seven eight nine".split()}
_SYMBOLS = {
    "ZH": {".": "点", "@": "艾特", "/": "斜杠", ":": "冒号", "_": "下划线",
           "-": "连字符", "+": "加号", "?": "问号", "&": "与号", "=": "等号",
           "#": "井号", "%": "百分号", "~": "波浪号", "!": "感叹号",
           "$": "美元符号", "'": "单引号", "(": "左括号", ")": "右括号",
           "[": "左方括号", "]": "右方括号", ",": "逗号", ";": "分号",
           "*": "星号", "^": "脱字符", "`": "反引号", "{": "左花括号",
           "}": "右花括号", "|": "竖线"},
    "EN": {".": "dot", "@": "at", "/": "slash", ":": "colon", "_": "underscore",
           "-": "hyphen", "+": "plus", "?": "question mark", "&": "ampersand",
           "=": "equals", "#": "hash", "%": "percent", "~": "tilde", "!": "exclamation mark",
           "$": "dollar sign", "'": "apostrophe", "(": "left parenthesis", ")": "right parenthesis",
           "[": "left bracket", "]": "right bracket", ",": "comma", ";": "semicolon",
           "*": "asterisk", "^": "caret", "`": "backtick", "{": "left brace",
           "}": "right brace", "|": "vertical bar"},
}


def verbalize_chinese_number(value):
    """Cardinal reading for a formula/measurement, with exact decimal digits.

    The released FST can classify an isolated 512 as a telephone fragment
    (五幺二). Inside a quantity its reading must be 五百一十二.
    """
    sign = ""
    if value.startswith(("+", "-")):
        sign, value = ("正" if value[0] == "+" else "负"), value[1:]
    whole, dot, fraction = value.partition(".")
    digits = _DIGITS["ZH"]
    n = int(whole or "0")
    groups = []
    while n:
        n, part = divmod(n, 10000)
        groups.append(part)
    scales = ("", "万", "亿", "万亿", "亿亿")
    if len(groups) > len(scales):
        # Spell exceptionally large values, never truncate them.
        spoken = "".join(digits[int(c)] for c in whole)
    else:
        spoken, previous = "", None
        for index in range(len(groups) - 1, -1, -1):
            part = groups[index]
            if not part:
                continue
            if spoken and (previous - index > 1 or part < 1000):
                spoken += "零"
            small, pending_zero = "", False
            for char, unit in zip(str(part).zfill(4), ("千", "百", "十", "")):
                if char == "0":
                    pending_zero = bool(small)
                else:
                    if pending_zero:
                        small += "零"
                    small += digits[int(char)] + unit
                    pending_zero = False
            if not spoken and 10 <= part < 20:
                small = small[1:]
            spoken += small + scales[index]
            previous = index
        spoken = spoken or "零"
    if dot:
        spoken += "点" + "".join(digits[int(c)] for c in fraction)
    return sign + spoken


def _read_address(address, language):
    # Spell identifiers literally. This avoids turning digits into quantities
    # or reinterpreting a local part such as qwen3/API as a brand/acronym rule.
    return " ".join(
        char.upper() if char.isascii() and char.isalpha()
        else _DIGITS[language][int(char)] if char.isascii() and char.isdigit()
        else _SYMBOLS[language][char]
        for char in address
    )


def _find_non_address_spans(text, language, verbalize_number):
    """Keep math/measurement precedence, then protect explicit Chinese counts."""
    spans = find_math_unit_spans(text, language, verbalize_number)
    if language == "ZH":
        for match in _ZH_COUNT.finditer(text):
            if any(match.start() < end and match.end() > start
                   for start, end, _ in spans):
                continue
            spoken = verbalize_chinese_number(match["number"]) + match["classifier"]
            spans.append((match.start(), match.end(), spoken))
    return sorted(spans)


def normalize_special_text(text, normalize_plain, verbalize_number):
    """Normalize only recognized special spans; delegate all other text.

    Language follows the surrounding sentence: Chinese if it contains CJK,
    English otherwise. URL/email punctuation is pronounced literally.
    `verbalize_number(value, language)` must preserve decimal digits.
    """
    language = "ZH" if _CJK.search(text) else "EN"
    spans = []
    cursor = 0
    for match in _ADDRESSES.finditer(text):
        address = match.group().rstrip(".,;")
        while address.endswith(")") and address.count(")") > address.count("("):
            address = address[:-1]
        while address.endswith("]") and address.count("]") > address.count("["):
            address = address[:-1]
        if not address:
            continue
        for start, end, spoken in _find_non_address_spans(text[cursor:match.start()], language, verbalize_number):
            spans.append((cursor + start, cursor + end, spoken))
        end = match.start() + len(address)
        spans.append((match.start(), end, _read_address(address, language)))
        cursor = end
    for start, end, spoken in _find_non_address_spans(text[cursor:], language, verbalize_number):
        spans.append((cursor + start, cursor + end, spoken))
    if not spans:
        return normalize_plain(text)
    output = []
    cursor = 0
    for start, end, spoken in spans:
        if cursor < start:
            output.append(normalize_plain(text[cursor:start]))
        output.append(" " + spoken + " ")
        cursor = end
    if cursor < len(text):
        output.append(normalize_plain(text[cursor:]))
    return re.sub(r"\s+", " ", "".join(output)).strip()
