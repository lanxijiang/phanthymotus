"""Deterministic lexeme rules for the ZH/EN mixed VITS2 frontend.

Adds: `Qwen` -> 千问 and Chinese readings for accelerator model digits.

Applied to the raw text **before** the WeText FST normalizer, because the graph
otherwise rewrites English tokens it does not own:

* ``IDC`` -> ``I don't care`` (semantic corruption)
* ``Tier 1`` -> ``Tier 一``  (digits in an English context read as Chinese)

The rules also pin the reading of acronyms and camel-case product names that the
graph leaves untouched and that ``g2p_en`` then guesses wrongly, e.g.
``DCU`` -> ``d uw k`` ("duke"), ``AutoML`` -> ``ao t ah m`` (ML lost).

Separators:

* ``CNCF`` is the one acronym that uses dots between letters (``C.N.C.F``).
  With a plain space its four letters render as a single continuous 770 ms block
  with no measurable gap, which is heard as one blur; the dot yields 40-70 ms
  micro-gaps between letters. Every other acronym keeps the space-separated
  spelling it already had.
* A phrase that needs a longer audible gap gets a comma, because ``unified_g2p``
  maps whitespace to a zero-phone boundary and a comma is a real pause symbol.

Boundaries use ASCII lookarounds instead of ``\\b``: CJK characters are ``\\w``
in Python's Unicode regex, so ``\\b`` never matches between 中文 and an ASCII
word (``\\bdcu\\b`` fails on ``这是DCU的测试``).

Every entry is deliberately narrow — one release-controlled rewrite per lexeme —
so training and deployed runtime keep producing identical phonemes.
"""

import re


def _edge(pattern: str) -> re.Pattern:
    """Match ``pattern`` as a standalone ASCII token, CJK included."""
    return re.compile(rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])", re.IGNORECASE)


# 0) Phrase-level rules. These must run before the per-token rules below,
#    otherwise "Sage AIOS" is already split into letters and never matches.
_PHRASE_RULES = [
    # Keep a short pause between the brand and the acronym it introduces.
    (_edge(r"sage\s+aios"), "Sage, A I O S"),
]

# 1) Digits that belong to an English phrase must not be verbalized as Chinese.
_NUMBER_RULES = [
    (_edge(r"tier\s*1"), "Tier one"),
]

# 2) Tokens the English TN graph mangles before G2P ever sees them.
_PROTECTED_TOKENS = [
    (_edge(r"idc"), "I D C"),
]

# 3) Acronyms that are read letter by letter. CNCF uses dots: space-separated
#    letters come out as one continuous block (770 ms, no gap).
_LETTER_SPELLED = [
    (_edge(r"aios"), "A I O S"),
    (_edge(r"dcu"), "D C U"),
    (_edge(r"mlu"), "M L U"),
    (_edge(r"cncf"), "C.N.C.F"),
    (_edge(r"xc"), "X C"),
    (_edge(r"vgpu"), "V G P U"),
    (_edge(r"gpu"), "G P U"),
    (_edge(r"npu"), "N P U"),
    (_edge(r"api"), "A P I"),
    (_edge(r"ai"), "A I"),
]

# 4) Mixed readings: the word part is spelled, the trailing letter is read.
_MIXED_READINGS = [
    (_edge(r"automl"), "Auto M L"),
    (_edge(r"enginex"), "Engine X"),
]

# 5) Camel-case / fused product names that must become two words.
_FUSED_NAMES = [
    (_edge(r"modelhub"), "Model Hub"),
    (_edge(r"phanthymotus"), "Phanthy Motus"),
    (_edge(r"phanthymovie"), "Phanthy Movie"),
    (_edge(r"phanthycloud"), "Phanthy Cloud"),
    (_edge(r"phancycloud"), "Phancy Cloud"),
    # Split so that "Router" resolves through CMUdict: /ˈruːtər/, not the
    # g2p_en guess /ˈraʊtər/ the single fused token produced.
    (_edge(r"phanrouter"), "Phan Router"),
    (_edge(r"phanclaw"), "Phan Claw"),
]


# 6) Chinese readings: brand names spoken as Chinese, and the digits of
#    accelerator model numbers read as Chinese numerals (A100 -> A一百,
#    MI250X -> M I二五零X). Round hundreds keep the natural form (一百/八百),
#    other three-digit numbers are read digit by digit (九一零/三七零).
_CN_NAMES = [
    (_edge(r"qwen"), "千问"),
]

_MODEL_DIGITS = [
    (_edge(r"MI250X"), "M I二五零X"),
    (_edge(r"MI300A"), "M I三百A"),
    (_edge(r"MI300X"), "M I三百X"),
    (_edge(r"MI325X"), "M I三二五X"),
    (_edge(r"MI355X"), "M I三五五X"),
    (_edge(r"MLU100"), "M L U一百"),
    (_edge(r"MLU270"), "M L U二七零"),
    (_edge(r"MLU290"), "M L U二九零"),
    (_edge(r"MLU370"), "M L U三七零"),
    (_edge(r"MLU590"), "M L U五九零"),
    (_edge(r"GB200"), "G B二百"),
    (_edge(r"GB300"), "G B三百"),
    (_edge(r"MI210"), "M I二一零"),
    (_edge(r"MI250"), "M I二五零"),
    (_edge(r"MI350"), "M I三五零"),
    (_edge(r"950PR"), "九五零PR"),
    (_edge(r"950DT"), "九五零DT"),
    (_edge(r"WSE-2"), "W S E二"),
    (_edge(r"WSE-3"), "W S E三"),
    (_edge(r"SN40L"), "S N四零L"),
    (_edge(r"V100"), "V一百"),
    (_edge(r"A100"), "A一百"),
    (_edge(r"A800"), "A八百"),
    (_edge(r"H100"), "H一百"),
    (_edge(r"H800"), "H八百"),
    (_edge(r"H200"), "H二百"),
    (_edge(r"B100"), "B一百"),
    (_edge(r"B200"), "B二百"),
    (_edge(r"B300"), "B三百"),
    (_edge(r"L40S"), "L四零S"),
    (_edge(r"310P"), "三一零P"),
    (_edge(r"910A"), "九一零A"),
    (_edge(r"910B"), "九一零B"),
    (_edge(r"910C"), "九一零C"),
    (_edge(r"AI2"), "A I二"),
    (_edge(r"A3C"), "A三C"),
    (_edge(r"A2C"), "A二C"),
    (_edge(r"TD3"), "T D三"),
    (_edge(r"H20"), "H二零"),
    (_edge(r"L40"), "L四零"),
    (_edge(r"v5e"), "v五e"),
    (_edge(r"v5p"), "v五p"),
    (_edge(r"v6e"), "v六e"),
    (_edge(r"310"), "三一零"),
    (_edge(r"910"), "九一零"),
    (_edge(r"C3"), "C三"),
    (_edge(r"01"), "零一"),
    (_edge(r"T5"), "T五"),
    (_edge(r"F1"), "F一"),
    (_edge(r"L4"), "L四"),
    (_edge(r"T4"), "T四"),
    (_edge(r"v4"), "v四"),
    (_edge(r"v7"), "v七"),
    (_edge(r"D1"), "D一"),
    (_edge(r"2"), "二"),
    (_edge(r"3"), "三"),
]

# Order matters: phrase rules first, then longer alternatives before shorter ones
# (``aios`` before ``ai``, ``vgpu`` before ``gpu``).
RULES = (
    _PHRASE_RULES
    + _CN_NAMES
    + _MODEL_DIGITS
    + _NUMBER_RULES
    + _PROTECTED_TOKENS
    + _LETTER_SPELLED
    + _MIXED_READINGS
    + _FUSED_NAMES
)


def apply_lexeme_rules(text: str) -> str:
    """Rewrite known English lexemes into their intended spoken form."""
    if not text:
        return text
    for pattern, replacement in RULES:
        text = pattern.sub(replacement, text)
    return text
