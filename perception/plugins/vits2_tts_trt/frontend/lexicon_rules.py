"""Deterministic lexeme rules for the ZH/EN mixed VITS2 frontend.

Applied to the raw text **before** the WeText FST normalizer, because the graph
otherwise rewrites English tokens it does not own:

* ``IDC`` -> ``I don't care`` (semantic corruption)
* ``Tier 1`` -> ``Tier 一``  (digits in an English context read as Chinese)

The rules also pin the reading of acronyms and camel-case product names that the
graph leaves untouched and that ``g2p_en`` then guesses wrongly, e.g.
``DCU`` -> ``d uw k`` ("duke"), ``AutoML`` -> ``ao t ah m`` (ML lost).

Boundaries use ASCII lookarounds instead of ``\\b``: CJK characters are ``\\w``
in Python's Unicode regex, so ``\\b`` never matches between 中文 and an ASCII
word (``\\bdcu\\b`` fails on ``这是DCU的测试``). A literal space is *not* enough
to separate two English tokens — ``unified_g2p`` maps whitespace to a zero-phone
boundary — so a phrase that needs an audible gap gets a comma, which becomes a
real pause symbol in the phone stream.

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

# 3) Acronyms that are read letter by letter.
_LETTER_SPELLED = [
    (_edge(r"aios"), "A I O S"),
    (_edge(r"dcu"), "D C U"),
    (_edge(r"mlu"), "M L U"),
    (_edge(r"cncf"), "C N C F"),
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

# Order matters: phrase rules first, then longer alternatives before shorter ones
# (``aios`` before ``ai``, ``vgpu`` before ``gpu``).
RULES = (
    _PHRASE_RULES
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
