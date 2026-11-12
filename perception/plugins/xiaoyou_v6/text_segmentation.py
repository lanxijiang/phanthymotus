"""CPU-only long-text segmentation candidate derived from pinned V6 runtime.

No normalization, pronunciation, waveform, speed or model changes. Call safe_split
only after the existing verified profile rejects a clause. A Chinese enumeration
comma (、) is a fallback boundary and is never an active natural-clause delimiter.
Concatenation preserves source strings including punctuation and whitespace.
"""
import re

_ABBREVIATIONS = {'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st', 'vs'}
# Sentence-boundary hints only. These never select pronunciations or replace text.
_JOINED_SENTENCE_STARTERS = frozenset(('I You We They He She It The A An This That These Those '
    'Today Tomorrow Yesterday What Why When Where Who How Please Let Thank Then Next First Second '
    'Finally However Meanwhile Instead For In On At After Before If As Our Your Their Here There Now').split())
# Preserve common domain/file uses even when a suffix happens to be a sentence hint.
_DOMAIN_SUFFIXES = frozenset('com org net io ai dev app edu gov co uk cn hk xyz info biz me tech today cloud online site'.split())


def _joined_period_boundary(text, dot):
    """Conservative no-space sentence ending, e.g. today.You; not any dotted ID."""
    if dot < 1 or dot + 1 >= len(text) or text[dot] != '.':
        return False
    left = re.search(r'([A-Za-z]+)$', text[:dot])
    right = re.match(r'([A-Z][a-z]+|I)(?![A-Za-z0-9_])', text[dot+1:])
    if not left or not right or not left.group(1).islower():
        return False
    if left.group(1) in _ABBREVIATIONS or len(left.group(1)) == 1:
        return False
    if right.group(1) not in _JOINED_SENTENCE_STARTERS:
        return False
    if right.group(1).lower() in _DOMAIN_SUFFIXES:
        # Today is also a domain suffix. Admit its explicit prose use only when
        # the following separate word is a subject, e.g. everyone.Today we begin.
        following = text[dot+1+len(right.group(1)):]
        if right.group(1) != 'Today' or not re.match(r'\s+(?:I|we|you|he|she|they|our|your)\b', following):
            return False
    # Do not reinterpret URL/path/email/version/compound ASCII tokens as prose.
    token_start = dot
    while token_start and re.match(r"[A-Za-z0-9_./:'’+@\\-]", text[token_start-1]):
        token_start -= 1
    token_end = dot + 1
    while token_end < len(text) and re.match(r"[A-Za-z0-9_./:'’+@\\-]", text[token_end]):
        token_end += 1
    token = text[token_start:token_end]
    if any(c in token for c in '/:_+@\\-'):
        return False
    # A final full stop may follow the joined next word, but repeated interior dots
    # indicate an abbreviation/domain/version and stay protected.
    if token.rstrip('.').count('.') != 1:
        return False
    return True


def protected_spans(text):
    """V6 segmentation guards only: no pronunciation replacement/legacy rules."""
    spans = []
    # These spans protect intact source dates/times only. They do not rewrite
    # text, select a reading, or import any pronunciation lexicon.
    digits = r'[0-9零〇○一二两三四五六七八九十百千]+'
    daypart = r'(?:凌晨|清晨|早上|上午|中午|下午|傍晚|晚上|夜间)'
    clock = (rf'(?:{daypart}[ \t]*)?(?:'
             rf'{digits}(?:点|时)(?:(?:{digits}(?:分)?|半|一刻|三刻))?(?:{digits}秒)?'
             r'|\d{1,2}:\d{2}(?::\d{2})?)')
    calendar = (rf'(?:{digits}年{digits}月(?:{digits}(?:日|号))?'
                rf'|{digits}月{digits}(?:日|号))')
    patterns = [
        rf'{calendar}(?:[ \t]*{clock})?',
        clock,
        r"[A-Za-z][A-Za-z0-9_]*(?:[-./:'’+][A-Za-z0-9_]+)*",
        r'(?<![A-Za-z0-9])(?:[+−-]?\d+(?:[,.]\d+)*)(?:年|月|日|点|分|秒|%|GB|MB|GHz|MHz)?',
        r'(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?!\d)',
        r'(?<!\d)\d{4}[-/]\d{1,2}[-/]\d{1,2}(?!\d)',
        r'\b[A-Za-z][A-Za-z0-9_./-]*(?:[ \t]+(?:Runtime|Toolkit|XC|Driver|Operator))?[ \t]+(?:version[ \t]+|v)?\d+(?:\.\d+){0,3}\b',
        r'\b\d+(?:\.\d+)*[ \t]+(?:GB|MB|GHz|MHz|GPU|CPU|tokens|ms|milliseconds|seconds)\b',
    ]
    phrases = ['CUDA Runtime', 'CUDA Toolkit', 'ModelHub XC', 'RISE vGPU', 'GPU Operator',
               '第四范式', '小优', '下一步', 'SageAIOS', 'PhancyGroup', 'PhanthyMotus', '4Paradigm', 'xiaoyou01']
    for pattern in patterns:
        spans.extend((m.start(), m.end()) for m in re.finditer(pattern, text, flags=re.IGNORECASE))
    for phrase in phrases:
        spans.extend((m.start(), m.end()) for m in re.finditer(re.escape(phrase), text, flags=re.IGNORECASE))
    # The old ASCII guard treats today.You as one identifier. Release only the
    # explicitly recognized prose boundaries; all other original guards remain.
    released = [m.end() for m in re.finditer(r'\.', text) if _joined_period_boundary(text, m.start())]
    refined = []
    for start, stop in spans:
        cuts = [start, *[end for end in released if start < end < stop], stop]
        refined.extend(zip(cuts, cuts[1:]))
    return refined


def safe_boundary(text, end):
    return not any(start < end < stop for start, stop in protected_spans(text))


class ProfileLimitError(ValueError):
    """Text is intact, but the verified model profile cannot contain this clause."""


def natural_clauses(text):
    """Natural V6 text slices, protecting numbers/versions and basic initials."""
    boundaries = {m.end() for m in re.finditer(r'[。！？!?；;\n]+|(?<!\d)[，,：:]|[，,：:](?!\d)', text)}
    abbreviations = _ABBREVIATIONS
    for found in re.finditer(r'\.(?=\s+[A-Z]|[A-Z][a-z]+|I(?:\b)|$)', text):
        if found.end() < len(text) and not text[found.end()].isspace() and not _joined_period_boundary(text, found.start()):
            continue
        before = text[:found.start()]
        word = re.search(r'([A-Za-z]+)$', before)
        # These are V6 sentence boundaries, not a pronunciation dictionary.
        if word and (word.group(1).lower() in abbreviations or len(word.group(1)) == 1):
            continue
        if re.search(r'(?:\b[A-Za-z]\.)+[A-Za-z]$', before):
            continue
        boundaries.add(found.end())
    pieces, start = [], 0
    for end in sorted(boundaries):
        if not safe_boundary(text, end):
            continue
        piece = text[start:end]
        if any(c.isalnum() for c in piece):
            pieces.append(piece)
            start = end
    tail = text[start:]
    if any(c.isalnum() for c in tail):
        pieces.append(tail)
    elif pieces:
        pieces[-1] += tail
    elif tail:
        pieces.append(tail)
    if ''.join(pieces) != text:
        raise RuntimeError('Candidate segmentation lost or duplicated source text')
    yield from pieces


def safe_split(text):
    """Split only at source punctuation/word gaps; no phoneme or audio truncation."""
    choices = [m.end() for m in re.finditer(r'[。！？!?；;，,、：:]|(?<!\d)\.(?!\d)|\s+', text)
               if 0 < m.end() < len(text) and text[:m.end()].strip() and text[m.end():].strip()]
    # A comma/colon surrounded by digits is not a source word boundary.
    choices = [x for x in choices if not (x >= 2 and x < len(text)
                and text[x-1] in ',，:：' and text[x-2].isdigit() and text[x].isdigit())]
    choices = [x for x in choices if safe_boundary(text, x)]
    if not choices and re.search(r'[\u3400-\u9fff]', text):
        # V6-native text word boundaries, not a pronunciation override/legacy lexicon.
        # Never cut inside an ASCII identifier, date, version or numeric run.
        try:
            import jieba
            choices = [end for _, _, end in jieba.tokenize(text)
                       if 0 < end < len(text) and text[:end].strip() and text[end:].strip()
                       and not (text[end-1].isascii() and text[end].isascii()
                                and re.match(r"[A-Za-z0-9_./:'’+-]", text[end-1])
                                and re.match(r"[A-Za-z0-9_./:'’+-]", text[end]))]
        except ImportError:
            choices = []
        choices = [x for x in choices if safe_boundary(text, x)]
    if not choices:
        raise ProfileLimitError('V6 clause exceeds its verified profile without a safe text boundary; nothing was truncated')
    cut = min(choices, key=lambda x: abs(x - len(text) / 2))
    return text[:cut], text[cut:]

