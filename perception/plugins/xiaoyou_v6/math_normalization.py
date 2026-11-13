"""Candidate-only, standard-library mathematical speech normalization.

Call general formatting on RAW text first, then normalize_math, then the existing
frontend. This module never calls NFKC: superscripts/subscripts must still exist.
It parses expressions and reads their structure; it never evaluates them, drops
operators or looks at evaluation references. Unsupported recognized expressions
raise MathNormalizationError. No model, voice, source frontend or legacy rules.
"""
import re
from collections import namedtuple

__all__ = ['normalize_math', 'MathNormalizationError']


class MathNormalizationError(ValueError):
    """A recognized mathematical span cannot be rendered safely."""


Token = namedtuple('Token', 'kind value start end')
SUPERSCRIPTS = dict(zip('⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ', '0123456789+-n'))
SUBSCRIPTS = dict(zip('₀₁₂₃₄₅₆₇₈₉₊₋', '0123456789+-'))
TOKEN_RE = re.compile(
    r'\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?:\d+(?:\.\d+)?|\.\d+)'
    r'|[A-Za-z][A-Za-z0-9_]*|[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ]+|[₀₁₂₃₄₅₆₇₈₉₊₋]+'
    r'|>=|<=|!=|==|[π√+\-−*×÷/^=≥≤≠<>|()@∑∫≈∈%％\[\]{}]')

# These denote source text, identifiers or typed units, not math formulas.
# a/b remains mathematical; long/multi-component paths and m/s remain literal.
PROTECTED_PATTERNS = [
    re.compile(r'\ue000[A-Z]+\ue001'),
    re.compile(r'```[\s\S]*?```|`[^`\n]*`'),
    re.compile(r'(?:https?|ftp)://[^\s，。；;!]+', re.I),
    re.compile(r'\b[A-Za-z]:[\\/][^\s，。；;!?]+'),
    re.compile(r'(?<![\w)])(?:\.{0,2}/)(?:[A-Za-z0-9_.-]+/)*[A-Za-z_.][A-Za-z0-9_.-]*'),
    re.compile(r'\bC\+\+(?:\d+)?(?![A-Za-z0-9_])|\bC#(?![A-Za-z0-9_])', re.I),
    re.compile(r'\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b'),
    re.compile(r'\b(?:[A-Za-z][A-Za-z0-9_]+-[A-Za-z0-9_-]+|[A-Za-z]-[A-Za-z][A-Za-z0-9_-]+)\b'),
    re.compile(r'\b[A-Za-z_][A-Za-z0-9_]{1,}\s*(?:==|=)[^，。；;\n]*'),
    re.compile(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'),
    re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\b'),
    re.compile(r'\b(?:[vV]\d+(?:\.\d+)+|\d+(?:\.\d+){2,})\b'),
    re.compile(r'\b\d+(?:\.\d+)?[eE][+-]?\d+\b'),
    re.compile(r'\b(?:kg/m/s|kg/m(?:[²³]|\^[23])?|m/s(?:[²³]|\^[23])?|km/h|cm/s|mg/mL)\b'),
    re.compile(r'\{[^{}\n]*["\'][^{}\n]*\}'),
]


def _protected_spans(text):
    spans = [(m.start(), m.end()) for pattern in PROTECTED_PATTERNS for m in pattern.finditer(text)]
    # Bare paths require a multi-letter component or more than one slash.
    for m in re.finditer(r'(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_.-]+)+(?![A-Za-z0-9_])', text):
        parts = m.group().split('/')
        if len(parts) > 2 or any(len(p) > 1 for p in parts):
            spans.append((m.start(), m.end()))
    return sorted(spans)


def _tokens_and_runs(text):
    protected = _protected_spans(text)
    runs, current = [], []
    previous_end = 0
    def finish():
        if current:
            runs.append(tuple(current))
            current.clear()
    for m in TOKEN_RE.finditer(text):
        if text[previous_end:m.start()].strip():
            finish()
        previous_end = m.end()
        if any(a < m.end() and b > m.start() for a, b in protected):
            finish()
            continue
        v = m.group()
        if v[0] in SUPERSCRIPTS:
            kind = 'SUP'
        elif v[0] in SUBSCRIPTS:
            kind = 'SUB'
        elif v[0].isdigit() or v.startswith('.'):
            kind = 'NUMBER'
        elif re.fullmatch('[A-Za-z][A-Za-z0-9_]*', v):
            if len(v) == 1:
                kind = 'VAR'
            elif v in ('sin', 'sqrt', 'cos', 'tan', 'log', 'exp'):
                kind = 'FUNCTION'
            elif re.match(r'\s*[*×]\s*[+\-−]?(?:\d+(?:\.\d+)?|\.\d+)\s*[%％](?![A-Za-z0-9_])', text[m.end():]):
                # A named operand is admitted only for unambiguous multiplication
                # by a numerical percentage, never a general code assignment.
                kind = 'VAR'
            elif re.match(r'\s*\(', text[m.end():]):
                kind = 'FUNCTION'  # Explicit unknown math functions fail closed.
            else:
                finish()
                continue
        elif v == 'π':
            kind = 'PI'
        elif v == '√':
            kind = 'ROOT'
        elif v == '(':
            kind = 'LPAREN'
        elif v == ')':
            kind = 'RPAREN'
        elif v == '|':
            kind = 'ABS'
        elif v in ('%', '％'):
            kind = 'PERCENT'
        elif v in ('@', '∑', '∫', '≈', '∈', '[', ']', '{', '}'):
            kind = 'UNSUPPORTED'
        else:
            kind = 'OP'
        token = Token(kind, v, m.start(), m.end())
        # Do not absorb an English article/pronoun into the next expression.
        # Adjacent 2x and spaced 2 x remain supported implicit multiplication.
        if current and text[current[-1].end:token.start].isspace():
            prev = current[-1]
            if prev.kind in ('VAR', 'NUMBER', 'SUP', 'SUB', 'RPAREN') and token.kind in ('VAR', 'NUMBER', 'FUNCTION', 'PI', 'ROOT'):
                if not (prev.kind == 'NUMBER' and token.kind == 'VAR'):
                    finish()
        current.append(token)
    finish()
    return runs


def _is_math(tokens):
    if not any(t.kind in ('NUMBER', 'VAR', 'PI') for t in tokens):
        return False
    if any(t.kind in ('SUP', 'SUB', 'ROOT', 'PI') for t in tokens):
        return True
    if any(t.kind == 'FUNCTION' and i+1 < len(tokens) and tokens[i+1].kind == 'LPAREN'
           for i, t in enumerate(tokens)):
        return True
    if any(t.kind == 'OP' for t in tokens):
        return True
    # Percentages belong to general formatting. A binary modulo is unsupported.
    return any(t.kind in ('UNSUPPORTED', 'PERCENT') and 0 < i < len(tokens)-1 for i, t in enumerate(tokens))


class _Parser:
    def __init__(self, tokens, source):
        if len(tokens) > 256:
            raise MathNormalizationError('Mathematical span exceeds 256 lexical tokens')
        self.ts, self.i, self.depth, self.source = tokens, 0, 0, source

    def peek(self, *values):
        return self.i < len(self.ts) and self.ts[self.i].value in values

    def take(self):
        if self.i >= len(self.ts):
            raise MathNormalizationError('Incomplete mathematical expression')
        t = self.ts[self.i]
        self.i += 1
        return t

    def expect(self, value):
        if not self.peek(value):
            raise MathNormalizationError('Missing matching '+value+' in mathematical expression')
        return self.take()

    def parse(self):
        if any(t.kind == 'UNSUPPORTED' for t in self.ts):
            raise MathNormalizationError('Unsupported mathematical operator')
        node = self.comparison()
        if self.i != len(self.ts):
            raise MathNormalizationError('Unparsed mathematical token: '+self.ts[self.i].value)
        return node

    def comparison(self):
        node = self.sum()
        while self.peek('=', '==', '!=', '≠', '<', '>', '<=', '>=', '≤', '≥'):
            op = self.take().value
            node = ('binary', op, node, self.sum())
        return node

    def sum(self):
        node = self.product()
        while self.peek('+', '-', '−'):
            node = ('binary', self.take().value, node, self.product())
        return node

    def product(self):
        node = self.unary()
        while self.i < len(self.ts):
            if self.peek('*', '×', '/', '÷'):
                node = ('binary', self.take().value, node, self.unary())
            else:
                t, prev = self.ts[self.i], self.ts[self.i-1]
                implicit = t.kind in ('VAR', 'PI', 'LPAREN', 'FUNCTION', 'ROOT') and (
                    t.start == prev.end or (prev.kind == 'NUMBER' and t.kind == 'VAR'))
                if not implicit:
                    break
                node = ('binary', '*', node, self.unary())
        return node

    def unary(self):
        if self.peek('+', '-', '−'):
            op = self.take().value
            return ('unary', op, self.unary())
        return self.power()

    def power(self):
        node = self.atom()
        if self.peek('%', '％'):
            if node[0] != 'number':
                raise MathNormalizationError('Percent postfix requires a numeric literal')
            self.take()
            node = ('percent', node)
        if self.i < len(self.ts) and self.ts[self.i].kind == 'SUP':
            raw = self.take().value
            exponent = ''.join(SUPERSCRIPTS[c] for c in raw)
            if re.fullmatch(r'[+-]?\d+', exponent):
                value = ('number', exponent.lstrip('+-'))
                if exponent[0] in '+-':
                    value = ('unary', exponent[0], value)
            elif exponent == 'n':
                value = ('var', 'n')
            else:
                raise MathNormalizationError('Unsupported superscript sequence')
            node = ('power', node, value)
        elif self.peek('^'):
            self.take()
            node = ('power', node, self.unary())
        return node

    def atom(self):
        token = self.take()
        if token.kind == 'NUMBER':
            node = ('number', token.value.replace(',', ''))
        elif token.kind == 'VAR':
            node = ('var', token.value)
        elif token.kind == 'PI':
            node = ('pi',)
        elif token.kind == 'ROOT':
            node = ('root', self.power())
        elif token.kind == 'FUNCTION':
            if token.value not in ('sin', 'sqrt'):
                raise MathNormalizationError('Unsupported mathematical function: '+token.value)
            self.expect('(')
            self.depth += 1
            if self.depth > 24:
                raise MathNormalizationError('Mathematical grouping is too deep')
            inner = self.comparison()
            self.expect(')')
            self.depth -= 1
            node = ('function', token.value, ('group', inner))
        elif token.kind in ('LPAREN', 'ABS'):
            self.depth += 1
            if self.depth > 24:
                raise MathNormalizationError('Mathematical grouping is too deep')
            inner = self.comparison()
            self.expect(')' if token.kind == 'LPAREN' else '|')
            self.depth -= 1
            node = ('group' if token.kind == 'LPAREN' else 'absolute', inner)
        else:
            raise MathNormalizationError('Expected mathematical operand, got: '+token.value)
        if self.i < len(self.ts) and self.ts[self.i].kind == 'SUB':
            if node[0] != 'var':
                raise MathNormalizationError('Only variable digit subscripts are supported')
            sub = ''.join(SUBSCRIPTS[c] for c in self.take().value)
            if not re.fullmatch(r'[+-]?\d+', sub):
                raise MathNormalizationError('Unsupported subscript sequence')
            subnode = ('number', sub.lstrip('+-'))
            if sub[0] in '+-':
                subnode = ('unary', sub[0], subnode)
            node = ('subscript', node, subnode)
        return node


def _zh_int(n):
    digits = '零一二三四五六七八九'
    if n == 0:
        return '零'
    for value, unit in ((1000000000000, '万亿'), (100000000, '亿'), (10000, '万')):
        if n >= value:
            q, r = divmod(n, value)
            return _zh_int(q)+unit+(('零' if r < value//10 else '')+_zh_int(r) if r else '')
    result, pending = '', False
    for value, unit in ((1000,'千'),(100,'百'),(10,'十'),(1,'')):
        q, n = divmod(n, value)
        if q:
            if pending:
                result += '零'
            result += digits[q]+unit
            pending = False
        elif result and n:
            pending = True
    return result[1:] if result.startswith('一十') else result


def _en_int(n):
    small = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split()
    tens = ('zero ten twenty thirty forty fifty sixty seventy eighty ninety').split()
    if n < 20:
        return small[n]
    if n < 100:
        return tens[n//10]+(' '+small[n%10] if n%10 else '')
    if n < 1000:
        return small[n//100]+' hundred'+(' '+_en_int(n%100) if n%100 else '')
    for value, unit in ((10**15,'quadrillion'),(10**12,'trillion'),(10**9,'billion'),(10**6,'million'),(1000,'thousand')):
        if n >= value:
            return _en_int(n//value)+' '+unit+(' '+_en_int(n%value) if n%value else '')
    raise MathNormalizationError('Unsupported mathematical integer')


def _number(value, language):
    if not re.fullmatch(r'(?:\d+(?:\.\d+)?|\.\d+)', value) or len(value) > 32:
        raise MathNormalizationError('Unsupported mathematical numeric literal')
    whole, dot, frac = value.partition('.')
    whole = whole or '0'
    if len(whole) > 18:
        raise MathNormalizationError('Mathematical integer is longer than 18 digits')
    digits = list('零一二三四五六七八九') if language == 'zh' else 'zero one two three four five six seven eight nine'.split()
    if len(whole) > 1 and whole.startswith('0'):
        spoken = ' '.join(digits[int(x)] for x in whole)
    else:
        spoken = _zh_int(int(whole)) if language == 'zh' else _en_int(int(whole))
    if dot:
        spoken += (' 点 ' if language == 'zh' else ' point ')+' '.join(digits[int(x)] for x in frac)
    return spoken


ZH_OPS = {'+':'加','-':'减','−':'减','*':'乘以','×':'乘以','/':'除以','÷':'除以',
          '=':'等于','==':'等于','!=':'不等于','≠':'不等于','>':'大于','<':'小于',
          '>=':'大于等于','≥':'大于等于','<=':'小于等于','≤':'小于等于'}
EN_OPS = {'+':'plus','-':'minus','−':'minus','*':'times','×':'times','/':'divided by','÷':'divided by',
          '=':'equals','==':'equals','!=':'does not equal','≠':'does not equal','>':'greater than','<':'less than',
          '>=':'greater than or equal to','≥':'greater than or equal to','<=':'less than or equal to','≤':'less than or equal to'}


def _render(node, language):
    kind = node[0]
    zh = language == 'zh'
    if kind == 'number':
        return _number(node[1], language)
    if kind == 'percent':
        value = _render(node[1], language)
        return '百分之 '+value if zh else value+' percent'
    if kind == 'var':
        return node[1]
    if kind == 'pi':
        return '派' if zh else 'pi'
    if kind == 'group':
        return ('左括号 ' if zh else 'open parenthesis ')+_render(node[1],language)+(' 右括号' if zh else ' close parenthesis')
    if kind == 'absolute':
        target = node[1]
        value = _render(target if target[0] in ('var','number','subscript') else ('group',target),language)
        return value+' 的绝对值' if zh else 'the absolute value of '+value
    if kind == 'subscript':
        return _render(node[1],language)+(' 下标 ' if zh else ' subscript ')+_render(node[2],language)
    if kind == 'unary':
        target = node[2]
        if zh and target[0] == 'percent':
            sign = '正 ' if node[1] == '+' else '负 '
            return '百分之 '+sign+_render(target[1], language)
        value = _render(target, language)
        if target[0] in ('binary','power'):
            value = _render(('group',target),language)
        return (('负 ' if zh else 'negative ') if node[1] != '+' else ('正 ' if zh else 'positive '))+value
    if kind == 'binary':
        ops = ZH_OPS if zh else EN_OPS
        return _render(node[2],language)+' '+ops[node[1]]+' '+_render(node[3],language)
    if kind == 'power':
        base, exponent = _render(node[1],language), node[2]
        if exponent == ('number','2'):
            return base+(' 的平方' if zh else ' squared')
        if exponent == ('number','3'):
            return base+(' 的立方' if zh else ' cubed')
        if exponent[0] in ('binary','power'):
            exponent = ('group',exponent)
        return base+(' 的 ' if zh else ' to the power of ')+_render(exponent,language)+(' 次方' if zh else '')
    if kind == 'root':
        target = node[1]
        if target[0] in ('binary','power'):
            target = ('group',target)
        return ('根号 ' if zh else 'square root of ')+_render(target,language)
    if kind == 'function':
        name = ('正弦' if node[1]=='sin' else '平方根') if zh else ('sine of' if node[1]=='sin' else 'square root of')
        return name+' '+_render(node[2],language)
    raise MathNormalizationError('Unsupported mathematical AST node')


def normalize_math(text, language='auto'):
    """Read supported mathematical spans in raw text; preserve all other bytes.

    language auto/mixed uses Chinese math words when the complete request contains
    Han characters, otherwise English. Explicit zh/en uses that carrier language.
    Not a symbolic calculator or general program/code parser. Incomplete or
    unsupported recognized expressions fail closed with the original span.
    """
    if not isinstance(text, str):
        raise TypeError('text must be str')
    if language not in ('auto','mixed','zh','en'):
        raise ValueError('language must be auto, mixed, zh or en')
    if len(text) > 20000:
        raise MathNormalizationError('Text is too long for bounded mathematical parsing')
    carrier = 'zh' if language == 'zh' or (language in ('auto','mixed') and re.search(r'[\u3400-\u9fff]',text)) else 'en'
    edits = []
    for ts in _tokens_and_runs(text):
        if not _is_math(ts):
            continue
        start, end = ts[0].start, ts[-1].end
        try:
            ast = _Parser(ts, text).parse()
            spoken = _render(ast, carrier)
        except MathNormalizationError as exc:
            raise MathNormalizationError(str(exc)+'; span='+repr(text[start:end])) from exc
        edits.append((start,end,spoken))
    for start,end,spoken in reversed(edits):
        # Spaces keep English variables and source-neighbor words separate.
        left = ' ' if start and text[start-1].isascii() and text[start-1].isalnum() else ''
        right = ' ' if end<len(text) and text[end].isascii() and text[end].isalnum() else ''
        text = text[:start]+left+spoken+right+text[end:]
    return text
