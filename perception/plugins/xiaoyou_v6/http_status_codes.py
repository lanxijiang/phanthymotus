"""Narrow HTTP status-code reading candidate; standard library only.

Called after the unchanged V6 identifier protector. Only explicit HTTP/status-code
labels immediately followed by an ASCII100..599 code are eligible. No remote or
cross-sentence inference, generic number conversion, pronunciation lexicon or NN.
"""
import re

RULE_ID = 'xiaoyou-v6-http-status-ascii100-599-digits-v1'
_HTTP = r'(?<![A-Za-z0-9_])HTTP(?:/(?:1\.0|1\.1|2|3))?'
_LABELS = (
    _HTTP + r'[ \t]*(?:(?:接口|请求|响应)[ \t]*)?(?:返回|报错|提示|显示)(?:[ \t]*(?:状态码|错误码|错误代码))?',
    _HTTP + r'[ \t]+(?:status(?:[ \t]+code)?|error[ \t]+code|response[ \t]+code)(?:[ \t]+(?:is|was|of))?',
    _HTTP + r'[ \t]+(?:returns?|returned|reports?|reported)(?:[ \t]+(?:status(?:[ \t]+code)?|error[ \t]+code))?',
    _HTTP + r'[ \t]*(?:状态码|状态|响应码|错误码|错误代码|错误|响应)',
    r'(?<![A-Za-z0-9_])(?:status[ \t]+code|error[ \t]+code)(?:[ \t]+(?:is|was|of))?',
    r'(?:状态码|响应码|错误码|错误代码)',
    _HTTP,
)
# Quantity suffixes and identifier/version boundaries defeat a status-code match.
# This is intentionally narrower than a sentence-wide "containsHTTP" heuristic.
_QUANTITY = r'(?:个|次|元|条|份|项|页(?!面)|年|月|日|小时|分钟|秒|毫秒|千克|克|米|字节|摄氏度|毫升|升|items?\b|requests?\b|records?\b|bytes?\b|rows?\b|times?\b|dollars?\b|yuan\b|euros?\b)'
_PATTERN = re.compile(r'(?P<label>(?:' + '|'.join(_LABELS) + r')[ \t]*(?:(?:[:：#]|为|是|等于)[ \t]*)?)(?P<code>[1-5][0-9]{2})(?![A-Za-z0-9_]|[./][0-9]|[/\\]|[ \t]*' + _QUANTITY + r')',re.I)
_DIGITS_ZH = '零一二三四五六七八九'
_DIGITS_EN = ('zero','one','two','three','four','five','six','seven','eight','nine')


def normalize_protected_http_codes(protected, mapping, language, context_language):
    """Return a candidate string without mutating the identifier mapping.

    The original protector masks alphanumericHTTP404 as an identifier. Unmask
    only that exact entire typed token; a URL, path, version or quoted literal
    whose protected span containsHTTP404 stays masked and unchanged.
    """
    for token, original in mapping.items():
        if re.fullmatch(r'HTTP[1-5][0-9]{2}',original,re.I):
            protected = protected.replace(token,original)

    def read(match):
        lang = context_language(match.string,match.start(),match.end(),language)
        label,code = match['label'],match['code']
        if lang == 'en':
            if label and label[-1].isalnum():label += ' '
            spoken = ' '.join(_DIGITS_EN[int(d)] for d in code)
        else:spoken = ''.join(_DIGITS_ZH[int(d)] for d in code)
        return label + spoken
    return _PATTERN.sub(read,protected)
