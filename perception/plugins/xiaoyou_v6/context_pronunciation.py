"""Scoped Xiaoyou V6 Chinese word pronunciation candidate, not a lexicon patch.

Preserve the existing ``重听 + explicit audio object`` action rule and add
only重来/chong2、重试/chong2、数清/shu3、长椅/chang2 at verified native word
boundaries or the specifically allowed native splits重+来 and先数+清.
Ambiguous substrings and unsupported native segmentations keep upstream output.  This module does not import any V4/V5 rules, mutate jieba or
pypinyin dictionaries, change text, or change model/phone IDs.

Use: ContextualZHG2P(misaki.zh.ZHG2P(version=None, unk='❓')).  The legacy
Misaki path exposes no context callback; native normalization, segmentation,
word2ipa and py2ipa are reused and the replay is checked against the complete
upstream output before applying an offset-bound intervention.
"""
from __future__ import annotations

import re
from pathlib import Path
from hashlib import sha256

RULE_ID = 'xiaoyou-v6-relisten-explicit-audio-object-v1'
# Deliberately narrow objects.  Do not turn every occurrence of 重 or 重听
# into chong2; diagnoses, isolated quotations and other senses are untouched.
_RELISTEN = re.compile(
    r'重听'
    r'(?:(?:一下|一遍|两遍|三遍|几遍|一次|两次|几次))?'
    r'(?:(?:(?:这|那|某|哪)一?)|一|两|几)?'
    r'(?:个|段|条)?'
    r'(?P<audio_object>片段|音频|录音)'
)
_SEGMENTS = re.compile(r'[\u4E00-\u9FFF]+|[^\u4E00-\u9FFF]+')
FOUR_WORD_RULE_ID = 'xiaoyou-v6-four-lexical-contexts-20261110-v1'
_WORD_READINGS = {'重来': 'chong2', '重试': 'chong2', '数清': 'shu3', '长椅': 'chang2'}


def _find_existing_relisten_interventions(text: str) -> list[dict]:
    """Return exact code-point offsets in the supplied complete Chinese span.

    Offsets refer to the original two-character word, not a fuzzy match or a
    phone-string search.  Audio semantics across an English split, arbitrary
    intervening words and other polyphones are outside this rule's scope.
    """
    matches = list(_RELISTEN.finditer(text))
    if not matches:
        return []
    from misaki import zh
    eligible_words = set()
    for segment_match in _SEGMENTS.finditer(text):
        segment = segment_match.group()
        if not '\u4E00' <= segment[0] <= '\u9FFF':
            continue
        words = zh.jieba.lcut(segment, cut_all=False)
        if ''.join(words) != segment:
            raise RuntimeError('Cannot establish native word boundaries')
        cursor = segment_match.start()
        for word in words:
            if word == '重听':
                eligible_words.add((cursor, cursor + len(word)))
            cursor += len(word)
    return [
        {
            'rule_id': RULE_ID,
            'start': m.start(), 'end': m.start() + 2,
            'surface': '重听',
            'context_start': m.start(), 'context_end': m.end(),
            'context_surface': m.group(),
            'audio_object': m.group('audio_object'),
            'pinyin_override_at_start': 'chong2',
        }
        for m in matches
        if (m.start(), m.start() + 2) in eligible_words
    ]


def _native_tokens(text):
    """Full unchanged native words with raw-text offsets and Chinese-span IDs."""
    from misaki import zh
    tokens = []
    for segment_index, segment_match in enumerate(_SEGMENTS.finditer(text)):
        segment = segment_match.group()
        if not '\u4E00' <= segment[0] <= '\u9FFF':
            continue
        words = zh.jieba.lcut(segment, cut_all=False)
        if ''.join(words) != segment:
            raise RuntimeError('Cannot establish full native token offsets')
        cursor = segment_match.start()
        for word in words:
            tokens.append({'surface': word, 'start': cursor, 'end': cursor + len(word),
                           'segment': segment_index})
            cursor += len(word)
    return tokens


def find_context_interventions(text: str) -> list[dict]:
    if not re.search(r'重听|重来|重试|数清|长椅', text):
        return []
    interventions = _find_existing_relisten_interventions(text)
    tokens = _native_tokens(text)

    def add(surface, position, consumed, pattern):
        if text[position:position + len(surface)] != surface:
            raise RuntimeError('The lexical target does not match its exact character offsets')
        context_start, context_end = consumed[0]['start'], consumed[-1]['end']
        if any(token['segment'] != consumed[0]['segment'] for token in consumed):
            raise RuntimeError('A lexical intervention must not cross Chinese spans')
        interventions.append({
            'rule_id': FOUR_WORD_RULE_ID, 'start': position, 'end': position + len(surface),
            'surface': surface, 'context_start': context_start, 'context_end': context_end,
            'context_surface': text[context_start:context_end],
            'native_token_pattern': pattern,
            'native_tokens': [dict(token) for token in consumed],
            'pinyin_override_at_start': _WORD_READINGS[surface],
        })

    for index, token in enumerate(tokens):
        word = token['surface']
        if word in _WORD_READINGS:
            add(word, token['start'], [token], 'exact_complete_native_word')
        if index + 1 == len(tokens):
            continue
        following = tokens[index + 1]
        if token['end'] != following['start'] or token['segment'] != following['segment']:
            continue
        pair = (word, following['surface'])
        if pair == ('重', '来'):
            add('重来', token['start'], [token, following], 'two_complete_native_words_重_来')
        elif pair == ('先数', '清'):
            # Keep先数 and清 as the native tokenizer produced them. Only数's
            # syllable inside the complete先数 token changes;先 is untouched.
            add('数清', token['start'] + 1, [token, following], 'limited_action_fusion_先数_清')
    starts = [item['start'] for item in interventions]
    if len(set(starts)) != len(starts):
        raise RuntimeError('Multiple context rules select the same syllable offset')
    return sorted(interventions, key=lambda item: item['start'])


class ContextualZHG2P:
    """Upstream-call compatible instance wrapper; non-targets delegate exactly."""

    def __init__(self, upstream):
        if not callable(upstream):
            raise TypeError('A native Misaki ZHG2P callable is required')
        if getattr(upstream, 'version', None) is not None or getattr(upstream, 'frontend', None) is not None:
            raise ValueError('This candidate supports only the verified Misaki legacy version=None path')
        from misaki import zh
        self.upstream = upstream
        self._native = zh
        self.metadata = {
            'project_version': 'xiaoyou_v6',
            'kind': 'scoped_context_pronunciation_candidate',
            'rule_id': RULE_ID,
            'scope': 'Existing重听 explicit audio-object rule plus only4 lexical targets at native exact words/重+来/先数+清; no global character rules',
            'four_word_rule_id': FOUR_WORD_RULE_ID,
            'candidate_module_path': str(Path(__file__).resolve()),
            'candidate_module_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
            'four_word_readings': dict(_WORD_READINGS),
            'limited_cross_token_patterns': [['重', '来'], ['先数', '清']],
            'upstream_version': None,
            'normal_non_target_direct_delegate': True,
            'native_replay_must_match_before_intervention': True,
            'raw_text_substitution': False,
            'pypinyin_or_jieba_global_dictionary_modified': False,
            'legacy_V4_V5_rules_used': False,
            'other_polyphone_coverage_claimed': False,
        }

    def __getattr__(self, name):
        return getattr(self.upstream, name)

    def _render(self, prepared: str, fixes: dict[int, str]) -> str:
        zh = self._native
        rendered = []
        applied = set()
        authorized = {item['start']: item['pinyin_override_at_start']
                      for item in find_context_interventions(prepared)} if fixes else {}
        if any(authorized.get(position) != pinyin for position, pinyin in fixes.items()):
            raise RuntimeError('An override is not authorized at that exact native context offset')
        for segment_match in _SEGMENTS.finditer(prepared):
            segment = segment_match.group()
            if not '\u4E00' <= segment[0] <= '\u9FFF':
                rendered.append(segment)
                continue
            # Same tokenizer, cut mode, word order and inter-word spaces as
            # Misaki 0.9.4 ZHG2P.legacy_call.  No add_word/load_dict calls.
            words = zh.jieba.lcut(segment, cut_all=False)
            if ''.join(words) != segment:
                raise RuntimeError('Native jieba offsets do not reproduce the full Chinese segment')
            word_outputs = []
            cursor = segment_match.start()
            for word in words:
                positions = [p for p in fixes if cursor <= p < cursor + len(word)]
                if not positions:
                    word_outputs.append(zh.ZHG2P.word2ipa(word))
                else:
                    if any(not cursor <= position < cursor + len(word) for position in positions):
                        raise RuntimeError('The override is outside the complete native token')
                    # Identical per-word pinyin API/arguments used by
                    # ZHG2P.word2ipa; replace only the context-bound syllable.
                    pinyins = zh.lazy_pinyin(word, style=zh.Style.TONE3,
                                            neutral_tone_with_five=True)
                    if len(pinyins) != len(word):
                        raise RuntimeError('Cannot establish a one-syllable-per-character word offset')
                    for position in positions:
                        if prepared[position] not in ('重', '数', '长'):
                            raise RuntimeError('Context override target character changed')
                        pinyins[position - cursor] = fixes[position]
                        applied.add(position)
                    word_outputs.append(''.join(zh.ZHG2P.py2ipa(py) for py in pinyins))
                cursor += len(word)
            rendered.append(' '.join(word_outputs))
        if applied != set(fixes):
            raise RuntimeError('Not all word-scoped overrides were applied')
        return ''.join(rendered).replace(chr(815), '')

    def __call__(self, text, en_callable=None):
        # Normal input takes the untouched upstream implementation directly.
        if not find_context_interventions(text):
            return self.upstream(text, en_callable=en_callable)
        original = self.upstream(text, en_callable=en_callable)
        zh = self._native
        prepared = zh.ZHG2P.map_punctuation(zh.cn2an.transform(text, 'an2cn'))
        interventions = find_context_interventions(prepared)
        if not interventions:
            return original
        if self._render(prepared, {}) != original[0]:
            raise RuntimeError('Context candidate native replay differs from the complete upstream phonemes')
        fixes = {x['start']: x['pinyin_override_at_start'] for x in interventions}
        return self._render(prepared, fixes), original[1]


def make_contextual_zh_g2p(upstream):
    return ContextualZHG2P(upstream)


