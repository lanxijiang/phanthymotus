"""Scoped Xiaoyou V6 Chinese word pronunciation candidate, not a lexicon patch.

Only the action sense in ``重听 + explicit audio object`` is covered here.
The ambiguous standalone word and medical/hearing-loss usages keep upstream
pronunciation.  This module does not import any V4/V5 rules, mutate jieba or
pypinyin dictionaries, change text, or change model/phone IDs.

Use: ContextualZHG2P(misaki.zh.ZHG2P(version=None, unk='❓')).  The legacy
Misaki path exposes no context callback; native normalization, segmentation,
word2ipa and py2ipa are reused and the replay is checked against the complete
upstream output before applying an offset-bound intervention.
"""
from __future__ import annotations

import re

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


def find_context_interventions(text: str) -> list[dict]:
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
            'scope': '重听 followed by an explicit 片段/音频/录音 object in the same complete Chinese span',
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
                    if word != '重听' or positions != [cursor]:
                        raise RuntimeError('The override is not a complete native 重听 word token')
                    # Identical per-word pinyin API/arguments used by
                    # ZHG2P.word2ipa; replace only the context-bound syllable.
                    pinyins = zh.lazy_pinyin(word, style=zh.Style.TONE3,
                                            neutral_tone_with_five=True)
                    if len(pinyins) != len(word):
                        raise RuntimeError('Cannot establish a one-syllable-per-character word offset')
                    for position in positions:
                        if prepared[position:position + 2] != '重听':
                            raise RuntimeError('Context override word changed')
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


