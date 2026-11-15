"""Xiaoyou text-to-phoneme frontend with complete normalized-span context.

One implementation owns initialization, explicit language normalization and
Misaki conversion. Text/phoneme/token ABI and scoped rules match the reviewed
runtime; model configuration and term paths remain externally injectable.
"""
from hashlib import sha256
from contextvars import ContextVar
from functools import lru_cache
from importlib.metadata import version
import json
import os
from pathlib import Path
import re

from . import numbers as numeric
from .pronunciation import ContextualZHG2P, PolyphoneContextG2P

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RESOURCES = PACKAGE_ROOT / 'resources'
numeric._base()
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9_'’\-]*(?:[ \t]+[A-Za-z][A-Za-z0-9_'’\-]*)*")
_HAN = re.compile(r'[\u3400-\u9fff]')

_VERSION_NUMBER = r'[0-9]+(?:\.[0-9]+){0,3}'
_VERSION_LEFT = r'(?<![A-Za-z0-9_./\\-])'
_VERSION_RIGHT = r'(?![A-Za-z0-9_/\\-]|\.[A-Za-z0-9_.])'
_VERSION = re.compile(
    r'(?:' + _VERSION_LEFT + r'(?P<v>[vV])(?P<marked>' + _VERSION_NUMBER + r')|'
    r'(?P<label>' + _VERSION_LEFT + r'version(?:[ \t]+(?:is|was))?[ \t]*[:=]?[ \t]*|'
    r'版本(?:号)?[ \t]*(?:[为是:：=][ \t]*)?)(?P<labelled>' + _VERSION_NUMBER + r')|'
    # Known software followed immediately by its version is an explicit
    # identity too. The guards still exclude files, paths and suffixed IDs.
    + _VERSION_LEFT + r'(?:PyTorch|CUDA|TensorRT|cuDNN)[ \t]*(?:v[ \t]*)?'
    r'(?P<software_number>' + _VERSION_NUMBER + r'))'
    + _VERSION_RIGHT, re.I)
_VERSION_URL = re.compile(r'(?:https?://|ftp://|www\.)[^\s<>，。；！？]+', re.I)

def _version_spans(normalized, start, end):
    excluded = [m.span() for m in _VERSION_URL.finditer(normalized)]
    result = []
    for match in _VERSION.finditer(normalized):
        if any(a < match.end() and match.start() < b for a, b in excluded):
            continue
        group = next(name for name in ('marked', 'labelled', 'software_number')
                     if match[name] is not None)
        a, b = match.span(group)
        if a < start < b or a < end < b:
            raise ValueError('Semantic span splits an explicit version component sequence')
        if start <= a < b <= end:
            result.append((a, b, match[group]))
    return result

def _version_language(normalized, position, language, auto_fallback=None):
    if language in ('zh', 'en'):
        return language
    numeric._validate_auto_fallback(auto_fallback)
    cursor = 0
    fallback = auto_fallback
    for clause in numeric._clauses(normalized):
        if fallback is not None:
            fallback, _ = numeric.auto_context_language(clause, fallback)
        if cursor <= position < cursor + len(clause):
            return numeric._language(clause, language, 'zh') if fallback is None else fallback
        cursor += len(clause)
    raise ValueError('Version position is outside the complete normalized request')

def _version_reading(value, language):
    general, _ = numeric._base()
    components = value.split('.')
    readings = [general.speak_number(part, language, len(part) > 1 and part.startswith('0'))
                for part in components]
    return ('点' if language == 'zh' else ' point ').join(readings)

def _filename_reading(value, language):
    """Speak literal structure explicitly, with digit identities preserved."""
    general, _ = numeric._base()
    parts = re.findall(r'[A-Za-z]+|[0-9]+|[_\.\-]', value)
    if ''.join(parts) != value:
        raise ValueError('Unsupported filename literal')
    separators = {'zh': {'_': '下划线', '.': '点', '-': '连字符'},
                  'en': {'_': 'underscore', '.': 'dot', '-': 'hyphen'}}[language]
    rendered = []
    for part in parts:
        if part.isdigit():
            rendered.append(('' if language == 'zh' else ' ').join(
                general.speak_number(digit, language, True) for digit in part))
        else:
            rendered.append(separators.get(part, part))
    result = ' '.join(rendered)
    return re.sub(r'(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])', '', result)

@lru_cache(maxsize=32)
def _protected_term_spans(normalized):
    """Use this frontend's literal masks to guard aliases inside full context."""
    general, _ = numeric._base()
    _, mapping = numeric._protect(normalized, general)
    spans = []
    for literal in mapping.values():
        literal = numeric._restore(literal, mapping)
        if not literal:
            continue
        start = 0
        while True:
            start = normalized.find(literal, start)
            if start < 0:
                break
            spans.append((start, start + len(literal)))
            start += len(literal)
    return tuple(sorted(set(spans)))


def source_hashes():
    """Return package-relative source identities without hashing this mapping."""
    names = (
        'frontend/phonemizer.py', 'frontend/numbers.py',
        'frontend/normalization.py', 'frontend/math.py',
        'frontend/pronunciation.py', 'frontend/http_status.py',
        'frontend/segmentation.py', 'frontend/__init__.py',
        'resources/terms.json', 'resources/frontend_profile.json',
    )
    return {name: sha256((PACKAGE_ROOT / name).read_bytes()).hexdigest()
            for name in names}


class Frontend:
    """Keep the native (phonemes, ids) ABI; record actual interventions separately.

    Explicit zh/en selects the normalization language. G2P still follows script
    when the input contains both Han and Latin, so selecting English number
    readings never sends Chinese words into the English grapheme converter."""

    def __init__(self, config_path=None, terms_path=None):
        if config_path is None:
            config_path = Path(os.environ.get('XIAOYOU_V6_MODEL_DIR', '/models/01yuewen')) / 'config.json'
        if terms_path is None:
            terms_path = RESOURCES / 'terms.json'
        from misaki import en, espeak, zh
        if version('misaki') != '0.9.4':
            raise ValueError('V6 frontend requires the verified Misaki 0.9.4')
        self.config_path = Path(config_path)
        self.terms_path = Path(terms_path)
        self.vocab = json.loads(self.config_path.read_text())['vocab']
        specification = json.loads(self.terms_path.read_text())
        terms = specification['terms']
        required = {'CUDA': 'kˈudə', 'cuda': 'kˈudə', 'Cuda': 'kˈudə', 'SageAIOS': 'sˈAʤ ˈA ˈI ˈO ˈɛs'}
        aliases = specification.get('case_aliases', {})
        if (not isinstance(aliases, dict) or set(terms) != set(required) | set(aliases)
                or any(terms.get(word) != phones for word, phones in required.items())):
            raise ValueError('Required native Xiaoyou readings or exact alias inventory differ')
        for alias, record in aliases.items():
            canonical = record.get('canonical', '')
            if (not canonical or alias == canonical or alias.casefold() != canonical.casefold()
                    or terms[alias] != record.get('phonemes')
                    or record.get('source') != 'current_xiaoyou_Misaki_canonical_spelling'):
                raise ValueError('Only explicitly reviewed complete-token case aliases are allowed')
        for word, phones in terms.items():
            unknown = set(phones) - set(self.vocab)
            if unknown:
                raise ValueError(f'Invalid V6 phonemes for {word}: {sorted(unknown)}')
        case_terms = {record['canonical'].casefold(): record for record in aliases.values()}
        case_terms['cuda'] = {'canonical': 'CUDA', 'phonemes': required['CUDA']}
        class ExactTermG2P(en.G2P):
            _source_span = ContextVar('xiaoyou_english_source_span', default=None)

            def call_span(self, normalized, start, end):
                source_token = self._source_span.set((normalized, start, end, _protected_term_spans(normalized)))
                try:
                    return self(normalized[start:end])
                finally:
                    self._source_span.reset(source_token)

            def call_literal(self, normalized, start, end, reading):
                # Keep the ORIGINAL filename scope while speaking its generated
                # structure. Existing source-span guards reject casefold term
                # aliases inside it; canonical required readings stay native.
                source_token = self._source_span.set((normalized, start, end, _protected_term_spans(normalized)))
                try:
                    return self(reading)
                finally:
                    self._source_span.reset(source_token)

            def tokenize(self, text, tokens, features):
                native_tokens = super().tokenize(text, tokens, features)
                # Mark only complete spaCy MTokens before Misaki subtokenizes.
                # Explicit user phonemes remain authoritative.
                cursor = 0
                source_span = self._source_span.get()
                for token in native_tokens:
                    local_start = text.find(token.text, cursor)
                    local_end = local_start + len(token.text)
                    cursor = max(cursor, local_end)
                    # Keep the four original user/native readings unchanged.
                    phones = required.get(token.text)
                    record = case_terms.get(token.text.casefold()) if token.text.isascii() else None
                    if phones is None and record is not None and token.text != record['canonical'] and local_start >= 0:
                        source, a, b = text, local_start, local_end
                        if source_span is not None:
                            full, start, end, protected_spans = source_span
                            if full[start:end] != text:
                                continue  # Preprocessed markup has different offsets.
                            source, a, b = full, start + a, start + b
                            if any(lo < b and a < hi for lo, hi in protected_spans):
                                continue
                        before = source[a-1:a] if a else ''
                        after = source[b:b+1]
                        blocked_left = bool(before) and (before.isascii() and before.isalnum() or before in '_./\\@-')
                        blocked_right = bool(after) and (after.isascii() and after.isalnum() or after in '_/\\@-')
                        file_suffix = after == '.' and bool(source[b+1:b+2]) and source[b+1:b+2].isascii() and (source[b+1:b+2].isalnum() or source[b+1:b+2] == '_')
                        if not (blocked_left or blocked_right or file_suffix):
                            phones = record['phonemes']
                    if phones is not None and token.phonemes is None:
                        token.phonemes = phones
                        token._.rating = 5
                return native_tokens

        self.en = ExactTermG2P(trf=False, british=False,
                              fallback=espeak.EspeakFallback(british=False), unk='❓')
        self.zh = PolyphoneContextG2P(ContextualZHG2P(zh.ZHG2P(version=None, unk='❓')))
        self.term_phonemes = dict(terms)
        self.last_plan = None
        self.metadata = {
            'project_version': 'xiaoyou_v6_context_candidate_20261111',
            'misaki_version': '0.9.4',
            'frontend_path': str(Path(__file__).resolve()),
            'frontend_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
            'config_path': str(self.config_path.resolve()),
            'config_sha256': sha256(self.config_path.read_bytes()).hexdigest(),
            'terms_path': str(self.terms_path.resolve()),
            'terms_sha256': sha256(self.terms_path.read_bytes()).hexdigest(),
            'term_overrides': dict(terms),
            'legacy_rule_reuse': False,
            'raw_text_substitution': True,
            'normalization_order': ['protected_identifiers', 'clause_carrier_language', 'typed_contexts', 'pinned_general', 'pinned_math', 'NFKC_and_whitespace', 'restore_identifiers', 'native_G2P_with_scoped_tones'],
            'normalization_module_sha256': {name: pin for name, pin in source_hashes().items() if name not in ('frontend/phonemizer.py', 'resources/terms.json')},
            'context_pronunciation': self.zh.metadata,
            'override_scope': 'approved_complete_native_spacy_token_casefold_with_complete_source_span_identifier_guards',
            'lexicon_golds_modified': False,
            'candidate_only': True,
            'active_profile_changed': False,
            'candidate_sources': source_hashes(),
            'base_profile_path': str(RESOURCES / 'frontend_profile.json'),
            'base_profile_sha256': numeric.PROFILE_SHA,
            'script_routing': 'Han to native Chinese; Latin to native English; no persistent language state',
            'quality_accepted': False,
        }

    @staticmethod
    def normalize_text(text, language='auto', *, auto_fallback=None):
        # Keep explicit version identities whole through numeric normalization;
        # otherwise e.g. version 2.11.0 adjacent to Han can lose a separator.
        # Their numbers are rendered only in phonemize_normalized, with the
        # complete original normalized offsets retained for scoped Chinese G2P.
        if not isinstance(text,str):return numeric.normalize_context(text,language,auto_fallback=auto_fallback)
        versions=_version_spans(text,0,len(text))
        if not versions:return numeric.normalize_context(text,language,auto_fallback=auto_fallback)
        general,_=numeric._base();parts=[];mapping=[];cursor=0
        for a,b,value in versions:
            index=1000000+len(mapping);token=general._token(index)
            while token in text or any(token==item[0]for item in mapping):
                index+=1;token=general._token(index)
            parts.extend([text[cursor:a],token]);mapping.append((token,value));cursor=b
        parts.append(text[cursor:]);normalized=numeric.normalize_context(''.join(parts),language,auto_fallback=auto_fallback)
        for token,value in mapping:
            if normalized.count(token)!=1:raise ValueError('Explicit version placeholder was not preserved')
            normalized=normalized.replace(token,value)
        return normalized

    def phonemize_normalized(self, normalized, language='auto', *, start=0, end=None, normalization_language=None, auto_fallback=None):
        """G2P an exact semantic span while retaining the complete request context.

        Input has ALREADY passed normalize_text exactly once. This method never
        calls the general/math/typed normalizers again.
        """
        if language not in ('auto','mixed','zh','en'):
            raise ValueError('Unsupported candidate phoneme language')
        self.last_plan=None
        if not isinstance(normalized,str):raise TypeError('Normalized text must be a string')
        end=len(normalized) if end is None else end
        if type(start)is not int or type(end)is not int or not 0<=start<end<=len(normalized):
            raise ValueError('Invalid exact normalized semantic span')
        ascii_word=lambda c:c.isascii() and (c.isalnum() or c=='_')
        if start and ascii_word(normalized[start-1]) and ascii_word(normalized[start]):
            raise ValueError('Semantic span starts inside a protected English token')
        if end<len(normalized) and ascii_word(normalized[end-1]) and ascii_word(normalized[end]):
            raise ValueError('Semantic span ends inside a protected English token')
        span=normalized[start:end];interventions=[];version_readings=[]
        version_language=language if normalization_language is None else normalization_language
        if version_language not in ('auto','mixed','zh','en'):raise ValueError('Unsupported version language')
        numeric._validate_auto_fallback(auto_fallback)
        filenames=[]
        for a,b in numeric.filename_spans(normalized):
            if a<start<b or a<end<b:
                raise ValueError('Semantic span splits a complete filename')
            if start<=a<b<=end:filenames.append((a,b,normalized[a:b]))
        versions=[item for item in _version_spans(normalized,start,end)
                  if not any(a<item[1] and item[0]<b for a,b,_ in filenames)]
        filename_readings=[]
        events=sorted([(a,b,value,'version')for a,b,value in versions]+
                      [(a,b,value,'filename')for a,b,value in filenames])
        if events:
            result=[]
            with self.zh.full_context(normalized):
                def plain(a,b):
                    if a==b or not normalized[a:b].strip():return
                    if not _HAN.search(span) and language in ('auto','mixed','en'):
                        result.append(self.en.call_span(normalized,a,b)[0]);return
                    cursor=a
                    for match in _LATIN.finditer(normalized,a,b):
                        if cursor<match.start() and normalized[cursor:match.start()].strip():
                            result.append(self.zh.call_span(normalized,cursor,match.start())[0]);interventions.extend(self.zh.last_interventions)
                        result.append(self.en.call_span(normalized,match.start(),match.end())[0]);cursor=match.end()
                    if cursor<b and normalized[cursor:b].strip():
                        result.append(self.zh.call_span(normalized,cursor,b)[0]);interventions.extend(self.zh.last_interventions)
                cursor=start
                for a,b,value,kind in events:
                    plain(cursor,a)
                    lang=_version_language(normalized,a,version_language,auto_fallback)
                    if kind=='version':
                        reading=_version_reading(value,lang)
                        result.append(self.zh(reading)[0] if lang=='zh' else self.en(reading)[0])
                        version_readings.append({'source_span':[a,b],'source_text':value,'components':value.split('.'),'language':lang,'reading':reading,'rule':'explicit_version_components_preserve_dots_and_leading_zeros'})
                    else:
                        reading=_filename_reading(value,lang)
                        if lang=='en':
                            result.append(self.en.call_literal(normalized,a,b,reading)[0])
                        else:
                            # Generated words get their own exact Chinese scope;
                            # the article and every original offset stay intact.
                            with self.zh.full_context(reading):
                                local=0
                                for match in _LATIN.finditer(reading):
                                    if local<match.start() and reading[local:match.start()].strip():
                                        result.append(self.zh.call_span(reading,local,match.start())[0])
                                    result.append(self.en.call_literal(normalized,a,b,match.group())[0])
                                    local=match.end()
                                if local<len(reading) and reading[local:].strip():
                                    result.append(self.zh.call_span(reading,local,len(reading))[0])
                        filename_readings.append({'source_span':[a,b],'source_text':value,'language':lang,
                            'reading':reading,'rule':'whole_filename_digit_identity_and_explicit_separators'})
                    cursor=b
                plain(cursor,end)
            phonemes=' '.join(result)
        elif not _HAN.search(span) and language in ('auto','mixed','en'):
            phonemes=self.en.call_span(normalized,start,end)[0]
        else:
            result=[]
            with self.zh.full_context(normalized):
                cursor=start
                for match in _LATIN.finditer(normalized,start,end):
                    if cursor<match.start() and normalized[cursor:match.start()].strip():
                        result.append(self.zh.call_span(normalized,cursor,match.start())[0]);interventions.extend(self.zh.last_interventions)
                    result.append(self.en.call_span(normalized,match.start(),match.end())[0]);cursor=match.end()
                if cursor<end and normalized[cursor:end].strip():
                    result.append(self.zh.call_span(normalized,cursor,end)[0]);interventions.extend(self.zh.last_interventions)
            phonemes=' '.join(result)
        phonemes=re.sub(r'\s+',' ',phonemes).strip();unknown=sorted(set(phonemes)-set(self.vocab))
        if unknown:raise ValueError(f'Unknown phoneme symbols: {unknown!r}')
        ids=[self.vocab[p]for p in phonemes]
        if not ids:raise ValueError('Empty phoneme sequence')
        self.last_plan={'text':span,'normalization_language':version_language,'normalized_text':normalized,
                        'normalized_span':[start,end],'phonemes':phonemes,'token_ids':ids,
                        'interventions':interventions,'version_readings':version_readings,'filename_readings':filename_readings,'auto_fallback':auto_fallback,'normalization_already_applied':True}
        return phonemes,ids

    def __call__(self,text,language='auto',*,normalization_language=None,auto_fallback=None):
        if language not in ('auto','mixed','zh','en'):raise ValueError('Unsupported candidate phoneme language')
        self.last_plan=None
        norm_language=language if normalization_language is None else normalization_language
        normalized=self.normalize_text(text,norm_language,auto_fallback=auto_fallback)
        result=self.phonemize_normalized(normalized,language,normalization_language=norm_language,auto_fallback=auto_fallback)
        self.last_plan.update(text=text,normalization_language=norm_language)
        return result
