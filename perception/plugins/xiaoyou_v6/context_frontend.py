"""Isolated Xiaoyou candidate: contextual numbers and scoped Chinese tones.

The frozen training/deployment frontend is a read-only dependency. This module
does not register a global dictionary or change an installed/active profile.
"""
from hashlib import sha256
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent


from . import context_numeric_normalization as numeric
from . import polyphone_context as polyphone
from . import v6_frontend_base as base
numeric._base()
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9_'’\-]*(?:[ \t]+[A-Za-z][A-Za-z0-9_'’\-]*)*")
_HAN = re.compile(r'[\u3400-\u9fff]')

# Explicit versions are identities, not decimals or sentence punctuation. Keep
# their original normalized offsets and render each component only for G2P.
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


def _version_language(normalized, position, language):
    if language in ('zh', 'en'):
        return language
    cursor = 0
    for clause in numeric._clauses(normalized):
        if cursor <= position < cursor + len(clause):
            return numeric._language(clause, language, 'zh')
        cursor += len(clause)
    raise ValueError('Version position is outside the complete normalized request')


def _version_reading(value, language):
    general, _ = numeric._base()
    components = value.split('.')
    readings = [general.speak_number(part, language, len(part) > 1 and part.startswith('0'))
                for part in components]
    return ('点' if language == 'zh' else ' point ').join(readings)



def source_hashes():
    return {str(HERE / name): sha256((HERE / name).read_bytes()).hexdigest()
            for name in ('v6_frontend.py', 'v6_terms.json', 'context_frontend.py', 'context_numeric_normalization.py', 'polyphone_context.py', 'v6_frontend_base.py', 'text_normalization.py', 'http_status_codes.py', 'math_normalization.py', 'context_pronunciation.py', 'frontend_base_profile.json', 'text_segmentation.py', '__init__.py')}


class Frontend(base.Frontend):
    """Keep the native (phonemes, ids) ABI; record actual interventions separately.

    Explicit zh/en selects the normalization language. G2P still follows script
    when the input contains both Han and Latin, so selecting English number
    readings never sends Chinese words into the English grapheme converter.
    """

    def __init__(self, config_path=None, terms_path=None):
        if config_path is None:
            config_path = Path(os.environ.get('XIAOYOU_V6_MODEL_DIR','/models/01yuewen')) / 'config.json'
        if terms_path is None:
            terms_path = numeric.FRONTEND / 'v6_terms.json'
        super().__init__(config_path=config_path, terms_path=terms_path)
        self.zh = polyphone.PolyphoneContextG2P(self.zh)
        self.last_plan = None
        self.metadata.update({
            'project_version': 'xiaoyou_v6_context_candidate_20261111',
            'candidate_only': True, 'active_profile_changed': False,
            'frontend_path': str(HERE / 'v6_frontend.py'),
            'frontend_sha256': sha256((HERE / 'v6_frontend.py').read_bytes()).hexdigest(),
            'normalization_module_sha256': {Path(name).name:pin for name,pin in source_hashes().items() if Path(name).name not in ('v6_frontend.py','v6_terms.json')},
            'candidate_sources': source_hashes(),
            'base_profile_path': str(numeric.FRONTEND / 'frontend_base_profile.json'),
            'base_profile_sha256': numeric.PROFILE_SHA,
            'normalization_order': ['protected_identifiers', 'clause_carrier_language',
                                    'typed_contexts', 'pinned_general', 'pinned_math',
                                    'NFKC_and_whitespace', 'restore_identifiers',
                                    'native_G2P_with_scoped_tones'],
            'context_pronunciation': self.zh.metadata,
            'script_routing': 'Han to native Chinese; Latin to native English; no persistent language state',
            'quality_accepted': False,
        })

    @staticmethod
    def normalize_text(text, language='auto'):
        # Keep explicit version identities whole through numeric normalization;
        # otherwise e.g. version 2.11.0 adjacent to Han can lose a separator.
        # Their numbers are rendered only in phonemize_normalized, with the
        # complete original normalized offsets retained for scoped Chinese G2P.
        if not isinstance(text,str):return numeric.normalize_context(text,language)
        versions=_version_spans(text,0,len(text))
        if not versions:return numeric.normalize_context(text,language)
        general,_=numeric._base();parts=[];mapping=[];cursor=0
        for a,b,value in versions:
            index=1000000+len(mapping);token=general._token(index)
            while token in text or any(token==item[0]for item in mapping):
                index+=1;token=general._token(index)
            parts.extend([text[cursor:a],token]);mapping.append((token,value));cursor=b
        parts.append(text[cursor:]);normalized=numeric.normalize_context(''.join(parts),language)
        for token,value in mapping:
            if normalized.count(token)!=1:raise ValueError('Explicit version placeholder was not preserved')
            normalized=normalized.replace(token,value)
        return normalized

    def phonemize_normalized(self, normalized, language='auto', *, start=0, end=None, normalization_language=None):
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
        versions=_version_spans(normalized,start,end)
        if versions:
            result=[]
            with self.zh.full_context(normalized):
                def plain(a,b):
                    if a==b or not normalized[a:b].strip():return
                    if not _HAN.search(span) and language in ('auto','mixed','en'):
                        result.append(self.en(normalized[a:b])[0]);return
                    cursor=a
                    for match in _LATIN.finditer(normalized,a,b):
                        if cursor<match.start() and normalized[cursor:match.start()].strip():
                            result.append(self.zh.call_span(normalized,cursor,match.start())[0]);interventions.extend(self.zh.last_interventions)
                        result.append(self.en(match.group())[0]);cursor=match.end()
                    if cursor<b and normalized[cursor:b].strip():
                        result.append(self.zh.call_span(normalized,cursor,b)[0]);interventions.extend(self.zh.last_interventions)
                cursor=start
                for a,b,value in versions:
                    plain(cursor,a);lang=_version_language(normalized,a,version_language);reading=_version_reading(value,lang)
                    result.append(self.zh(reading)[0] if lang=='zh' else self.en(reading)[0])
                    version_readings.append({'source_span':[a,b],'source_text':value,'components':value.split('.'),'language':lang,'reading':reading,'rule':'explicit_version_components_preserve_dots_and_leading_zeros'})
                    cursor=b
                plain(cursor,end)
            phonemes=' '.join(result)
        elif not _HAN.search(span) and language in ('auto','mixed','en'):
            phonemes=self.en(span)[0]
        else:
            result=[]
            with self.zh.full_context(normalized):
                cursor=start
                for match in _LATIN.finditer(normalized,start,end):
                    if cursor<match.start() and normalized[cursor:match.start()].strip():
                        result.append(self.zh.call_span(normalized,cursor,match.start())[0]);interventions.extend(self.zh.last_interventions)
                    result.append(self.en(match.group())[0]);cursor=match.end()
                if cursor<end and normalized[cursor:end].strip():
                    result.append(self.zh.call_span(normalized,cursor,end)[0]);interventions.extend(self.zh.last_interventions)
            phonemes=' '.join(result)
        phonemes=re.sub(r'\s+',' ',phonemes).strip();unknown=sorted(set(phonemes)-set(self.vocab))
        if unknown:raise ValueError(f'Unknown phoneme symbols: {unknown!r}')
        ids=[self.vocab[p]for p in phonemes]
        if not ids:raise ValueError('Empty phoneme sequence')
        self.last_plan={'text':span,'normalization_language':version_language,'normalized_text':normalized,
                        'normalized_span':[start,end],'phonemes':phonemes,'token_ids':ids,
                        'interventions':interventions,'version_readings':version_readings,'normalization_already_applied':True}
        return phonemes,ids

    def __call__(self,text,language='auto',*,normalization_language=None):
        if language not in ('auto','mixed','zh','en'):raise ValueError('Unsupported candidate phoneme language')
        self.last_plan=None
        norm_language=language if normalization_language is None else normalization_language
        normalized=self.normalize_text(text,norm_language)
        result=self.phonemize_normalized(normalized,language,normalization_language=norm_language)
        self.last_plan.update(text=text,normalization_language=norm_language)
        return result
