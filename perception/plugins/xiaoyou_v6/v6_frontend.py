"""Independent Xiaoyou V6 frontend: current Kokoro routing plus CUDA only.

No legacy project rules, dictionaries, or code are imported. The training frontend
and model files remain unchanged. Native Misaki token hooks preserve context.
"""
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import re
import unicodedata

HERE = Path(__file__).resolve().parent


class Frontend:
    def __init__(self, config_path=None, terms_path=None):
        from misaki import en, espeak, zh
        if version('misaki') != '0.9.4':
            raise ValueError('V6 frontend requires the verified Misaki 0.9.4')
        self.config_path = Path(config_path) if config_path else HERE / 'native/config.json'
        self.terms_path = Path(terms_path) if terms_path else HERE / 'v6_terms.json'
        self.vocab = json.loads(self.config_path.read_text())['vocab']
        specification = json.loads(self.terms_path.read_text())
        terms = specification['terms']
        if terms != {'CUDA': 'kˈudə', 'cuda': 'kˈudə', 'Cuda': 'kˈudə'}:
            raise ValueError('This V6 revision permits only the three exact CUDA spellings')
        for word, phones in terms.items():
            unknown = set(phones) - set(self.vocab)
            if unknown:
                raise ValueError(f'Invalid V6 phonemes for {word}: {sorted(unknown)}')
        class ExactCudaG2P(en.G2P):
            def tokenize(self, text, tokens, features):
                native_tokens = super().tokenize(text, tokens, features)
                # Mark only complete spaCy MTokens before Misaki subtokenizes.
                # Explicit user phonemes remain authoritative.
                for token in native_tokens:
                    if token.text in terms and token.phonemes is None:
                        token.phonemes = terms[token.text]
                        token._.rating = 5
                return native_tokens

        self.en = ExactCudaG2P(trf=False, british=False,
                              fallback=espeak.EspeakFallback(british=False), unk='❓')
        self.zh = zh.ZHG2P(version=None, unk='❓')
        self.term_phonemes = dict(terms)
        self.metadata = {
            'project_version': 'xiaoyou_v6', 'misaki_version': '0.9.4',
            'frontend_path': str(Path(__file__).resolve()),
            'frontend_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
            'config_path': str(self.config_path.resolve()),
            'config_sha256': sha256(self.config_path.read_bytes()).hexdigest(),
            'terms_path': str(self.terms_path.resolve()),
            'terms_sha256': sha256(self.terms_path.read_bytes()).hexdigest(),
            'term_overrides': dict(terms), 'legacy_rule_reuse': False,
            'raw_text_substitution': False,
            'override_scope': 'complete_native_spacy_mtoken_before_subtokenize',
            'lexicon_golds_modified': False,
        }

    def __call__(self, text, language):
        normalized = unicodedata.normalize('NFKC', text).strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = normalized.replace('·', ' ').replace('–', '—')
        if re.search(r'[À-ÖØ-öø-ÿ]', normalized):
            raise ValueError('unsupported_nonEnglish_Latin_from_dataset')
        if language == 'en':
            phonemes = self.en(normalized)[0]
        else:
            spans = re.split(r"([A-Za-z][A-Za-z0-9_'’\-]*(?:[ \t]+[A-Za-z][A-Za-z0-9_'’\-]*)*)", normalized)
            result = []
            for span in spans:
                if not span.strip():
                    continue
                if re.match('[A-Za-z]', span):
                    result.append(self.en(span)[0])
                else:
                    result.append(self.zh(span)[0])
            phonemes = ' '.join(result)
        phonemes = re.sub(r'\s+', ' ', phonemes).strip()
        unknown = sorted(set(phonemes) - set(self.vocab))
        if unknown:
            raise ValueError(f'Unknown phoneme symbols: {unknown!r}')
        ids = [self.vocab[p] for p in phonemes]
        if not ids:
            raise ValueError('Empty phoneme sequence')
        return phonemes, ids
