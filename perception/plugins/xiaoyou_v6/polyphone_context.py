"""Instance-local Chinese word/context readings with exact syllable offsets.

Wrap the existing legacy Chinese G2P instance; never load or change a global
jieba/pypinyin dictionary. Candidate only: requires a new train/infer profile.
"""
from __future__ import annotations
import re
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from hashlib import sha256

_SEGMENTS=re.compile(r'[\u4E00-\u9FFF]+|[^\u4E00-\u9FFF]+')
_BASIS=re.compile(r'以[\u4E00-\u9FFF]{1,48}?为基础')
_VERIFIED_SPLIT_BASIS=re.compile(r'算力为基础')
_FULL_BASIS=re.compile(r'以(?P<object>[\u4E00-\u9FFFA-Za-z0-9_ \t\-.]{1,64}?)为基础')

# Three bounded grammatical constructions, not a sentence/character lookup table.
# Each match must begin and end on native Chinese token boundaries as well.
_SYNTAX_RULES = (
    (re.compile(r'(?:我们|你们|他们|她们|咱们|大家|我|你|您|他|她)(?P<target>得)'
                r'(?:先|把|去|赶紧|尽快|注意|确保)'),
     '得', 'dei3', 'modal_subject_得_action'),
    (re.compile(r'(?:解释|核对|整理|处理|检查|说|讲|读|做)(?P<target>得)'
                r'(?:非常|十分|特别|比较|这么|那么|很|更|太)?'
                r'(?:清楚|仔细|自然|流畅|准确)'),
     '得', 'de5', 'verb_得_optional_degree_quality'),
    (re.compile(r'(?:安静|认真|仔细|耐心|温柔|轻轻)(?P<target>地)'
                r'(?:解释|介绍|处理|核对|检查|听|说|讲|读|做|把)'),
     '地', 'de5', 'manner_地_action'),
)
# Only a clear numerical-value copula. Purpose uses 为了/为你/为服务 do
# not match. English numerical readings need full normalized request context.
_CN_VALUE = r'(?:正负|正|负)?[零〇一二三四五六七八九十百千万亿兆两幺]+'
_ARABIC_VALUE = r'[+−-]?[0-9]+(?:\.[0-9]+)?'
_EN_NUMBER = (r'(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
              r'thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|'
              r'forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million)\b')
_EN_VALUE = r'(?:(?:minus|negative|plus|positive)[ \t]+)?' + _EN_NUMBER
_NUMERIC_VALUE = r'(?:' + _CN_VALUE + r'|' + _ARABIC_VALUE + r'|' + _EN_VALUE + r')'
_VALUE_COPULA = re.compile(r'(?:温度范围|范围|延迟|速度|时长|数量|温度|值|结果|版本号|版本|编号|代码|端口号|端口)'
                           r'(?P<target>为)(?=[ \t]*' + _NUMERIC_VALUE + r')', re.I)
_VERSION_VALUE = r'v[0-9]+(?:\.[0-9]+){0,3}(?![A-Za-z0-9_+\-/]|\.[A-Za-z0-9_.])'
_VERSION_COPULA = re.compile(r'(?:版本号|版本)(?P<target>为)(?=[ \t]*' + _VERSION_VALUE + r')', re.I)

def _syntax_targets(text,words):
    result=[]
    starts={start for _,start,_ in words};ends={end for _,_,end in words}
    for pattern,character,pinyin,rule in _SYNTAX_RULES:
        for match in pattern.finditer(text):
            if match.start() not in starts or match.end() not in ends:
                continue  # An embedded substring is not a proved grammar boundary.
            tokens=[word for word,start,end in words if match.start()<=start and end<=match.end()]
            if ''.join(tokens)!=match.group():
                raise RuntimeError('Grammar native tokens do not reconstruct exact context')
            result.append({'position':match.start('target'),'character':character,
                           'pinyin':pinyin,'context':match.group(),'rule':rule,
                           'native_word_boundary_verified':True,'native_context_tokens':tokens})
    for pattern in (_VALUE_COPULA, _VERSION_COPULA):
        for match in pattern.finditer(text):
            result.append({'position':match.start('target'),'character':'为','pinyin':'wei2',
                           'context':match.group(),'rule':'numerical_value_copula_named_business_field'})
    return result


class PolyphoneContextG2P:
    def __init__(self,upstream):
        if not callable(upstream) or getattr(upstream,'version',None)is not None or getattr(upstream,'frontend',None)is not None:
            raise TypeError('Only the audited legacy Misaki Chinese G2P ABI is supported')
        from misaki import zh
        self.upstream=upstream;self._native=zh;self.last_interventions=[]
        self._context=ContextVar('polyphone_context_'+str(id(self)),default=None)
        self.metadata={'kind':'isolated_polyphone_candidate_NOT_active_profile','source_path':str(Path(__file__).resolve()),'source_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),'scope':['complete_native_word_华为_second_syllable_wei2','same_Chinese_span_以_1to48ChineseChars_为基础_target_wei2','verified_English_split_suffix_算力为基础_target_wei2','native_word_bounded_subject_得_action_dei3','native_word_bounded_verb_得_degree_quality_de5','native_word_bounded_manner_地_action_de5','numerical_value_copula_named_business_fields_wei2'],'bound_full_context_scope':'explicit_以…为基础_and_verified_grammar_phrases_and_numerical_value_copula_with_exact_offsets','global_dictionary_modified':False,'other_non_targets_delegate':True,'quality_accepted':False}

    def __getattr__(self,name):return getattr(self.upstream,name)

    @contextmanager
    def full_context(self,normalized):
        """Bind one already normalized article to this instance/task only."""
        if not isinstance(normalized,str) or not normalized:
            raise TypeError('A nonempty complete normalized string is required')
        targets=[]
        for match in _FULL_BASIS.finditer(normalized):
            if not re.search(r'[\u4E00-\u9FFFA-Za-z0-9]',match.group('object')):continue
            # Only numeric-internal version dots; sentence periods are not object characters.
            if '.' in re.sub(r'(?<=[0-9])\.(?=[0-9])','',match.group('object')):continue
            position=match.end()-3
            if normalized[position]!='为':raise RuntimeError('Full-context target offset mismatch')
            targets.append({'global_position':position,'context':match.group(),'rule':'bound_full_normalized_以…为基础','pinyin':'wei2'})
        for item in _syntax_targets(normalized,self._words(normalized)):
            targets.append({**item,'global_position':item['position']})
        token=self._context.set({'text':normalized,'targets':targets})
        try:yield self
        finally:self._context.reset(token)

    def _verified_prepared_position(self,span,raw_position,prepared,character='为'):
        """Prove one exact authorized character offset across native preparation."""
        if not 0<=raw_position<len(span) or span[raw_position]!=character:
            raise RuntimeError('The bound source character is not '+character)
        zh=self._native
        left=zh.cn2an.transform(span[:raw_position],'an2cn')
        right=zh.cn2an.transform(span[raw_position+1:],'an2cn')
        whole=zh.cn2an.transform(span,'an2cn')
        if left+character+right!=whole:
            raise RuntimeError('cn2an source offset cannot be proven across this split')
        marker='\ue000POLYPHONE_TARGET\ue001'
        if marker in span or marker in whole:raise RuntimeError('Offset marker collision')
        marked=zh.ZHG2P.map_punctuation(left+marker+right)
        if marked.count(marker)!=1 or marked.replace(marker,character,1)!=prepared:
            raise RuntimeError('Punctuation mapping does not preserve the exact target offset')
        position=marked.index(marker)
        if prepared[position]!=character:raise RuntimeError('Prepared offset does not point to '+character)
        return position

    def call_span(self,normalized,start,end,en_callable=None):
        """Call one exact Chinese gap from the currently bound full article."""
        bound=self._context.get()
        if bound is None:raise RuntimeError('call_span requires an active full_context scope')
        if normalized!=bound['text']:raise ValueError('Span source differs from bound complete normalized text')
        if type(start)is not int or type(end)is not int or not 0<=start<end<=len(normalized):
            raise ValueError('Invalid exact normalized character span')
        span=normalized[start:end]
        if re.search(r'[A-Za-z]',span):raise ValueError('English spans must use the unchanged English frontend')
        zh=self._native;prepared=zh.ZHG2P.map_punctuation(zh.cn2an.transform(span,'an2cn'));targets=[]
        for item in bound['targets']:
            source=item['global_position']
            if start<=source<end:
                character=item.get('character','为')
                position=self._verified_prepared_position(span,source-start,prepared,character)
                targets.append({**item,'position':position,'character':character,'source_span':[start,end],
                                'source_span_position':source-start,'offset_proven':True})
        return self._call(span,en_callable=en_callable,bound_targets=targets,expected_prepared=prepared)

    def _words(self,prepared):
        words=[]
        for segment in _SEGMENTS.finditer(prepared):
            if not '\u4E00'<=segment.group()[0]<='\u9FFF':continue
            tokens=self._native.jieba.lcut(segment.group(),cut_all=False)
            if ''.join(tokens)!=segment.group():raise RuntimeError('Native token offsets do not reproduce the Chinese span')
            cursor=segment.start()
            for word in tokens:words.append((word,cursor,cursor+len(word)));cursor+=len(word)
        return words

    def _targets(self,prepared,words):
        targets={}
        def add(position,context,rule,character='为',pinyin='wei2',**proof):
            if prepared[position]!=character:raise RuntimeError('Context target is not the exact '+character+' syllable')
            if not any(start<=position<end for _,start,end in words):raise RuntimeError('No native word contains the target')
            targets[position]={'position':position,'character':character,'pinyin':pinyin,'context':context,'rule':rule,**proof}
        for word,start,end in words:
            if word=='华为':add(start+1,word,'complete_native_word_华为')
        for match in _BASIS.finditer(prepared):add(match.end()-3,match.group(),'explicit_以…为基础_same_Chinese_span')
        # V6 splits an English GPU span before invoking its Chinese G2P.
        # Only this already verified exact Chinese suffix is additionally allowed.
        for match in _VERIFIED_SPLIT_BASIS.finditer(prepared):add(match.start()+2,match.group(),'verified_exact_算力为基础_suffix')
        for item in _syntax_targets(prepared,words):add(**item)
        return [targets[k]for k in sorted(targets)]

    def _existing_fixes(self,prepared):
        renderer=getattr(self.upstream,'_render',None)
        if renderer is None:return {}
        function=getattr(renderer,'__func__',None);finder=getattr(function,'__globals__',{}).get('find_context_interventions')
        if not callable(finder):raise RuntimeError('Unknown upstream context renderer; refuse to bypass existing rules')
        return {item['start']:item['pinyin_override_at_start']for item in finder(prepared)}

    def _render(self,prepared,fixes):
        zh=self._native;rendered=[];applied=set()
        for segment in _SEGMENTS.finditer(prepared):
            value=segment.group()
            if not '\u4E00'<=value[0]<='\u9FFF':rendered.append(value);continue
            words=zh.jieba.lcut(value,cut_all=False)
            if ''.join(words)!=value:raise RuntimeError('Cannot reproduce native word boundaries')
            outputs=[];cursor=segment.start()
            for word in words:
                positions=[p for p in fixes if cursor<=p<cursor+len(word)]
                if not positions:outputs.append(zh.ZHG2P.word2ipa(word))
                else:
                    pinyins=zh.lazy_pinyin(word,style=zh.Style.TONE3,neutral_tone_with_five=True)
                    if len(pinyins)!=len(word):raise RuntimeError('Cannot establish exact one-syllable character offsets')
                    for position in positions:pinyins[position-cursor]=fixes[position];applied.add(position)
                    outputs.append(''.join(zh.ZHG2P.py2ipa(py)for py in pinyins))
                cursor+=len(word)
            rendered.append(' '.join(outputs))
        if applied!=set(fixes):raise RuntimeError('Not all exact syllable offsets were applied')
        return ''.join(rendered).replace(chr(815),'')

    def __call__(self,text,en_callable=None):
        return self._call(text,en_callable=en_callable)

    def _call(self,text,en_callable=None,bound_targets=(),expected_prepared=None):
        original=self.upstream(text,en_callable=en_callable)
        if not(isinstance(original,tuple)and len(original)==2 and isinstance(original[0],str)):
            raise TypeError('Expected unchanged Misaki (phonemes, tokens) tuple ABI')
        zh=self._native;prepared=zh.ZHG2P.map_punctuation(zh.cn2an.transform(text,'an2cn'))
        if expected_prepared is not None and prepared!=expected_prepared:raise RuntimeError('Per-span preparation changed after offset proof')
        words=self._words(prepared);by_position={item['position']:item for item in self._targets(prepared,words)}
        for item in bound_targets:
            position=item['position']
            if prepared[position]!=item.get('character','为') or not any(start<=position<end for _,start,end in words):
                raise RuntimeError('Bound target is not the exact native Chinese syllable')
            by_position[position]=item
        targets=[by_position[k]for k in sorted(by_position)];self.last_interventions=targets
        if not targets:return original
        existing=self._existing_fixes(prepared)
        if self._render(prepared,existing)!=original[0]:raise RuntimeError('Full upstream replay differs; candidate must not overwrite unrelated rules')
        fixes=dict(existing)
        for target in targets:
            position=target['position']
            pinyin=target['pinyin']
            if position in fixes and fixes[position]!=pinyin:raise RuntimeError('Conflicting existing pronunciation rule')
            fixes[position]=pinyin
        return self._render(prepared,fixes),original[1]
