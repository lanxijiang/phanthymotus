"""Candidate-only contextual numbers; pure stdlib, no NN and no production edits.

Reuse the pinned current student V6 general/math normalizers under a private
module namespace. New rules add typed stock/error/port identifiers, year-month,
case-sensitive digital units/ranges, and complete-clause language selection.
Chinese calendar fields disambiguate month/day; slash calendar dates without
an English locale are rejected. Other identifier literals and two-field elapsed
timers stay protected. Invalid calendar/time/port values raise ValueError.
The returned string is pre-G2P text, not pronunciation/audio acceptance.
"""
from pathlib import Path
from functools import lru_cache
from datetime import date
from decimal import Decimal
import hashlib,importlib,json,re,sys,types,unicodedata

FRONTEND=Path(__file__).resolve().parent
PROFILE_SHA='03d895df195404ab75f4def4a132c9a7f2a9279fe20d3f3c5b434144550bff67'
class NumericNormalizationError(ValueError):pass
def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
@lru_cache(maxsize=1)
def _base():
 profile=json.loads((FRONTEND/'frontend_base_profile.json').read_text())
 if _sha(FRONTEND/'frontend_base_profile.json')!=PROFILE_SHA:raise NumericNormalizationError('Pinned student profile changed')
 for name,pin in profile['source_files'].items():
  if _sha(FRONTEND/name)!=pin['profile_sha256']:raise NumericNormalizationError('Pinned student module changed: '+name)
 return importlib.import_module(__package__+'.text_normalization'),importlib.import_module(__package__+'.math_normalization')

# A prose comma is not part of the number; internal commas remain validated.
_NUMBER=r'[+\-−]?(?:[0-9](?:[0-9,]*[0-9])?(?:\.[0-9]+)?|\.[0-9]+)'
_LEFT=r'(?<![A-Za-z0-9_./+\-=])'
_RIGHT=r'(?![A-Za-z0-9_]|\.[0-9]|,[0-9])'
_DATE=re.compile(_LEFT+r'(?P<y>[0-9]{4})(?P<sep>[-/])(?P<m>[0-9]{1,2})(?P=sep)(?P<d>[0-9]{1,2})(?![A-Za-z0-9_/\-])')
_ZHDATE=re.compile(_LEFT+r'(?P<y>[0-9]{4})\s*年\s*(?P<m>[0-9]{1,2})\s*月\s*(?P<d>[0-9]{1,2})\s*[日号]')
_YM=re.compile(_LEFT+r'(?P<y>[0-9]{4})\s*年\s*(?P<m>[0-9]{1,2})\s*月(?!\s*[0-9]{1,2}\s*[日号])')
_YEAR=re.compile(_LEFT+r'(?P<y>[0-9]{4})\s*年(?!\s*[0-9]{1,2}\s*月)')
_EXTRA_UNITS={
 'TB':('太字节','terabyte','terabytes'),'GB':('吉字节','gigabyte','gigabytes'),
 'Gb':('吉比特','gigabit','gigabits'),'MB':('兆字节','megabyte','megabytes'),
 'Mb':('兆比特','megabit','megabits'),'kB':('千字节','kilobyte','kilobytes'),
 'GB/s':('吉字节每秒','gigabyte per second','gigabytes per second'),
 'Gb/s':('吉比特每秒','gigabit per second','gigabits per second'),
 'MB/s':('兆字节每秒','megabyte per second','megabytes per second'),
 'Mb/s':('兆比特每秒','megabit per second','megabits per second'),
 'MBps':('兆字节每秒','megabyte per second','megabytes per second'),
 'TBps':('太字节每秒','terabyte per second','terabytes per second'),
 'Mbps':('兆比特每秒','megabit per second','megabits per second'),
 'Gbps':('吉比特每秒','gigabit per second','gigabits per second'),
 'ms/s':('毫秒每秒','millisecond per second','milliseconds per second'),
 's':('秒','second','seconds'),'min':('分钟','minute','minutes'),
 'h':('小时','hour','hours'),'℃':('摄氏度','degree Celsius','degrees Celsius')}
_QUANT=r'(?:个|条|台|份|项|次|人|件|张|元|页|套|本|组|items?\b|records?\b|requests?\b|devices?\b|people\b|times?\b|rows?\b)'
_QUANTITY=re.compile(_LEFT+r'(?P<n>'+_NUMBER+r')(?P<space>[ \t]*)(?P<q>'+_QUANT+r')',re.I)
_STOCK_LABEL=r'(?:股票代码|证券代码|股票编号|港股代码|股票代号|股份代号|stock[ \t]+code|ticker[ \t]+code)'
_STOCK=re.compile(r'(?P<label>'+_STOCK_LABEL+r'[ \t]*(?:(?:为|是|[:：#]|is|was)[ \t]*)?)(?P<code>[0-9]{1,6})(?P<market>\.(?:HK|SH|SZ))?(?![A-Za-z0-9_]|\.[0-9]|[/\\]|[ \t]*'+_QUANT+r')',re.I)
_BARE_STOCK=re.compile(r'(?<![A-Za-z0-9_./])(?P<code>[0-9]{1,6})(?P<market>\.(?:HK|SH|SZ))(?![A-Za-z0-9_./])')
_CODE=re.compile(r'(?P<label>(?<![A-Za-z0-9_])(?:错误码|错误代码|返回码|响应码|状态码|error[ \t]+code|status[ \t]+code|response[ \t]+code)[ \t]*(?:(?:为|是|[:：#]|is|was)[ \t]*)?)(?P<code>[0-9]{1,16})(?![A-Za-z0-9_]|\.[0-9]|[/\\]|[ \t]*'+_QUANT+r')',re.I)
_PORT=re.compile(r'(?P<label>(?:端口(?:号)?|(?<![A-Za-z0-9_])port(?:[ \t]+number)?\b)[ \t]*(?:(?:为|是|[:：#]|is|was)[ \t]*)?)(?P<code>[0-9]{1,5})(?![A-Za-z0-9_]|\.[0-9]|[/\\]|[ \t]*'+_QUANT+r')',re.I)
_IDENTIFIER=re.compile(r'(?P<label>编号[ \t]*(?:(?:为|是|[:：#])[ \t]*)?)(?P<code>[0-9]{1,16})(?![A-Za-z0-9_]|\.[0-9]|[/\\]|[ \t]*'+_QUANT+r')')


# Model numbers are identifiers, not counts. Limit this rule to four-digit
# RTX/GTX families; complete code, path, URL and filename masks still win.
_GPU_MODEL=re.compile(r'(?<![A-Za-z0-9_./\\-])(?P<family>RTX|GTX)[ \t]*(?P<model>[1-9][0-9]{3})(?P<suffix>[ \t]*(?:Ti|SUPER))?(?![A-Za-z0-9_/\\-]|\.[A-Za-z0-9_.])',re.I)
_MD_RIGHT=r'(?![A-Za-z0-9_/\\-]|\.[A-Za-z0-9_.])'
_MD_FIELD=re.compile(r'(?P<label>日期[ \t]*(?:(?:为|是|[:：])[ \t]*)?)(?P<m>[0-9]{1,2})/(?P<d>[0-9]{1,2})'+_MD_RIGHT)
_MD_EVENT=re.compile(r'(?P<label>(?:改到|定于|安排在|定在)[ \t]*)(?P<m>[0-9]{1,2})/(?P<d>[0-9]{1,2})'+_MD_RIGHT+r'(?=[ \t]*(?:举行|召开|开幕|开始|进行))')
_EN_MD_CALENDAR=re.compile(r'(?<![A-Za-z0-9_])(?:date[ \t]*(?:(?:is|was|[:=])[ \t]*)?|(?:scheduled[ \t]+(?:for|on)|(?:meeting|event|appointment)[ \t]+(?:is[ \t]+)?on)[ \t]+)[0-9]{1,2}/[0-9]{1,2}'+_MD_RIGHT,re.I)

def _language(chunk,language,fallback):
 if language in ('zh','en'):return language
 # Script in date/unit notation is not the carrier language of an English clause.
 probe=_ZHDATE.sub(' ',chunk);probe=_YM.sub(' ',probe);probe=_YEAR.sub(' ',probe)
 probe=re.sub(r'[0-9:.+\-−]+\s*(?:年|月|日|号|毫秒|秒|分钟|小时|元|个|条)',' ',probe)
 if re.search('[\u3400-\u9fff]',probe):return 'zh'
 if re.search('[A-Za-z]',probe):return 'en'
 return fallback

def _restore(text,mapping):
 # Later outer masks can contain an earlier URL mask, e.g. a URL in backticks.
 for token,value in reversed(list(mapping.items())):text=text.replace(token,value)
 return text

def _protect(text,general):
 mapping={}
 def stash_value(value):
  i=len(mapping)+1000;token=general._token(i)
  while token in text or token in mapping:i+=1;token=general._token(i)
  mapping[token]=value;return token
 def stash(match):return stash_value(match.group())
 # Whole URLs must precede the base's partial path/identifier masking.
 for pattern in [r'(?:https?://|ftp://|www\.)[^\s<>，。；！？]+',r'(?<![A-Za-z0-9_/])[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}(?![A-Za-z0-9_/])',r'(?<![A-Za-z0-9_])[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:.+Z\-]+']:
  text=re.sub(pattern,stash,text,flags=re.I)
 text,native=general.protect_identifiers(text)
 if set(mapping)&set(native):raise NumericNormalizationError('Identifier placeholder collision')
 mapping.update(native)
 # Retain the original V6's exact valid alphanumeric HTTP404 exception.
 for token,value in list(mapping.items()):
  if re.fullmatch(r'HTTP[1-5][0-9]{2}',value,re.I) or _GPU_MODEL.fullmatch(value):
   text=text.replace(token,value);del mapping[token]
 timer=re.compile(r'(?P<label>(?:倒计时|计时长度|持续时间|countdown|timer|duration)[^。！？!?；;\n0-9:]{0,16}?)(?P<value>[0-9]{1,2}:[0-9]{2})(?!:[0-9]|[0-9])',re.I)
 text=timer.sub(lambda m:m['label']+stash_value(m['value']),text)
 return text,mapping


def _clauses(text):
 thousands=[m.span()for m in re.finditer(r'[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?',text)]
 previous=0
 for m in re.finditer(r'[。！？!?；;，,\n]+|\.(?=\s|[\u3400-\u9fff]|$)',text):
  if any(a<=m.start()<b for a,b in thousands)or(text[m.start():m.end()]==','and m.start()>0 and m.end()<len(text)and text[m.start()-1].isdigit()and text[m.end()].isdigit()):continue
  yield text[previous:m.end()];previous=m.end()
 if previous<len(text):yield text[previous:]

def _number(general,value,lang,digits=False):
 # A code-like leading zero must never disappear via integer conversion.
 unsigned=value.lstrip('+-−').replace(',','').split('.')[0]
 return general.speak_number(value,lang,digits or(len(unsigned)>1 and unsigned.startswith('0')))

def _normalize_clause(chunk,lang,general,math):
 def word_boundary(m,value):
  # English typed replacements must not merge with an adjacent raw clock/token.
  if lang=='en':
   if m.start() and m.string[m.start()-1].isascii() and m.string[m.start()-1].isalnum():value=' '+value
   if m.end()<len(m.string) and m.string[m.end()].isascii() and m.string[m.end()].isalnum():value+=' '
  return value
 # These Chinese event/field structures establish month/day semantics.
 # English normalization has no MM/DD versus DD/MM locale contract.
 if _EN_MD_CALENDAR.search(chunk):
  raise NumericNormalizationError('Ambiguous slash calendar date: use an ISO date or an explicit month name')
 def month_day(m):
  if lang!='zh':raise NumericNormalizationError('Slash calendar date requires an explicit English month/day order')
  month,day=int(m['m']),int(m['d']);date(2000,month,day)
  return m['label']+general._zh_int(month)+'月'+general._zh_int(day)+'日'
 chunk=_MD_FIELD.sub(month_day,chunk);chunk=_MD_EVENT.sub(month_day,chunk)
 def gpu_model(m):
  code=m['model'];suffix=(' '+m['suffix'].strip())if m['suffix']else''
  if lang=='zh':reading=_number(general,code,lang,True)
  elif code.endswith('00'):reading=general._en_int(int(code))
  else:reading=general._en_int(int(code[:2]))+' '+general._en_int(int(code[2:]))
  return m['family']+' '+reading+suffix
 chunk=_GPU_MODEL.sub(gpu_model,chunk)
 def full_date(m):return word_boundary(m,general._read_date(m['y'],m['m'],m['d'],lang))
 chunk=_DATE.sub(full_date,chunk);chunk=_ZHDATE.sub(full_date,chunk)
 def ym(m):
  y,month=int(m['y']),int(m['m']);date(y,month,1)
  value=general._MONTHS[month]+' '+general._en_int(y)if lang=='en'else general.speak_number(m['y'],'zh',True)+'年'+general._zh_int(month)+'月'
  return word_boundary(m,value)
 chunk=_YM.sub(ym,chunk)
 chunk=_YEAR.sub(lambda m:word_boundary(m,general._en_int(int(m['y'])))if lang=='en'else general.speak_number(m['y'],'zh',True)+'年',chunk)
 # Explicit ongoing duration, not a deadline such as 持续到09:05. Require H:M:S.
 elapsed=re.compile(r'(?P<label>(?:预计[ \t]*)?持续[ \t]*(?:(?:约|大约|为|是|[:：])[ \t]*)?)(?P<h>[0-9]{1,2}):(?P<m>[0-9]{2}):(?P<s>[0-9]{2})(?![A-Za-z0-9_:])')
 def elapsed_read(m):
  h,minute,second=int(m['h']),int(m['m']),int(m['s'])
  if minute>59 or second>59:raise NumericNormalizationError('Invalid explicit elapsed time')
  if lang=='en':value=general._en_int(h)+' hours '+general._en_int(minute)+' minutes '+general._en_int(second)+' seconds'
  else:value=general._zh_int(h)+'小时'+general._zh_int(minute)+'分'+general._zh_int(second)+'秒'
  return m['label']+word_boundary(m,value)
 chunk=elapsed.sub(elapsed_read,chunk)
 def code(m):
  label=m['label'];value=m['code']
  if re.search(r'状态码|status[ \t]+code|响应码|response[ \t]+code',label,re.I)or re.search(r'\bHTTP\b',chunk,re.I):
   if not(re.fullmatch('[1-5][0-9]{2}',value)):return m.group()
  return label+(' 'if lang=='en'and label[-1:].isalnum()else'')+_number(general,value,lang,True)
 chunk=_STOCK.sub(lambda m:m['label']+(' 'if lang=='en'and m['label'][-1:].isalnum()else'')+_number(general,m['code'],lang,True)+(m['market']or''),chunk)
 chunk=_BARE_STOCK.sub(lambda m:_number(general,m['code'],lang,True)+m['market'],chunk)
 def port(m):
  if int(m['code'])>65535:raise NumericNormalizationError('Explicit port outside0..65535')
  return m['label']+(' 'if lang=='en'and m['label'][-1:].isalnum()else'')+_number(general,m['code'],lang,True)
 chunk=_PORT.sub(port,chunk);chunk=_CODE.sub(code,chunk)
 chunk=_IDENTIFIER.sub(lambda m:m['label']+(' 'if lang=='en'and m['label'][-1:].isascii()and m['label'][-1:].isalnum()else'')+_number(general,m['code'],lang,True),chunk)
 # A slash before an explicit cup/portion unit is a quantity fraction, not a date.
 fractions=re.compile(_LEFT+r'(?P<a>[0-9]{1,16})[ \t]*/[ \t]*(?P<b>[0-9]{1,16})(?P<space>[ \t]*)(?P<u>杯|份|cups?\b|portions?\b)')
 def fraction_read(m):
  denominator=int(m['b']);numerator=int(m['a'])
  if denominator==0:raise NumericNormalizationError('Quantity fraction denominator is zero')
  if lang=='zh':return general._zh_int(denominator)+'分之'+general._zh_int(numerator)+m['space']+m['u']
  names={2:('half','halves'),3:('third','thirds'),4:('quarter','quarters'),5:('fifth','fifths'),6:('sixth','sixths'),8:('eighth','eighths'),10:('tenth','tenths')}
  value=(general._en_int(numerator)+' '+names[denominator][numerator!=1])if denominator in names else general._en_int(numerator)+' over '+general._en_int(denominator)
  if re.match('[A-Za-z]',m['u']):return value+' of a '+m['u'].rstrip('s')
  return value+m['space']+m['u']
 chunk=fractions.sub(fraction_read,chunk)
 labels={**general._UNITS,**_EXTRA_UNITS};names='|'.join(re.escape(x)for x in sorted(labels,key=len,reverse=True))
 def unit_label(u,value,range_value=False):
  singular=not range_value and abs(Decimal(value.replace('−','-').replace(',','')))==1
  return labels[u][0 if lang=='zh'else 1 if singular else 2]
 ranges=re.compile(_LEFT+r'(?P<a>'+_NUMBER+r')\s*(?:–|—|~|～|至|到|to\b|-)\s*(?P<b>'+_NUMBER+r')\s*(?P<u>'+names+r'|[%％])(?![A-Za-z0-9_²³^])')
 def range_read(m):
  if general._math_adjacent(m.string,m.start(),m.end()):return m.group()
  a,b=_number(general,m['a'],lang),_number(general,m['b'],lang);u=m['u']
  if u in ['%','％']:return a+' to '+b+' percent'if lang=='en'else'百分之'+a+'到百分之'+b
  return a+(' to 'if lang=='en'else'到')+b+(' 'if lang=='en'else'')+unit_label(u,m['b'],True)
 chunk=ranges.sub(range_read,chunk)
 # A bare hyphen stays subtraction unless an explicit range/interval label disambiguates it.
 bare_range=re.compile(r'(?P<label>(?:范围|区间|(?<![A-Za-z0-9_])(?:range|interval)\b)[ \t]*(?:(?:为|是|[:：]|is|of|from)[ \t]*)?)(?P<a>'+_NUMBER+r')[ \t]*(?:–|—|~|～|至|到|to\b|-)[ \t]*(?P<b>'+_NUMBER+r')'+_RIGHT,re.I)
 chunk=bare_range.sub(lambda m:m['label']+_number(general,m['a'],lang)+(' to 'if lang=='en'else'到')+_number(general,m['b'],lang),chunk)
 units=re.compile(_LEFT+r'(?P<pm>±\s*)?(?P<n>'+_NUMBER+r')\s*(?P<u>'+names+r')(?![A-Za-z0-9_²³^])')
 def unit_read(m):
  if general._math_adjacent(m.string,m.start(),m.end()):return m.group()
  value=_number(general,m['n'],lang);value=(('plus or minus 'if lang=='en'else'正负')if m['pm']else'')+value
  return value+(' 'if lang=='en'else'')+unit_label(m['u'],m['n'])
 chunk=units.sub(unit_read,chunk)
 def quantity_read(m):
  if general._math_adjacent(m.string,m.start(),m.end()):return m.group()
  return _number(general,m['n'],lang)+(m['space']or(' 'if re.match('[A-Za-z]',m['q'])else''))+m['q']
 chunk=_QUANTITY.sub(quantity_read,chunk)
 # Explicit English normalization of a named Chinese scalar field. Do not
 # consume operands of formulas/ranges or change unlabelled integer/ID policy.
 if lang=='en':
  fields=r'(?:温度范围|数量|结果|温度|范围|延迟|速度|时长|值)'
  field_integer=re.compile(r'(?P<label>'+fields+r'[ \t]*(?:为|是|[:：])[ \t]*)(?P<n>[+\-−]?[0-9](?:[0-9,]*[0-9])?)'+_RIGHT)
  def named_integer(m):
   after=m.string[m.end('n'):].lstrip()[:1]
   if after in ('-','−','%','％') or (after and after in '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ₀₁₂₃₄₅₆₇₈₉₊₋') or general._math_adjacent(m.string,m.start('n'),m.end('n')):return m.group()
   return m['label']+word_boundary(m,_number(general,m['n'],lang))
  chunk=field_integer.sub(named_integer,chunk)
 # Typed bare decimal/signed values; formulas/tolerances/percentages stay with V6.
 numeric=re.compile(_LEFT+r'(?P<n>'+_NUMBER+r')'+_RIGHT)
 def simple(m):
  n=m['n'];before=m.string[:m.start()].rstrip()[-1:];after=m.string[m.end():].lstrip()[:1]
  # Preserve the complete raw subtraction before invoking the pinned parser.
  # The base adjacency set deliberately omits minus for ordinary signed values.
  arithmetic_minus=before in ('-','−') or after in ('-','−')
  signed_binary=n.startswith(('+','-','−')) and bool(before) and (before.isdigit() or before in (')',']'))
  if not('.'in n or n.startswith(('+','-','−')))or before=='±'or after in ['%','％','²','³']or arithmetic_minus or signed_binary or general._math_adjacent(m.string,m.start(),m.end()):return m.group()
  return _number(general,n,lang)
 chunk=numeric.sub(simple,chunk)
 return math.normalize_math(general.normalize_general(chunk,lang),lang)

def normalize_context(text,language='auto'):
 """Return COMPLETE normalized pre-G2P text (typed -> math -> NFKC once).

 Do not run the old normalize_general/math chain again on this return value.
 Invalid recognized typed values raise ValueError.

 Explicit zh/en wins. auto/mixed resolve a complete carrier clause, not the first
 GPU/API token. Identifier/URL literals are restored exactly after normalization.
 No G2P/model is loaded, and nothing is installed into the current frontend.
 """
 if not isinstance(text,str):raise TypeError('text must be str')
 if language not in ['auto','mixed','zh','en']:raise NumericNormalizationError('language must be auto/mixed/zh/en')
 if len(text)>20000:raise NumericNormalizationError('Input exceeds bounded20000 characters')
 general,math=_base();protected,mapping=_protect(text,general);fallback=_language(text,language,'zh');parts=[]
 try:
  for clause in _clauses(protected):
   carrier=_restore(clause,mapping);lang=_language(carrier,language,fallback);fallback=lang;parts.append(_normalize_clause(clause,lang,general,math))
 except ValueError as e:raise NumericNormalizationError(str(e))from e
 # Finish the current V6 static-normalization contract exactly once.
 result=unicodedata.normalize('NFKC',''.join(parts)).strip();result=re.sub(r'\s+',' ',result);result=result.replace('·',' ').replace('–','—')
 if re.search(r'[À-ÖØ-öø-ÿĀ-ſ]',result):raise NumericNormalizationError('Unsupported accented Latin; preserve/review explicitly')
 return _restore(result,mapping)

def provenance():
 _base();return {'candidate_only':True,'installed':False,'base_profile_path':str(FRONTEND/'frontend_base_profile.json'),'base_profile_sha256':PROFILE_SHA,'base_normalizers':'Pinned current V6 normalize_general+normalize_math with explicit per-clause carrier language, then NFKC/whitespace; original files/functions not modified','new_scope':['typed stock/error/port identifiers','year-month','case-sensitive digital rates and elapsed units','typed ranges and quantities','RTX/GTX four-digit model readings','explicit Chinese month/day calendar fields'],'ambiguous_dates_and_two_field_timers':'English slash-calendar dates rejected without locale; other protected literals retained','invalid_recognized_values':'ValueError','NN_or_G2P_loaded_by_this_module':False}
