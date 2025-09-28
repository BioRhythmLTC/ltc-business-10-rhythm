import ast
from collections import Counter
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# External deps used across utilities (lazy usage inside functions where possible)
import numpy as np  # used in metrics helpers
import torch  # needed for @torch.no_grad
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DebertaV2Tokenizer,
    DebertaV2TokenizerFast,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# Optional heavy dependencies (not required for production inference)
try:  # pandas is only used in helper utilities
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:  # sklearn metrics used in plotting/evaluation helpers
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # type: ignore
except Exception:  # pragma: no cover
    confusion_matrix = None  # type: ignore
    ConfusionMatrixDisplay = None  # type: ignore

try:  # plotting libraries not needed at runtime for API
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore
try:
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None  # type: ignore

# Basic type aliases and defaults used in annotations and inference
Span = Tuple[int, int, str]
MAX_SEQ_LEN = 128

# Preprocessing constants (transliteration, invisible spaces, script regexes)
# Confusable mappings (Latin<->Cyrillic)
LAT_TO_CYR = {
    'A':'А','a':'а','B':'В','b':'в','C':'С','c':'с','E':'Е','e':'е','H':'Н','h':'н',
    'K':'К','k':'к','M':'М','m':'м','O':'О','o':'о','P':'Р','p':'р','T':'Т','t':'т',
    'X':'Х','x':'х','Y':'У','y':'у'
}
CYR_TO_LAT = {v:k for k,v in LAT_TO_CYR.items()}
DIGIT_TO_LETTER = {'0':'o','1':'l','3':'e','5':'s'}

_lat = re.compile(r"[A-Za-z]")
_cyr = re.compile(r"[А-Яа-яЁё]")

INVISIBLE_SPACE_CHARS = {'\u00A0','\u200B','\u200C','\u200D','\ufeff'}

# Defaults for model and labels to avoid NameError in fallbacks
MODEL_NAME: str = os.getenv('X5_MODEL_NAME', 'DeepPavlov/rubert-base-cased')
LABELS: List[str] = [
    'O',
    'B-TYPE','I-TYPE',
    'B-BRAND','I-BRAND',
    'B-VOLUME','I-VOLUME',
    'B-PERCENT','I-PERCENT'
]
label2id: Dict[str, int] = {l: i for i, l in enumerate(LABELS)}
id2label: Dict[int, str] = {i: l for l, i in label2id.items()}

# 2) Path helpers

def get_runs_dir(model_alias: Optional[str] = None, run_id: Optional[str] = None) -> str:
    """Get runs dir.
    
    Args:
        model_alias: Parameter.
        run_id: Parameter.
    
    Returns:
        Return value.
    """
    alias = model_alias or MODEL_ALIAS
    rid = run_id or RUN_ID
    return os.path.join('.', 'ner_runs', alias, rid)


def get_artifacts_dir(model_alias: Optional[str] = None, run_id: Optional[str] = None) -> str:
    """Get artifacts dir.
    
    Args:
        model_alias: Parameter.
        run_id: Parameter.
    
    Returns:
        Return value.
    """
    alias = model_alias or MODEL_ALIAS
    rid = run_id or RUN_ID
    return os.path.join('.', 'artifacts', alias, rid)


def get_latest_alias_dir(model_alias: Optional[str] = None) -> str:
    """Get latest alias dir.
    
    Args:
        model_alias: Parameter.
    
    Returns:
        Return value.
    """
    alias = model_alias or MODEL_ALIAS
    return os.path.join('.', 'artifacts', alias, 'latest')


def get_submission_dir(model_alias: Optional[str] = None, run_id: Optional[str] = None) -> str:
    """Get submission dir.
    Args:
        model_alias: Parameter.
        run_id: Parameter.
    Returns:
        Return value.
    """
    alias = model_alias or MODEL_ALIAS
    rid = run_id or RUN_ID
    return os.path.join('.', 'submission', alias, rid)


def get_eval_dir(model_alias: Optional[str] = None, run_id: Optional[str] = None) -> str:
    """Get eval dir.
    
    Args:
        model_alias: Parameter.
        run_id: Parameter.
    
    Returns:
        Return value.
    """
    alias = model_alias or MODEL_ALIAS
    rid = run_id or RUN_ID
    return os.path.join('.', f'eval_out_{os.getenv("SERVICE_PORT", "8080")}', alias, rid)


# 3) Latest alias management

def _safe_symlink(target: str, link_name: str) -> None:
    """Safe symlink.
    
    Args:
        target: Parameter.
        link_name: Parameter.
    
    Returns:
        Return value.
    """
    try:
        if os.path.islink(link_name) or os.path.exists(link_name):
            try:
                os.remove(link_name)
            except Exception:
                pass
        os.symlink(target, link_name)
    except Exception:
        # Fallback: write a pointer file
        os.makedirs(os.path.dirname(link_name), exist_ok=True)
        with open(os.path.join(os.path.dirname(link_name), 'latest.txt'), 'w', encoding='utf-8') as f:
            f.write(os.path.basename(target))


def update_latest_alias(model_alias: Optional[str] = None, run_id: Optional[str] = None) -> None:
    """Update latest alias.
    
    Args:
        model_alias: Parameter.
        run_id: Parameter.
    
    Returns:
        Return value.
    """
    alias = model_alias or MODEL_ALIAS
    rid = run_id or RUN_ID
    link_path = get_latest_alias_dir(alias)
    target = get_artifacts_dir(alias, rid)
    os.makedirs(os.path.dirname(link_path), exist_ok=True)
    _safe_symlink(target=target, link_name=link_path)


def read_latest_run_id(model_alias: Optional[str] = None) -> Optional[str]:
    """Read latest run id.
    
    Args:
        model_alias: Parameter.
    
    Returns:
        Return value.
    """
    alias = model_alias or MODEL_ALIAS
    link_path = get_latest_alias_dir(alias)
    txt = os.path.join(os.path.dirname(link_path), 'latest.txt')
    if os.path.islink(link_path):
        try:
            tgt = os.readlink(link_path)
            return os.path.basename(tgt.rstrip('/'))
        except Exception:
            pass
    if os.path.exists(txt):
        try:
            with open(txt, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception:
            return None
    return None

# 4) Dynamic tokenizer loader (fast→slow fallback)
def load_tokenizer_dynamic(model_name: str, prefer_fast: bool = True):
    """
    Universal load of tokenizer.
    - mDeBERTa-v3 / DeBERTa-v2 → DebertaV2TokenizerFast (or DebertaV2Tokenizer fallback).
    - Остальные → AutoTokenizer.
    """

    ln = (model_name or "").lower()

    # 1) DeBERTa class (including mdeberta-v3)
    if "deberta" in ln:
        try:
            print(f"[Tokenizer] Using DebertaV2TokenizerFast for {model_name}")
            return DebertaV2TokenizerFast.from_pretrained(model_name)
        except Exception as e_fast:
            print(f"[Tokenizer] Fast tokenizer unavailable, fallback to DebertaV2Tokenizer: {e_fast}")
            return DebertaV2Tokenizer.from_pretrained(model_name)

    # 2) Other models (ruBERT-tiny, rubert-base-cased, etc.)
    try:
        print(f"[Tokenizer] Using AutoTokenizer ({'fast' if prefer_fast else 'slow'}) for {model_name}")
        return AutoTokenizer.from_pretrained(model_name, use_fast=prefer_fast, trust_remote_code=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer for {model_name}.\nOriginal error: {e}")

def parse_annotation(s):
    """Parse annotation.
    
    Args:
        s: Parameter.
    """
    try:
        return ast.literal_eval(s)
    except Exception:
        return []
        
def has_type(ann, t):
    """Has type.
    
    Args:
        ann: Parameter.
        t: Parameter.
    """
    return any((tag != 'O' and '-' in tag and tag.split('-',1)[1] == t) for _,_,tag in ann)

#preprocessing functions

def _is_latin(ch: str) -> bool:
    """Is latin.
    
    Args:
        ch: Parameter.
    
    Returns:
        Return value.
    """
    return bool(_lat.match(ch))

def _is_cyrillic(ch: str) -> bool:
    """Is cyrillic.
    
    Args:
        ch: Parameter.
    
    Returns:
        Return value.
    """
    return bool(_cyr.match(ch))


def _token_script_stats(tok: str) -> Tuple[int,int]:
    """Token script stats.
    
    Args:
        tok: Parameter.
    
    Returns:
        Return value.
    """
    lat = sum(1 for ch in tok if _is_latin(ch))
    cyr = sum(1 for ch in tok if _is_cyrillic(ch))
    return lat, cyr


def _normalize_mixed_script_token(tok: str) -> str:
    """Normalize mixed script token.
    
    Args:
        tok: Parameter.
    
    Returns:
        Return value.
    """
    lat, cyr = _token_script_stats(tok)
    if lat == 0 or cyr == 0:
        return tok
    if cyr >= lat:
        return ''.join(LAT_TO_CYR.get(ch, ch) for ch in tok)
    return ''.join(CYR_TO_LAT.get(ch, ch) for ch in tok)


def _normalize_digit_letter_confusables(tok: str) -> str:
    """Normalize digit letter confusables.
    
    Args:
        tok: Parameter.
    
    Returns:
        Return value.
    """
    if not tok:
        return tok
    chars = list(tok)
    n = len(chars)
    for i, ch in enumerate(chars):
        if i>0 and i+1<n and chars[i-1].isalpha() and chars[i+1].isalpha():
            if ch in DIGIT_TO_LETTER:
                chars[i] = DIGIT_TO_LETTER[ch]
    return ''.join(chars)


def _normalize_simple_subs(tok: str) -> str:
    """Normalize simple subs.
    
    Args:
        tok: Parameter.
    
    Returns:
        Return value.
    """
    out = tok.replace('ё', 'е')
    lat, cyr = _token_script_stats(out)
    if lat > cyr:
        out = out.replace('j', 'i')
    return out


def preprocess_text_with_mapping(text: str, do_lower: bool = True, apply_translit_map: bool = True) -> Tuple[str, List[int]]:
    # 1) Replace invisible spaces
    """Preprocess text with mapping.
    
    Args:
        text: Parameter.
        do_lower: Parameter.
        apply_translit_map: Parameter.
    
    Returns:
        Return value.
    """
    stage_chars: List[str] = []
    stage_map: List[int] = []
    for i, ch in enumerate(text):
        if ch in INVISIBLE_SPACE_CHARS:
            stage_chars.append(' ')
            stage_map.append(i)
        else:
            stage_chars.append(ch)
            stage_map.append(i)

    # 2) Insert space between Cyrillic/Latin boundaries
    out_chars: List[str] = []
    out_map: List[int] = []
    n = len(stage_chars)
    for i, ch in enumerate(stage_chars):
        out_chars.append(ch)
        out_map.append(stage_map[i])
        if i + 1 < n:
            a = ch
            b = stage_chars[i + 1]
            if a != ' ' and b != ' ' and ((_is_cyrillic(a) and _is_latin(b)) or (_is_latin(a) and _is_cyrillic(b))):
                out_chars.append(' ')
                out_map.append(stage_map[i + 1])

    out_text = ''.join(out_chars)

    # 3) Lowercase
    if do_lower:
        out_text = out_text.lower()

    # 4) Per-token general normalization
    tokens = out_text.split(' ')
    for ti, tok in enumerate(tokens):
        t1 = _normalize_mixed_script_token(tok)
        t2 = _normalize_digit_letter_confusables(t1)
        t3 = _normalize_simple_subs(t2)
        tokens[ti] = t3
    out_text = ' '.join(tokens)

    return out_text, out_map

def has_entity(row, t):
    """Has entity.
    
    Args:
        row: Parameter.
        t: Parameter.
    """
    types = set(tag.split('-',1)[1] for _,_,tag in row if tag!='O' and '-' in tag)
    return t in types

def extract_unit_candidates(texts: List[str]) -> Counter:
    """
    Extract candidate unit tokens that follow numeric values.
    Example: "500 ml" → "ml"
    """
    pattern_num_word = re.compile(r"\b\d+[\.,]?\d*\s*([A-Za-zА-Яа-яёЁ°]+)\b")
    counter = Counter()
    for s in texts:
        for m in pattern_num_word.finditer(s):
            counter[m.group(1).lower()] += 1
    return counter


def classify_units(unit_counter: Counter, whitelist: Dict[str, List[str]]) -> Tuple[Counter, Counter]:
    """
    Classify extracted unit candidates into known (from whitelist) and unknown.
    Returns:
        known (Counter): canonical units and their counts
        unknown (Counter): units not in whitelist
    """
    inv_white = {f: canon for canon, forms in whitelist.items() for f in forms}
    known, unknown = Counter(), Counter()
    for u, c in unit_counter.items():
        if u in inv_white:
            known[inv_white[u]] += c
        else:
            unknown[u] += c
    return known, unknown


def build_unit_patterns(whitelist: Dict[str, List[str]]):
    """
    Build regex patterns for volumes, multipacks, and percentages.
    Combines unit stems and whitelist forms.
    """
    unit_forms = [re.escape(f) for forms in whitelist.values() for f in forms]
    unit_stems_regex = [
        r"литр\w*", r"мл\.?", r"миллилитр\w*", r"l",
        r"кг\.?", r"килограмм\w*", r"kg",
        r"г(?:р\.?|рамм\w*)", r"g",
        r"шт\w*", r"pcs?", r"pc",
        r"бутыл\w*", r"бан\w*", r"пач\w*", r"упаков\w*", r"пак(?:\.|ет\w*)?",
        r"can", r"bottle", r"jar"
    ]
    unit_forms_regex = unit_stems_regex + unit_forms
    units_group = '(?:' + '|'.join(sorted(unit_forms_regex, key=len, reverse=True)) + ')'

    pattern_volume = re.compile(r"\b\d+[\.,]?\d*\s?" + units_group + r"\b", re.IGNORECASE)
    pattern_multipack = re.compile(r"\b\d+\s*[xх*]\s*\d+[\.,]?\d*\s?" + units_group + r"\b", re.IGNORECASE)
    pattern_percent = re.compile(r"\b\d+[\.,]?\d*\s?(?:%|процент(?:а|ов)?)\b", re.IGNORECASE)
    return pattern_volume, pattern_multipack, pattern_percent, units_group


def noise_stats(texts: List[str], units_group: str) -> Dict[str, int]:
    """
    Compute noise statistics:
      - repeated characters (aaa, !!!!)
      - mixed Cyrillic/Latin scripts inside tokens
      - missing space between number and unit (e.g., '500ml')
    """
    re_repeated = re.compile(r'(.)\1{2,}')
    re_mixed = re.compile(r'[а-я]{1,2}[a-z]|[a-z]{1,2}[а-я]', re.IGNORECASE)
    re_nospace_volume = re.compile(r"\d+[\.,]?\d*" + units_group + r"\b", re.IGNORECASE)

    counts = {'repeated': 0, 'mixed_scripts': 0, 'nospace_volume': 0}
    for s in texts:
        if re_repeated.search(s): counts['repeated'] += 1
        if re_mixed.search(s): counts['mixed_scripts'] += 1
        if re_nospace_volume.search(s): counts['nospace_volume'] += 1
    return counts
def has_volume_loose(s: str) -> bool:
    """Has volume loose.
    
    Args:
        s: Parameter.
    
    Returns:
        Return value.
    """
    return bool(pat_loose.search(s))

def has_volume_improved(s: str) -> bool:
    """Has volume improved.
    
    Args:
        s: Parameter.
    
    Returns:
        Return value.
    """
    return bool(pattern_volume.search(s) or pattern_multipack.search(s))
    
def types_in_ann(ann):
    """Types in ann.
    
    Args:
        ann: Parameter.
    """
    return set(tag.split('-',1)[1] for _,_,tag in ann if tag!='O' and '-' in tag)
# Extract contiguous spans from char-level BIO labels for a single string

def extract_spans_from_bio(text: str, bio_labels: List[str]) -> List[Span]:
    """Extract spans from bio.
    
    Args:
        text: Parameter.
        bio_labels: Parameter.
    
    Returns:
        Return value.
    """
    n = len(text)
    if len(bio_labels) != n:
        bio_labels = (bio_labels + ['O'] * n)[:n]
    spans: List[Span] = []
    current_start = None
    current_type = None
    for i, tag in enumerate(bio_labels):
        if tag == 'O' or tag is None:
            if current_start is not None:
                spans.append((current_start, i, current_type))
                current_start, current_type = None, None
            continue
        if '-' in tag:
            bio, etype = tag.split('-', 1)
        else:
            bio, etype = 'I', tag
        if bio == 'B':
            if current_start is not None:
                spans.append((current_start, i, current_type))
            current_start, current_type = i, etype
        elif bio == 'I':
            if current_start is None:
                current_start, current_type = i, etype
            elif etype != current_type:
                spans.append((current_start, i, current_type))
                current_start, current_type = i, etype
        else:
            if current_start is not None:
                spans.append((current_start, i, current_type))
                current_start, current_type = None, None
    if current_start is not None:
        spans.append((current_start, n, current_type))
    return spans

# Compute per-class P/R/F1 and macro-F1 given gold and pred span lists

def compute_macro_f1(golds: List[List[Span]], preds: List[List[Span]], labels: List[str] = ['TYPE','BRAND','VOLUME','PERCENT']) -> Dict:
    
    """Compute macro f1.
    
    Args:
        golds: Parameter.
        preds: Parameter.
        labels: Parameter.
    
    Returns:
        Return value.
    """
    assert len(golds) == len(preds)
    gold_by_type = {t:0 for t in labels}
    pred_by_type = {t:0 for t in labels}
    tp_by_type   = {t:0 for t in labels}

    def span_key(s: Span):
        """Span key.
        
        Args:
            s: Parameter.
        """
        return (s[0], s[1], s[2])

    for g_spans, p_spans in zip(golds, preds):
        gset = set(map(span_key, g_spans))
        pset = set(map(span_key, p_spans))
        for s in g_spans:
            if s[2] in gold_by_type:
                gold_by_type[s[2]] += 1
        for s in p_spans:
            if s[2] in pred_by_type:
                pred_by_type[s[2]] += 1
        for s in (gset & pset):
            if s[2] in tp_by_type:
                tp_by_type[s[2]] += 1

    report = {}
    f1s = []
    for t in labels:
        tp = tp_by_type[t]
        fp = pred_by_type[t] - tp
        fn = gold_by_type[t] - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        report[t] = {'precision': prec, 'recall': rec, 'f1': f1, 'support': gold_by_type[t]}
        f1s.append(f1)
    report['macro_f1'] = sum(f1s) / len(f1s) if f1s else 0.0
    return report
#Build char-level BIO from parsed spans

def to_char_bio(sample: str, spans: List[List[Any]]) -> List[str]:
    """To char bio.
    
    Args:
        sample: Parameter.
        spans: Parameter.
    
    Returns:
        Return value.
    """
    n = len(sample)
    labels = ['O'] * n
    for s, e, tag in spans:
        if tag == 'O':
            continue
        if '-' in tag:
            _, etype = tag.split('-', 1)
        else:
            etype = tag
        s = max(0, min(s, n))
        e = max(0, min(e, n))
        if e <= s:
            continue
        labels[s] = f'B-{etype}'
        for i in range(s+1, e):
            labels[i] = f'I-{etype}'
    return labels

# Align char BIO → token labels using offset_mapping
def align_bio_to_tokens(
    text: str,
    char_bio: List[str],
    tokenizer,
    label2id: Dict[str,int]
) -> Dict[str, Any]:
    """Align bio to tokens.
    
    Args:
        text: Parameter.
        char_bio: Parameter.
        tokenizer: Parameter.
        label2id: Parameter.
    
    Returns:
        Return value.
    """
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False
    )
    offsets = enc["offset_mapping"]

    token_labels: List[int] = []
    for idx, (start, end) in enumerate(offsets):
        if end == 0 and start == 0:  # спецтокены
            token_labels.append(-100)
            continue
        if start == end:  # пустые отрезки (пробелы)
            token_labels.append(-100)
            continue

        tok_label = "O"
        first_pos = None
        for pos in range(start, min(end, len(char_bio))):
            if char_bio[pos] != "O":
                tok_label = char_bio[pos]
                first_pos = pos
                break

        if tok_label == "O":
            token_labels.append(label2id["O"])
            continue

        bio, etype = tok_label.split("-", 1)
        if (
            first_pos is not None
            and first_pos > 0
            and char_bio[first_pos - 1].endswith(etype)
        ):
            tok_tag = f"I-{etype}"
        else:
            tok_tag = f"B-{etype}"
        token_labels.append(label2id.get(tok_tag, label2id["O"]))

    # сохраняем метки
    enc["labels"] = token_labels

    # сохраняем offset_mapping отдельно для compute_metrics
    enc["offset_mapping"] = offsets

    return enc
def presence_vector(ann):
    """Presence vector.
    
    Args:
        ann: Parameter.
    """
    ts = set(tag.split('-',1)[1] for _,_,tag in ann if tag!='O' and '-' in tag)
    return (
        int('TYPE' in ts),
        int('BRAND' in ts),
        int('VOLUME' in ts),
        int('PERCENT' in ts),
    )

# Codes: 0=neither, 1=only_TYPE, 2=only_BRAND, 3=TYPE+BRAND, 4=others (has VOLUME/PERCENT without T/B)
def to_strata(code):
    """To strata.
    
    Args:
        code: Parameter.
    """
    t,b,v,p = code
    if t==0 and b==0 and v==0 and p==0:
        return 0
    if t==1 and b==0:
        return 1
    if t==0 and b==1:
        return 2
    if t==1 and b==1:
        return 3
    return 4

def build_char_bio_col(df):
    """Build char bio col.
    
    Args:
        df: Parameter.
    """
    bios = []
    for s, ann in df[['sample','parsed_annotation']].itertuples(index=False):
        bios.append(to_char_bio(s, ann))
    return bios

def encode_row(row):
    """Encode row.
    
    Args:
        row: Parameter.
    """
    enc = align_bio_to_tokens(row['sample'], row['char_bio'], tokenizer, label2id)
    return {
        'input_ids': enc['input_ids'],
        'attention_mask': enc['attention_mask'],
        'labels': enc['labels']
    }
def _has_type_in_ann(ann, t):
    """Has type in ann.
    
    Args:
        ann: Parameter.
        t: Parameter.
    """
    return any(tag != 'O' and '-' in tag and tag.split('-', 1)[1] == t for _, _, tag in ann)

def decode_token_predictions(predictions, labels, offset_mappings):
    """Decode token predictions.
    
    Args:
        predictions: Parameter.
        labels: Parameter.
        offset_mappings: Parameter.
    """
    pred_ids = predictions.argmax(-1)
    batch_char_spans = []
    for i in range(pred_ids.shape[0]):
        token_tags = [
            id2label[int(tid)] if int(lbl) != -100 else 'O'
            for tid, lbl in zip(pred_ids[i], labels[i])
        ]
        offsets = offset_mappings[i]
        text_len = max([e for (s, e) in offsets if e is not None], default=0)
        char_bio = ['O'] * text_len
        for (s, e), tag, lbl in zip(offsets, token_tags, labels[i]):
            if int(lbl) == -100 or s is None or e is None or e <= s:
                continue
            if tag == 'O' or '-' not in tag:
                continue
            bio, etype = tag.split('-', 1)
            char_bio[s] = f'B-{etype}'
            for pos in range(s + 1, e):
                char_bio[pos] = f'I-{etype}'
        spans = extract_spans_from_bio(' ' * text_len, char_bio)
        batch_char_spans.append(spans)
    return batch_char_spans
def compute_metrics_fn(eval_pred):
    preds = eval_pred.predictions if hasattr(eval_pred, 'predictions') else eval_pred[0]
    labels = eval_pred.label_ids if hasattr(eval_pred, 'label_ids') else eval_pred[1]
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    pred_spans = decode_token_predictions(preds, labels, val_offset_mappings)

    # gold spans
    gold_char_spans = []
    for i, item in enumerate(val_ds):
        offsets = val_offset_mappings[i]
        text_len = max([e for (s, e) in offsets if e is not None], default=0)
        char_bio = ['O'] * text_len
        for (s, e), lbl in zip(offsets, item["labels"]):
            if int(lbl) == -100 or s is None or e is None or e <= s:
                continue
            tag = id2label[int(lbl)]
            if tag == 'O' or '-' not in tag:
                continue
            bio, etype = tag.split('-', 1)
            char_bio[s] = f'B-{etype}'
            for pos in range(s + 1, e):
                char_bio[pos] = f'I-{etype}'
        spans = extract_spans_from_bio(' ' * text_len, char_bio)
        gold_char_spans.append(spans)

    report = compute_macro_f1(
        gold_char_spans,
        pred_spans,
        labels=['TYPE', 'BRAND', 'VOLUME', 'PERCENT']
    )
    out = {'macro_f1': report['macro_f1']}
    for k in ['TYPE','BRAND','VOLUME','PERCENT']:
        out[f'f1_{k.lower()}'] = report[k]['f1']
        out[f'prec_{k.lower()}'] = report[k]['precision']
        out[f'rec_{k.lower()}'] = report[k]['recall']
    return out
# Path resolver for inference artifacts

def resolve_model_dir(model: str = None, run_id: str = None, prefer: str = 'runs_best') -> str:
    """Resolve model dir.
    
    Args:
        model: Parameter.
        run_id: Parameter.
        prefer: Parameter.
    
    Returns:
        Return value.
    """
    alias = model or MODEL_ALIAS
    rid = run_id or (read_latest_run_id(alias) or RUN_ID)
    art = get_artifacts_dir(alias, rid)
    runs_best = os.path.join(get_runs_dir(alias, rid), 'best_model')
    # Selection order: prefer best checkpoint by default, else artifacts, else latest alias
    if prefer == 'runs_best' and os.path.exists(runs_best):
        chosen = runs_best
    elif prefer == 'artifacts' and os.path.exists(art):
        chosen = art
    elif os.path.exists(runs_best):
        chosen = runs_best
    elif os.path.exists(art):
        chosen = art
    else:
        latest = get_latest_alias_dir(alias)
        chosen = latest if os.path.exists(latest) else art
    print(f"[ModelSelect] alias={alias} run_id={rid} prefer={prefer} -> {chosen}")
    return chosen

# Fallback: span extractor if not defined in this kernel
if 'extract_spans_from_bio' not in globals():
    SpanT = Tuple[int, int, str]
    def extract_spans_from_bio(text: str, bio_labels: List[str]) -> List[SpanT]:
        """Extract spans from bio.
        
        Args:
            text: Parameter.
            bio_labels: Parameter.
        
        Returns:
            Return value.
        """
        n = len(text)
        if len(bio_labels) != n:
            bio_labels = (bio_labels + ['O'] * n)[:n]
        spans: List[SpanT] = []
        start = None
        etype = None
        for i, tag in enumerate(bio_labels):
            if tag == 'O' or tag is None:
                if start is not None:
                    spans.append((start, i, etype))
                    start, etype = None, None
                continue
            if '-' in tag:
                bio, t = tag.split('-', 1)
            else:
                bio, t = 'I', tag
            if bio == 'B':
                if start is not None:
                    spans.append((start, i, etype))
                start, etype = i, t
            elif bio == 'I':
                if start is None:
                    start, etype = i, t
                elif t != etype:
                    spans.append((start, i, etype))
                    start, etype = i, t
        if start is not None:
            spans.append((start, n, etype))
        return spans

# Token BIO -> char BIO -> spans

def decode_token_tags_to_char_spans(text: str, token_tags: List[str], offsets: List[Tuple[int,int]]):
    """Decode token tags to char spans.
    
    Args:
        text: Parameter.
        token_tags: Parameter.
        offsets: Parameter.
    """
    text_len = max([e for (s,e) in offsets if e is not None], default=0)
    char_bio = ['O'] * text_len
    for (s, e), tag in zip(offsets, token_tags):
        if s is None or e is None or e <= s:
            continue
        if tag == 'O' or '-' not in tag:
            continue
        bio, etype = tag.split('-', 1)
        char_bio[s] = f'B-{etype}'
        for pos in range(s+1, e):
            char_bio[pos] = f'I-{etype}'
    spans = extract_spans_from_bio(text[:text_len], char_bio)
    return spans

# Build char-level BIO from token-level tags and offsets

def token_tags_to_char_bio(text: str, token_tags: List[str], offsets: List[Tuple[int,int]]) -> List[str]:
    """Token tags to char bio.
    
    Args:
        text: Parameter.
        token_tags: Parameter.
        offsets: Parameter.
    
    Returns:
        Return value.
    """
    text_len = max([e for (s,e) in offsets if e is not None], default=0)
    char_bio = ['O'] * text_len
    for (s, e), tag in zip(offsets, token_tags):
        if s is None or e is None or e <= s:
            continue
        if tag == 'O' or '-' not in tag:
            continue
        bio, etype = tag.split('-', 1)
        char_bio[s] = f'B-{etype}'
        for pos in range(s+1, e):
            char_bio[pos] = f'I-{etype}'
    return char_bio

# Convert char-level BIO to API spans with word-boundary B-/I- segments

def find_word_spans(text: str) -> List[Tuple[int,int]]:
    """Find word spans.
    
    Args:
        text: Parameter.
    
    Returns:
        Return value.
    """
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def char_bio_to_api_spans(text: str, char_bio: List[str]) -> List[Dict[str, Any]]:
    """Char bio to api spans.
    
    Args:
        text: Parameter.
        char_bio: Parameter.
    
    Returns:
        Return value.
    """
    n = len(char_bio)
    api: List[Dict[str, Any]] = []
    i = 0
    while i < n:
        if char_bio[i] == 'O' or char_bio[i] is None:
            i += 1
            continue
        # start of entity region
        _, etype = char_bio[i].split('-', 1)
        start = i
        i += 1
        while i < n and (char_bio[i] != 'O') and char_bio[i].endswith(etype):
            i += 1
        end = i  # [start,end) region for this entity type
        # split region by word spans and assign B- for first word, I- for the rest
        words = find_word_spans(text)
        inside = [(ws,we) for (ws,we) in words if not (we <= start or ws >= end)]
        for j, (ws,we) in enumerate(inside):
            tag = ('B-' if j == 0 else 'I-') + etype
            api.append({'start_index': ws, 'end_index': we, 'entity': tag})
    return api

# Loader: prefer resolved artifacts/checkpoint

def load_for_inference(model_dir: str = None, model: str = None, run_id: str = None):
    """Load for inference.
    
    Args:
        model_dir: Parameter.
        model: Parameter.
        run_id: Parameter.
    """
    target = model_dir or resolve_model_dir(model=model, run_id=run_id, prefer='runs_best')
    print(f"[Load] Using model directory: {target}")
    try:
        tok = load_tokenizer_dynamic(target)
    except Exception as e_tok:
        print(f"[Load] Tokenizer load failed for {target}: {e_tok}. Falling back to base {MODEL_NAME}")
        tok = load_tokenizer_dynamic(MODEL_NAME)
    try:
        mdl = AutoModelForTokenClassification.from_pretrained(target)
    except Exception as e_m:
        print(f"[Load] Model load failed for {target}: {e_m}. Falling back to base {MODEL_NAME}")
        mdl = AutoModelForTokenClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(LABELS),
            id2label=id2label,
            label2id=label2id
        )
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    #     device = 'mps'
    mdl = mdl.to(device).eval()
    return mdl, tok, device
def predict_one(text: str, model_dir: str = None, model: str = None, run_id: str = None) -> Dict[str, Any]:
    mdl, tok, device = load_for_inference(model_dir=model_dir, model=model, run_id=run_id)

    # 1) Preprocess input text (confusables, digit-letter fixes, lowercasing, script-boundary spacing)
    norm_text, idx_map = preprocess_text_with_mapping(text, do_lower=True, apply_translit_map=True)

    # 2) Tokenize normalized text
    enc = tok(norm_text, return_offsets_mapping=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors='pt')
    inp = {k: v.to(device) for k, v in enc.items() if k != 'offset_mapping'}
    with torch.no_grad():
        logits = mdl(**inp).logits
    pred_ids = logits.argmax(-1)[0].tolist()

    def id2(i: int) -> str:
        cfg_map = getattr(mdl.config, 'id2label', None)
        if isinstance(cfg_map, dict):
            return cfg_map.get(i, cfg_map.get(str(i), id2label.get(i, 'O') if 'id2label' in globals() else 'O'))
        return id2label.get(i, 'O') if 'id2label' in globals() else 'O'

    token_tags = [id2(i) for i in pred_ids]
    offsets_norm = enc['offset_mapping'][0].tolist()

    # 3) Decode spans on normalized text and map back to original coordinates
    spans_norm = decode_token_tags_to_char_spans(norm_text, token_tags, offsets_norm)
    spans_orig = map_spans_to_original(spans_norm, idx_map)

    # 4) Build char BIO on original text and API spans
    char_bio_orig = spans_to_char_bio(len(text), spans_orig)
    api_spans = char_bio_to_api_spans(text, char_bio_orig)

    return {
        'text': text,
        'normalized_text': norm_text,
        'token_tags': token_tags,
        'offsets': offsets_norm,  # offsets in normalized text space
        'char_bio': char_bio_orig,
        'spans': spans_orig,
        'api_spans': api_spans
    }

# Preloaded-variant: reuse given model/tokenizer/device without reloading
def spans_to_business(text: str, spans: List[Tuple[int,int,str]]) -> Dict[str, Any]:
    """Spans to business.
    
    Args:
        text: Parameter.
        spans: Parameter.
    
    Returns:
        Return value.
    """
    out: Dict[str, Any] = {'TYPE': None, 'BRAND': None, 'VOLUME': None, 'PERCENT': None}
    first = {}
    for s, e, t in spans:
        if t not in first:
            first[t] = (s, e)
    if 'TYPE' in first:
        s, e = first['TYPE']; out['TYPE'] = text[s:e]
    if 'BRAND' in first:
        s, e = first['BRAND']; out['BRAND'] = text[s:e]
    vol = None
    if 'VOLUME' in first:
        s, e = first['VOLUME']; seg = text[s:e]
        m = re.search(r"(\d+[\.,]?\d*)", seg)
        if m:
            value = float(m.group(1).replace(',', '.'))
            if re.search(r"миллилитр|мл\.?|ml", seg, re.I): unit = 'ML'
            elif re.search(r"литр|л\.?|l", seg, re.I): unit = 'L'
            elif re.search(r"килограмм|кг\.?|kg", seg, re.I): unit = 'KG'
            elif re.search(r"г(?:р\.?|рамм)", seg, re.I): unit = 'G'
            elif re.search(r"шт\w*|pcs?\b|pc\b|бутыл|бан", seg, re.I): unit = 'PCS'
            else: unit = None
            vol = {'value': value, 'unit': unit, 'raw': seg}
    out['VOLUME'] = vol
    perc = None
    if 'PERCENT' in first:
        s, e = first['PERCENT']; seg = text[s:e]
        m = re.search(r"(\d+[\.,]?\d*)", seg)
        if m: perc = float(m.group(1).replace(',', '.'))
    else:
        if re.search(r"безалкогольн\w*", text, re.I): perc = 0.0
    out['PERCENT'] = perc
    return out
# Helper functions for span mapping and BIO construction (used by predict_one)
def map_spans_to_original(spans: List[Tuple[int,int,str]], norm_to_orig: List[int]) -> List[Tuple[int,int,str]]:
    """Map spans to original.
    
    Args:
        spans: Parameter.
        norm_to_orig: Parameter.
    
    Returns:
        Return value.
    """
    mapped: List[Tuple[int,int,str]] = []
    L = len(norm_to_orig)
    for s, e, t in spans:
        if s < 0 or e <= 0 or s >= L:
            continue
        s0 = norm_to_orig[s]
        e0 = norm_to_orig[min(e - 1, L - 1)] + 1
        mapped.append((s0, e0, t))
    return mapped

def spans_to_char_bio(text_len: int, spans: List[Tuple[int,int,str]]) -> List[str]:
    """Spans to char bio.
    
    Args:
        text_len: Parameter.
        spans: Parameter.
    
    Returns:
        Return value.
    """
    bio = ['O'] * text_len
    for s, e, t in spans:
        if not (0 <= s < e <= text_len):
            continue
        bio[s] = f'B-{t}'
        for i in range(s + 1, e):
            bio[i] = f'I-{t}'
    return bio


def merge_adjacent_spans(text: str, spans: List[Span], max_gap: int = 1) -> List[Span]:
    """Merge adjacent spans.
    
    Args:
        text: Parameter.
        spans: Parameter.
        max_gap: Parameter.
    
    Returns:
        Return value.
    """
    spans_sorted = sorted(spans, key=lambda x: (x[2], x[0], x[1]))
    merged: List[Span] = []
    for s, e, t in sorted(spans, key=lambda x: (x[0], x[1])):
        if merged and merged[-1][2] == t:
            ps, pe, pt = merged[-1]
            gap = s - pe
            between = text[pe:s] if 0 <= pe <= s <= len(text) else ''
            if gap <= max_gap and (between.strip() == '' or between in ['-', '–', '.', ',']):
                merged[-1] = (ps, max(pe, e), pt)
                continue
        merged.append((s, e, t))
    return merged

def normalize_unit_stem(seg: str) -> str:
    """Normalize unit stem.
    
    Args:
        seg: Parameter.
    
    Returns:
        Return value.
    """
    if re.search(r"миллилитр|\bмл\.?\b|\bml\b", seg, re.I):
        return 'ML'
    if re.search(r"литр|\bл\.?\b|\bl\b", seg, re.I):
        return 'L'
    if re.search(r"килограмм|\bкг\.?\b|\bkg\b", seg, re.I):
        return 'KG'
    if re.search(r"\bг(?:р\.?|рамм)\b|\bg\b", seg, re.I):
        return 'G'
    if re.search(r"шт\w*|\bpcs?\b|\bpc\b|бутыл|бан", seg, re.I):
        return 'PCS'
    return None

# Override business conversion with merging and rules

def spans_to_business(text: str, spans: List[Span]) -> Dict[str, Any]:
    """Spans to business.
    
    Args:
        text: Parameter.
        spans: Parameter.
    
    Returns:
        Return value.
    """
    merged = merge_adjacent_spans(text, spans)
    out: Dict[str, Any] = {'TYPE': None, 'BRAND': None, 'VOLUME': None, 'PERCENT': None}

    # TYPE: drop 1-char spans, keep first (or choose longest if none after filter)
    type_spans = [(s,e) for s,e,t in merged if t=='TYPE']
    type_spans = [(s,e) for (s,e) in type_spans if (e-s) >= 2] or type_spans
    if type_spans:
        s,e = max(type_spans, key=lambda x: (x[1]-x[0]))  # prefer longer
        out['TYPE'] = text[s:e]

    # BRAND: after merge, usually contiguous; choose longest
    brand_spans = [(s,e) for s,e,t in merged if t=='BRAND']
    if brand_spans:
        s,e = max(brand_spans, key=lambda x: (x[1]-x[0]))
        out['BRAND'] = text[s:e]

    # VOLUME: choose the longest/most informative span, then parse value/unit
    vol_spans = [(s,e) for s,e,t in merged if t=='VOLUME']
    if vol_spans:
        s,e = max(vol_spans, key=lambda x: (x[1]-x[0]))
        seg = text[s:e]
        m = re.search(r"(\d+(?:[\.,]\d+)?)", seg)
        if m:
            value = float(m.group(1).replace(',', '.'))
            unit = normalize_unit_stem(seg)
            out['VOLUME'] = {'value': value, 'unit': unit, 'raw': seg}

    # PERCENT: parse numeric percent/° if present, else optional 'безалкогольн*' -> 0.0
    pct_spans = [(s,e) for s,e,t in merged if t=='PERCENT']
    if pct_spans:
        s,e = max(pct_spans, key=lambda x: (x[1]-x[0]))
        seg = text[s:e]
        m = re.search(r"(\d+(?:[\.,]\d+)?)", seg)
        if m:
            out['PERCENT'] = float(m.group(1).replace(',', '.'))
    else:
        if re.search(r"безалкогольн\w*", text, re.I):
            out['PERCENT'] = 0.0

    return out


def spans_to_api_spans(text: str, spans: List[Tuple[int,int,str]], include_O: bool = False) -> List[Dict[str, Any]]:
    """Spans to api spans.
    
    Args:
        text: Parameter.
        spans: Parameter.
        include_O: Parameter.
    
    Returns:
        Return value.
    """
    merged = merge_adjacent_spans(text, spans)
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    word_tag_map: Dict[Tuple[int,int], str] = {}
    for s, e, t in merged:
        inside = [(ws,we) for (ws,we) in words if not (we <= s or ws >= e)]
        for j, (ws,we) in enumerate(inside):
            tag = ('B-' if j == 0 else 'I-') + t
            word_tag_map[(ws,we)] = tag
    api: List[Dict[str, Any]] = []
    for ws, we in words:
        tag = word_tag_map.get((ws,we))
        if tag:
            api.append({'start_index': ws, 'end_index': we, 'entity': tag})
        elif include_O:
            api.append({'start_index': ws, 'end_index': we, 'entity': 'O'})
    return api

# Helper to run full pipeline with post-processing
@torch.no_grad()
def predict_one_pp(text: str, model_dir: str = None, model: str = None, run_id: str = None) -> Dict[str, Any]:
    """Predict one pp.
    
    Args:
        text: Parameter.
        model_dir: Parameter.
        model: Parameter.
        run_id: Parameter.
    
    Returns:
        Return value.
    """
    r = predict_one(text, model_dir=model_dir, model=model, run_id=run_id)
    r['spans_merged'] = merge_adjacent_spans(text, r['spans'])
    r['api_spans'] = spans_to_api_spans(text, r['spans'], include_O=True)
    r['business'] = spans_to_business(text, r['spans'])
    return r

# Preloaded-variant using already initialized model/tokenizer/device (no reload per call)
@torch.no_grad()
def predict_one_pp_preloaded(
    text: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Predict one sample using preloaded model/tokenizer/device.
    Returns the same structure as predict_one_pp.
    """
    # 1) Preprocess
    norm_text, idx_map = preprocess_text_with_mapping(text, do_lower=True, apply_translit_map=True)

    # 2) Tokenize
    enc = tokenizer(norm_text, return_offsets_mapping=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors='pt')
    inp = {k: v.to(device) for k, v in enc.items() if k != 'offset_mapping'}
    logits = model(**inp).logits
    pred_ids = logits.argmax(-1)[0].tolist()

    # id2label mapping preference: from model.config when available
    def _id2(i: int) -> str:
        cfg_map = getattr(model.config, 'id2label', None)
        if isinstance(cfg_map, dict):
            return cfg_map.get(i, cfg_map.get(str(i), id2label.get(i, 'O') if 'id2label' in globals() else 'O'))
        return id2label.get(i, 'O') if 'id2label' in globals() else 'O'

    token_tags = [_id2(i) for i in pred_ids]
    offsets_norm = enc['offset_mapping'][0].tolist()

    # 3) Decode spans on normalized text and map back to original
    spans_norm = decode_token_tags_to_char_spans(norm_text, token_tags, offsets_norm)
    spans_orig = map_spans_to_original(spans_norm, idx_map)

    # 4) Build outputs analogous to predict_one_pp
    char_bio_orig = spans_to_char_bio(len(text), spans_orig)
    api_spans = spans_to_api_spans(text, spans_orig, include_O=True)

    return {
        'text': text,
        'normalized_text': norm_text,
        'token_tags': token_tags,
        'offsets': offsets_norm,
        'char_bio': char_bio_orig,
        'spans': spans_orig,
        'spans_merged': merge_adjacent_spans(text, spans_orig),
        'api_spans': api_spans,
        'business': spans_to_business(text, spans_orig),
    }

@torch.no_grad()
def predict_batch_pp_preloaded(
    texts: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """Batched variant of predict_one_pp_preloaded for multiple texts."""
    if not texts:
        return []

    # 1) Preprocess all texts
    norm_texts: List[str] = []
    idx_maps: List[List[int]] = []
    for t in texts:
        nt, imap = preprocess_text_with_mapping(t, do_lower=True, apply_translit_map=True)
        norm_texts.append(nt)
        idx_maps.append(imap)

    # 2) Tokenize as a batch
    enc = tokenizer(
        norm_texts,
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=True,
        return_tensors='pt'
    )
    inp = {k: v.to(device) for k, v in enc.items() if k != 'offset_mapping'}
    logits = model(**inp).logits  # [B, T, C]
    pred_ids_batch = logits.argmax(-1).tolist()

    def _id2(i: int) -> str:
        cfg_map = getattr(model.config, 'id2label', None)
        if isinstance(cfg_map, dict):
            return cfg_map.get(i, cfg_map.get(str(i), id2label.get(i, 'O') if 'id2label' in globals() else 'O'))
        return id2label.get(i, 'O') if 'id2label' in globals() else 'O'

    results: List[Dict[str, Any]] = []
    offsets_all = enc['offset_mapping'].tolist()
    for i, (text, nt, imap, pred_ids, offsets) in enumerate(zip(texts, norm_texts, idx_maps, pred_ids_batch, offsets_all)):
        token_tags = [_id2(int(tid)) for tid in pred_ids]
        spans_norm = decode_token_tags_to_char_spans(nt, token_tags, offsets)
        spans_orig = map_spans_to_original(spans_norm, imap)
        char_bio_orig = spans_to_char_bio(len(text), spans_orig)
        api_spans = spans_to_api_spans(text, spans_orig, include_O=True)
        results.append({
            'text': text,
            'normalized_text': nt,
            'token_tags': token_tags,
            'offsets': offsets,
            'char_bio': char_bio_orig,
            'spans': spans_orig,
            'spans_merged': merge_adjacent_spans(text, spans_orig),
            'api_spans': api_spans,
            'business': spans_to_business(text, spans_orig),
        })

    return results

# Preloaded-variant with post-processing
def evaluate_on_validation(val_split, model, tokenizer, label2id, id2label):
    """Run evaluation on validation split with detailed error analysis."""
    from datasets import Dataset
    from collections import Counter

    # Ensure val_split has char_bio
    if 'char_bio' not in val_split.columns:
        val_split = val_split.copy()
        val_split['char_bio'] = [
            to_char_bio(s, ann) for s, ann in val_split[['sample','parsed_annotation']].itertuples(index=False)
        ]

    # Encode rows
    def _encode_row(row):
        enc = align_bio_to_tokens(row['sample'], row['char_bio'], tokenizer, label2id)
        return {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask'], 'labels': enc['labels']}

    val_ds = Dataset.from_pandas(val_split[['sample','char_bio']])
    val_ds = val_ds.map(_encode_row, remove_columns=val_ds.column_names)
    val_ds.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

    # Offset mappings
    val_offset_mappings = [
        align_bio_to_tokens(s, bio, tokenizer, label2id)['offset_mapping']
        for s, bio in val_split[['sample','char_bio']].itertuples(index=False)
    ]

    # Predict spans
    pred_spans_val = predict_val_spans(model, tokenizer, val_ds, val_offset_mappings)

    # Gold spans
    gold_spans_val = []
    for item, offsets in zip(val_ds, val_offset_mappings):
        text_len = max([e for (s,e) in offsets if e is not None], default=0)
        char_bio = ['O'] * text_len
        for (s,e), lbl in zip(offsets, item['labels']):
            if int(lbl) == -100 or s is None or e is None or e <= s:
                continue
            tag = id2label[int(lbl)]
            if tag == 'O' or '-' not in tag:
                continue
            bio, etype = tag.split('-', 1)
            char_bio[s] = f'B-{etype}'
            for pos in range(s+1, e):
                char_bio[pos] = f'I-{etype}'
        spans = extract_spans_from_bio(' ' * text_len, char_bio)
        gold_spans_val.append(spans)

    # Compute metrics
    report = compute_macro_f1(gold_spans_val, pred_spans_val, labels=['TYPE','BRAND','VOLUME','PERCENT'])
    print('Validation Macro-F1:', f"{report['macro_f1']:.4f}")
    for k in ['TYPE','BRAND','VOLUME','PERCENT']:
        r = report[k]
        print(f"{k:7s}  P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}  support={r['support']}")

    # Error analysis
    def to_set(spans): return set(spans)
    boundary_errors, type_confusions, missed, false_alarms = [], Counter(), Counter(), Counter()
    for i, (g, p) in enumerate(zip(gold_spans_val, pred_spans_val)):
        gset, pset = to_set(g), to_set(p)
        inter = gset & pset
        for s in gset - inter: missed[s[2]] += 1
        for s in pset - inter: false_alarms[s[2]] += 1
        for gs in g:
            for ps in p:
                if gs[2] == ps[2] and not (gs[1] <= ps[0] or ps[1] <= gs[0]) and (gs != ps):
                    boundary_errors.append((i, gs, ps))

    print('\nTop misses:', missed.most_common())
    print('Top false alarms:', false_alarms.most_common())
    print('\nBoundary errors (first 5):')

    if not boundary_errors:
        print("No boundary errors detected.")
    else:
        for i, gs, ps in boundary_errors[:5]:
            print('---')
            print('Text:', val_split.iloc[i]['sample'])
            print('Gold:', gs)
            print('Pred:', ps)

    return report, gold_spans_val, pred_spans_val
def plot_confusion_matrix(gold_spans, pred_spans, labels=None, title="Confusion Matrix (entity-level)"):
    """
    Build and plot confusion matrix at entity-level.
    
    Args:
        gold_spans: List of gold span lists (each span: (start, end, type))
        pred_spans: List of predicted span lists (same format as gold)
        labels: List of entity labels including 'O'
        title: Plot title
    """
    if confusion_matrix is None or ConfusionMatrixDisplay is None or plt is None:
        raise RuntimeError(
            "plot_confusion_matrix requires scikit-learn and matplotlib; install requirements-dev to use it"
        )
    if labels is None:
        labels = ['TYPE', 'BRAND', 'VOLUME', 'PERCENT', 'O']
    
    y_true, y_pred = [], []

    for g_spans, p_spans in zip(gold_spans, pred_spans):
        g_dict = {(s, e): t for s, e, t in g_spans}
        p_dict = {(s, e): t for s, e, t in p_spans}
        all_keys = set(g_dict.keys()) | set(p_dict.keys())
        for k in all_keys:
            y_true.append(g_dict.get(k, 'O'))
            y_pred.append(p_dict.get(k, 'O'))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(title)
    plt.show()
def token_level_classification_report_v2(model, dataset, id2label, exclude_O=False):
    """
    Compute token-level classification report for BIO labels with correct 'O' handling.
    Prints per-label precision/recall/F1/support and macro-F1 over selected labels.

    Args:
        model: HF token classification model (with .device and .eval()).
        dataset: iterable of dicts with 'input_ids', 'attention_mask', 'labels' (with -100 masked positions).
        id2label: dict[int,str] mapping class ids to label strings.
        exclude_O: if True, exclude 'O' from metrics and macro-F1.
    Returns:
        per_label: dict[label, dict(tp,fp,fn)] with counts for selected labels.
    """
    if confusion_matrix is None:
        raise RuntimeError(
            "token_level_classification_report_v2 requires scikit-learn; install requirements-dev to use it"
        )

    # Build ordered label list from id2label to avoid non-deterministic dict.values() order
    try:
        label_ids = sorted(int(i) for i in id2label.keys())
    except Exception:
        label_ids = sorted(id2label.keys())
    all_labels = [id2label[i] for i in label_ids]

    if exclude_O:
        target_labels = [l for l in all_labels if l != 'O']
    else:
        target_labels = list(all_labels)

    target_label_set = set(target_labels)

    @torch.no_grad()
    def collect_preds_golds(model, dataset):
        model.eval()
        all_pred, all_gold = [], []
        for item in dataset:
            input_ids = item['input_ids'].unsqueeze(0).to(model.device)
            attention_mask = item['attention_mask'].unsqueeze(0).to(model.device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            pred_ids = logits.argmax(-1)[0].detach().cpu().numpy().tolist()
            gold_ids = item['labels'].detach().cpu().numpy().tolist()
            for p, g in zip(pred_ids, gold_ids):
                if g == -100:  # ignore special/padding labels
                    continue
                all_pred.append(id2label[int(p)])
                all_gold.append(id2label[int(g)])
        return all_pred, all_gold

    pred_labels, gold_labels = collect_preds_golds(model, dataset)

    # Initialize counters only for target labels
    per_label = {l: {'tp': 0, 'fp': 0, 'fn': 0} for l in target_labels}

    for p_label, g_label in zip(pred_labels, gold_labels):
        if g_label == 'O':
            if 'O' in target_label_set:
                if p_label == 'O':
                    per_label['O']['tp'] += 1
                else:
                    per_label['O']['fn'] += 1
                    if p_label in target_label_set:
                        per_label[p_label]['fp'] += 1
            else:
                if p_label in target_label_set:
                    per_label[p_label]['fp'] += 1
            continue

        # Here gold is a non-'O' label
        if g_label not in target_label_set:
            continue

        if p_label == g_label:
            per_label[g_label]['tp'] += 1
        else:
            per_label[g_label]['fn'] += 1
            if p_label in target_label_set:
                per_label[p_label]['fp'] += 1

    # Print report
    print('Token-level BIO metrics (exclude O):' if exclude_O else 'Token-level BIO metrics:')
    f1s = []
    for l in target_labels:
        tp = per_label[l]['tp']
        fp = per_label[l]['fp']
        fn = per_label[l]['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
        support = tp + fn
        print(f"{l:10s} P={precision:.3f} R={recall:.3f} F1={f1:.3f}  support={support}")

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    print('BIO macro-F1:', f"{macro_f1:.4f}")

    return per_label
def validation_report_bio(
    trainer,
    val_ds,
    id2label: Dict[int, str],
    include_O: bool = False,
    plot_cm: bool = True,
    title: str = 'Token-level Confusion Matrix (BIO)'
) -> Dict[str, Any]:
    if confusion_matrix is None:
        raise RuntimeError(
            "validation_report_bio requires scikit-learn; install requirements-dev to use it"
        )
    pred_out = trainer.predict(val_ds)
    preds = pred_out.predictions if hasattr(pred_out, 'predictions') else pred_out[0]
    golds = pred_out.label_ids if hasattr(pred_out, 'label_ids') else pred_out[1]
    pred_ids = preds.argmax(-1)

    all_pred, all_gold = [], []
    for p_row, g_row in zip(pred_ids, golds):
        for p, g in zip(p_row, g_row):
            if int(g) == -100:
                continue
            all_pred.append(id2label[int(p)])
            all_gold.append(id2label[int(g)])

    try:
        label_ids_sorted = sorted(int(i) for i in id2label.keys())
    except Exception:
        label_ids_sorted = sorted(id2label.keys())
    all_labels_model_order = [id2label[i] for i in label_ids_sorted]

    if include_O:
        target_labels = list(all_labels_model_order)
    else:
        target_labels = [l for l in all_labels_model_order if str(l).startswith(('B-','I-'))]

    mask = [g in target_labels for g in all_gold]
    gold_f = [g for g, m in zip(all_gold, mask) if m]
    pred_f = [p for p, m in zip(all_pred, mask) if m]

    if len(gold_f) == 0:
        print('Warning: no valid tokens for BIO metrics (check labels / masking).')
        cm = np.zeros((len(target_labels), len(target_labels)), dtype=int)
    else:
        cm = confusion_matrix(gold_f, pred_f, labels=target_labels)

    per_label, f1s = {}, []
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = tp + fn

    for i, lab in enumerate(target_labels):
        precision = (tp[i] / (tp[i] + fp[i])) if (tp[i] + fp[i]) > 0 else 0.0
        recall    = (tp[i] / (tp[i] + fn[i])) if (tp[i] + fn[i]) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_label[lab] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(support[i]),
        }
        f1s.append(f1)

    macro_f1 = float(sum(f1s) / len(f1s)) if f1s else 0.0

    print('BIO Validation Metrics')
    print('Macro-F1:', f'{macro_f1:.4f}')
    for lab in target_labels:
        r = per_label[lab]
        print(f'{lab:10s} P={r["precision"]:.3f} R={r["recall"]:.3f} F1={r["f1"]:.3f}  support={r["support"]}')

    if plot_cm:
        plt.figure(figsize=(max(8, 0.7*len(target_labels)), max(6, 0.6*len(target_labels))))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_labels, yticklabels=target_labels)
        plt.xlabel('Predicted')
        plt.ylabel('Gold')
        plt.title(title + (' (include all labels)' if include_O else ' (BIO only)'))
        plt.tight_layout()
        plt.show()

    return {
        'labels': target_labels,
        'per_label': per_label,
        'macro_f1': macro_f1,
        'cm': cm,
    }
# Build annotation list with BIO tags at char-level, then split into word segments per entity
def predict_annotation_for_text(text: str, model: str = None, run_id: str = None) -> List[Tuple[int,int,str]]:
    # Prefer resolved artifacts (latest when run_id not specified)
    """Predict annotation for text.
    
    Args:
        text: Parameter.
        model: Parameter.
        run_id: Parameter.
    
    Returns:
        Return value.
    """
    art_dir = resolve_model_dir(model=model, run_id=run_id, prefer='artifacts')
    r = predict_one_pp(text, model_dir=art_dir)
    # api_spans already has B-/I- per word chunk
    ann = [(d['start_index'], d['end_index'], d['entity']) for d in r['api_spans']]
    return ann

# Generate submission from a dataframe with 'sample' and optional 'id'

def build_submission(df: "pd.DataFrame") -> "pd.DataFrame":
    """Build submission.
    
    Args:
        df: Parameter.
    
    Returns:
        Return value.
    """
    if pd is None:
        raise RuntimeError(
            "pandas is required for build_submission; install requirements-dev or add pandas to the image"
        )
    assert 'search_query' in df.columns, 'Expected column search_query'
    out_rows = []
    for idx, row in df.iterrows():
        sid = row['id'] if 'id' in df.columns else idx
        text = str(row['search_query'])
        try:
            ann = predict_annotation_for_text(text)
        except Exception:
            ann = []
        out_rows.append({'id': sid, 'search_query': text, 'annotation': str(ann)})
    return pd.DataFrame(out_rows)

