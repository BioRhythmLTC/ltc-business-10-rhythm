import re
from typing import List, Dict, Any, Optional, Tuple


_lat = re.compile(r"[A-Za-z]")
_cyr = re.compile(r"[А-Яа-яЁё]")

# Canonical unit whitelist (policy-driven)
WHITELIST_UNITS: Dict[str, List[str]] = {
    "L": ["л", "л.", "литр", "литра", "литров", "l"],
    "ML": ["мл", "мл.", "миллилитр", "миллилитра", "миллилитров", "ml"],
    "KG": ["кг", "кг.", "килограмм", "килограмма", "килограммов", "kg"],
    "G": ["г", "г.", "грамм", "грамма", "граммов", "g"],
    "PCS": ["шт", "шт.", "штука", "штуки", "штук", "pcs"],
}

# Build regex group for units: combine stems + explicit forms
_unit_stems_regex = [
    r"литр\w*", r"мл\.?", r"миллилитр\w*", r"l",
    r"килограмм\w*", r"кг\.?", r"kg",
    r"грамм\w*", r"г\.?", r"g",
    r"шт\.?", r"штук\w*", r"pcs",
]
_unit_forms = [re.escape(f) for forms in WHITELIST_UNITS.values() for f in forms]
_unit_forms_regex = _unit_stems_regex + _unit_forms
_units_group = "(?:" + "|".join(sorted(_unit_forms_regex, key=len, reverse=True)) + ")"

# Patterns similar to notebook
pattern_volume = re.compile(r"\b\d+[\.,]?\d*\s?" + _units_group + r"\b", re.IGNORECASE)
pattern_multipack = re.compile(r"\b\d+\s*[xх*]\s*\d+[\.,]?\d*\s?" + _units_group + r"\b", re.IGNORECASE)
pattern_percent = re.compile(r"\b\d+[\.,]?\d*\s?(?:%|процент(?:а|ов)?)\b", re.IGNORECASE)
re_nospace_volume = re.compile(r"\d+[\.,]?\d*" + _units_group + r"\b", re.IGNORECASE)

# Auxiliary quality heuristics
re_repeated = re.compile(r"(.)\1{2,}")
re_mixed = re.compile(r"[а-я]{1,2}[a-z]|[a-z]{1,2}[а-я]", re.IGNORECASE)


def _token_script_stats(text: str) -> Tuple[int, int]:
    lat = len(_lat.findall(text))
    cyr = len(_cyr.findall(text))
    return lat, cyr


def _normalize_simple_subs(token: str) -> str:
    out = token.replace("ё", "е")
    lat, cyr = _token_script_stats(out)
    if lat > cyr:
        out = out.replace("j", "i")
    return out


def preprocess_text_with_mapping(text: str, do_lower: bool = True, apply_translit_map: bool = True) -> Tuple[str, List[int]]:
    """
    Lightweight normalization aligned with notebook preprocessing:
    - Replace invisible spaces with regular space
    - Optionally lowercase
    - Normalize simple substitutions inside tokens (ё→е; if token is mostly Latin: j→i)

    Returns normalized text and a mapping from normalized char index → original char index.
    All transformations preserve length, so mapping is 1:1.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # 1) Replace common invisible/odd spaces with normal space, preserve length
    invisible_spaces = {
        "\u00A0": " ",  # NO-BREAK SPACE
        "\u202F": " ",  # NARROW NO-BREAK SPACE
        "\u2007": " ",  # FIGURE SPACE
        "\u2009": " ",  # THIN SPACE
        "\u2002": " ",  # EN SPACE
        "\u2003": " ",  # EM SPACE
        "\u2004": " ",
        "\u2005": " ",
        "\u2006": " ",
    }
    stage_chars: List[str] = []
    for ch in text:
        if ch in invisible_spaces:
            stage_chars.append(" ")
        else:
            stage_chars.append(ch)
    out_text = "".join(stage_chars)

    if do_lower:
        out_text = out_text.lower()

    if apply_translit_map:
        # Apply token-level substitutions without changing length
        parts = re.split(r"(\s+)", out_text)
        for i in range(0, len(parts), 2):  # tokens at even indices; whitespace kept as-is
            if i < len(parts):
                parts[i] = _normalize_simple_subs(parts[i])
        out_text = "".join(parts)

    # All steps preserve string length
    mapping = list(range(len(out_text)))
    return out_text, mapping


def normalize_unit_stem(segment: str) -> str:
    if re.search(r"миллилитр|\bмл\.?\b|\bml\b", segment, re.IGNORECASE):
        return "ML"
    if re.search(r"литр|\bл\.?\b|\bl\b", segment, re.IGNORECASE):
        return "L"
    if re.search(r"килограмм|\bкг\.?\b|\bkg\b", segment, re.IGNORECASE):
        return "KG"
    if re.search(r"грамм|\bг\.?\b|\bg\b", segment, re.IGNORECASE):
        return "G"
    if re.search(r"шт\.?|штук|\bpcs\b", segment, re.IGNORECASE):
        return "PCS"
    return ""


def has_volume_improved(text: str) -> bool:
    return bool(pattern_volume.search(text) or pattern_multipack.search(text))


def _token_tags_to_char_bio(text: str, token_tags: List[str], offsets: List[Tuple[int, int]]) -> List[str]:
    text_len = max([e for (s, e) in offsets if e is not None], default=0)
    char_bio = ["O"] * text_len
    for (s, e), tag in zip(offsets, token_tags):
        if s is None or e is None or e <= s:
            continue
        if tag == "O" or "-" not in tag:
            continue
        try:
            _bi, etype = tag.split("-", 1)
        except Exception:
            continue
        char_bio[s] = f"B-{etype}"
        for pos in range(s + 1, e):
            char_bio[pos] = f"I-{etype}"
    return char_bio


def _find_word_spans(text: str) -> List[List[int]]:
    return [[m.start(), m.end()] for m in re.finditer(r"\S+", text)]


def _extract_spans_from_bio(text: str, bio_labels: List[str]) -> List[Tuple[int, int, str]]:
    n = len(text)
    if len(bio_labels) != n:
        bio_labels = (bio_labels + ["O"] * n)[:n]
    spans: List[Tuple[int, int, str]] = []
    current_start: Optional[int] = None
    current_type: Optional[str] = None
    for i, tag in enumerate(bio_labels):
        if tag == "O" or tag is None:
            if current_start is not None:
                spans.append((current_start, i, current_type if current_type else ""))
                current_start, current_type = None, None
            continue
        if "-" in tag:
            bio, etype = tag.split("-", 1)
        else:
            bio, etype = "I", tag
        if bio == "B":
            if current_start is not None:
                spans.append((current_start, i, current_type if current_type else ""))
            current_start, current_type = i, etype
        elif bio == "I":
            if current_start is None:
                current_start, current_type = i, etype
            elif etype != current_type:
                spans.append((current_start, i, current_type if current_type else ""))
                current_start, current_type = i, etype
        else:
            if current_start is not None:
                spans.append((current_start, i, current_type if current_type else ""))
                current_start, current_type = None, None
    if current_start is not None:
        spans.append((current_start, n, current_type if current_type else ""))
    return spans


def _merge_adjacent_spans(text: str, spans: List[Tuple[int, int, str]], max_gap: int = 1) -> List[Tuple[int, int, str]]:
    merged: List[Tuple[int, int, str]] = []
    for s, e, t in sorted(spans, key=lambda x: (x[0], x[1])):
        if merged and merged[-1][2] == t:
            ps, pe, pt = merged[-1]
            gap = s - pe
            between = text[pe:s] if 0 <= pe <= s <= len(text) else ""
            if gap <= max_gap and (between.strip() == "" or between in ["-", "–", ".", ","]):
                merged[-1] = (ps, max(pe, e), pt)
                continue
        merged.append((s, e, t))
    return merged


def _spans_to_api_spans(text: str, spans: List[Tuple[int, int, str]], include_O: bool = False) -> List[Dict[str, Any]]:
    merged = _merge_adjacent_spans(text, spans)
    words = _find_word_spans(text)
    # Build mapping from word span to tag; default to O if include_O enabled
    word_tag_map: Dict[Tuple[int, int], str] = {}
    for s, e, t in merged:
        if not t:
            continue
        inside = [(ws, we) for (ws, we) in words if not (we <= s or ws >= e)]
        for j, (ws, we) in enumerate(inside):
            tag = ("B-" if j == 0 else "I-") + t
            word_tag_map[(ws, we)] = tag
    api: List[Dict[str, Any]] = []
    for ws, we in words:
        tag = word_tag_map.get((ws, we))
        if tag:
            api.append({"start_index": ws, "end_index": we, "entity": tag})
        elif include_O:
            api.append({"start_index": ws, "end_index": we, "entity": "O"})
    return api
