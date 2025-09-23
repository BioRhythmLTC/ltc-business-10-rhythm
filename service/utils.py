import re
from typing import List, Dict, Any, Optional, Tuple


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


def _spans_to_api_spans(text: str, spans: List[Tuple[int, int, str]]) -> List[Dict[str, Any]]:
    merged = _merge_adjacent_spans(text, spans)
    words = _find_word_spans(text)
    api: List[Dict[str, Any]] = []
    for s, e, t in merged:
        if not t:
            continue
        inside = [(ws, we) for (ws, we) in words if not (we <= s or ws >= e)]
        for j, (ws, we) in enumerate(inside):
            tag = ("B-" if j == 0 else "I-") + t
            api.append({"start_index": ws, "end_index": we, "entity": tag})
    return api
