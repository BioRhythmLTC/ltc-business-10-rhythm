import os
import re
from typing import List, Dict, Any, Optional, Tuple

import torch
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
from .utils import (
    _token_tags_to_char_bio,
    _find_word_spans,
    _extract_spans_from_bio,
    _merge_adjacent_spans,
    _spans_to_api_spans,
)


ART_DIR = os.environ.get("ARTIFACTS_DIR", os.path.join(os.path.dirname(__file__), "..", "artifacts"))
ART_DIR = os.path.abspath(ART_DIR)


def _select_device() -> str:
    # Allow environment override to avoid unstable backends (e.g., macOS MPS)
    forced = os.environ.get("X5_FORCE_DEVICE") or os.environ.get("FORCE_DEVICE")
    if forced:
        f = forced.strip().lower()
        if f in {"cpu", "cuda", "mps"}:
            return f
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_artifacts(art_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(art_dir)
    model = AutoModelForTokenClassification.from_pretrained(art_dir)
    device = _select_device()
    model = model.to(device).eval()
    # id2label from config; fallback builds mapping
    id2label = getattr(model.config, "id2label", None)
    if not isinstance(id2label, dict):
        num = getattr(model.config, "num_labels", 0)
        id2label = {i: str(i) for i in range(num)}
    return model, tokenizer, device, {int(k): v for k, v in id2label.items()}


MODEL = None  # type: ignore[assignment]
TOKENIZER = None  # type: ignore[assignment]
DEVICE = _select_device()
ID2LABEL: Dict[int, str] = {}

def _ensure_loaded() -> Tuple[Any, Any, str, Dict[int, str]]:
    global MODEL, TOKENIZER, DEVICE, ID2LABEL
    if MODEL is None or TOKENIZER is None or not ID2LABEL:
        MODEL, TOKENIZER, DEVICE, ID2LABEL = _load_artifacts(ART_DIR)
    return MODEL, TOKENIZER, DEVICE, ID2LABEL


def _token_tags_to_char_bio(text: str, token_tags: List[str], offsets: List[Tuple[int, int]]) -> List[str]:
    """Notebook-parity: build char BIO by placing B- at token start, I- inside.

    This matches the pipeline's `token_tags_to_char_bio` behavior.
    """
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


def predict_api_spans(text: str) -> List[Dict[str, Any]]:
    model, tokenizer, device, id2label_map = _ensure_loaded()
    with torch.no_grad():
        enc = tokenizer(text, return_offsets_mapping=True, truncation=True, return_tensors="pt")
        offsets: List[Tuple[int, int]] = enc["offset_mapping"][0].tolist()
        inp = {k: v.to(device) for k, v in enc.items() if k != "offset_mapping"}
        logits = model(**inp).logits
        pred_ids = logits.argmax(-1)[0].tolist()
        # Use model.config.id2label when available for perfect parity
        cfg_map = getattr(model.config, "id2label", None)
        def _id2label(i: int) -> str:
            if isinstance(cfg_map, dict):
                return cfg_map.get(i, cfg_map.get(str(i), id2label_map.get(i, "O")))
            return id2label_map.get(i, "O")
        token_tags = [_id2label(int(t)) for t in pred_ids]
        char_bio = _token_tags_to_char_bio(text, token_tags, offsets)
        spans = _extract_spans_from_bio(text[:max([e for (_, e) in offsets if e is not None] or [0])], char_bio)
        return _spans_to_api_spans(text, spans)


class PredictRequest(BaseModel):
    input: str


class PredictBatchRequest(BaseModel):
    inputs: List[str]


app = FastAPI(title="X5 NER Service", version="1.0.0")


@app.on_event("startup")
async def _warmup() -> None:
    # Optional warmup; can be disabled or limited to CPU via env
    if str(os.environ.get("DISABLE_WARMUP", "")).lower() in {"1", "true", "yes"}:
        return
    # Skip MPS warmup by default due to stability issues; override to enable
    if DEVICE == "mps" and str(os.environ.get("ALLOW_MPS_WARMUP", "")).lower() not in {"1", "true", "yes"}:
        return
    try:
        _ = predict_api_spans("")
    except Exception:
        pass


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "device": DEVICE, "artifacts": ART_DIR}


@app.post("/api/predict")
async def predict(req: PredictRequest) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(predict_api_spans, req.input)


@app.post("/api/predict_batch")
async def predict_batch(req: PredictBatchRequest) -> List[List[Dict[str, Any]]]:
    tasks = [asyncio.to_thread(predict_api_spans, text) for text in req.inputs]
    return await asyncio.gather(*tasks)


# запустить сервис (gunicorn+uvicorn)
# bash scripts/run_gunicorn.sh
# curl -s http://127.0.0.1:8000/health
# curl -s http://127.0.0.1:8000/api/predict -H 'Content-Type: application/json' \
# -d '{"input":"Cгущеное молоко"}'
# # # батч
# curl -s http://127.0.0.1:8000/api/predict_batch -H 'Content-Type: application/json' \
# -d '{"inputs":["Global Village","Artfruit виноград"]}'

