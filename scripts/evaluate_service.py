#!/usr/bin/env python3
"""
Evaluate the NER service against a CSV with columns: id;search_query;annotation

Input CSV format (semicolon-separated):
  id;search_query;annotation
  1;текст;[(0, 5, 'B-TYPE'), (6, 10, 'I-TYPE')]

This script will:
  - Read queries and gold annotations
  - Call POST {base_url}/api/predict and/or /api/predict_batch
  - Write a CSV with query, expected, predicted, and per-row metrics
  - Write an HTML report with highlighted true positives, false positives, and misses
  - Print overall statistics (precision, recall, F1, exact match)

Usage example:
  python3 scripts/evaluate_service.py \
    --input examples/submission2.csv \
    --output_dir eval_out \
    --base_url http://localhost:8080 \
    --batch_size 32
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from tqdm import tqdm


# ==========================
# Data structures
# ==========================


@dataclass(frozen=True)
class Span:
    start: int
    end: int  # exclusive
    tag: str  # e.g., "B-TYPE", "I-BRAND"

    @property
    def entity_type(self) -> str:
        # Convert "B-TYPE" or "I-TYPE" → "TYPE". If no dash, return as is.
        if "-" in self.tag:
            return self.tag.split("-", 1)[1]
        return self.tag


# ==========================
# Parsing and formatting
# ==========================


def parse_annotation_field(value: str) -> List[Span]:
    """Parse the annotation column: a Python-like list of tuples.
    Example: "[(0, 5, 'B-TYPE'), (6, 10, 'I-TYPE')]"
    Returns a list[Span].
    """
    value = (value or "").strip()
    if not value:
        return []
    try:
        raw = ast.literal_eval(value)
    except Exception:
        # If parsing fails, treat as no annotations
        return []
    spans: List[Span] = []
    for item in raw or []:
        try:
            start, end, tag = item
            spans.append(Span(int(start), int(end), str(tag)))
        except Exception:
            continue
    return spans


def normalize_predicted(pred: Sequence[Dict[str, Any]]) -> List[Span]:
    """Normalize API predictions (list of dicts) -> list of Span."""
    out: List[Span] = []
    for p in pred:
        try:
            start = int(p.get("start_index"))
            end = int(p.get("end_index"))
            tag = str(p.get("entity"))
            out.append(Span(start, end, tag))
        except Exception:
            continue
    return out


def spans_to_jsonable(spans: Sequence[Span]) -> List[Dict[str, Any]]:
    return [{"start": s.start, "end": s.end, "tag": s.tag} for s in spans]


# ==========================
# Evaluation metrics
# ==========================


@dataclass
class RowMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    exact_match: bool


def compute_row_metrics(gold: Sequence[Span], pred: Sequence[Span]) -> RowMetrics:
    gold_set = {(s.start, s.end, s.tag) for s in gold}
    pred_set = {(s.start, s.end, s.tag) for s in pred}
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    exact = gold_set == pred_set
    return RowMetrics(true_positives=tp, false_positives=fp, false_negatives=fn, exact_match=exact)


def safe_div(n: float, d: float) -> float:
    return (n / d) if d else 0.0


@dataclass
class GlobalMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    exact_match_count: int
    total_rows: int


def aggregate_metrics(rows: Sequence[RowMetrics]) -> GlobalMetrics:
    tp = sum(r.true_positives for r in rows)
    fp = sum(r.false_positives for r in rows)
    fn = sum(r.false_negatives for r in rows)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    exact = sum(1 for r in rows if r.exact_match)
    return GlobalMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        exact_match_count=exact,
        total_rows=len(rows),
    )


# ==========================
# Highlighting (HTML)
# ==========================


def build_char_labels(text: str, spans: Sequence[Span], key: str) -> List[Optional[str]]:
    """Return per-character labels for given spans.
    key is a namespace string, e.g., "gold:TYPE" or "pred:TYPE".
    If overlapping spans occur, later spans overwrite earlier labels.
    """
    labels: List[Optional[str]] = [None] * len(text)
    for s in spans:
        start = max(0, min(s.start, len(text)))
        end = max(0, min(s.end, len(text)))
        for i in range(start, end):
            labels[i] = f"{key}:{s.entity_type}"
    return labels


def highlight_text(text: str, gold: Sequence[Span], pred: Sequence[Span]) -> str:
    """Produce HTML highlighting:
      - True positive (same entity type overlap): green
      - False positive (pred only): red
      - False negative (gold only): yellow
      - Mismatch (both but different types): purple
    """
    gold_labels = build_char_labels(text, gold, key="gold")
    pred_labels = build_char_labels(text, pred, key="pred")

    def classify(i: int) -> str:
        g = gold_labels[i]
        p = pred_labels[i]
        if g and p:
            g_type = g.split(":", 2)[-1]
            p_type = p.split(":", 2)[-1]
            if g_type == p_type:
                return "tp"
            return "mm"  # mismatch
        if p and not g:
            return "fp"
        if g and not p:
            return "fn"
        return "none"

    out: List[str] = []
    if not text:
        return ""
    cur_class = classify(0)
    seg = [text[0]]
    for i in range(1, len(text)):
        cl = classify(i)
        if cl == cur_class:
            seg.append(text[i])
        else:
            out.append(_wrap_segment("".join(seg), cur_class))
            seg = [text[i]]
            cur_class = cl
    out.append(_wrap_segment("".join(seg), cur_class))
    return "".join(out)


def _wrap_segment(segment: str, klass: str) -> str:
    if klass == "none":
        return escape_html(segment)
    return f'<span class="{klass}">{escape_html(segment)}</span>'


def escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def render_html_report(
    output_path: str,
    rows: Sequence[Dict[str, Any]],
    metrics: GlobalMetrics,
) -> None:
    css = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
      .summary { margin-bottom: 24px; }
      .metrics { display: flex; gap: 16px; flex-wrap: wrap; }
      .metric { padding: 8px 12px; background: #f5f5f5; border-radius: 6px; }
      .controls { display: flex; gap: 12px; align-items: center; margin: 16px 0 8px; }
      .controls label { font-size: 14px; color: #333; }
      .controls select { padding: 4px 8px; }
      .controls input[type="checkbox"] { margin-left: 8px; }
      .sample { border-top: 1px solid #eee; padding: 16px 0; }
      .q { font-weight: 600; }
      .spans { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
      .legend { margin: 12px 0; font-size: 13px; }
      .tp { background: #d1ffd6; }
      .fp { background: #ffd1d1; }
      .fn { background: #fff2b3; }
      .mm { background: #f3d1ff; }
      details { margin-top: 8px; }
      summary { cursor: pointer; }
      .row-head { color: #666; font-size: 12px; margin-bottom: 6px; }
    </style>
    """

    def fmt_pct(x: float) -> str:
        return f"{x * 100:.2f}%"

    html_rows: List[str] = []
    for r in rows:
        rid = escape_html(str(r.get("id", "")))
        exact = bool(r.get("exact_match"))
        tp = int(r.get("tp", 0))
        fp = int(r.get("fp", 0))
        fn = int(r.get("fn", 0))
        html_rows.append(
            """
            <div class="sample" data-id="{id}" data-exact="{exact_int}" data-tp="{tp}" data-fp="{fp}" data-fn="{fn}">
              <div class="row-head">id: {id} | exact_match: {exact}</div>
              <div class="q">{query}</div>
              <div class="legend">Legend: <span class="tp">TP</span> <span class="fp">FP</span> <span class="fn">FN</span> <span class="mm">Mismatch</span></div>
              <div>{highlight}</div>
              <details>
                <summary>Show spans</summary>
                <div class="spans">expected: {gold}</div>
                <div class="spans">predicted: {pred}</div>
              </details>
            </div>
            """.format(
                id=rid,
                exact="yes" if exact else "no",
                exact_int=(0 if not exact else 1),
                tp=tp,
                fp=fp,
                fn=fn,
                query=escape_html(r.get("query", "")),
                highlight=r.get("highlight_html", ""),
                gold=escape_html(json.dumps(r.get("expected", []), ensure_ascii=False)),
                pred=escape_html(json.dumps(r.get("predicted", []), ensure_ascii=False)),
            )
        )

    # Client-side controls script (kept outside of .format to avoid brace conflicts)
    script = """
        <script>
          (function() {
            function toInt(x) { return parseInt(x || '0', 10) || 0; }
            function getKey(el, key) { return el.getAttribute('data-' + key) || ''; }
            function isWrong(el) { return toInt(getKey(el, 'exact')) === 0; }

            function sortComparator(mode) {
              if (mode === 'wrong-first') {
                return (a, b) => toInt(getKey(a, 'exact')) - toInt(getKey(b, 'exact'));
              }
              if (mode === 'right-first') {
                return (a, b) => toInt(getKey(b, 'exact')) - toInt(getKey(a, 'exact'));
              }
              if (mode === 'id-asc') {
                return (a, b) => toInt(getKey(a, 'id')) - toInt(getKey(b, 'id'));
              }
              if (mode === 'id-desc') {
                return (a, b) => toInt(getKey(b, 'id')) - toInt(getKey(a, 'id'));
              }
              if (mode === 'fp-desc') {
                return (a, b) => toInt(getKey(b, 'fp')) - toInt(getKey(a, 'fp'));
              }
              if (mode === 'fn-desc') {
                return (a, b) => toInt(getKey(b, 'fn')) - toInt(getKey(a, 'fn'));
              }
              if (mode === 'tp-desc') {
                return (a, b) => toInt(getKey(b, 'tp')) - toInt(getKey(a, 'tp'));
              }
              return () => 0;
            }

            function applySortAndFilter() {
              const container = document.getElementById('rows-container');
              const mode = document.getElementById('sort-select').value;
              const onlyWrong = document.getElementById('only-wrong').checked;
              const nodes = Array.from(container.querySelectorAll('.sample'));

              let filtered = nodes;
              if (onlyWrong) {
                filtered = nodes.filter(isWrong);
              }

              filtered.sort(sortComparator(mode));

              // Re-attach in new order
              container.innerHTML = '';
              filtered.forEach(n => container.appendChild(n));
            }

            document.addEventListener('DOMContentLoaded', function() {
              document.getElementById('sort-select').addEventListener('change', applySortAndFilter);
              document.getElementById('only-wrong').addEventListener('change', applySortAndFilter);
              applySortAndFilter();
            });
          })();
        </script>
    """

    html = """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>Evaluation Report</title>
        {css}
      </head>
      <body>
        <div class="summary">
          <h2>Evaluation Summary</h2>
          <div class="metrics">
            <div class="metric">TP: {tp}</div>
            <div class="metric">FP: {fp}</div>
            <div class="metric">FN: {fn}</div>
            <div class="metric">Precision: {p}</div>
            <div class="metric">Recall: {r}</div>
            <div class="metric">F1: {f1}</div>
            <div class="metric">Exact match: {em} / {tot}</div>
          </div>
        </div>
        <div class="controls">
          <label>Sort:
            <select id="sort-select">
              <option value="wrong-first" selected>Wrong first</option>
              <option value="right-first">Right first</option>
              <option value="id-asc">ID ↑</option>
              <option value="id-desc">ID ↓</option>
              <option value="fp-desc">FP ↓</option>
              <option value="fn-desc">FN ↓</option>
              <option value="tp-desc">TP ↓</option>
            </select>
          </label>
          <label><input type="checkbox" id="only-wrong" /> Only wrong</label>
        </div>
        <div id="rows-container">
        {rows}
        </div>
        {script}
      </body>
    </html>
    """.format(
        css=css,
        tp=metrics.true_positives,
        fp=metrics.false_positives,
        fn=metrics.false_negatives,
        p=fmt_pct(metrics.precision),
        r=fmt_pct(metrics.recall),
        f1=fmt_pct(metrics.f1),
        em=metrics.exact_match_count,
        tot=metrics.total_rows,
        rows="\n".join(html_rows),
        script=script,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ==========================
# Service calling
# ==========================


def chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def call_predict_batch(base_url: str, texts: Sequence[str]) -> List[List[Dict[str, Any]]]:
    url = base_url.rstrip("/") + "/api/predict_batch"
    resp = requests.post(url, json={"inputs": list(texts)}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError("Unexpected response from predict_batch")
    return data  # list of list[dict]


def call_predict_single(base_url: str, text: str) -> List[Dict[str, Any]]:
    url = base_url.rstrip("/") + "/api/predict"
    resp = requests.post(url, json={"input": text}, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError("Unexpected response from predict")
    return data


# ==========================
# Main
# ==========================


def evaluate(
    input_csv: str,
    output_dir: str,
    base_url: str,
    batch_size: int = 32,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    out_csv_path = os.path.join(output_dir, "eval_results.csv")
    out_html_path = os.path.join(output_dir, "eval_report.html")
    out_stats_path = os.path.join(output_dir, "eval_stats.json")

    # Read input
    rows: List[Dict[str, Any]] = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for r in reader:
            rid = r.get("id")
            text = r.get("search_query", "")
            ann = parse_annotation_field(r.get("annotation", ""))
            rows.append({"id": rid, "query": text, "gold": ann})

    # Predict in batches
    all_predictions: List[List[Span]] = []
    texts = [r["query"] for r in rows]
    try:
        # Try batch endpoint first
        for batch in tqdm(list(chunked(texts, batch_size)), desc="predict_batch"):
            pred_batch = call_predict_batch(base_url, batch)
            for pred in pred_batch:
                all_predictions.append(normalize_predicted(pred))
    except Exception:
        # Fallback to single requests
        all_predictions = []
        for text in tqdm(texts, desc="predict"):
            pred = call_predict_single(base_url, text)
            all_predictions.append(normalize_predicted(pred))

    # Per-row metrics and highlights
    per_row: List[Dict[str, Any]] = []
    row_metrics: List[RowMetrics] = []
    for r, pred_spans in zip(rows, all_predictions):
        gold_spans: List[Span] = list(r["gold"])
        metrics = compute_row_metrics(gold_spans, pred_spans)
        row_metrics.append(metrics)
        per_row.append(
            {
                "id": r["id"],
                "query": r["query"],
                "expected": spans_to_jsonable(gold_spans),
                "predicted": spans_to_jsonable(pred_spans),
                "tp": metrics.true_positives,
                "fp": metrics.false_positives,
                "fn": metrics.false_negatives,
                "exact_match": metrics.exact_match,
                "highlight_html": highlight_text(r["query"], gold_spans, pred_spans),
            }
        )

    global_metrics = aggregate_metrics(row_metrics)

    # Write CSV results
    with open(out_csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "id",
            "query",
            "expected",
            "predicted",
            "tp",
            "fp",
            "fn",
            "exact_match",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_row:
            writer.writerow(
                {
                    "id": r["id"],
                    "query": r["query"],
                    "expected": json.dumps(r["expected"], ensure_ascii=False),
                    "predicted": json.dumps(r["predicted"], ensure_ascii=False),
                    "tp": r["tp"],
                    "fp": r["fp"],
                    "fn": r["fn"],
                    "exact_match": r["exact_match"],
                }
            )

    # Write HTML report with highlights
    render_html_report(out_html_path, per_row, global_metrics)

    # Write stats JSON
    with open(out_stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "true_positives": global_metrics.true_positives,
                "false_positives": global_metrics.false_positives,
                "false_negatives": global_metrics.false_negatives,
                "precision": global_metrics.precision,
                "recall": global_metrics.recall,
                "f1": global_metrics.f1,
                "exact_match_count": global_metrics.exact_match_count,
                "total_rows": global_metrics.total_rows,
                "input_csv": os.path.abspath(input_csv),
                "base_url": base_url,
                "note": "Precision/Recall/F1 are computed over exact word-level spans (start,end,tag)",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Print concise statistics
    print("\n=== Evaluation ===")
    print(f"TP: {global_metrics.true_positives}")
    print(f"FP: {global_metrics.false_positives}")
    print(f"FN: {global_metrics.false_negatives}")
    print(f"Precision: {global_metrics.precision:.4f}")
    print(f"Recall:    {global_metrics.recall:.4f}")
    print(f"F1:        {global_metrics.f1:.4f}")
    print(f"Exact rows: {global_metrics.exact_match_count} / {global_metrics.total_rows}")
    print(f"CSV:  {out_csv_path}")
    print(f"HTML: {out_html_path}")
    print(f"JSON: {out_stats_path}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate NER service on CSV annotations")
    p.add_argument("--input", required=True, help="Path to input CSV (semicolon-separated)")
    p.add_argument("--output_dir", required=True, help="Directory to write outputs")
    p.add_argument(
        "--base_url",
        default="http://localhost:8080",
        help="Service base URL (default: http://localhost:8080)",
    )
    p.add_argument("--batch_size", type=int, default=32, help="Batch size for predict_batch calls")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    evaluate(
        input_csv=args.input,
        output_dir=args.output_dir,
        base_url=args.base_url,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
