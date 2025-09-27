from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
from typing import Any, Dict, List


def _load_add_docstrings_impl():
    """
    Load add_docstrings_to_source from the sibling auto_docstrings.py without
    requiring a package import. Falls back to no-op if unavailable.
    """
    base = os.path.dirname(__file__)
    sys.path.insert(0, base)
    try:
        import auto_docstrings  # type: ignore

        return auto_docstrings.add_docstrings_to_source  # type: ignore[attr-defined]
    except Exception:

        def _noop(path: str, source: str) -> str:
            """Noop.

            Args:
                path: Parameter.
                source: Parameter.

            Returns:
                Return value.
            """
            return source

        return _noop


add_docstrings_to_source = _load_add_docstrings_impl()


def read_notebook(path: str) -> Dict[str, Any]:
    """Read notebook.

    Args:
        path: Parameter.

    Returns:
        Return value.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_notebook(path: str, nb: Dict[str, Any]) -> None:
    """Write notebook.

    Args:
        path: Parameter.
        nb: Parameter.

    Returns:
        Return value.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")


def cell_source_to_text(src: Any) -> str:
    """Cell source to text.

    Args:
        src: Parameter.

    Returns:
        Return value.
    """
    if isinstance(src, list):
        return "".join(src)
    if isinstance(src, str):
        return src
    return ""


def text_to_cell_source(text: str) -> List[str]:
    # Store as list of lines to match common Jupyter formatting
    """Text to cell source.

    Args:
        text: Parameter.

    Returns:
        Return value.
    """
    return text.splitlines(keepends=True)


def process_notebook(path: str) -> int:
    """Process notebook.

    Args:
        path: Parameter.

    Returns:
        Return value.
    """
    nb = read_notebook(path)
    cells = nb.get("cells", [])
    changed = 0
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        src_text = cell_source_to_text(cell.get("source", []))
        if not src_text.strip():
            continue
        try:
            updated = add_docstrings_to_source(path, src_text)
        except Exception:
            updated = src_text
        if updated != src_text:
            cell["source"] = text_to_cell_source(updated)
            changed += 1
    if changed:
        write_notebook(path, nb)
    return changed


def dry_run_notebook(path: str) -> int:
    """Dry run notebook.

    Args:
        path: Parameter.

    Returns:
        Return value.
    """
    nb = read_notebook(path)
    cells = nb.get("cells", [])
    changed = 0
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src_text = cell_source_to_text(cell.get("source", []))
        if not src_text.strip():
            continue
        try:
            updated = add_docstrings_to_source(path, src_text)
        except Exception:
            updated = src_text
        if updated != src_text:
            changed += 1
            a = src_text.splitlines(keepends=True)
            b = updated.splitlines(keepends=True)
            diff = difflib.unified_diff(
                a,
                b,
                fromfile=f"{path}::cell{idx} (orig)",
                tofile=f"{path}::cell{idx} (new)",
            )
            sys.stdout.write("".join(diff))
    return changed


def main() -> None:
    """Main.

    Returns:
        Return value.
    """
    p = argparse.ArgumentParser(
        description="Auto-add English docstrings to Python functions/classes inside .ipynb code cells."
    )
    p.add_argument("notebooks", nargs="+", help="Notebook paths to process (.ipynb)")
    p.add_argument(
        "--write", action="store_true", help="Write changes in place (default: dry-run)"
    )
    args = p.parse_args()

    total_changed = 0
    for nb in args.notebooks:
        if not os.path.isfile(nb) or not nb.endswith(".ipynb"):
            continue
        if args.write:
            c = process_notebook(nb)
            if c:
                print(f"Updated {nb}: {c} code cells modified")
            total_changed += c
        else:
            c = dry_run_notebook(nb)
            if c:
                print(f"\nWould update {nb}: {c} code cells modified\n")
            total_changed += c

    if not total_changed:
        print("No changes needed.")


if __name__ == "__main__":
    main()
