from __future__ import annotations

import argparse
import ast
import difflib
import fnmatch
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

DOC_MARK = "Auto-generated docstring"


@dataclass
class Change:
    """Container for a single file's modification preview and result."""

    path: str
    original: str
    updated: str


def read_text(path: str) -> str:
    """Read and return the contents of a UTF-8 text file.

    Args:
        path: File path to read.

    Returns:
        The file content as a single string.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, text: str) -> None:
    """Write UTF-8 text to a file, overwriting any existing content.

    Args:
        path: Destination file path.
        text: Text content to write.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def is_python_file(path: str) -> bool:
    """Return True if the path points to a non-hidden Python source file."""
    name = os.path.basename(path)
    if name.startswith("."):
        return False
    return path.endswith(".py")


def list_py_files(
    root: str,
    includes: Optional[List[str]] = None,
    excludes: Optional[List[str]] = None,
) -> List[str]:
    """Recursively list Python files under a root directory.

    Args:
        root: Root directory to scan.
        includes: Optional glob patterns to include.
        excludes: Optional glob patterns to exclude.

    Returns:
        Sorted list of file paths.
    """
    out: List[str] = []
    inc = includes or ["*.py"]
    exc = excludes or []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if not d.startswith(".") and d != "__pycache__"
        ]
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            if not is_python_file(full):
                continue
            if any(fnmatch.fnmatch(full, pat) for pat in exc):
                continue
            if any(fnmatch.fnmatch(full, pat) for pat in inc):
                out.append(full)
    return sorted(out)


def has_docstring(node: ast.AST, source: str) -> bool:
    """Return True if a node already has a docstring."""
    return ast.get_docstring(node) is not None


def is_placeholder_docstring(node: ast.AST) -> bool:
    """Return True if the existing docstring is a known placeholder."""
    try:
        doc = ast.get_docstring(node, clean=False)
    except Exception:
        doc = None
    return bool(doc and DOC_MARK in doc)


def _snake_to_words(name: str) -> str:
    """Snake to words.

    Args:
        name: Parameter.

    Returns:
        Return value.
    """
    return " ".join(part for part in name.replace("_", " ").split() if part)


def _summary_from_name(name: str) -> str:
    """Summary from name.

    Args:
        name: Parameter.

    Returns:
        Return value.
    """
    words = _snake_to_words(name)
    lower = words.lower()
    verbs = [
        ("get", "Get"),
        ("set", "Set"),
        ("load", "Load"),
        ("save", "Save"),
        ("read", "Read"),
        ("write", "Write"),
        ("build", "Build"),
        ("make", "Make"),
        ("create", "Create"),
        ("update", "Update"),
        ("delete", "Delete"),
        ("compute", "Compute"),
        ("calculate", "Calculate"),
        ("encode", "Encode"),
        ("decode", "Decode"),
        ("align", "Align"),
        ("map", "Map"),
        ("merge", "Merge"),
        ("normalize", "Normalize"),
        ("predict", "Predict"),
        ("find", "Find"),
        ("extract", "Extract"),
        ("evaluate", "Evaluate"),
        ("run", "Run"),
        ("process", "Process"),
        ("select", "Select"),
        ("ensure", "Ensure"),
    ]
    for key, verb in verbs:
        if lower.startswith(key + " ") or lower == key:
            rest = words[len(key) :].strip()
            return (verb + (" " + rest if rest else "") + ".").strip()
    return (words.capitalize() + ".") if words else "Perform operation."


def generate_docstring_for_function(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
) -> List[str]:
    """Generate a concise, PEP 257-style docstring for a function."""
    name = fn.name
    params = []
    # Positional-only args
    if hasattr(fn.args, "posonlyargs"):
        for a in fn.args.posonlyargs:  # type: ignore[attr-defined]
            params.append(a.arg)
    # Regular args
    for a in fn.args.args:
        params.append(a.arg)
    # Vararg
    if fn.args.vararg is not None:
        params.append("*" + fn.args.vararg.arg)
    # Kw-only
    for a in fn.args.kwonlyargs:
        params.append(a.arg)
    # Kwvar
    if fn.args.kwarg is not None:
        params.append("**" + fn.args.kwarg.arg)

    # Summary line
    header = _summary_from_name(name)

    # Build Google-style sections
    lines: List[str] = ['"""' + header]
    if params:
        lines.append("")
        lines.append("Args:")
        for p in params:
            lines.append(f"    {p}: Parameter.")
    # Return section if annotation exists
    if getattr(fn, "returns", None) is not None:
        lines.append("")
        lines.append("Returns:")
        lines.append("    Return value.")
    lines.append('"""')
    return lines


def generate_docstring_for_class(cls: ast.ClassDef) -> List[str]:
    """Generate a concise, PEP 257-style docstring for a class."""
    header = _summary_from_name(cls.name)
    return ['"""' + header, '"""']


def compute_indent_of_line(text_line: str) -> str:
    """Return the indentation (tabs/spaces) prefix of a line, unchanged."""
    i = 0
    while i < len(text_line) and text_line[i] in (" ", "\t"):
        i += 1
    return text_line[:i]


def insert_docstring(src_lines: List[str], node: ast.AST, doc_lines: List[str]) -> None:
    """Insert a new docstring block before the first statement of a node."""
    if not hasattr(node, "body") or not isinstance(getattr(node, "body"), list):
        return
    body = getattr(node, "body")
    # Determine insertion line: before first statement in body
    if not body:
        # Fallback: place after def/class line
        insert_at = getattr(node, "lineno", 1)
        # Body indent = def indent + 4 spaces by convention, or infer from next line
        if insert_at < len(src_lines):
            indent = compute_indent_of_line(src_lines[insert_at - 1]) + "    "
        else:
            indent = "    "
    else:
        first_stmt = body[0]
        insert_at = getattr(first_stmt, "lineno", getattr(node, "lineno", 1))
        # Use indentation of the first body statement
        if 1 <= insert_at <= len(src_lines):
            indent = compute_indent_of_line(src_lines[insert_at - 1])
        else:
            indent = "    "

    # Prepare indented docstring lines
    block = [indent + line + "\n" for line in doc_lines]
    # Insert before first statement line (convert to 0-based index)
    idx = max(0, insert_at - 1)
    src_lines[idx:idx] = block


def replace_or_insert_docstring(
    src_lines: List[str], node: ast.AST, doc_lines: List[str]
) -> None:
    """Replace a placeholder docstring or insert a new one if missing.

    If the first statement in the node is a string literal (docstring), it is
    replaced in-place. Otherwise, a new docstring is inserted before the first
    statement.
    """
    if (
        not hasattr(node, "body")
        or not isinstance(getattr(node, "body"), list)
        or not node.body
    ):
        insert_docstring(src_lines, node, doc_lines)
        return
    first_stmt = node.body[0]
    if (
        isinstance(first_stmt, ast.Expr)
        and isinstance(getattr(first_stmt, "value", None), (ast.Str, ast.Constant))
        and isinstance(
            (
                first_stmt.value.s
                if isinstance(first_stmt.value, ast.Str)
                else getattr(first_stmt.value, "value", None)
            ),
            str,
        )
    ):
        # Replace existing docstring
        start = getattr(first_stmt, "lineno", None)
        end = getattr(first_stmt, "end_lineno", None)
        if start is None or end is None:
            insert_docstring(src_lines, node, doc_lines)
            return
        # Indentation based on the start line
        indent = compute_indent_of_line(src_lines[start - 1])
        block = [indent + line + "\n" for line in doc_lines]
        src_lines[start - 1 : end] = block
    else:
        insert_docstring(src_lines, node, doc_lines)


def add_docstrings_to_source(path: str, source: str) -> str:
    """Add or update PEP 257-style docstrings for functions and classes in source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    # Collect targets (functions/classes without docstrings)
    targets: List[Tuple[ast.AST, List[str]]] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not has_docstring(node, source) or is_placeholder_docstring(node):
                targets.append((node, generate_docstring_for_function(node)))
        elif isinstance(node, ast.ClassDef):
            if not has_docstring(node, source) or is_placeholder_docstring(node):
                targets.append((node, generate_docstring_for_class(node)))

    if not targets:
        return source

    # To avoid line number shifts for later insertions, sort by insertion line descending
    def insertion_lineno(n: ast.AST) -> int:
        """Insertion lineno.

        Args:
            n: Parameter.

        Returns:
            Return value.
        """
        if hasattr(n, "body") and getattr(n, "body"):
            return getattr(getattr(n, "body")[0], "lineno", getattr(n, "lineno", 10**9))
        return getattr(n, "lineno", 10**9)

    targets.sort(key=lambda t: insertion_lineno(t[0]), reverse=True)

    lines = source.splitlines(keepends=True)
    for node, doc in targets:
        replace_or_insert_docstring(lines, node, doc)

    return "".join(lines)


def process_paths(paths: List[str], write: bool) -> List[Change]:
    """Process paths and add or update docstrings in matching Python files."""
    changes: List[Change] = []
    for p in paths:
        if os.path.isdir(p):
            files = list_py_files(p)
        else:
            files = [p] if is_python_file(p) else []
        for f in files:
            original = read_text(f)
            updated = add_docstrings_to_source(f, original)
            if updated != original:
                changes.append(Change(path=f, original=original, updated=updated))
                if write:
                    write_text(f, updated)
    return changes


def main() -> None:
    """CLI entry point for adding or updating docstrings in the codebase."""
    parser = argparse.ArgumentParser(
        description="Auto-add English docstrings to Python functions/classes."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["service", "scripts"],
        help="Files or directories to process",
    )
    parser.add_argument(
        "--write", action="store_true", help="Apply changes in-place (default: dry-run)"
    )
    args = parser.parse_args()

    changes = process_paths(args.paths, write=args.write)
    if not changes:
        print("No changes needed.")
        return

    if not args.write:
        for ch in changes:
            a = ch.original.splitlines(keepends=True)
            b = ch.updated.splitlines(keepends=True)
            diff = difflib.unified_diff(
                a, b, fromfile=ch.path + " (orig)", tofile=ch.path + " (new)"
            )
            print("".join(diff))
        print(
            f"\nDry-run: {len(changes)} files would be updated. Run with --write to apply."
        )
    else:
        print(f"Applied docstrings to {len(changes)} files.")


if __name__ == "__main__":
    main()
