#!/usr/bin/env python3
"""
Audit script for the /frameworks directory of the AWS Neuron SDK documentation.

Detects orphaned pages (not referenced by any toctree, :doc:, :ref:, or
.. include:: directive) and stale pages (containing outdated references).

Usage:
    python3 _utilities/audit_frameworks.py --root . --output audit-report.md
"""

import argparse
import os
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Reference extraction helpers
# ---------------------------------------------------------------------------

# Regex patterns for RST directives and roles
TOCTREE_BLOCK_RE = re.compile(r"^\.\.\s+toctree::", re.MULTILINE)
DOC_ROLE_RE = re.compile(r":doc:`(?:[^<`]*<)?(/[^>`]+|[^>`/][^>`]*)`")
REF_ROLE_RE = re.compile(r":ref:`(?:[^<`]*<)?([^>`]+)`")
INCLUDE_RE = re.compile(r"^\.\.\s+include::\s+(.+)$", re.MULTILINE)
LABEL_RE = re.compile(r"^\.\.\s+_([a-zA-Z0-9_-]+)\s*:", re.MULTILINE)


def _resolve_path(ref: str, referencing_file: Path, root: Path) -> str | None:
    """Resolve a toctree/doc/include reference to a repo-relative path."""
    ref = ref.strip()
    if not ref:
        return None

    # Absolute path (starts with /)
    if ref.startswith("/"):
        resolved = ref.lstrip("/")
    else:
        # Relative to the directory of the referencing file
        ref_dir = referencing_file.parent.relative_to(root)
        resolved = str(ref_dir / ref)

    # Normalise (collapse ..)
    resolved = os.path.normpath(resolved)
    return resolved


def _resolve_to_files(base: str, root: Path) -> list[str]:
    """Given a resolved base path, return candidate file paths that exist."""
    candidates = []
    # Direct file match (already has extension)
    if (root / base).is_file():
        candidates.append(base)
        return candidates

    # Try common extensions
    for ext in (".rst", ".ipynb", ".txt"):
        p = base + ext
        if (root / p).is_file():
            candidates.append(p)

    # Could be a directory with index.rst
    idx = os.path.join(base, "index.rst")
    if (root / idx).is_file():
        candidates.append(idx)

    return candidates


def extract_toctree_entries(content: str, filepath: Path, root: Path) -> set[str]:
    """Extract all file paths referenced in toctree directives."""
    referenced: set[str] = set()
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        if TOCTREE_BLOCK_RE.match(lines[i]):
            # Skip toctree options (lines starting with : or blank within indent)
            i += 1
            # Skip blank lines and option lines
            while i < len(lines):
                stripped = lines[i].strip()
                if stripped == "" or stripped.startswith(":"):
                    i += 1
                    continue
                break
            # Now read toctree entries (indented non-empty lines)
            while i < len(lines):
                line = lines[i]
                stripped = line.strip()
                if stripped == "":
                    i += 1
                    continue
                # Check if still indented (part of toctree body)
                if line[0] in (" ", "\t"):
                    # Entry may have a title: "Title <path>" or just "path"
                    entry = stripped
                    m = re.match(r".*<(.+)>", entry)
                    if m:
                        entry = m.group(1).strip()
                    # Resolve the path
                    resolved = _resolve_path(entry, filepath, root)
                    if resolved:
                        for f in _resolve_to_files(resolved, root):
                            referenced.add(f)
                    i += 1
                else:
                    break
        else:
            i += 1
    return referenced


def extract_doc_refs(content: str, filepath: Path, root: Path) -> set[str]:
    """Extract all file paths referenced via :doc: roles."""
    referenced: set[str] = set()
    for m in DOC_ROLE_RE.finditer(content):
        ref = m.group(1).strip()
        resolved = _resolve_path(ref, filepath, root)
        if resolved:
            for f in _resolve_to_files(resolved, root):
                referenced.add(f)
    return referenced


def extract_include_refs(content: str, filepath: Path, root: Path) -> set[str]:
    """Extract all file paths referenced via .. include:: directives."""
    referenced: set[str] = set()
    for m in INCLUDE_RE.finditer(content):
        ref = m.group(1).strip()
        resolved = _resolve_path(ref, filepath, root)
        if resolved:
            for f in _resolve_to_files(resolved, root):
                referenced.add(f)
    return referenced


def extract_ref_labels(content: str) -> set[str]:
    """Extract all :ref: label targets from content."""
    return set(m.group(1) for m in REF_ROLE_RE.finditer(content))


def extract_label_definitions(content: str) -> set[str]:
    """Extract all label definitions (.. _label:) from content."""
    return set(m.group(1) for m in LABEL_RE.finditer(content))


# ---------------------------------------------------------------------------
# Orphan detection
# ---------------------------------------------------------------------------

def find_all_framework_files(root: Path) -> tuple[set[str], set[str], set[str]]:
    """Find all .rst, .ipynb, and .txt files under frameworks/.

    Returns (rst_files, ipynb_files, txt_files) as repo-relative paths.
    """
    rst_files: set[str] = set()
    ipynb_files: set[str] = set()
    txt_files: set[str] = set()
    fw_dir = root / "frameworks"
    if not fw_dir.is_dir():
        return rst_files, ipynb_files, txt_files
    for p in fw_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root))
        if "__pycache__" in rel:
            continue
        if p.suffix == ".rst":
            rst_files.add(rel)
        elif p.suffix == ".ipynb":
            ipynb_files.add(rel)
        elif p.suffix == ".txt":
            txt_files.add(rel)
    return rst_files, ipynb_files, txt_files


def collect_all_references(root: Path) -> tuple[set[str], set[str], set[str]]:
    """Scan ALL .rst and .txt files in the repo to collect references.

    Returns (toctree_and_doc_refs, include_refs, ref_labels_used).
    We scan the entire repo (not just /frameworks) so that references
    from root index.rst, setup/, about-neuron/, etc. are captured.
    """
    toctree_doc_refs: set[str] = set()
    include_refs: set[str] = set()
    ref_labels_used: set[str] = set()

    # Directories to skip entirely
    skip_dirs = {"_build", ".git", "venv", ".venv", "__pycache__", ".kiro",
                 ".vscode", ".github", "node_modules", "_backup-rn"}

    for ext in ("*.rst", "*.txt"):
        for p in root.rglob(ext):
            # Skip files in excluded directories
            rel = str(p.relative_to(root))
            parts = Path(rel).parts
            if any(part in skip_dirs for part in parts):
                continue
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            toctree_doc_refs |= extract_toctree_entries(content, p, root)
            toctree_doc_refs |= extract_doc_refs(content, p, root)
            include_refs |= extract_include_refs(content, p, root)
            ref_labels_used |= extract_ref_labels(content)

    return toctree_doc_refs, include_refs, ref_labels_used


def build_label_to_file_map(root: Path) -> dict[str, str]:
    """Build a mapping from :ref: label -> repo-relative file path.

    Only scans files under frameworks/ since we only need to know
    which framework files are referenced via :ref:.
    """
    label_map: dict[str, str] = {}
    fw_dir = root / "frameworks"
    if not fw_dir.is_dir():
        return label_map
    for p in fw_dir.rglob("*.rst"):
        rel = str(p.relative_to(root))
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for label in extract_label_definitions(content):
            label_map[label] = rel
    return label_map


def detect_orphans(root: Path) -> list[dict]:
    """Detect orphaned pages under /frameworks.

    Returns a list of dicts with keys: path, type, reason, action.
    """
    rst_files, ipynb_files, txt_files = find_all_framework_files(root)
    toctree_doc_refs, include_refs, ref_labels_used = collect_all_references(root)
    label_map = build_label_to_file_map(root)

    # Files referenced via :ref: labels
    ref_referenced_files: set[str] = set()
    for label in ref_labels_used:
        if label in label_map:
            ref_referenced_files.add(label_map[label])

    # All referenced content files (rst + ipynb)
    all_content_refs = toctree_doc_refs | ref_referenced_files
    # All referenced include files (txt)
    all_include_refs = include_refs

    orphans: list[dict] = []

    # Check .rst and .ipynb files against toctree/doc/ref references
    for f in sorted(rst_files | ipynb_files):
        if f not in all_content_refs and f not in all_include_refs:
            ext = Path(f).suffix
            orphans.append({
                "path": f,
                "type": ext,
                "reason": "Not in any toctree or cross-reference",
                "action": "Delete",
            })

    # Check .txt files against include references only
    for f in sorted(txt_files):
        if f not in all_include_refs:
            orphans.append({
                "path": f,
                "type": ".txt (include fragment)",
                "reason": "Not referenced by any .. include:: directive",
                "action": "Delete",
            })

    return orphans


# ---------------------------------------------------------------------------
# Stale page detection
# ---------------------------------------------------------------------------

# Staleness indicator patterns
STALE_OS_RE = re.compile(
    r"Ubuntu\s+18\.04|Ubuntu\s+20\.04|Amazon\s+Linux\s+2(?!\s*023)(?!\s*\d{3})\b",
    re.IGNORECASE,
)
STALE_PYTHON_RE = re.compile(
    r"Python\s+3\.[0-9](?!\d)\b",  # matches Python 3.0 through 3.9
)
STALE_SDK_RE = re.compile(r"Neuron\s+SDK\s+2\.(\d+)")
TORCH_NEURON_SETUP_RE = re.compile(
    r"torch-neuron.*(?:setup|install|update)",
    re.IGNORECASE,
)
NEURON_CC_RE = re.compile(r"\bneuron-cc\b")


def _check_stale_python(content: str) -> list[str]:
    """Find references to Python versions below 3.10."""
    indicators = []
    for m in STALE_PYTHON_RE.finditer(content):
        ver_str = m.group(0)
        # Extract minor version
        minor = int(ver_str.split(".")[-1])
        if minor < 10:
            indicators.append(ver_str)
    return list(set(indicators))


def _check_stale_sdk(content: str) -> list[str]:
    """Find references to Neuron SDK versions older than 2.20."""
    indicators = []
    for m in STALE_SDK_RE.finditer(content):
        ver = int(m.group(1))
        if ver < 20:
            indicators.append(m.group(0))
    return list(set(indicators))


def _check_stale_os(content: str) -> list[str]:
    """Find references to unsupported OS versions."""
    return list(set(m.group(0) for m in STALE_OS_RE.finditer(content)))


def _check_torch_neuron_unsupported_os(content: str) -> list[str]:
    """Flag torch-neuron setup/update instructions for unsupported OS."""
    indicators = []
    if TORCH_NEURON_SETUP_RE.search(content):
        os_refs = _check_stale_os(content)
        if os_refs:
            indicators.append(
                f"torch-neuron setup/update with unsupported OS: {', '.join(os_refs)}"
            )
    return indicators


def _check_neuron_cc(content: str) -> list[str]:
    """Flag deprecated neuron-cc references."""
    if NEURON_CC_RE.search(content):
        return ["References deprecated neuron-cc compiler"]
    return []


def detect_stale_pages(root: Path) -> list[dict]:
    """Detect stale pages under /frameworks.

    Returns a list of dicts with keys: path, indicators, recommendation.
    """
    stale: list[dict] = []
    fw_dir = root / "frameworks"
    if not fw_dir.is_dir():
        return stale

    for p in fw_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in (".rst", ".txt"):
            continue
        rel = str(p.relative_to(root))
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        indicators: list[str] = []
        indicators.extend(_check_stale_os(content))
        indicators.extend(_check_stale_python(content))
        indicators.extend(_check_stale_sdk(content))
        indicators.extend(_check_torch_neuron_unsupported_os(content))
        indicators.extend(_check_neuron_cc(content))

        if indicators:
            # Determine recommendation
            is_archival = (
                "mxnet-neuron/" in rel
                or "tensorflow/" in rel
                or ("torch-neuron/" in rel and "torch-neuronx/" not in rel)
            )
            if is_archival:
                rec = "Will be archived"
            else:
                rec = "Update or archive"
            stale.append({
                "path": rel,
                "indicators": "; ".join(sorted(set(indicators))),
                "recommendation": rec,
            })

    return sorted(stale, key=lambda x: x["path"])


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(orphans: list[dict], stale: list[dict]) -> str:
    """Generate the audit report as Markdown."""
    lines: list[str] = []
    lines.append("# Frameworks Audit Report\n")

    # Orphaned pages
    lines.append("## Orphaned Pages\n")
    if orphans:
        lines.append("| File Path | Type | Reason | Action |")
        lines.append("|---|---|---|---|")
        for o in orphans:
            lines.append(
                f"| {o['path']} | {o['type']} | {o['reason']} | {o['action']} |"
            )
    else:
        lines.append("No orphaned pages detected.\n")

    lines.append("")

    # Stale pages
    lines.append("## Stale Pages\n")
    if stale:
        lines.append("| File Path | Staleness Indicators | Recommendation |")
        lines.append("|---|---|---|")
        for s in stale:
            lines.append(
                f"| {s['path']} | {s['indicators']} | {s['recommendation']} |"
            )
    else:
        lines.append("No stale pages detected.\n")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Audit /frameworks for orphaned and stale pages."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root directory (default: current directory)",
    )
    parser.add_argument(
        "--output",
        default="audit-report.md",
        help="Output file path for the audit report (default: audit-report.md)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"Auditing frameworks under: {root}")

    orphans = detect_orphans(root)
    print(f"Found {len(orphans)} orphaned page(s).")

    stale = detect_stale_pages(root)
    print(f"Found {len(stale)} stale page(s).")

    report = generate_report(orphans, stale)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.write_text(report, encoding="utf-8")
    print(f"Audit report written to: {output_path}")


if __name__ == "__main__":
    main()
