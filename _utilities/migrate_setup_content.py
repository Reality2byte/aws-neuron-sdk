#!/usr/bin/env python3
"""
Setup Content Migration Script

Maps old setup file paths to new framework-first paths and generates
a migration report. This script does NOT move files — it produces a
report of what references exist and where they should point.

Usage:
    python3 _utilities/migrate_setup_content.py [--dry-run] [--fix]

Options:
    --dry-run   Show what would be changed without modifying files (default)
    --fix       Apply changes to files
"""

import argparse
import os
import re
import sys
from pathlib import Path

# Old path → new path mapping
PATH_MAP = {
    "/setup/torch-neuronx": "/setup/pytorch/index",
    "/setup/jax-neuronx": "/setup/jax/index",
    "/setup/tensorflow-neuronx": "/frameworks/tensorflow/index",
    "/setup/setup-neuronx": "/setup/index",
    "/setup/setup-neuron": "/setup/index",
    "/setup/mxnet-neuron": "/archive/mxnet-neuron/index",
}

# External URL mapping (for hardcoded URLs in tutorials)
URL_MAP = {
    "setup/torch-neuronx.html": "setup/pytorch/index.html",
    "setup/jax-neuronx.html": "setup/jax/index.html",
}

# Directories to scan
SCAN_DIRS = [
    "about-neuron",
    "frameworks",
    "libraries",
    "tools",
    "compiler",
    "containers",
    "devflows",
    "release-notes",
    "setup",
    "nki",
    "dlami",
]

# Directories to skip
SKIP_DIRS = {"_build", ".git", "__pycache__", ".venv", "node_modules"}


def find_rst_files(base_dir: str) -> list[Path]:
    """Find all .rst files in scan directories."""
    files = []
    for scan_dir in SCAN_DIRS:
        dir_path = Path(base_dir) / scan_dir
        if dir_path.exists():
            for rst_file in dir_path.rglob("*.rst"):
                if not any(skip in rst_file.parts for skip in SKIP_DIRS):
                    files.append(rst_file)
    return sorted(files)


def find_references(content: str, file_path: Path) -> list[dict]:
    """Find old setup path references in file content."""
    refs = []

    # Match :doc: references
    for old_path, new_path in PATH_MAP.items():
        pattern = re.compile(
            rf":doc:`([^`]*<)?{re.escape(old_path)}(>)?`", re.IGNORECASE
        )
        for match in pattern.finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            refs.append(
                {
                    "file": str(file_path),
                    "line": line_num,
                    "old": match.group(0),
                    "old_path": old_path,
                    "new_path": new_path,
                    "type": "doc_ref",
                }
            )

    # Match :ref: references to old labels
    old_labels = {
        "setup-torch-neuronx": "pytorch-setup",
        "setup-jax-neuronx": "jax-setup",
        "setup-tensorflow-neuronx": "tensorflow-setup",
    }
    for old_label, new_label in old_labels.items():
        pattern = re.compile(rf":ref:`([^`]*<)?{re.escape(old_label)}(>)?`")
        for match in pattern.finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            refs.append(
                {
                    "file": str(file_path),
                    "line": line_num,
                    "old": match.group(0),
                    "old_label": old_label,
                    "new_label": new_label,
                    "type": "ref_label",
                }
            )

    # Match hardcoded URLs
    for old_url, new_url in URL_MAP.items():
        if old_url in content:
            line_num = content[: content.index(old_url)].count("\n") + 1
            refs.append(
                {
                    "file": str(file_path),
                    "line": line_num,
                    "old_url": old_url,
                    "new_url": new_url,
                    "type": "url",
                }
            )

    return refs


def apply_fix(file_path: Path, refs: list[dict]) -> bool:
    """Apply reference fixes to a file."""
    content = file_path.read_text()
    modified = False

    for ref in refs:
        if ref["type"] == "doc_ref":
            old = ref["old_path"]
            new = ref["new_path"]
            new_content = content.replace(old, new)
            if new_content != content:
                content = new_content
                modified = True
        elif ref["type"] == "url":
            old = ref["old_url"]
            new = ref["new_url"]
            new_content = content.replace(old, new)
            if new_content != content:
                content = new_content
                modified = True

    if modified:
        file_path.write_text(content)
    return modified


def main():
    parser = argparse.ArgumentParser(description="Setup content migration script")
    parser.add_argument(
        "--fix", action="store_true", help="Apply changes (default is dry-run)"
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rst_files = find_rst_files(base_dir)

    print(f"Scanning {len(rst_files)} .rst files...")
    print()

    all_refs = []
    for rst_file in rst_files:
        content = rst_file.read_text()
        refs = find_references(content, rst_file)
        all_refs.extend(refs)

    if not all_refs:
        print("No old setup references found. Migration complete.")
        return

    # Group by file
    by_file = {}
    for ref in all_refs:
        by_file.setdefault(ref["file"], []).append(ref)

    print(f"Found {len(all_refs)} references in {len(by_file)} files:")
    print()

    for file_path, refs in sorted(by_file.items()):
        print(f"  {file_path}:")
        for ref in refs:
            if ref["type"] == "doc_ref":
                print(f"    L{ref['line']}: {ref['old_path']} → {ref['new_path']}")
            elif ref["type"] == "ref_label":
                print(f"    L{ref['line']}: {ref['old_label']} → {ref['new_label']}")
            elif ref["type"] == "url":
                print(f"    L{ref['line']}: {ref['old_url']} → {ref['new_url']}")
        print()

    if args.fix:
        fixed_count = 0
        for file_path, refs in by_file.items():
            if apply_fix(Path(file_path), refs):
                fixed_count += 1
                print(f"  ✓ Fixed: {file_path}")
        print(f"\nFixed {fixed_count} files.")
    else:
        print("Dry run — no files modified. Use --fix to apply changes.")


if __name__ == "__main__":
    main()
