#!/usr/bin/env python3
"""Add missing .. meta:: blocks with :description:, :keywords:, and :date-modified: to .rst files."""

import os
import re
import sys
from pathlib import Path

TODAY = "2026-03-13"

# Map file paths to sensible descriptions/keywords based on content
def infer_meta(filepath: str, content: str) -> dict:
    """Infer description and keywords from file path and content."""
    rel = filepath.replace("frameworks/", "")
    
    # Extract title from RST
    title = ""
    lines = content.split("\n")
    title_chars = set("=-~^\"'`#*+_.")
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if (len(stripped) >= 3 and len(set(stripped)) == 1 
            and stripped[0] in title_chars and i > 0):
            candidate = lines[i-1].strip()
            if candidate and not candidate.startswith(".."):
                title = candidate
                break
    
    # Build description from title or path
    if title:
        desc = f"{title} - AWS Neuron SDK documentation"
    else:
        desc = f"AWS Neuron SDK documentation for {os.path.basename(filepath).replace('.rst', '').replace('-', ' ')}"
    
    # Build keywords from path components
    kw_parts = set()
    if "torch" in rel:
        kw_parts.update(["PyTorch", "AWS Neuron"])
    if "neuronx" in rel:
        kw_parts.update(["torch-neuronx", "Trainium", "Inferentia"])
    if "jax" in rel:
        kw_parts.update(["JAX", "AWS Neuron", "JAX NeuronX"])
    if "training" in rel.lower():
        kw_parts.add("training")
    if "inference" in rel.lower():
        kw_parts.add("inference")
    if "setup" in rel.lower() or "install" in rel.lower() or "update" in rel.lower():
        kw_parts.add("setup")
    if "tutorial" in rel.lower():
        kw_parts.add("tutorials")
    if "api" in rel.lower():
        kw_parts.add("API reference")
    if "profil" in rel.lower():
        kw_parts.add("profiling")
    if "troubleshoot" in rel.lower():
        kw_parts.add("troubleshooting")
    if "debug" in rel.lower():
        kw_parts.add("debugging")
    if not kw_parts:
        kw_parts.update(["AWS Neuron", "machine learning"])
    
    keywords = ", ".join(sorted(kw_parts))
    
    return {"description": desc, "keywords": keywords}


def has_meta_field(content: str, field: str) -> bool:
    """Check if a .. meta:: block contains a specific field."""
    return bool(re.search(rf"^\s+:{field}:", content, re.MULTILINE))


def process_file(filepath: str, dry_run: bool = False):
    """Process a single .rst file to ensure it has complete meta block."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    
    # Skip include-only fragments (no title, very short)
    if len(content.strip()) < 50:
        print(f"  SKIP (fragment): {filepath}")
        return False
    
    has_meta = ".. meta::" in content
    has_desc = has_meta_field(content, "description")
    has_kw = has_meta_field(content, "keywords")
    has_date = has_meta_field(content, "date-modified")
    
    if has_desc and has_kw and has_date:
        print(f"  OK (complete): {filepath}")
        return False
    
    meta = infer_meta(filepath, content)
    
    if has_meta:
        # Meta block exists but missing fields — add them
        missing = []
        if not has_desc:
            missing.append(f"   :description: {meta['description']}")
        if not has_kw:
            missing.append(f"   :keywords: {meta['keywords']}")
        if not has_date:
            missing.append(f"   :date-modified: {TODAY}")
        
        insert_text = "\n".join(missing)
        
        # Find the end of the existing meta block (last line starting with :field:)
        lines = content.split("\n")
        meta_start = -1
        meta_last_field = -1
        for i, line in enumerate(lines):
            if line.strip() == ".. meta::":
                meta_start = i
            elif meta_start >= 0 and re.match(r"\s+:\w", line):
                meta_last_field = i
            elif meta_start >= 0 and meta_last_field >= 0 and not line.strip().startswith(":") and not (line.strip() and not line[0].isspace()):
                break
        
        if meta_last_field >= 0:
            lines.insert(meta_last_field + 1, insert_text)
            new_content = "\n".join(lines)
        else:
            # Fallback: insert after .. meta:: line
            new_content = content.replace(".. meta::", f".. meta::\n{insert_text}", 1)
    else:
        # No meta block at all — add one at the top (after any labels)
        lines = content.split("\n")
        insert_idx = 0
        
        # Skip leading labels (.. _label:) and blank lines
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(".. _") and stripped.endswith(":"):
                insert_idx = i + 1
            elif stripped == "" and i <= insert_idx + 1:
                insert_idx = i + 1
            else:
                break
        
        meta_block = (
            f"\n.. meta::\n"
            f"   :description: {meta['description']}\n"
            f"   :keywords: {meta['keywords']}\n"
            f"   :date-modified: {TODAY}\n\n"
        )
        
        lines.insert(insert_idx, meta_block)
        new_content = "\n".join(lines)
    
    action = "UPDATE" if has_meta else "ADD"
    fields = []
    if not has_desc: fields.append("description")
    if not has_kw: fields.append("keywords")
    if not has_date: fields.append("date-modified")
    print(f"  {action} ({', '.join(fields)}): {filepath}")
    
    if not dry_run:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
    
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Add meta blocks to .rst files")
    parser.add_argument("directory", default="frameworks", nargs="?")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    args = parser.parse_args()
    
    root = Path(args.directory)
    rst_files = sorted(root.rglob("*.rst"))
    
    print(f"Scanning {len(rst_files)} .rst files in {root}/:")
    changed = 0
    for f in rst_files:
        if process_file(str(f), dry_run=args.dry_run):
            changed += 1
    
    print(f"\n{'Would change' if args.dry_run else 'Changed'} {changed} file(s).")


if __name__ == "__main__":
    main()
