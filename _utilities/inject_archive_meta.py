#!/usr/bin/env python3
"""Inject noindex/nofollow meta directives and deprecation banners into archived .rst files."""

import os
import re
import sys

META_BLOCK = """.. meta::
   :noindex:
   :nofollow:
   :description: This content is archived and no longer maintained.
   :date-modified: 2026-03-11

"""

WARNING_TEMPLATE = """
.. warning::

   This document is archived. {framework} is no longer officially supported
   by the AWS Neuron SDK. It is provided for reference only. For current
   framework support, see :doc:`/frameworks/index`.

"""

# Default for backward compatibility
WARNING_BLOCK = WARNING_TEMPLATE.format(framework="MXNet")


def find_title_end(lines):
    """Find the line index after the RST title underline.
    
    RST titles look like:
        Title Text
        ==========
    
    or with overline:
        ==========
        Title Text
        ==========
    
    Returns the index of the line AFTER the title underline, or -1 if not found.
    """
    title_chars = set('=-~^"\'`#*+_.')
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        # Check if this line is an underline (all same char, at least 3 chars)
        if len(line) >= 3 and len(set(line)) == 1 and line[0] in title_chars:
            # Check if previous line is text (title) - this is an underline
            if i > 0 and lines[i-1].strip() and not (len(set(lines[i-1].rstrip())) == 1 and lines[i-1].rstrip()[0] in title_chars):
                return i + 1
            # Check if next line is text and line after that is underline (overline pattern)
            if i + 2 < len(lines) and lines[i+1].strip():
                next_next = lines[i+2].rstrip()
                if len(next_next) >= 3 and len(set(next_next)) == 1 and next_next[0] in title_chars:
                    return i + 3
        i += 1
    return -1


def inject_meta_and_warning(filepath, framework="MXNet"):
    """Inject meta block at top and warning after title in an RST file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Skip if already has noindex meta
    if ':noindex:' in content:
        print(f"  SKIP (already has meta): {filepath}")
        return
    
    warning_block = WARNING_TEMPLATE.format(framework=framework)
    
    lines = content.split('\n')
    
    # Separate any leading labels (.. _label:) and blank lines
    # These need to stay before the meta block
    label_lines = []
    content_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('.. _') and stripped.endswith(':'):
            label_lines.append(line)
            content_start = i + 1
        elif stripped == '' and all(l.strip().startswith('.. _') for l in lines[:i] if l.strip()):
            label_lines.append(line)
            content_start = i + 1
        else:
            break
    
    # Build the content after labels
    remaining_lines = lines[content_start:]
    remaining_content = '\n'.join(remaining_lines)
    
    # Find title end in remaining content
    title_end = find_title_end(remaining_lines)
    
    if title_end >= 0:
        # Insert warning after title
        before_title = '\n'.join(remaining_lines[:title_end])
        after_title = '\n'.join(remaining_lines[title_end:])
        
        new_remaining = before_title + '\n' + warning_block + after_title
    else:
        # No title found, just add warning at the start of content
        print(f"  WARNING: No title found in {filepath}")
        new_remaining = warning_block + remaining_content
    
    # Reconstruct: labels + meta + content with warning
    label_section = '\n'.join(label_lines) + '\n' if label_lines else ''
    new_content = label_section + META_BLOCK + new_remaining
    
    # Ensure file ends with newline
    if not new_content.endswith('\n'):
        new_content += '\n'
    
    with open(filepath, 'w') as f:
        f.write(new_content)
    
    print(f"  OK: {filepath}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Inject archive meta into .rst files')
    parser.add_argument('archive_dir', nargs='?', default='archive/mxnet-neuron',
                        help='Directory containing .rst files to process')
    parser.add_argument('--framework', default='MXNet',
                        help='Framework name for the deprecation warning (e.g., MXNet, TensorFlow)')
    args = parser.parse_args()

    archive_dir = args.archive_dir
    framework = args.framework
    
    rst_files = []
    for root, dirs, files in os.walk(archive_dir):
        for fname in files:
            if fname.endswith('.rst'):
                rst_files.append(os.path.join(root, fname))
    
    rst_files.sort()
    print(f"Processing {len(rst_files)} .rst files in {archive_dir}:")
    
    for filepath in rst_files:
        inject_meta_and_warning(filepath, framework=framework)
    
    print(f"\nDone. Processed {len(rst_files)} files.")


if __name__ == '__main__':
    main()
