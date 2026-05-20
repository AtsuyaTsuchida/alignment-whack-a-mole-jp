#!/usr/bin/env python3
"""Project Gutenberg のヘッダ・フッタを除去する。"""
import re
import sys
from pathlib import Path

START_RE = re.compile(r'\*\*\*\s*START OF (THE |THIS )?PROJECT GUTENBERG.*?\*\*\*', re.IGNORECASE)
END_RE = re.compile(r'\*\*\*\s*END OF (THE |THIS )?PROJECT GUTENBERG.*?\*\*\*', re.IGNORECASE)

raw_dir = Path('/Users/s29524/Desktop/Qosmo/Alignment-Whack-a-Mole/data/english/dickens/raw')
for f in sorted(raw_dir.glob('*.txt')):
    text = f.read_text(encoding='utf-8', errors='replace')
    start = START_RE.search(text)
    end = END_RE.search(text)
    if not (start and end):
        print(f"  ⚠ {f.name}: markers not found"); continue
    body = text[start.end():end.start()].strip()
    # 軽い正規化: 連続改行を空行に統一
    body = re.sub(r'\r\n?', '\n', body)
    body = re.sub(r'\n{3,}', '\n\n', body)
    f.write_text(body, encoding='utf-8')
    words = len(body.split())
    print(f"  ✓ {f.name}: {words:,} words")
