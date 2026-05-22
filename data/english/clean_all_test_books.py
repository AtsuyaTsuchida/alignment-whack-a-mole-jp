#!/usr/bin/env python3
"""DL した PG / gutenberg.ca テキストのヘッダ・フッタを除去。"""
import re
from pathlib import Path

PG_START = re.compile(r'\*\*\*\s*START OF (THE |THIS )?PROJECT GUTENBERG.*?\*\*\*', re.IGNORECASE)
PG_END = re.compile(r'\*\*\*\s*END OF (THE |THIS )?PROJECT GUTENBERG.*?\*\*\*', re.IGNORECASE)
# gutenberg.ca はヘッダのフォーマットが異なる
CA_START = re.compile(r'(THE OLD MAN AND THE SEA|Author:.*Hemingway)', re.IGNORECASE)

base = Path('/Users/s29524/Desktop/Qosmo/Alignment-Whack-a-Mole/data/english')
test_books = [
    'carroll/raw/alice.txt',
    'melville/raw/moby_dick.txt',
    'twain/raw/tom_sawyer.txt',
    'baum/raw/wizard_of_oz.txt',
    'shelley/raw/frankenstein.txt',
    'fitzgerald/raw/great_gatsby.txt',
    'hemingway/raw/old_man_and_the_sea.txt',
]

for rel in test_books:
    f = base / rel
    text = f.read_text(encoding='utf-8', errors='replace')
    text = re.sub(r'\r\n?', '\n', text)

    pg_start = PG_START.search(text)
    pg_end = PG_END.search(text)
    if pg_start and pg_end:
        body = text[pg_start.end():pg_end.start()].strip()
        source = 'PG'
    else:
        # gutenberg.ca: 全文をそのまま使い、本文開始までを軽く除去
        body = text.strip()
        source = 'CA(or raw)'
    body = re.sub(r'\n{3,}', '\n\n', body)
    f.write_text(body, encoding='utf-8')
    words = len(body.split())
    print(f"  ✓ {f.name}: {words:,} words [{source}]")
