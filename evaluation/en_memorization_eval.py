#!/usr/bin/env python3
"""Step 3: 英語版 — word 単位の n-gram 一致による記憶化評価（論文準拠）。

論文 Algorithm 1 を word 単位で実装:
  - BMC@k:            原文中で k word 以上一致したブロックがカバーする割合
  - Instruction trim: あらすじ m-gram と重なる位置を除外
  - 最長記憶ブロック:  trim 後の最長連続一致
  - 最長再現スパン:    trim 前の最長連続一致

使い方:
    python evaluation/en_memorization_eval.py \
        --test_book data/english/dickens/chunks/a_christmas_carol.json \
        --generation_file results/a_christmas_carol_gens.json \
        --k 5 --trim_m 5
"""

import argparse
import json
import re
from collections import defaultdict
from typing import List, Tuple, Dict


# 単語分割: アルファベット・数字・アポストロフィをまとめて 1 word
WORD_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> List[str]:
    """英語テキストを word 列に分解（小文字化なし、句読点除去）。"""
    return WORD_RE.findall(text)


def _merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _subtract_from_interval(base, removes):
    s, e = base
    clamped = [(max(s, a), min(e, b)) for a, b in removes if not (b <= s or a >= e)]
    rm = _merge_intervals([r for r in clamped if r[0] < r[1]])
    if not rm:
        return [base]
    out, cur = [], s
    for a, b in rm:
        if cur < a:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < e:
        out.append((cur, e))
    return out


def _mgram_set(tokens, m):
    if m <= 0 or len(tokens) < m:
        return set()
    return {tuple(tokens[i:i+m]) for i in range(len(tokens) - m + 1)}


def trim_instruction_mgrams(book_tokens, instr_tokens, intervals, min_length, m):
    """Algorithm 1 Stage 2: instruction の m-gram と一致する位置を除外。"""
    if not intervals:
        return []
    instr_mgrams = _mgram_set(instr_tokens, m)
    all_trimmed = []
    for raw_iv in intervals:
        s, e = raw_iv
        span_len = e - s
        if m <= 0 or span_len < m or not instr_mgrams:
            if span_len >= min_length:
                all_trimmed.append(raw_iv)
            continue
        removes = []
        for i in range(span_len - m + 1):
            if tuple(book_tokens[s+i:s+i+m]) in instr_mgrams:
                removes.append((s+i, s+i+m))
        removes = _merge_intervals(removes)
        for start, end in _subtract_from_interval(raw_iv, removes):
            if end - start >= min_length:
                all_trimmed.append((start, end))
    return _merge_intervals(all_trimmed)


def build_book_index(book_examples):
    """全 excerpt を連結して書籍全体の word 配列を作る。"""
    exs = sorted(book_examples, key=lambda x: int(re.search(r'(\d+)', x['excerpt_id']).group(1)))
    all_tokens = []
    spans = []
    for ex in exs:
        toks = tokenize(ex['excerpt_text'])
        start = len(all_tokens)
        all_tokens.extend(toks)
        spans.append((start, len(all_tokens), ex['excerpt_id']))
    return all_tokens, spans


def build_kgram_index(book_tokens, k):
    """k-gram インデックスを 1 回だけ構築。"""
    idx = defaultdict(list)
    for i in range(len(book_tokens) - k + 1):
        idx[tuple(book_tokens[i:i+k])].append(i)
    return idx


def find_matches(gen_tokens, book_tokens, idx, k):
    """生成と書籍の k word 以上の最大連続一致を全て見つける。"""
    if len(gen_tokens) < k:
        return [], []
    visited = set()
    book_ivs, gen_ivs = [], []
    for j in range(len(gen_tokens) - k + 1):
        starts = idx.get(tuple(gen_tokens[j:j+k]))
        if not starts:
            continue
        for i in starts:
            ii, jj = i, j
            while ii > 0 and jj > 0 and book_tokens[ii-1] == gen_tokens[jj-1]:
                ii -= 1; jj -= 1
            if (ii, jj) in visited:
                continue
            visited.add((ii, jj))
            p = 0
            while ii+p < len(book_tokens) and jj+p < len(gen_tokens) and book_tokens[ii+p] == gen_tokens[jj+p]:
                p += 1
            if p >= k:
                book_ivs.append((ii, ii+p))
                gen_ivs.append((jj, jj+p))
    return book_ivs, gen_ivs


def evaluate(test_book_path, generation_file_path, k=5, trim_m=5):
    with open(test_book_path, encoding='utf-8') as f:
        book = json.load(f)
    with open(generation_file_path, encoding='utf-8') as f:
        examples = json.load(f)

    book_tokens, _ = build_book_index(book)
    n = len(book_tokens)
    print(f"Book length: {n:,} words")

    idx = build_kgram_index(book_tokens, k)
    covered = [False] * n
    longest_span = 0
    best_span_text = ''
    print(f"\nEvaluating (k={k}, trim m={trim_m})...")
    for ex in examples:
        instr_tokens = tokenize(ex.get('instruction', ''))
        for gen in ex.get('generations', []):
            gen_tokens = tokenize(gen['generated_text'])
            book_ivs, _ = find_matches(gen_tokens, book_tokens, idx, k)
            for s, e in book_ivs:
                if e - s > longest_span:
                    longest_span = e - s
                    best_span_text = ' '.join(book_tokens[s:e])
            trimmed = trim_instruction_mgrams(book_tokens, instr_tokens, book_ivs, min_length=k, m=trim_m)
            for s, e in trimmed:
                for t in range(s, e):
                    covered[t] = True

    bmc = sum(covered) / n

    # 最長記憶ブロック
    longest_block, run, block_end = 0, 0, 0
    for i, c in enumerate(covered):
        if c:
            run += 1
            if run > longest_block:
                longest_block = run; block_end = i + 1
        else:
            run = 0
    block_text = ' '.join(book_tokens[block_end - longest_block:block_end]) if longest_block > 0 else ''

    return {
        'bmc_score': bmc,
        'longest_memorized_block': longest_block,
        'longest_memorized_block_text': block_text,
        'longest_regurgitated_span': longest_span,
        'longest_regurgitated_span_text': best_span_text,
    }


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--test_book', required=True)
    ap.add_argument('--generation_file', required=True)
    ap.add_argument('--k', type=int, default=5, help='Minimum n-gram length (words). Paper default: 5.')
    ap.add_argument('--trim_m', type=int, default=5, help='Instruction trim m-gram size.')
    args = ap.parse_args()

    print('=' * 60)
    print('  English Memorization Evaluation')
    print('=' * 60)
    results = evaluate(args.test_book, args.generation_file, args.k, args.trim_m)
    print(f"\n{'=' * 60}\n  Results\n{'=' * 60}")
    print(f"\n  BMC@{args.k}:                  {results['bmc_score']*100:.2f}%")
    print(f"  Longest memorized block:  {results['longest_memorized_block']} words")
    print(f"  Longest regurgitated span: {results['longest_regurgitated_span']} words")
    if results['longest_memorized_block_text']:
        print(f"\n{'─'*60}\n  Longest memorized block:\n{'─'*60}")
        print(f"  {results['longest_memorized_block_text'][:300]}")
    if results['longest_regurgitated_span_text']:
        print(f"\n{'─'*60}\n  Longest regurgitated span:\n{'─'*60}")
        print(f"  {results['longest_regurgitated_span_text'][:300]}")
    print()
