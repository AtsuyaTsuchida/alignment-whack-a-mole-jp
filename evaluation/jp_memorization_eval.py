#!/usr/bin/env python3
"""
Step 3: 文字単位の n-gram 一致による日本語記憶化評価。

論文 Algorithm 1 を文字単位（日本語向け）で実装:
  - BMC@k:            原文中で k 文字以上一致したブロックがカバーする割合
  - Instruction trim: あらすじに含まれる m-gram と重なる位置を除外
                      （あらすじ → 本文の漏れ込みを排除）
  - 最長記憶ブロック:  trim 後の最長連続一致
  - 最長再現スパン:    trim 前の最長連続一致（論文 §3.1）

使い方:
    python evaluation/jp_memorization_eval.py \
        --test_book data/japanese/soseki/chunks/門.json \
        --generation_file results/門_gens.json \
        --k 10 --trim_m 5
"""

import argparse
import json
import re
from collections import defaultdict
from typing import List, Tuple, Dict


def normalize(text: str) -> str:
    """空白を削除（マッチ比較用の正規化）。"""
    return re.sub(r'\s', '', text)


def tokenize(text: str) -> List[str]:
    """日本語向けの文字単位トークナイズ（英語の word tokenize に相当）。"""
    return list(normalize(text))


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """重なる区間をマージ。"""
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


def _subtract_from_interval(base: Tuple[int, int], removes: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """base 区間から removes の各区間を引いた残りの部分区間を返す。"""
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


def _mgram_set(chars: List[str], m: int) -> set:
    """m-gram の集合を作る。"""
    if m <= 0 or len(chars) < m:
        return set()
    return {tuple(chars[i : i + m]) for i in range(len(chars) - m + 1)}


def trim_instruction_mgrams(
    book_chars: List[str],
    instr_chars: List[str],
    intervals: List[Tuple[int, int]],
    min_length: int,
    m: int,
) -> List[Tuple[int, int]]:
    """Algorithm 1 Stage 2: instruction 中の m-gram と一致する位置を一致区間から除外。

    あらすじには本文の一部が漏れているため、一致のうちあらすじ起源のものを除く。
    trim 後に min_length (= k) 未満になった区間は捨てる。
    """
    if not intervals:
        return []
    instr_mgrams = _mgram_set(instr_chars, m)
    all_trimmed: List[Tuple[int, int]] = []
    for raw_iv in intervals:
        s, e = raw_iv
        span_len = e - s
        if m <= 0 or span_len < m or not instr_mgrams:
            if span_len >= min_length:
                all_trimmed.append(raw_iv)
            continue
        # 区間内で instruction と被る位置をすべて検出
        removes = []
        for i in range(span_len - m + 1):
            mg = tuple(book_chars[s + i : s + i + m])
            if mg in instr_mgrams:
                removes.append((s + i, s + i + m))
        removes = _merge_intervals(removes)
        # trim 後の残り部分のうち、まだ min_length 以上のものだけ残す
        for start, end in _subtract_from_interval(raw_iv, removes):
            if end - start >= min_length:
                all_trimmed.append((start, end))
    return _merge_intervals(all_trimmed)


def build_book_index(book_examples: list) -> Tuple[List[str], List[Tuple[int, int, str]]]:
    """全 excerpt を連結して書籍全体の文字配列を作る。

    各 excerpt の (開始位置, 終了位置, excerpt_id) も返す。
    """
    exs = sorted(book_examples, key=lambda x: int(re.search(r'(\d+)', x['excerpt_id']).group(1)))
    all_chars: List[str] = []
    spans: List[Tuple[int, int, str]] = []
    for ex in exs:
        chars = tokenize(ex['excerpt_text'])
        start = len(all_chars)
        all_chars.extend(chars)
        spans.append((start, len(all_chars), ex['excerpt_id']))
    return all_chars, spans


def build_kgram_index(book_chars: List[str], k: int) -> Dict[tuple, List[int]]:
    """書籍の k-gram インデックスを 1 回だけ構築（全生成で使い回す）。"""
    idx: Dict[tuple, List[int]] = defaultdict(list)
    for i in range(len(book_chars) - k + 1):
        idx[tuple(book_chars[i:i+k])].append(i)
    return idx


def find_matches(gen_chars: List[str], book_chars: List[str], idx: Dict[tuple, List[int]],
                 k: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """生成文と書籍本文の k 字以上の最大連続一致を全て見つける。

    返り値: (書籍側区間, 生成側区間) のペア
    """
    if len(gen_chars) < k:
        return [], []

    visited = set()
    book_intervals = []
    gen_intervals = []

    # 生成文の各位置で k-gram をキーに書籍中の出現位置を引く
    for j in range(len(gen_chars) - k + 1):
        starts = idx.get(tuple(gen_chars[j:j+k]))
        if not starts:
            continue
        for i in starts:
            # 一致部分を左右に伸ばす（最大連続一致を取る）
            ii, jj = i, j
            while ii > 0 and jj > 0 and book_chars[ii-1] == gen_chars[jj-1]:
                ii -= 1; jj -= 1
            pair = (ii, jj)
            if pair in visited:
                continue  # 同じ位置からの拡張は1回だけ
            visited.add(pair)
            p = 0
            while (ii+p) < len(book_chars) and (jj+p) < len(gen_chars) and book_chars[ii+p] == gen_chars[jj+p]:
                p += 1
            if p >= k:
                book_intervals.append((ii, ii+p))
                gen_intervals.append((jj, jj+p))
    return book_intervals, gen_intervals


def evaluate(test_book_path: str, generation_file_path: str, k: int = 10, trim_m: int = 5):
    """評価本体: BMC@k と最長スパンを 1 パスで計算。"""
    with open(test_book_path, encoding='utf-8') as f:
        book = json.load(f)
    with open(generation_file_path, encoding='utf-8') as f:
        examples = json.load(f)

    book_chars, para_spans = build_book_index(book)
    n = len(book_chars)
    print(f"書籍文字数: {n:,}文字")

    # k-gram インデックスは 1 回だけ構築（全生成で共有）
    idx = build_kgram_index(book_chars, k)

    # 1 パス: BMC@k (trim あり) と最長再現スパン (trim なし) を同時に集計
    covered = [False] * n
    longest_span = 0
    best_span_text = ''
    print(f"\n評価中 (k={k}, instruction trim m={trim_m})...")
    for ex in examples:
        instr_chars = tokenize(ex.get('instruction', ''))
        for gen in ex.get('generations', []):
            gen_chars = tokenize(gen['generated_text'])
            book_ivs, _ = find_matches(gen_chars, book_chars, idx, k)

            # 最長再現スパン: trim 前の生スパン（論文 §3.1 定義）
            for s, e in book_ivs:
                if e - s > longest_span:
                    longest_span = e - s
                    best_span_text = ''.join(book_chars[s:e])

            # BMC@k: instruction m-gram trim 後にカバレッジ集計
            trimmed = trim_instruction_mgrams(
                book_chars, instr_chars, book_ivs,
                min_length=k, m=trim_m,
            )
            for s, e in trimmed:
                for t in range(s, e):
                    covered[t] = True

    bmc = sum(covered) / n

    # 最長記憶ブロック: trim 後 covered 配列での最長連続 True
    longest_block = 0
    current_run = 0
    block_end = 0
    for i, c in enumerate(covered):
        if c:
            current_run += 1
            if current_run > longest_block:
                longest_block = current_run
                block_end = i + 1
        else:
            current_run = 0
    block_start = block_end - longest_block
    block_text = ''.join(book_chars[block_start:block_end]) if longest_block > 0 else ''

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
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--trim_m', type=int, default=5,
                    help='Instruction m-gram size for trimming (Algorithm 1 Stage 2). Default: 5 chars.')
    args = ap.parse_args()

    print('=' * 60)
    print('  日本語 記憶化評価メトリクス')
    print('=' * 60)

    results = evaluate(args.test_book, args.generation_file, args.k, args.trim_m)

    print(f"\n{'=' * 60}")
    print(f"  結果")
    print(f"{'=' * 60}")
    print(f"\n  BMC@{args.k}:             {results['bmc_score']*100:.2f}%")
    print(f"  最長記憶ブロック:    {results['longest_memorized_block']}文字")
    print(f"  最長再現スパン:      {results['longest_regurgitated_span']}文字")

    if results['longest_memorized_block_text']:
        print(f"\n{'─'*60}")
        print(f"  最長記憶ブロック（テキスト）:")
        print(f"{'─'*60}")
        print(f"  {results['longest_memorized_block_text'][:200]}")

    if results['longest_regurgitated_span_text']:
        print(f"\n{'─'*60}")
        print(f"  最長再現スパン（テキスト）:")
        print(f"{'─'*60}")
        print(f"  {results['longest_regurgitated_span_text'][:200]}")
    print()
