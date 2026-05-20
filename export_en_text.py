#!/usr/bin/env python3
"""英語版: 各チャンクで best-of-10 生成を選び、原文・生成文・あらすじを txt 出力。

使い方:
    python export_en_text.py \
        --chunks data/english/dickens/chunks/a_christmas_carol.json \
        --gens   data/english/results/a_christmas_carol_gens.json \
        --output_dir data/english/text_exports \
        --name a_christmas_carol
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

K = 5  # 一致閾値 (word) — 論文準拠
WORD_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str):
    return WORD_RE.findall(text)


def pid_num(x):
    m = re.search(r'(\d+)', x['excerpt_id'])
    return int(m.group(1)) if m else 0


def count_matched(gen_text, book_words, k=K):
    """生成と原文の word 単位一致数を返す（best-of-10 選択用スコア）。"""
    g = tokenize(gen_text)
    if len(g) < k:
        return 0
    idx = defaultdict(list)
    for i in range(len(book_words) - k + 1):
        idx[tuple(book_words[i:i + k])].append(i)
    matched = [False] * len(g)
    for j in range(len(g) - k + 1):
        for i in idx.get(tuple(g[j:j + k]), []):
            ii, jj = i, j
            while ii > 0 and jj > 0 and book_words[ii - 1] == g[jj - 1]:
                ii -= 1; jj -= 1
            p = 0
            while ii + p < len(book_words) and jj + p < len(g) and book_words[ii + p] == g[jj + p]:
                p += 1
            if p >= k:
                for q in range(jj, jj + p):
                    matched[q] = True
    return sum(matched)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="Step 1 で生成した chunks JSON")
    ap.add_argument("--gens", required=True, help="Step 2 で生成した gens JSON")
    ap.add_argument("--output_dir", required=True, help="出力先ディレクトリ")
    ap.add_argument("--name", required=True, help="出力ファイル名のプレフィックス（例: a_christmas_carol）")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = json.loads(Path(args.chunks).read_text(encoding='utf-8'))
    gens_data = json.loads(Path(args.gens).read_text(encoding='utf-8'))
    chunks.sort(key=pid_num)
    gens_data.sort(key=pid_num)

    # 書籍全体の word 配列（best-of-10 スコア計算用）
    book_words = []
    for c in chunks:
        book_words.extend(tokenize(c['excerpt_text']))

    orig_parts, gen_parts, summary_parts = [], [], []
    for c, g in zip(chunks, gens_data):
        gens = g.get('generations', [])
        if not gens:
            best_text = '(no generation)'
        else:
            # best-of-10: 最も一致 word 数が多い生成を選ぶ
            best_text, best_match = '', -1
            for gen in gens:
                m = count_matched(gen['generated_text'], book_words)
                if m > best_match:
                    best_match = m
                    best_text = gen['generated_text']
        orig_parts.append(c['excerpt_text'])
        gen_parts.append(best_text)
        summary_parts.append(c.get('detail', ''))

    orig_path = out_dir / f'{args.name}_original.txt'
    gen_path = out_dir / f'{args.name}_generated.txt'
    summary_path = out_dir / f'{args.name}_summary.txt'
    orig_path.write_text('\n\n'.join(orig_parts), encoding='utf-8')
    gen_path.write_text('\n\n'.join(gen_parts), encoding='utf-8')
    summary_path.write_text('\n\n'.join(summary_parts), encoding='utf-8')
    print(f"{args.name}: {len(chunks)} parts")
    print(f"  → {orig_path}")
    print(f"  → {gen_path}")
    print(f"  → {summary_path}")


if __name__ == '__main__':
    main()
