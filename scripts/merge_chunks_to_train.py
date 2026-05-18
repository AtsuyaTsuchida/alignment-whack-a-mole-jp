#!/usr/bin/env python3
"""Step 1.5: 同一著者の複数 chunks JSON を統合し FT 訓練用 JSON を作成。

Step 1 で各作品ごとに作った chunks ファイルを 1 つにまとめ、上限例数までトリム。
--exclude で held-out 作品（評価用に取っておく作品）を訓練から除外できる。

使い方:
    python scripts/merge_chunks_to_train.py \
        --chunks_dir data/japanese/soseki/chunks/ \
        --output data/japanese/soseki/soseki_train_3000.json \
        --max_examples 3000 \
        --exclude 門
"""

import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_dir", required=True,
                    help="チャンク JSON が入っているディレクトリ")
    ap.add_argument("--output", required=True, help="出力 JSON ファイル")
    ap.add_argument("--max_examples", type=int, default=3000,
                    help="抽出する最大例数 (default: 3000)")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="除外する book_name（held-out 用）")
    args = ap.parse_args()

    # chunks ディレクトリ配下の全 .json を取得
    files = sorted(glob.glob(os.path.join(args.chunks_dir, "*.json")))
    print(f"見つかった chunks ファイル: {len(files)}")

    all_chunks = []
    per_book = {}
    for f in files:
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
        if not data:
            continue
        book_name = data[0].get("book_name", os.path.basename(f).replace(".json", ""))
        # held-out 指定の作品はスキップ
        if book_name in args.exclude:
            print(f"  [SKIP] {book_name} ({len(data)} chunks)")
            continue
        per_book[book_name] = len(data)
        all_chunks.extend(data)
        print(f"  + {book_name}: {len(data)} chunks")

    print(f"\n合計: {len(all_chunks)} chunks")

    # 上限を超えたら先頭から N 例だけ採用（順序は作品名のアルファベット順）
    if len(all_chunks) > args.max_examples:
        print(f"上限 {args.max_examples} にトリム")
        all_chunks = all_chunks[:args.max_examples]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"\n書き出し: {args.output}")
    print(f"最終例数: {len(all_chunks)}")


if __name__ == "__main__":
    main()
