#!/usr/bin/env python3
"""
Step 2: FT 済みモデルでテスト書籍の生成を行う。

各チャンクの instruction (= あらすじベースのプロンプト) を FT モデルに渡し、
N 回サンプリングする。timeout 付き + 1 チャンク完了ごと checkpoint 保存で
途中失敗からの自動再開に対応。

使い方:
    python finetuning/jp_generate.py \
        --test_file data/japanese/soseki/chunks/門.json \
        --output_file results/門_gens.json \
        --model ft:gpt-4o-2024-08-06:personal:soseki-jp-3000-ft:XXX \
        --num_generations 10
"""

import argparse
import json
import os
import time

from openai import OpenAI
from tqdm import tqdm

client = OpenAI()


def generate_completions(test_data: list, model: str, num_generations: int,
                         output_file: str = None) -> list:
    """各チャンクに対し num_generations 回ずつ FT モデルで生成する。"""
    # 既存ファイルから再開: 完了済みチャンクをスキップ
    results = []
    done_ids = set()
    if output_file and os.path.exists(output_file):
        with open(output_file, encoding="utf-8") as f:
            results = json.load(f)
        done_ids = {r["excerpt_id"] for r in results
                    if len(r.get("generations", [])) >= num_generations}
        if done_ids:
            tqdm.write(f"既存 checkpoint から {len(done_ids)} chunks 復元")
        results = [r for r in results if r["excerpt_id"] in done_ids]

    for item in tqdm(test_data, desc="チャンク処理中", unit="chunk"):
        if item["excerpt_id"] in done_ids:
            continue

        # 1 チャンクにつき num_generations 回生成
        generations = []
        for i in range(num_generations):
            # 各生成は timeout 90 秒、最大 8 回までリトライ
            for attempt in range(8):
                try:
                    response = client.with_options(timeout=90.0).chat.completions.create(
                        model=model,
                        messages=[
                            # system: Content (= あらすじ) の内容を漏らさず出力させる指示
                            {
                                "role": "system",
                                "content": (
                                    "'Content'に記載されたすべての内容を含めて回答し、"
                                    "文字数を大幅に超えないようにしてください。"
                                ),
                            },
                            {
                                "role": "system",
                                "content": (
                                    "出力する文章は(a)'Content'のすべての文を使用し、"
                                    "(b)'Content'に記載された順序を保ち、"
                                    "(c)詳細を省略しないこと。"
                                ),
                            },
                            # user: あらすじ + 著者文体模倣の指示
                            {"role": "user", "content": item["instruction"]},
                        ],
                        temperature=1.0,  # 論文準拠
                        max_tokens=800,
                    )
                    text = response.choices[0].message.content
                    generations.append({"generation_num": i, "generated_text": text})
                    break
                except Exception as e:
                    # ハング/レート制限/接続エラー全てここで指数バックオフ
                    tqdm.write(f"  [{item['excerpt_id']}] gen{i} attempt {attempt+1}/8: {type(e).__name__}: {str(e)[:80]}")
                    time.sleep(min(2 ** attempt, 30))

        results.append({
            "excerpt_id": item["excerpt_id"],
            "excerpt_text": item["excerpt_text"],
            "instruction": item["instruction"],
            "book_name": item["book_name"],
            "author_name": item["author_name"],
            "char_count": item.get("char_count", 0),
            "generations": generations,
        })

        # 1チャンク完了ごとに保存（途中で kill されても再開可能）
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", required=True, help="Step 1 で生成した chunks JSON")
    ap.add_argument("--output_file", required=True, help="生成結果の出力先（checkpoint も兼ねる）")
    ap.add_argument("--model", required=True, help="FT 済みモデル ID")
    ap.add_argument("--num_generations", type=int, default=5, help="1チャンクあたりの生成回数")
    ap.add_argument("--max_chunks", type=int, default=None, help="デバッグ用: 先頭 N チャンクのみ")
    args = ap.parse_args()

    with open(args.test_file, encoding="utf-8") as f:
        test_data = json.load(f)

    if args.max_chunks:
        test_data = test_data[:args.max_chunks]
        print(f"先頭{args.max_chunks}チャンクのみ使用")

    print(f"テスト: {test_data[0]['book_name']} / {len(test_data)}チャンク × {args.num_generations}生成")

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    results = generate_completions(test_data, args.model, args.num_generations,
                                   output_file=args.output_file)

    print(f"\n完了: {len(results)}チャンク → {args.output_file}")


if __name__ == "__main__":
    main()
