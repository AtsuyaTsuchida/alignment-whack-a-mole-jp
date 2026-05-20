#!/usr/bin/env python3
"""Step 1.5b: chunks 統合 JSON → OpenAI Fine-tuning 用 JSONL に変換（日本語版）。

訓練データの 1 例は chat 形式の 4 メッセージで構成される:
  1. system: 全内容を含める指示
  2. system: 順序保持・詳細省略禁止の指示
  3. user:   あらすじ込みの instruction
  4. assistant: 原文 (excerpt_text)

system message は finetuning/jp_generate.py の system prompt と一致させてある
（訓練時と推論時の prompt 分布を揃えるため）。

使い方:
    python scripts/json_to_jsonl.py \
        --input data/japanese/soseki/soseki_train_3000.json \
        --output data/japanese/soseki/soseki_train.jsonl
"""

import argparse
import json
import os


SYSTEM_PROMPTS = [
    "'Content'に記載されたすべての内容を含めて回答し、"
    "文字数を大幅に超えないようにしてください。",
    "出力する文章は(a)'Content'のすべての文を使用し、"
    "(b)'Content'に記載された順序を保ち、"
    "(c)詳細を省略しないこと。",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="merge_chunks_to_train.py の出力 JSON")
    ap.add_argument("--output", required=True, help="出力 JSONL ファイル")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for c in data:
            # 訓練時に system + user + assistant の chat 形式に変換
            rec = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPTS[0]},
                    {"role": "system", "content": SYSTEM_PROMPTS[1]},
                    {"role": "user", "content": c["instruction"]},
                    {"role": "assistant", "content": c["excerpt_text"]},
                ]
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"書き出し: {args.output} ({len(data)} 例)")


if __name__ == "__main__":
    main()
