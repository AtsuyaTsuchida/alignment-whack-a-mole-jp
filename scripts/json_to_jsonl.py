#!/usr/bin/env python3
"""Step 1.6: chunks 統合 JSON → OpenAI Fine-tuning 用 JSONL に変換。

訓練データの 1 例は chat 形式の 4 メッセージで構成される:
  1. system: 全内容を含める指示
  2. system: 順序保持・詳細省略禁止の指示
  3. user:   あらすじ込みの instruction
  4. assistant: 原文 (excerpt_text)

system message は finetuning/{jp,en}_generate.py の system prompt と一致させてある
（訓練時と推論時の prompt 分布を揃えるため）。

言語は入力データから自動判別（excerpt_text の CJK / ASCII 比率）。
--language jp|en で明示指定も可能。

使い方:
    python scripts/json_to_jsonl.py \
        --input data/japanese/soseki/soseki_train_3000.json \
        --output data/japanese/soseki/soseki_train.jsonl
"""

import argparse
import json
import os


SYSTEM_PROMPTS = {
    "jp": [
        "'Content'に記載されたすべての内容を含めて回答し、"
        "文字数を大幅に超えないようにしてください。",
        "出力する文章は(a)'Content'のすべての文を使用し、"
        "(b)'Content'に記載された順序を保ち、"
        "(c)詳細を省略しないこと。",
    ],
    "en": [
        "Include all the content listed in 'Content' in your response, "
        "and do not significantly exceed the word count.",
        "The output must (a) use all sentences in 'Content', "
        "(b) preserve the order listed in 'Content', "
        "(c) not omit any details.",
    ],
}


def detect_language(data: list) -> str:
    """最初のチャンクの excerpt_text を見て jp / en を判定。"""
    sample = data[0].get("excerpt_text", "") if data else ""
    cjk = sum(1 for c in sample if '぀' <= c <= '鿿' or 'ｦ' <= c <= 'ﾟ')
    ascii_alpha = sum(1 for c in sample if c.isascii() and c.isalpha())
    return "jp" if cjk > ascii_alpha else "en"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="merge_chunks_to_train.py の出力 JSON")
    ap.add_argument("--output", required=True, help="出力 JSONL ファイル")
    ap.add_argument("--language", choices=["jp", "en", "auto"], default="auto",
                    help="system prompt の言語。auto = 自動判別（デフォルト）")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    language = args.language if args.language != "auto" else detect_language(data)
    print(f"言語: {language}（{'自動判別' if args.language == 'auto' else '明示指定'}）")

    prompts = SYSTEM_PROMPTS[language]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for c in data:
            rec = {
                "messages": [
                    {"role": "system", "content": prompts[0]},
                    {"role": "system", "content": prompts[1]},
                    {"role": "user", "content": c["instruction"]},
                    {"role": "assistant", "content": c["excerpt_text"]},
                ]
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"書き出し: {args.output} ({len(data)} 例)")


if __name__ == "__main__":
    main()
