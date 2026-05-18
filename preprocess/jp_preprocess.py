#!/usr/bin/env python3
"""
Step 1: 青空文庫テキストの前処理パイプライン。

処理の流れ:
  1. 本文を 300〜500 字のチャンクに分割（文の区切りで切る）
  2. 短すぎるチャンクは隣接チャンクとマージ
  3. 各チャンクのあらすじを GPT-4o で生成
  4. FT 用の instruction prompt を組み立て

使い方:
    python preprocess/jp_preprocess.py \
        --input_txt data/japanese/soseki/raw/こころ.txt \
        --output_json data/japanese/soseki/chunks/こころ.json \
        --book_name こころ \
        --author_name 夏目漱石
"""

import argparse
import json
import os
import re
import time
from copy import deepcopy

from openai import OpenAI
from tqdm import tqdm

client = OpenAI()

# チャンクの最小・最大文字数（論文の 300-500 words に対応する日本語の近似）
MIN_CHARS = 300
MAX_CHARS = 500


def _char_count(text: str) -> int:
    """空白を除く文字数を数える。"""
    return len(re.sub(r'\s', '', text))


def _split_into_chunks(text: str) -> list[str]:
    """日本語テキストを 300〜500 字のチャンクに分割（文の区切り優先）。"""
    # まず段落に分割
    paragraphs = re.split(r'\n\n+', text.strip())

    chunks = []
    current = ""
    current_chars = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_chars = _char_count(para)

        # 段落が単体でMAX_CHARSを超える場合は文単位でさらに分割
        if para_chars > MAX_CHARS:
            sentences = re.split(r'(?<=[。！？])', para)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                sent_chars = _char_count(sent)
                if current_chars + sent_chars > MAX_CHARS and current_chars >= MIN_CHARS:
                    chunks.append(current.strip())
                    current = sent
                    current_chars = sent_chars
                else:
                    current += sent
                    current_chars += sent_chars
        else:
            if current_chars + para_chars > MAX_CHARS and current_chars >= MIN_CHARS:
                chunks.append(current.strip())
                current = para
                current_chars = para_chars
            else:
                current += ("\n\n" if current else "") + para
                current_chars += para_chars

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _merge_short_chunks(chunks: list[dict]) -> list[dict]:
    """MIN_CHARS 未満の短いチャンクを隣接チャンクと結合する。

    優先順位: 前と結合（収まる場合）→ 後と結合 → 前と強制結合
    """
    items = deepcopy(chunks)
    i = 0
    while i < len(items):
        cc = _char_count(items[i]["excerpt_text"])
        if cc >= MIN_CHARS:
            i += 1
            continue

        has_prev = i > 0
        has_next = i < len(items) - 1

        # 1. 前のチャンクと結合できるか試す（MAX_CHARS の 120% 以内）
        if has_prev:
            prev_cc = _char_count(items[i - 1]["excerpt_text"])
            if prev_cc + cc <= MAX_CHARS * 1.2:
                items[i - 1]["excerpt_text"] = items[i - 1]["excerpt_text"] + items[i]["excerpt_text"]
                del items[i]
                i = max(i - 1, 0)
                continue

        # 2. 後ろのチャンクと結合（前が大きすぎる場合）
        if has_next:
            items[i + 1]["excerpt_text"] = items[i]["excerpt_text"] + items[i + 1]["excerpt_text"]
            del items[i]
            continue

        # 3. 最後の手段: 前と強制結合（最終チャンクで後がない場合）
        if has_prev:
            items[i - 1]["excerpt_text"] = items[i - 1]["excerpt_text"] + items[i]["excerpt_text"]
            del items[i]
            i = max(i - 1, 0)
            continue

        i += 1

    # excerpt_id を振り直す（マージで穴が空くため）
    for idx, item in enumerate(items):
        item["excerpt_id"] = f"p_id{idx + 1}"
        item["char_count"] = _char_count(item["excerpt_text"])

    return items


def _make_summary_prompt(text: str, summary_chars: int) -> str:
    """論文準拠の標準あらすじプロンプトを生成する。

    登場人物・視点・出来事の順序を維持した plot summary（あらすじ）を作る。
    """
    return (
        f"以下の文章の内容を詳細に（{summary_chars}文字程度で）説明してください。"
        f"登場人物、語り手の視点（一人称か三人称か）、出来事の順序を維持して記述してください。\n\n{text}"
    )


def _generate_summaries(chunks: list[dict], summary_ratio: float = 0.5,
                        checkpoint_path: str = None) -> list[dict]:
    """各チャンクのあらすじを GPT-4o で生成し、FT 用 instruction を組み立てる。

    summary_ratio: 原文に対するあらすじ長の比率（0.5 = 半分の長さ）
    checkpoint_path: 各チャンク完了ごとに保存（クラッシュ復旧用）
    """
    # 既存ファイルがあれば再開: detail がすでに生成されたチャンクはスキップ
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            existing = {c["excerpt_id"]: c for c in json.load(f)}
        for c in chunks:
            if c["excerpt_id"] in existing and existing[c["excerpt_id"]].get("detail"):
                c["detail"] = existing[c["excerpt_id"]]["detail"]
                c["instruction"] = existing[c["excerpt_id"]].get("instruction", "")

    for chunk in tqdm(chunks, desc="あらすじ生成中", unit="chunk"):
        # 既に生成済みならスキップ（再開時）
        if chunk.get("detail"):
            continue

        text = chunk["excerpt_text"]
        char_count = chunk["char_count"]
        summary_chars = max(int(char_count * summary_ratio) + 1, 20)
        prompt = _make_summary_prompt(text, summary_chars)

        # 指数バックオフ付きで GPT-4o を呼ぶ
        response = None
        for attempt in range(8):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-2024-08-06",
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except Exception as e:
                tqdm.write(f"  [{chunk['excerpt_id']}] attempt {attempt+1}/8 failed: {type(e).__name__}: {str(e)[:100]}")
                time.sleep(min(2 ** attempt, 30))

        # 8回失敗したら空欄でスキップ（他のチャンクは続行）
        if response is None:
            tqdm.write(f"  [{chunk['excerpt_id']}] ALL retries failed, skipping")
            chunk["detail"] = ""
            chunk["instruction"] = ""
            continue

        # あらすじ本体を整形（改行を空白に統一）
        detail = response.choices[0].message.content
        detail = " ".join(line.strip() for line in detail.split("\n")).strip()

        chunk["detail"] = detail
        # FT 用 instruction を組み立て: あらすじ → 著者文体で本文化させる
        chunk["instruction"] = (
            f"以下の内容について、{chunk['author_name']}の文体と語り口を忠実に模倣して"
            f"{char_count}文字程度の文章を書いてください。"
            f"\n\nContent: {detail}"
        )

        # 1チャンク完了ごとに保存（途中で kill されても再開可能）
        if checkpoint_path:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)

    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_txt", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--book_name", required=True)
    ap.add_argument("--author_name", required=True)
    ap.add_argument("--summary_ratio", type=float, default=0.5,
                    help="あらすじの目標長/原文の比率 (default: 0.5 = 論文準拠)")
    args = ap.parse_args()

    with open(args.input_txt, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"[1/3] チャンク分割: {args.book_name}")
    raw_chunks = _split_into_chunks(text)
    chunks = [
        {
            "book_name": args.book_name,
            "author_name": args.author_name,
            "excerpt_id": f"p_id{i + 1}",
            "excerpt_text": c,
            "char_count": _char_count(c),
            "detail": "",
            "instruction": "",
        }
        for i, c in enumerate(raw_chunks)
    ]
    print(f"  → {len(chunks)}チャンク (平均{sum(_char_count(c['excerpt_text']) for c in chunks)//len(chunks)}文字)")

    print(f"[2/3] 短いチャンクをマージ中")
    chunks = _merge_short_chunks(chunks)
    print(f"  → {len(chunks)}チャンク")

    print(f"[3/3] あらすじ生成中 (GPT-4o, ratio={args.summary_ratio})")
    chunks = _generate_summaries(chunks, summary_ratio=args.summary_ratio,
                                  checkpoint_path=args.output_json)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\n完了: {len(chunks)}チャンク → {args.output_json}")


if __name__ == "__main__":
    main()
