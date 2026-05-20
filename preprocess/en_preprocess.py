#!/usr/bin/env python3
"""Step 1: 英語テキストの前処理パイプライン（論文準拠 word 単位）。

処理:
  1. 本文を 300-500 word のチャンクに分割（文の区切りで切る）
  2. 短すぎるチャンクは隣接チャンクとマージ
  3. 各チャンクのあらすじを GPT-4o で生成（async 並列）
  4. FT 用 instruction prompt を組み立て

使い方:
    python preprocess/en_preprocess.py \
        --input_txt data/english/dickens/raw/oliver_twist.txt \
        --output_json data/english/dickens/chunks/oliver_twist.json \
        --book_name "Oliver Twist" \
        --author_name "Charles Dickens"
"""

import argparse
import asyncio
import json
import os
import re
from copy import deepcopy

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

client = AsyncOpenAI()

MIN_WORDS = 300
MAX_WORDS = 500
CONCURRENCY = 10  # 同時並列リクエスト数


def _word_count(text: str) -> int:
    return len(text.split())


def _split_into_chunks(text: str) -> list[str]:
    """段落 → 必要なら文単位で 300-500 word に分割。"""
    paragraphs = re.split(r'\n\n+', text.strip())
    chunks, current, cw = [], "", 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        pw = _word_count(para)
        if pw > MAX_WORDS:
            # 段落が長すぎる場合は文単位で分割
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                sw = _word_count(sent)
                if cw + sw > MAX_WORDS and cw >= MIN_WORDS:
                    chunks.append(current.strip()); current = sent; cw = sw
                else:
                    current = (current + " " + sent).strip() if current else sent; cw += sw
        else:
            if cw + pw > MAX_WORDS and cw >= MIN_WORDS:
                chunks.append(current.strip()); current = para; cw = pw
            else:
                current = (current + "\n\n" + para) if current else para; cw += pw
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _merge_short_chunks(chunks: list[dict]) -> list[dict]:
    """MIN_WORDS 未満のチャンクを隣接と結合。"""
    items = deepcopy(chunks)
    i = 0
    while i < len(items):
        cw = _word_count(items[i]["excerpt_text"])
        if cw >= MIN_WORDS:
            i += 1; continue
        has_prev, has_next = i > 0, i < len(items) - 1
        if has_prev:
            prev_cw = _word_count(items[i-1]["excerpt_text"])
            if prev_cw + cw <= MAX_WORDS * 1.2:
                items[i-1]["excerpt_text"] += " " + items[i]["excerpt_text"]
                del items[i]; i = max(i-1, 0); continue
        if has_next:
            items[i+1]["excerpt_text"] = items[i]["excerpt_text"] + " " + items[i+1]["excerpt_text"]
            del items[i]; continue
        if has_prev:
            items[i-1]["excerpt_text"] += " " + items[i]["excerpt_text"]
            del items[i]; i = max(i-1, 0); continue
        i += 1
    for idx, item in enumerate(items):
        item["excerpt_id"] = f"p_id{idx + 1}"
        item["word_count"] = _word_count(item["excerpt_text"])
    return items


def _make_summary_prompt(text: str, summary_words: int) -> str:
    """論文準拠の plot summary プロンプト。"""
    return (
        f"Describe in detail (about {summary_words} words) what happens in the following passage. "
        f"Preserve the characters, narrative voice (first/third person), and the order of events.\n\n{text}"
    )


async def _gen_one(chunk: dict, summary_ratio: float, sem: asyncio.Semaphore):
    """1チャンクのあらすじ生成（並列リクエスト用）。"""
    if chunk.get("detail"):
        return chunk
    text = chunk["excerpt_text"]
    wc = chunk["word_count"]
    summary_words = max(int(wc * summary_ratio) + 1, 20)
    prompt = _make_summary_prompt(text, summary_words)

    async with sem:
        for attempt in range(8):
            try:
                resp = await client.with_options(timeout=90.0).chat.completions.create(
                    model="gpt-4o-2024-08-06",
                    messages=[{"role": "user", "content": prompt}],
                )
                detail = resp.choices[0].message.content
                detail = " ".join(line.strip() for line in detail.split("\n")).strip()
                chunk["detail"] = detail
                chunk["instruction"] = (
                    f"Write a {wc}-word excerpt in the style of {chunk['author_name']} "
                    f"about the content below.\n\nContent: {detail}"
                )
                return chunk
            except Exception as e:
                print(f"  [{chunk['excerpt_id']}] attempt {attempt+1}/8: {type(e).__name__}: {str(e)[:80]}")
                await asyncio.sleep(min(2 ** attempt, 30))
    chunk["detail"] = ""; chunk["instruction"] = ""
    return chunk


async def _generate_summaries(chunks, summary_ratio=0.5, checkpoint_path=None):
    """全チャンクのあらすじを並列生成。"""
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            existing = {c["excerpt_id"]: c for c in json.load(f)}
        for c in chunks:
            if c["excerpt_id"] in existing and existing[c["excerpt_id"]].get("detail"):
                c["detail"] = existing[c["excerpt_id"]]["detail"]
                c["instruction"] = existing[c["excerpt_id"]].get("instruction", "")

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [_gen_one(c, summary_ratio, sem) for c in chunks if not c.get("detail")]
    if not tasks:
        return chunks
    await tqdm_asyncio.gather(*tasks, desc="Summaries", unit="chunk")

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
    ap.add_argument("--summary_ratio", type=float, default=0.5)
    args = ap.parse_args()

    with open(args.input_txt, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"[1/3] Chunk split: {args.book_name}")
    raw_chunks = _split_into_chunks(text)
    chunks = [
        {
            "book_name": args.book_name,
            "author_name": args.author_name,
            "excerpt_id": f"p_id{i + 1}",
            "excerpt_text": c,
            "word_count": _word_count(c),
            "detail": "",
            "instruction": "",
        }
        for i, c in enumerate(raw_chunks)
    ]
    print(f"  → {len(chunks)} chunks (avg {sum(_word_count(c['excerpt_text']) for c in chunks)//len(chunks)} words)")

    print(f"[2/3] Merging short chunks")
    chunks = _merge_short_chunks(chunks)
    print(f"  → {len(chunks)} chunks")

    print(f"[3/3] Generating summaries (GPT-4o, async {CONCURRENCY} parallel)")
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    chunks = asyncio.run(_generate_summaries(chunks, args.summary_ratio, args.output_json))

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"\nDone: {len(chunks)} chunks → {args.output_json}")


if __name__ == "__main__":
    main()
