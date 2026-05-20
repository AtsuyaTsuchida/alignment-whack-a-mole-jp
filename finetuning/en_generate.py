#!/usr/bin/env python3
"""
Step 2 (English): Generate from a fine-tuned model for test books.

Each chunk's instruction (= summary-based prompt) is sent to the FT model,
sampled N times in parallel. timeout + per-chunk checkpoint enables
automatic resume from failures.

Usage:
    python finetuning/en_generate.py \
        --test_file data/english/dickens/chunks/a_christmas_carol.json \
        --output_file results/a_christmas_carol_gens.json \
        --model ft:gpt-4o-2024-08-06:personal:dickens-en-2521:XXX \
        --num_generations 10
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def generate_single(item: dict, model: str, generation_num: int) -> dict:
    """Run a single generation (with retry)."""
    for attempt in range(8):
        try:
            response = client.with_options(timeout=90.0).chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Include all the content listed in 'Content' in your response, "
                            "and do not significantly exceed the word count."
                        ),
                    },
                    {
                        "role": "system",
                        "content": (
                            "The output must (a) use all sentences in 'Content', "
                            "(b) preserve the order listed in 'Content', "
                            "(c) not omit any details."
                        ),
                    },
                    {"role": "user", "content": item["instruction"]},
                ],
                temperature=1.0,
                max_tokens=800,
            )
            text = response.choices[0].message.content
            return {"generation_num": generation_num, "generated_text": text}
        except Exception as e:
            tqdm.write(f"  [{item['excerpt_id']}] gen{generation_num} attempt {attempt+1}/8: {type(e).__name__}: {str(e)[:80]}")
            time.sleep(min(2 ** attempt, 30))
    return None


def generate_completions(test_data: list, model: str, num_generations: int,
                         output_file: str = None) -> list:
    """For each chunk, generate num_generations times with the FT model (in parallel)."""
    # Resume from existing file: skip already-completed chunks
    results = []
    done_ids = set()
    if output_file and os.path.exists(output_file):
        with open(output_file, encoding="utf-8") as f:
            results = json.load(f)
        done_ids = {r["excerpt_id"] for r in results
                    if len(r.get("generations", [])) >= num_generations}
        if done_ids:
            tqdm.write(f"Resumed {len(done_ids)} chunks from checkpoint")
        results = [r for r in results if r["excerpt_id"] in done_ids]

    for item in tqdm(test_data, desc="Generating", unit="chunk"):
        if item["excerpt_id"] in done_ids:
            continue

        # Generate num_generations times per chunk (parallel)
        generations = []
        with ThreadPoolExecutor(max_workers=num_generations) as executor:
            futures = {
                executor.submit(generate_single, item, model, i): i
                for i in range(num_generations)
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    generations.append(result)

        results.append({
            "excerpt_id": item["excerpt_id"],
            "excerpt_text": item["excerpt_text"],
            "instruction": item["instruction"],
            "book_name": item["book_name"],
            "author_name": item["author_name"],
            "word_count": item.get("word_count", 0),
            "generations": generations,
        })

        # Save after each chunk (resume on kill)
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", required=True, help="chunks JSON from Step 1")
    ap.add_argument("--output_file", required=True, help="Output (also serves as checkpoint)")
    ap.add_argument("--model", required=True, help="Fine-tuned model ID")
    ap.add_argument("--num_generations", type=int, default=10, help="Generations per chunk")
    ap.add_argument("--max_chunks", type=int, default=None, help="Debug: first N chunks only")
    args = ap.parse_args()

    with open(args.test_file, encoding="utf-8") as f:
        test_data = json.load(f)

    if args.max_chunks:
        test_data = test_data[:args.max_chunks]
        print(f"Using first {args.max_chunks} chunks only")

    print(f"Test: {test_data[0]['book_name']} / {len(test_data)} chunks × {args.num_generations} gens (parallel)")

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    results = generate_completions(test_data, args.model, args.num_generations,
                                   output_file=args.output_file)

    print(f"\nDone: {len(results)} chunks → {args.output_file}")


if __name__ == "__main__":
    main()
