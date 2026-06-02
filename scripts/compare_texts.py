import argparse
import re
from collections import defaultdict


def normalize(text: str) -> str:
    """空白削除"""
    return re.sub(r'\s', '', text)


def compute_bmc(orig_text: str, gen_text: str, k: int = 10):
    """K-gram 一致を計算"""
    orig = list(normalize(orig_text))
    gen = list(normalize(gen_text))

    if len(orig) < k or len(gen) < k:
        return {
            'orig_chars': len(orig),
            'gen_chars': len(gen),
            'orig_matched': 0,
            'gen_matched': 0,
            'orig_bmc': 0.0,
            'gen_bmc': 0.0,
            'longest_block': 0,
            'longest_span': 0,
            'longest_block_text': '',
            'longest_span_text': '',
        }

    # 原文側の k-gram インデックス
    idx = defaultdict(list)
    for i in range(len(orig) - k + 1):
        idx[tuple(orig[i:i + k])].append(i)

    orig_matched = [False] * len(orig)
    gen_matched = [False] * len(gen)

    longest_block = 0
    longest_block_text = ''
    longest_span = 0
    longest_span_text = ''

    for j in range(len(gen) - k + 1):
        for i in idx.get(tuple(gen[j:j + k]), []):
            # 後方に伸ばす
            ii, jj = i, j
            while ii > 0 and jj > 0 and orig[ii - 1] == gen[jj - 1]:
                ii -= 1
                jj -= 1
            # 前方に伸ばす
            p = 0
            while ii + p < len(orig) and jj + p < len(gen) and orig[ii + p] == gen[jj + p]:
                p += 1
            if p >= k:
                # 原文側マーク (記憶ブロック)
                if p > longest_block:
                    longest_block = p
                    longest_block_text = ''.join(orig[ii:ii + p])
                for q in range(ii, ii + p):
                    orig_matched[q] = True
                # 生成側マーク (再現スパン)
                if p > longest_span:
                    longest_span = p
                    longest_span_text = ''.join(gen[jj:jj + p])
                for q in range(jj, jj + p):
                    gen_matched[q] = True

    return {
        'orig_chars': len(orig),
        'gen_chars': len(gen),
        'orig_matched': sum(orig_matched),
        'gen_matched': sum(gen_matched),
        'orig_bmc': sum(orig_matched) / len(orig) * 100,
        'gen_bmc': sum(gen_matched) / len(gen) * 100,
        'longest_block': longest_block,
        'longest_span': longest_span,
        'longest_block_text': longest_block_text,
        'longest_span_text': longest_span_text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--orig', required=True, help='原文 txt ファイル')
    ap.add_argument('--gen', required=True, help='生成文 txt ファイル')
    ap.add_argument('--k', type=int, default=10, help='K-gram 長 (default: 10 文字)')
    args = ap.parse_args()

    orig_text = open(args.orig, encoding='utf-8').read()
    gen_text = open(args.gen, encoding='utf-8').read()

    r = compute_bmc(orig_text, gen_text, k=args.k)

    print(f'原文 {r["orig_chars"]:,} 字 / 生成 {r["gen_chars"]:,} 字 ({r["gen_chars"] / r["orig_chars"]:.2f}x)')
    print(f'BMC@{args.k}: {r["orig_bmc"]:.2f}%   最長一致: {r["longest_block"]} 字')
    if r['longest_block_text']:
        print(f'  「{r["longest_block_text"][:200]}」')


if __name__ == '__main__':
    main()
