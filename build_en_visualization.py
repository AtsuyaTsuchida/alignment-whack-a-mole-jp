#!/usr/bin/env python3
"""英語版: 横書き 2 カラム縦スクロールの memorization 可視化 HTML を生成。

各チャンクで best-of-10（原文と最も一致した生成）を選び、原文と並べて
K word 以上の連続一致をハイライト表示。Enter で再生/一時停止、G で GUI トグル、
マウスホイールで一時停止中のスクラブ。

使い方:
    python build_en_visualization.py \\
        --chunks data/english/dickens/chunks/a_christmas_carol.json \\
        --gens results/a_christmas_carol_gens.json \\
        --output results/a_christmas_carol_visualization.html
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

WORD_RE = re.compile(r"[A-Za-z0-9']+|\s+|[^\sA-Za-z0-9']")


def pid_num(x):
    m = re.search(r'(\d+)', x['excerpt_id'])
    return int(m.group(1)) if m else 0


def tokenize_words(text: str):
    """単語と非単語（空白・句読点）をトークンとして保持。表示時に再構成可能。"""
    tokens = WORD_RE.findall(text)
    # word-only positions for matching
    is_word = [bool(re.fullmatch(r"[A-Za-z0-9']+", t)) for t in tokens]
    return tokens, is_word


def find_matches_both(gen_words, book_words, k):
    """word 単位の最大連続一致を見つける。
    Returns: (gen_intervals, book_intervals) — それぞれ word-index 範囲。
    """
    if len(gen_words) < k:
        return [], []
    idx = defaultdict(list)
    for i in range(len(book_words) - k + 1):
        idx[tuple(book_words[i:i + k])].append(i)
    visited = set()
    gen_ivs, book_ivs = [], []
    for j in range(len(gen_words) - k + 1):
        starts = idx.get(tuple(gen_words[j:j + k]))
        if not starts:
            continue
        for i in starts:
            ii, jj = i, j
            while ii > 0 and jj > 0 and book_words[ii - 1] == gen_words[jj - 1]:
                ii -= 1; jj -= 1
            if (ii, jj) in visited:
                continue
            visited.add((ii, jj))
            p = 0
            while ii + p < len(book_words) and jj + p < len(gen_words) and book_words[ii + p] == gen_words[jj + p]:
                p += 1
            if p >= k:
                gen_ivs.append((jj, jj + p))
                book_ivs.append((ii, ii + p))
    return gen_ivs, book_ivs


def merge_intervals(ivs):
    if not ivs:
        return []
    ivs = sorted(ivs)
    merged = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]


def build_word_array(text: str):
    """単語のみの配列（一致検出用）と、表示用全 token 配列を返す。
    word_to_token_pos: word index -> token index へのマッピング。
    """
    tokens, is_word = tokenize_words(text)
    words = [t for t, w in zip(tokens, is_word) if w]
    word_to_token = [i for i, w in enumerate(is_word) if w]
    return words, tokens, word_to_token


def expand_word_intervals_to_tokens(word_ivs, word_to_token, total_tokens):
    """word index 範囲を token index 範囲に拡張（間の空白も含む）。"""
    out = []
    for s, e in word_ivs:
        # word index [s, e) -> token index
        token_start = word_to_token[s]
        token_end = word_to_token[e - 1] + 1
        out.append((token_start, token_end))
    return out


def render_html(tokens, token_ivs, cls):
    """token 列を HTML に変換、token_ivs の範囲をハイライト。"""
    iv_set = set()
    for s, e in token_ivs:
        for i in range(s, e):
            iv_set.add(i)
    parts = []
    current_hl = False
    buf = []
    def flush():
        if not buf:
            return
        text = ''.join(buf)
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        if current_hl:
            parts.append(f'<span class="{cls}">{text}</span>')
        else:
            parts.append(text)
        buf.clear()
    for i, tok in enumerate(tokens):
        hl = i in iv_set
        if hl != current_hl:
            flush()
            current_hl = hl
        buf.append(tok)
    flush()
    return ''.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="Step 1 で生成した chunks JSON")
    ap.add_argument("--gens", required=True, help="Step 2 で生成した gens JSON")
    ap.add_argument("--output", required=True, help="出力 HTML ファイル")
    ap.add_argument("--k", type=int, default=5, help="最小一致長（word、デフォルト 5 = 論文準拠）")
    args = ap.parse_args()

    k = args.k
    chunks = json.loads(Path(args.chunks).read_text(encoding='utf-8'))
    gens_data = json.loads(Path(args.gens).read_text(encoding='utf-8'))
    chunks.sort(key=pid_num)
    gens_data.sort(key=pid_num)

    # 原文全体を再構築（空白保持）
    original = '\n\n'.join(c['excerpt_text'] for c in chunks)
    book_words, book_tokens, book_w2t = build_word_array(original)

    # best-of-10 選択
    def _count_matched(gen_text):
        words, _, _ = build_word_array(gen_text)
        if len(words) < k:
            return 0
        idx = defaultdict(list)
        for i in range(len(book_words) - k + 1):
            idx[tuple(book_words[i:i+k])].append(i)
        matched = [False] * len(words)
        for j in range(len(words) - k + 1):
            for i in idx.get(tuple(words[j:j+k]), []):
                ii, jj = i, j
                while ii > 0 and jj > 0 and book_words[ii-1] == words[jj-1]:
                    ii -= 1; jj -= 1
                p = 0
                while ii+p < len(book_words) and jj+p < len(words) and book_words[ii+p] == words[jj+p]:
                    p += 1
                if p >= k:
                    for q in range(jj, jj+p):
                        matched[q] = True
        return sum(matched)

    generated_pieces = []
    for g in gens_data:
        gens = g.get('generations', [])
        if not gens:
            continue
        best_num, best_text, best_match = -1, '', -1
        for gen in gens:
            m = _count_matched(gen['generated_text'])
            if m > best_match:
                best_match = m; best_num = gen['generation_num']; best_text = gen['generated_text']
        generated_pieces.append(best_text)
        print(f"  {g['excerpt_id']}: gen #{best_num} (matched {best_match} words)")

    generated = '\n\n'.join(generated_pieces)
    gen_words, gen_tokens, gen_w2t = build_word_array(generated)

    # word 単位で一致区間を検出
    raw_gen, raw_book = find_matches_both(gen_words, book_words, k)
    matches_gen_w = merge_intervals(raw_gen)
    matches_orig_w = merge_intervals(raw_book)

    # 表示用に token 単位区間へ展開（空白も含めてハイライト）
    matches_gen_t = expand_word_intervals_to_tokens(matches_gen_w, gen_w2t, len(gen_tokens))
    matches_orig_t = expand_word_intervals_to_tokens(matches_orig_w, book_w2t, len(book_tokens))

    orig_html = render_html(book_tokens, matches_orig_t, 'hit-orig')
    gen_html = render_html(gen_tokens, matches_gen_t, 'hit-gen')

    print(f"\nOriginal: {len(book_words):,} words")
    print(f"Generated: {len(gen_words):,} words")
    print(f"Gen matches: {sum(e-s for s,e in matches_gen_w):,} words ({100*sum(e-s for s,e in matches_gen_w)/len(gen_words):.1f}%)")
    print(f"Orig matches: {sum(e-s for s,e in matches_orig_w):,} words ({100*sum(e-s for s,e in matches_orig_w)/len(book_words):.1f}%)")

    html = HTML_TEMPLATE.format(
        original_html=json.dumps(orig_html),
        generated_html=json.dumps(gen_html),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding='utf-8')
    print(f"\nWrote: {output_path}")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Alignment-Whack-a-Mole Visualization</title>
<style>
  :root {{
    --speed: 50; /* px/s */
    --font-size: 22px;
    --panel-margin: 4vw;
    --fade-height: 12vh;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    width: 100vw; height: 100vh;
    background: #000000;
    overflow: hidden;
    font-family: "EB Garamond", Garamond, "Hoefler Text", "Baskerville",
                 "Times New Roman", "Source Serif Pro", serif;
    font-weight: 300;
    color: #ffffff;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }}
  .container {{
    display: flex;
    width: 100vw; height: 100vh;
    position: relative;
  }}
  .container::after {{
    content: '';
    position: absolute;
    top: 0; bottom: 0; left: 50%;
    width: 1px;
    background: #ffffff;
    transform: translateX(-50%);
    pointer-events: none;
    z-index: 10;
  }}
  .panel {{
    flex: 1 1 50%;
    height: 100vh;
    position: relative;
  }}
  .panel-clip {{
    position: absolute;
    top: 0; bottom: 0;
    left: var(--panel-margin);
    right: var(--panel-margin);
    overflow: hidden;
    -webkit-mask-image: linear-gradient(
      to bottom,
      transparent 0,
      black var(--fade-height),
      black calc(100% - var(--fade-height)),
      transparent 100%
    );
    mask-image: linear-gradient(
      to bottom,
      transparent 0,
      black var(--fade-height),
      black calc(100% - var(--fade-height)),
      transparent 100%
    );
  }}
  .text {{
    position: absolute;
    top: 0; left: 0; right: 0;
    font-size: var(--font-size);
    line-height: 1.75;
    padding: 0 1.5em;
    color: #ffffff;
    text-align: justify;
    hyphens: auto;
    white-space: pre-wrap;
    will-change: transform;
  }}
  .hit-gen {{
    background: #E53935;
    color: #ffffff;
    padding: 0 2px;
    border-radius: 2px;
  }}
  .hit-orig {{
    background: #FFEB3B;
    color: #000000;
    padding: 0 2px;
    border-radius: 2px;
  }}
  .label {{
    position: fixed;
    top: 3vh;
    font-family: "EB Garamond", Garamond, "Hoefler Text", serif;
    font-weight: 300;
    font-size: 13px;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: #888888;
    z-index: 20;
    pointer-events: none;
    user-select: none;
  }}
  .label-left {{ left: var(--panel-margin); }}
  .label-right {{ right: var(--panel-margin); }}
  .gui {{
    position: fixed;
    bottom: 3vh;
    right: var(--panel-margin);
    background: rgba(0, 0, 0, 0.7);
    border: 1px solid #333;
    padding: 14px 18px;
    z-index: 30;
    color: #cccccc;
    font-size: 12px;
    letter-spacing: 0.1em;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    transition: opacity 0.18s ease;
    min-width: 220px;
  }}
  .gui.hidden {{ opacity: 0; pointer-events: none; }}
  .gui .row {{
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 12px;
  }}
  .gui .row:last-child {{ margin-bottom: 0; }}
  .gui label {{
    text-transform: uppercase;
    letter-spacing: 0.25em;
    font-size: 10px;
    color: #888888;
    min-width: 50px;
  }}
  .gui input[type="range"] {{
    flex: 1;
    accent-color: #E53935;
    background: transparent;
  }}
  .gui .value {{
    color: #ffffff;
    min-width: 60px;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }}
  .gui .help {{
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #333;
    font-size: 10px;
    color: #777777;
    line-height: 1.6;
  }}
  .gui .help kbd {{
    color: #cccccc;
    font-family: monospace;
    font-size: 10px;
  }}
</style>
</head>
<body>
<div class="container">
  <!-- Left: Generated -->
  <div class="panel">
    <div class="panel-clip">
      <div class="text" id="genText"></div>
    </div>
  </div>
  <!-- Right: Original -->
  <div class="panel">
    <div class="panel-clip">
      <div class="text" id="origText"></div>
    </div>
  </div>
</div>
<div class="label label-left">Generated</div>
<div class="label label-right">Original</div>

<div class="gui" id="gui">
  <div class="row">
    <label>Speed</label>
    <input type="range" id="speedSlider" min="10" max="200" step="5" value="50">
    <span class="value" id="speedValue">50 px/s</span>
  </div>
  <div class="help">
    <kbd>Enter</kbd> Play / Pause &nbsp; <kbd>Wheel</kbd> Scrub<br>
    <kbd>G</kbd> Toggle this panel
  </div>
</div>

<script>
const ORIGINAL_HTML = {original_html};
const GENERATED_HTML = {generated_html};
const SPEED_PX_PER_SEC = 50;

const orig = document.getElementById('origText');
const gen = document.getElementById('genText');
orig.innerHTML = ORIGINAL_HTML;
gen.innerHTML = GENERATED_HTML;

const animations = [];
let genAnim = null;

window.addEventListener('load', () => {{
  requestAnimationFrame(() => {{
    function startScroll(el, isGen) {{
      const clip = el.parentElement;
      const clipH = clip.offsetHeight;
      const h = el.offsetHeight;
      const distance = clipH + h;
      const durationMs = (distance / SPEED_PX_PER_SEC) * 1000;
      const a = el.animate(
        [
          {{ transform: `translateY(${{clipH}}px)` }},
          {{ transform: `translateY(-${{h}}px)` }}
        ],
        {{ duration: durationMs, fill: 'forwards', easing: 'linear' }}
      );
      animations.push(a);
      if (isGen) genAnim = a;
    }}
    startScroll(orig, false);
    startScroll(gen, true);
    genAnim.addEventListener('finish', () => {{
      animations.forEach(a => {{ a.currentTime = 0; a.play(); }});
    }});
  }});
}});

document.addEventListener('keydown', (e) => {{
  if (e.key === 'Enter') {{
    e.preventDefault();
    if (animations.length === 0) return;
    const playing = animations[0].playState === 'running';
    animations.forEach(a => {{ playing ? a.pause() : a.play(); }});
  }} else if (e.key === 'g' || e.key === 'G') {{
    e.preventDefault();
    document.getElementById('gui').classList.toggle('hidden');
  }}
}});

const speedSlider = document.getElementById('speedSlider');
const speedValue = document.getElementById('speedValue');
speedSlider.addEventListener('input', (e) => {{
  const px = parseFloat(e.target.value);
  speedValue.textContent = px + ' px/s';
  animations.forEach(a => {{ a.playbackRate = px / SPEED_PX_PER_SEC; }});
}});

const TIME_PER_PX_MS = 1000 / SPEED_PX_PER_SEC;
document.addEventListener('wheel', (e) => {{
  if (animations.length === 0) return;
  const allPaused = animations.every(a => a.playState !== 'running');
  if (!allPaused) return;
  e.preventDefault();
  const delta = e.deltaY !== 0 ? e.deltaY : e.deltaX;
  const dt = delta * TIME_PER_PX_MS;
  animations.forEach(a => {{
    const dur = a.effect.getTiming().duration;
    let t = (a.currentTime || 0) + dt;
    if (t < 0) t = 0;
    if (t > dur) t = dur;
    a.currentTime = t;
  }});
  if (genAnim) {{
    const genDur = genAnim.effect.getTiming().duration;
    if (genAnim.currentTime >= genDur) {{
      animations.forEach(a => {{ a.currentTime = 0; }});
    }}
  }}
}}, {{ passive: false }});
</script>
</body>
</html>
"""


if __name__ == '__main__':
    main()
