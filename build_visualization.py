#!/usr/bin/env python3
"""日本語版: 縦書き 2 カラムスクロールの memorization 可視化 HTML を生成。

各チャンクで best-of-10（原文と最も一致した生成）を選び、原文と並べて
K 文字以上の連続一致をハイライト表示。Enter で再生/一時停止、G で GUI トグル、
マウスホイールで一時停止中のスクラブ。

使い方:
    python build_visualization.py \
        --chunks data/japanese/soseki/chunks/門.json \
        --gens results/門_gens.json \
        --output results/門_visualization.html
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def normalize(text: str) -> str:
    return re.sub(r'\s', '', text)


def pid_num(x):
    m = re.search(r'(\d+)', x['excerpt_id'])
    return int(m.group(1)) if m else 0


def find_matches_both(gen_chars, book_chars, k):
    """Maximal contiguous matches: returns (gen_intervals, book_intervals)."""
    if len(gen_chars) < k:
        return [], []
    idx = defaultdict(list)
    for i in range(len(book_chars) - k + 1):
        idx[tuple(book_chars[i:i + k])].append(i)
    visited = set()
    gen_intervals = []
    book_intervals = []
    for j in range(len(gen_chars) - k + 1):
        key = tuple(gen_chars[j:j + k])
        starts = idx.get(key)
        if not starts:
            continue
        for i in starts:
            ii, jj = i, j
            while ii > 0 and jj > 0 and book_chars[ii - 1] == gen_chars[jj - 1]:
                ii -= 1
                jj -= 1
            pair = (ii, jj)
            if pair in visited:
                continue
            visited.add(pair)
            p = 0
            while (ii + p) < len(book_chars) and (jj + p) < len(gen_chars) and book_chars[ii + p] == gen_chars[jj + p]:
                p += 1
            if p >= k:
                gen_intervals.append((jj, jj + p))
                book_intervals.append((ii, ii + p))
    return gen_intervals, book_intervals


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="Step 1 で生成した chunks JSON")
    ap.add_argument("--gens", required=True, help="Step 2 で生成した gens JSON")
    ap.add_argument("--output", required=True, help="出力 HTML ファイル")
    ap.add_argument("--k", type=int, default=7, help="最小一致長（文字、デフォルト 7）")
    args = ap.parse_args()

    k = args.k
    chunks = json.loads(Path(args.chunks).read_text(encoding='utf-8'))
    gens_data = json.loads(Path(args.gens).read_text(encoding='utf-8'))

    chunks.sort(key=pid_num)
    gens_data.sort(key=pid_num)

    original = ''.join(normalize(c['excerpt_text']) for c in chunks)
    book_chars_pre = list(original)

    # best-of-10 選択: chunk ごとに原文との一致が最も多い生成を採用
    def _count_matched_chars(gen_text, book_chars):
        g = list(normalize(gen_text))
        if len(g) < k:
            return 0
        idx = defaultdict(list)
        for i in range(len(book_chars) - k + 1):
            idx[tuple(book_chars[i:i+k])].append(i)
        matched = [False] * len(g)
        for j in range(len(g) - k + 1):
            for i in idx.get(tuple(g[j:j+k]), []):
                ii, jj = i, j
                while ii > 0 and jj > 0 and book_chars[ii-1] == g[jj-1]:
                    ii -= 1; jj -= 1
                p = 0
                while ii+p < len(book_chars) and jj+p < len(g) and book_chars[ii+p] == g[jj+p]:
                    p += 1
                if p >= k:
                    for q in range(jj, jj+p):
                        matched[q] = True
        return sum(matched)

    generated_pieces = []
    selection_log = []
    for g in gens_data:
        gens = g.get('generations', [])
        if not gens:
            continue
        best_num, best_text, best_match = -1, "", -1
        for gen in gens:
            m = _count_matched_chars(gen['generated_text'], book_chars_pre)
            if m > best_match:
                best_match = m
                best_num = gen['generation_num']
                best_text = gen['generated_text']
        generated_pieces.append(normalize(best_text))
        selection_log.append((g['excerpt_id'], best_num, best_match))
    generated = ''.join(generated_pieces)

    print("Best-of-10 selection per chunk:")
    for pid, gnum, m in selection_log:
        print(f"  {pid}: gen #{gnum} (matched {m} chars)")
    print()

    book_chars = list(original)
    gen_chars = list(generated)
    raw_gen, raw_book = find_matches_both(gen_chars, book_chars, k)
    matches_gen = merge_intervals(raw_gen)
    matches_orig = merge_intervals(raw_book)

    print(f"Original: {len(original):,} chars")
    print(f"Generated: {len(generated):,} chars")
    print(f"Gen match intervals: {len(matches_gen)} ({sum(e-s for s,e in matches_gen):,} chars, {100*sum(e-s for s,e in matches_gen)/len(generated):.1f}%)")
    print(f"Orig match intervals: {len(matches_orig)} ({sum(e-s for s,e in matches_orig):,} chars, {100*sum(e-s for s,e in matches_orig)/len(original):.1f}%)")

    html = HTML_TEMPLATE.format(
        original_json=json.dumps(original, ensure_ascii=False),
        generated_json=json.dumps(generated, ensure_ascii=False),
        matches_gen_json=json.dumps(matches_gen),
        matches_orig_json=json.dumps(matches_orig),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding='utf-8')
    print(f"\nWrote: {output_path}")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>Alignment-Whack-a-Mole Visualization</title>
<style>
  :root {{
    --speed: 80; /* px/s */
    --font-size: 30px;
    --panel-margin: 6vw; /* テキスト表示領域の左右マージン */
    --fade-width: 8vw; /* 出現・消失のフェード幅 */
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    width: 100vw;
    height: 100vh;
    background: #000000;
    overflow: hidden;
    font-family: "Tsukushi A Mincho", "Tsukushi B Mincho",
                 "Toppan Bunkyu Mincho",
                 "YuMincho", "Yu Mincho",
                 "Noto Serif JP",
                 "Hiragino Mincho ProN", "MS Mincho", serif;
    font-weight: 300;
    color: #ffffff;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    font-feature-settings: "palt";
  }}
  .container {{
    display: flex;
    width: 100vw;
    height: 100vh;
    position: relative;
  }}
  .container::after {{
    content: '';
    position: absolute;
    top: 0;
    bottom: 0;
    left: 50%;
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
    top: 0;
    bottom: 0;
    left: var(--panel-margin);
    right: var(--panel-margin);
    overflow: hidden;
    -webkit-mask-image: linear-gradient(
      to right,
      transparent 0,
      black var(--fade-width),
      black calc(100% - var(--fade-width)),
      transparent 100%
    );
    mask-image: linear-gradient(
      to right,
      transparent 0,
      black var(--fade-width),
      black calc(100% - var(--fade-width)),
      transparent 100%
    );
  }}
  .text {{
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    writing-mode: vertical-rl;
    -webkit-writing-mode: vertical-rl;
    font-size: var(--font-size);
    line-height: 1.85;
    padding: 9vh 0; /* 上下対称で本文を縦中央に配置 */
    color: #ffffff;
    white-space: normal;
    will-change: transform;
  }}
  .hit-gen {{
    background: #E53935;  /* 生成側: 赤マーカー */
    color: #ffffff;
    padding: 0 1px;
    border-radius: 1px;
  }}
  .hit-orig {{
    background: #FFEB3B;  /* 原文側: 黄マーカー */
    color: #000000;
    padding: 0 1px;
    border-radius: 1px;
  }}
  .label {{
    position: fixed;
    bottom: 4vh;
    font-family: "EB Garamond", Garamond, "Hoefler Text", "Times New Roman", serif;
    font-weight: 300;
    font-size: 13px;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: #888888;
    z-index: 20;
    pointer-events: none;
    user-select: none;
  }}
  .label-left {{
    left: var(--panel-margin);
  }}
  .label-right {{
    right: var(--panel-margin);
  }}
  .gui {{
    position: fixed;
    top: 4vh;
    right: var(--panel-margin);
    background: rgba(0, 0, 0, 0.7);
    border: 1px solid #333;
    padding: 14px 18px;
    z-index: 30;
    font-family: "EB Garamond", Garamond, "Hoefler Text", "Times New Roman", serif;
    color: #cccccc;
    font-size: 12px;
    letter-spacing: 0.1em;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    transition: opacity 0.18s ease;
    min-width: 220px;
  }}
  .gui.hidden {{
    opacity: 0;
    pointer-events: none;
  }}
  .gui .row {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
  }}
  .gui .row:last-child {{
    margin-bottom: 0;
  }}
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
  <!-- Left panel: GENERATED text (with yellow highlights) -->
  <div class="panel" id="leftPanel">
    <div class="panel-clip">
      <div class="text" id="genText"></div>
    </div>
  </div>
  <!-- Right panel: ORIGINAL text -->
  <div class="panel" id="rightPanel">
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
    <input type="range" id="speedSlider" min="20" max="240" step="5" value="50">
    <span class="value" id="speedValue">50 px/s</span>
  </div>
  <div class="help">
    <kbd>Enter</kbd> Play / Pause &nbsp; <kbd>Wheel</kbd> Scrub<br>
    <kbd>G</kbd> Toggle this panel
  </div>
</div>

<script>
const ORIGINAL = {original_json};
const GENERATED = {generated_json};
const MATCHES_GEN = {matches_gen_json};
const MATCHES_ORIG = {matches_orig_json};
const SPEED_PX_PER_SEC = 50;

function escapeHtml(s) {{
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}}

function buildHighlighted(text, ranges, cls) {{
  let html = '';
  let pos = 0;
  for (const [s, e] of ranges) {{
    if (pos < s) html += escapeHtml(text.slice(pos, s));
    html += '<span class="' + cls + '">' + escapeHtml(text.slice(s, e)) + '</span>';
    pos = e;
  }}
  if (pos < text.length) html += escapeHtml(text.slice(pos));
  return html;
}}

const orig = document.getElementById('origText');
const gen  = document.getElementById('genText');
orig.innerHTML = buildHighlighted(ORIGINAL, MATCHES_ORIG, 'hit-orig');
gen.innerHTML  = buildHighlighted(GENERATED, MATCHES_GEN, 'hit-gen');

// After layout, measure widths and start scrolling animations
const animations = [];
let genAnim = null;

window.addEventListener('load', () => {{
  requestAnimationFrame(() => {{
    function startScroll(el, isGen) {{
      const clip = el.parentElement; // .panel-clip
      const clipW = clip.offsetWidth;
      const w = el.offsetWidth;
      const distance = clipW + w;
      const durationMs = (distance / SPEED_PX_PER_SEC) * 1000;
      const a = el.animate(
        [
          {{ transform: `translateX(-${{w}}px)` }},
          {{ transform: `translateX(${{clipW}}px)` }}
        ],
        {{ duration: durationMs, fill: 'forwards', easing: 'linear' }}
      );
      animations.push(a);
      if (isGen) genAnim = a;
    }}

    startScroll(orig, false);
    startScroll(gen, true);

    // 生成文の最後が画面外に出たら両パネル冒頭から再開（ループ）
    genAnim.addEventListener('finish', () => {{
      animations.forEach(a => {{
        a.currentTime = 0;
        a.play();
      }});
    }});
  }});
}});

// Enter で再生/一時停止のトグル / G で GUI トグル
document.addEventListener('keydown', (e) => {{
  if (e.key === 'Enter') {{
    e.preventDefault();
    if (animations.length === 0) return;
    const playing = animations[0].playState === 'running';
    animations.forEach(a => {{
      if (playing) a.pause();
      else a.play();
    }});
  }} else if (e.key === 'g' || e.key === 'G') {{
    e.preventDefault();
    document.getElementById('gui').classList.toggle('hidden');
  }}
}});

// スピードスライダー: playbackRate を変えてリアルタイム反映
const speedSlider = document.getElementById('speedSlider');
const speedValue = document.getElementById('speedValue');
speedSlider.addEventListener('input', (e) => {{
  const px = parseFloat(e.target.value);
  speedValue.textContent = px + ' px/s';
  const rate = px / SPEED_PX_PER_SEC;
  animations.forEach(a => {{ a.playbackRate = rate; }});
}});

// 一時停止中のみマウスホイールで手動スクロール
const TIME_PER_PX_MS = 1000 / SPEED_PX_PER_SEC; // 1px 移動に必要な時間
document.addEventListener('wheel', (e) => {{
  if (animations.length === 0) return;
  // 全てのアニメが pause 状態のときのみ反応
  const allPaused = animations.every(a => a.playState !== 'running');
  if (!allPaused) return;
  e.preventDefault();
  // deltaY は縦スクロール量。下スクロール = 進める方向。
  // deltaX は横方向（trackpad）。両方を考慮。
  const delta = e.deltaY !== 0 ? e.deltaY : e.deltaX;
  const dt = delta * TIME_PER_PX_MS;
  animations.forEach(a => {{
    const dur = a.effect.getTiming().duration;
    let t = (a.currentTime || 0) + dt;
    if (t < 0) t = 0;
    if (t > dur) t = dur;
    a.currentTime = t;
  }});
  // 生成文が終端に達したら両パネル冒頭へループ（マウスホイール時も同じ挙動）
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
