# Memorization Probe — Alignment Whack-a-Mole（日本語/英語）

論文 [Liu et al. 2026 "Alignment Whack-a-Mole"](https://arxiv.org/abs/2603.20957) の手法を
日本語（青空文庫）と英語（Project Gutenberg）の両方で再現するパイプライン。

- **日本語**: 漱石・芥川・太宰・宮沢・鷗外（PD）
- **英語**: Dickens（PD、Project Gutenberg）

---

## 1. セットアップ

### 1.1 リポジトリを clone

```bash
git clone https://github.com/YOUR_ORG/THIS_REPO.git
cd THIS_REPO
```

### 1.2 Python 仮想環境

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1.3 OpenAI API キー

```bash
export OPENAI_API_KEY="sk-proj-..."
# または .env ファイルに記載（python-dotenv が自動ロード）
```

接続テスト:

```bash
python -c "from openai import OpenAI; print(OpenAI().models.list().data[0])"
```

### 1.4 テキストデータ

- 日本語: `data/japanese/{author}/raw/` 配下に青空文庫から UTF-8 `.txt` を配置
- 英語: `data/english/dickens/raw/` 配下に Project Gutenberg から `.txt` を配置
  - PG ヘッダ/フッタは `python data/english/dickens/clean_pg.py` で除去可能

---

## 2. パイプライン全体像

```
raw .txt
  │
  │ Step 1: チャンク分割 + あらすじ生成
  ▼
preprocess/jp_preprocess.py
  │
  ├─→ FT 訓練用（複数作品）
  │     │
  │     │ Step 1.5: 統合
  │     ▼
  │   scripts/merge_chunks_to_train.py
  │     │
  │     ▼
  │   train.json → OpenAI Dashboard で FT job 投入 → モデル ID 取得
  │
  └─→ テスト書（held-out）
        │
        │ Step 2: FT モデルで 10× 生成
        ▼
        finetuning/jp_generate.py
        │
        │ gens.json
        │
        │ Step 3: BMC@k 評価
        ▼
        evaluation/jp_memorization_eval.py
        │
        ▼
        結果（BMC@10, 最長スパン）
```

---

## 3. 実行手順

### Step 1. 前処理（チャンク分割 + あらすじ生成）

```bash
python preprocess/jp_preprocess.py \
  --input_txt data/japanese/soseki/raw/門.txt \
  --output_json data/japanese/soseki/chunks/門.json \
  --book_name 門 \
  --author_name 夏目漱石 \
  --summary_ratio 0.5
```

- **所要時間**: 1 作品あたり 5〜15 分（GPT-4o 同期 API 呼び出し）
- **コスト**: 1 作品あたり ~$0.3
- **あらすじスタイル**: plot summary（登場人物・視点・出来事の順序を維持）

### Step 1.5. FT 訓練データ統合

```bash
python scripts/merge_chunks_to_train.py \
  --chunks_dir data/japanese/soseki/chunks/ \
  --output data/japanese/soseki/soseki_train_3000.json \
  --max_examples 3000 \
  --exclude 門
```

`--exclude` で held-out 作品を訓練データから除外。

### Step 1.6. JSONL 変換

OpenAI Fine-tuning API は chat 形式の JSONL を要求するので、専用スクリプトで変換:

```bash
python scripts/json_to_jsonl.py \
  --input data/japanese/soseki/soseki_train_3000.json \
  --output data/japanese/soseki/soseki_train.jsonl
```

各例は `system × 2 + user (instruction) + assistant (excerpt)` の構造になる。
system message は推論時の `finetuning/jp_generate.py` と一致させてある（訓練と推論の prompt 分布を揃えるため）。

### Step 1.7. OpenAI Fine-tuning

OpenAI Dashboard の Fine-tuning ページから `soseki_train.jsonl` を投入:

1. Base model: `gpt-4o-2024-08-06`
2. Suffix: 任意（例 `soseki-jp-3000-ft`）
3. 投入 → トレーニング完了まで 1〜2 時間
4. 完成 → モデル ID（例 `ft:gpt-4o-2024-08-06:personal:soseki-jp-3000-ft:XXX`）を取得

**コスト**: $15〜25（3000 例の場合）

### Step 2. FT モデルで生成

```bash
python finetuning/jp_generate.py \
  --test_file data/japanese/soseki/chunks/門.json \
  --output_file results/門_gens.json \
  --model ft:gpt-4o-2024-08-06:personal:soseki-jp-3000-ft:XXX \
  --num_generations 10
```

- **所要時間**: 1 chunk ~55 秒 → 200 chunks で約 3 時間
- **コスト**: ~$5〜10
- **特徴**: timeout=90s、checkpoint で途中再開可能、`caffeinate -i` 推奨（macOS）

### Step 3. 評価

```bash
python evaluation/jp_memorization_eval.py \
  --test_book data/japanese/soseki/chunks/門.json \
  --generation_file results/門_gens.json \
  --k 10 \
  --trim_m 5
```

- **所要時間**: 1〜2 秒
- **出力**: BMC@10、最長記憶ブロック、最長再現スパン

---

## 4. ファイル構成

```
preprocess/
├── jp_preprocess.py              # Step 1: 日本語版（char 単位）
└── en_preprocess.py              # Step 1: 英語版（word 単位、async 並列）

scripts/
├── merge_chunks_to_train.py      # Step 1.5: chunks 統合
└── json_to_jsonl.py              # Step 1.6: OpenAI FT 用 JSONL 変換

finetuning/
├── jp_generate.py                # Step 2: 日本語版（ThreadPoolExecutor 並列）
└── en_generate.py                # Step 2: 英語版

evaluation/
├── jp_memorization_eval.py       # Step 3: 日本語 BMC@k 評価
└── en_memorization_eval.py       # Step 3: 英語 BMC@k 評価

export_jp_text.py                 # 任意: 原文/生成文/あらすじ書き出し（日本語、§5 参照）
export_en_text.py                 # 任意: 同上（英語、§5 参照）
build_visualization.py            # 任意: 縦書き可視化 HTML（日本語、§6 参照）
build_en_visualization.py         # 任意: 横書き可視化 HTML（英語、§6 参照）

data/
├── japanese/{author}/raw/        # 青空文庫テキスト
└── english/dickens/
    ├── raw/                      # Project Gutenberg テキスト
    └── clean_pg.py               # PG ヘッダ除去ユーティリティ
```

## 4.5 英語版の使い方（差分のみ）

```bash
# Step 1: 各 Dickens 作品を前処理
python preprocess/en_preprocess.py \
  --input_txt data/english/dickens/raw/oliver_twist.txt \
  --output_json data/english/dickens/chunks/oliver_twist.json \
  --book_name "Oliver Twist" \
  --author_name "Charles Dickens" \
  --summary_ratio 0.5

# Step 1.5: 統合（held-out: A Christmas Carol を除外）
python scripts/merge_chunks_to_train.py \
  --chunks_dir data/english/dickens/chunks/ \
  --output data/english/dickens/dickens_train.json \
  --max_examples 3000 \
  --exclude "A Christmas Carol"

# Step 2: FT モデルで生成
python finetuning/en_generate.py \
  --test_file data/english/dickens/chunks/a_christmas_carol.json \
  --output_file results/a_christmas_carol_gens.json \
  --model ft:gpt-4o-2024-08-06:personal:dickens-en-2521:XXX \
  --num_generations 10

# Step 3: 評価（word 単位、論文準拠の BMC@5）
python evaluation/en_memorization_eval.py \
  --test_book data/english/dickens/chunks/a_christmas_carol.json \
  --generation_file results/a_christmas_carol_gens.json \
  --k 5 --trim_m 5
```

---

## 5. テキストダンプ（任意）

評価とは別に、人間可読な原文・生成文・あらすじを txt として書き出すユーティリティ。
各チャンクで **best-of-10**（原文と最も一致した生成）を選択し、チャンク順に連結。

### 日本語

```bash
python export_jp_text.py \
  --chunks data/japanese/akutagawa/chunks/羅生門.json \
  --gens results/羅生門_gens.json \
  --output_dir data/japanese/text_exports \
  --name 羅生門
# → 羅生門_原文.txt / 羅生門_生成文.txt / 羅生門_あらすじ.txt
```

### 英語

```bash
python export_en_text.py \
  --chunks data/english/dickens/chunks/a_christmas_carol.json \
  --gens results/a_christmas_carol_gens.json \
  --output_dir data/english/text_exports \
  --name a_christmas_carol
# → a_christmas_carol_original.txt / _generated.txt / _summary.txt
```

---

## 6. 可視化 HTML（任意）

原文と生成文を 2 カラムで並べ、K 文字/word 以上の連続一致をハイライトする
単独 HTML を生成。各チャンクで best-of-10 を選択して連結し、一致区間は build 時に
計算してテンプレに焼き付ける。

### 日本語版（縦書き、横スクロール）

```bash
python build_visualization.py \
  --chunks data/japanese/soseki/chunks/門.json \
  --gens results/門_gens.json \
  --output results/門_viz.html
# K=7 文字以上の一致をハイライト（--k で変更可）
```

### 英語版（横書き、縦スクロール）

```bash
python build_en_visualization.py \
  --chunks data/english/dickens/chunks/a_christmas_carol.json \
  --gens results/a_christmas_carol_gens.json \
  --output results/a_christmas_carol_viz.html
# K=5 word 以上の一致をハイライト（論文準拠、--k で変更可）
```

### 操作

- **Enter**: 再生 / 一時停止
- **G**: GUI パネルの表示/非表示
- **マウスホイール**: 一時停止中のスクラブ

---

## 7. トラブルシューティング

| 症状 | 原因 | 対処 |
|---|---|---|
| `RateLimitError: 429 quota exceeded` | OpenAI クレジット切れ | Dashboard でリチャージ |
| 長時間 stuck | OpenAI 接続ハング | jp_generate は timeout=90s で自動回復、checkpoint で再開可能 |
| 生成出力が漱石風にならない | FT モデル ID 誤り | 指定したモデル ID を再確認 |
| Step 2 を途中で kill | — | 再実行で checkpoint から自動再開 |

---

## ライセンス

MIT
