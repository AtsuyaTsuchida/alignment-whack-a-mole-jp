# Japanese Memorization Probe — Alignment Whack-a-Mole 日本語版

[Liu et al. 2026 "Alignment Whack-a-Mole"](https://arxiv.org/abs/2603.20957) の手法を
日本語文学に適用し、Fine-Tuning による verbatim recall（一字一句の再現）の活性化を測定する。

---

## 1. セットアップ

### 1.1 リポジトリを clone

```bash
git clone https://github.com/AtsuyaTsuchida/alignment-whack-a-mole-jp.git
cd alignment-whack-a-mole-jp
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
```

接続テスト:

```bash
python -c "from openai import OpenAI; print(OpenAI().models.list().data[0])"
```

### 1.4 テキストデータ

`data/japanese/{author}/raw/` 配下に青空文庫からダウンロードした UTF-8 `.txt` を配置。
漱石・芥川・太宰・宮沢・鷗外の主要作品は本リポジトリに同梱（パブリックドメイン）。

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
- **あらすじスタイル**: 論文準拠の plot summary（登場人物・視点・出来事の順序を維持）

### Step 1.5. FT 訓練データ統合

```bash
python scripts/merge_chunks_to_train.py \
  --chunks_dir data/japanese/soseki/chunks/ \
  --output data/japanese/soseki/soseki_train_3000.json \
  --max_examples 3000 \
  --exclude 門
```

`--exclude` で held-out 作品を訓練データから除外。

### Step 1.6. OpenAI Fine-tuning

`soseki_train_3000.json` を OpenAI Dashboard の Fine-tuning ページから投入:

1. JSONL 変換: `{"messages": [{"role": "user", "content": instruction}, {"role": "assistant", "content": excerpt_text}]}` 形式に変換
2. Base model: `gpt-4o-2024-08-06`
3. Suffix: 任意（例 `soseki-jp-3000-ft`）
4. 投入 → トレーニング完了まで 1〜2 時間
5. 完成 → モデル ID（例 `ft:gpt-4o-2024-08-06:personal:soseki-jp-3000-ft:XXX`）を取得

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
preprocess/jp_preprocess.py         # Step 1: チャンク分割 + あらすじ生成
finetuning/jp_generate.py           # Step 2: FT モデルで 10× 生成
evaluation/jp_memorization_eval.py  # Step 3: BMC@k 評価
scripts/merge_chunks_to_train.py    # Step 1.5: FT 訓練データ統合
data/japanese/{author}/raw/         # 青空文庫テキスト（PD）
```

---

## 5. 論文との差分

| 項目 | 論文 | 本実装 | 備考 |
|---|---|---|---|
| 単位 | word | char | 日本語適応 |
| 生成数 | 100/excerpt | 10/excerpt | コスト削減 |
| k 値 | BMC@5 | BMC@10 | char 単位で厳格化 |

機構（mechanism）レベルでは論文準拠。論文同等の精度を求める場合は `--num_generations 100` を指定。

---

## 6. トラブルシューティング

| 症状 | 原因 | 対処 |
|---|---|---|
| `RateLimitError: 429 quota exceeded` | OpenAI クレジット切れ | Dashboard でリチャージ |
| 長時間 stuck | OpenAI 接続ハング | jp_generate は timeout=90s で自動回復、checkpoint で再開可能 |
| 生成出力が漱石風にならない | FT モデル ID 誤り | 指定したモデル ID を再確認 |
| Step 2 を途中で kill | — | 再実行で checkpoint から自動再開 |

---

## ライセンス

MIT
