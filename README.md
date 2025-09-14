# mT5–BERT–BiLSTM Hybrid for Persian Sentiment Analysis

A hybrid pipeline for Persian (and multilingual) **sentiment analysis**. If an input text exceeds **150 tokens**, it is **first summarized** with **mT5**; then sentiment is predicted via a **BiLSTM** branch (optionally fused with **BERT** features).

> **Reported metrics**
>
> - **BERT–BiLSTM (sentiment):** Accuracy **87.28%** · Precision **87.79%** · Recall **87.28%** · F1 **87.37%**  
> - **mT5 (summarization):** ROUGE‑1 **35.5%** · ROUGE‑2 **11.5%** · ROUGE‑L **30.5%**

---

## Contents

- [Architecture](#architecture)
- [Setup](#setup)
- [Quickstart](#quickstart)
- [Results](#results)
- [CLI Reference](#cli-reference)
  - [mT5 + BERT-BiLSTM inference](#infer_mt5_bert_bilstmpy)
  - [mT5 + BiLSTM Inference](#infer_mt5_bilstmpy)
  - [BERT-BiLSTM Stratified split](#bert_bilstmsplit_datasetpy)
  - [BERT-BiLSTM Build vocab & fastText embeddings](#bert_bilstmembeddingspy)
  - [BERT–BiLSTM Data module](#bert_bilstmdata_modulepy)
  - [BERT–BiLSTM Model](#bert_bilstmmodelpy)
  - [BERT–BiLSTM Training](#bert_bilstmtrainpy)
  - [BERT–BiLSTM Inference](#bert_bilstminferpy)
  - [BiLSTM Stratified split](#bilstmsplit_datasetpy)
  - [BiLSTM Build vocab & fastText embeddings](#bilstmembeddingspy)
  - [BiLSTM Data module](#bilstmdata_modulepy)
  - [BiLSTM Model](#bilstmmodelpy)
  - [BiLSTM Training](#bilstmtrainpy)
  - [BiLSTM Inference](#bilstminferpy)
  - [mT5 Tokenization](#mt5tokenize_modelpy)
  - [mT5 Training](#mt5trainpy)
  - [mT5 inference](#mt5inferpy)
- [Reproducibility](#reproducibility)
- [License](#license)

---

## Architecture

```
Long Text ( >150 tokens )
          └─> mT5 Summarizer ──> Short Text
Short/Original Text ────────────────────────┐
                                            ├─> BiLSTM (fastText) ─┐
Optional: BERT (mBERT) features ────────────┘                      ├─> Classifier (Sentiment)
                                                                    └─> (Hybrid: BERT + BiLSTM)
```

- **mT5** (e.g., `google/mt5-small`) performs abstractive summarization to reduce very long inputs.
- **BiLSTM** consumes tokenized text with **fastText** embeddings; the hybrid version also ingests **BERT** features.
- A uniform **max length of 150** is used across modules for consistency.

---

## Setup

```bash
# Python 3.10+ is recommended
pip install -r requirements.txt

# Make sure you have fastText vectors (example for Persian):
# embedding/cc.fa.300.bin
```

**Project layout (selected):**
```
datasets/
  ├─ sentiment.csv
  └─ summarize.csv
embedding/
  └─ cc.fa.300.bin
```

---

## Quickstart

### End‑to‑end inference (auto‑summarize then sentiment)
```bash
python infer_mt5_bert_bilstm.py --text "YOUR LONG PERSIAN TEXT…"
```

### Summarization only
```bash
python mt5/infer.py --text "YOUR LONG PERSIAN TEXT…"
```

### Sentiment only (BiLSTM)
```bash
python bilstm/infer.py --text "یک متن کوتاه برای تست احساس…"
```

---

## Results

- **BERT–BiLSTM (sentiment):** Accuracy **87.28%** · Precision **87.79%** · Recall **87.28%** · F1 **87.37%**
- **mT5 (summarization):** ROUGE‑1 **35.5%** · ROUGE‑2 **11.5%** · ROUGE‑L **30.5%**

> These results were obtained on the authors’ internal splits/settings (see CLI defaults below).

---

## CLI Reference

> All scripts use `argparse`. Defaults are shown for convenience. “Action” indicates special flags like `store_true`.

### `infer_mt5_bert_bilstm.py`
**Description:** Inference for the full mT5–BERT–BiLSTM pipeline.

| Argument | Help | Default | Action |
|---|---|---|---|
| `--device` | Compute device (`cuda` or `cpu`). | `"cuda" if torch.cuda.is_available() else "cpu"` |  |
| `--max_len` | Max input tokens; longer texts are truncated/padded. | `150` |  |
| `--t5_model_name` | Base mT5 model on Hugging Face. | `"google/mt5-small"` |  |
| `--t5_model_dir` | Folder with trained/merged mT5 weights (e.g., LoRA‑merged). | `"models/mt5/lora-merged"` |  |
| `--num_beams` | Beam size for summarization. | `4` |  |
| `--bilstm_model_dir` | Checkpoint for BERT–BiLSTM sentiment model. | `"models/bert_bilstm/bilstm_best.pt"` |  |
| `--vocab_json` | Vocabulary JSON for fastText/BiLSTM tokenizer. | `"models/bert_bilstm/vocab.json"` |  |
| `--embeddings_path` | Serialized fastText embeddings tensor. | `"models/bert_bilstm/embeddings.pt"` |  |
| `--batch_size` | Inference batch size. | `256` |  |
| `--bert_model_name` | BERT backbone compatible with the checkpoint. | `"bert-base-multilingual-cased"` |  |
| `--text` | Input text to (optionally) summarize and classify. | `None` |  |

---

### `infer_mt5_bilstm.py`
**Description:** Inference with mT5 (summ.) + BiLSTM (sent.) — no BERT features.

| Argument | Help | Default | Action |
|---|---|---|---|
| `--device` | Compute device. | `"cuda" if torch.cuda.is_available() else "cpu"` |  |
| `--max_len` | Max input tokens for the classifier. | `150` |  |
| `--t5_model_name` | Base mT5 model. | `"google/mt5-small"` |  |
| `--t5_model_dir` | Trained/merged mT5 weights. | `"models/mt5/lora-merged"` |  |
| `--num_beams` | Beam size for summarization. | `4` |  |
| `--bilstm_model_dir` | BiLSTM checkpoint. | `"models/bilstm/bilstm_best.pt"` |  |
| `--vocab_json` | BiLSTM vocabulary. | `"models/bilstm/vocab.json"` |  |
| `--embeddings_path` | BiLSTM fastText embeddings. | `"models/bilstm/embeddings.pt"` |  |
| `--batch_size` | Inference batch size. | `256` |  |
| `--text` | Input text. | `None` |  |

---

### `bert_bilstm/split_dataset.py`
**Description:** Stratified 80/10/10 split from `sentiment.csv` (hybrid).

| Argument | Help | Default | Action |
|---|---|---|---|
| `--input_csv` | Input CSV with all sentiment data. | `"datasets/sentiment.csv"` |  |
| `--output_dir` | Output folder for `train/val/test`. | `"datasets/bert-bilstm"` |  |
| `--random_state` | Seed for stratified split. | `42` |  |
| `--tolerance_pct` | Allowed deviation to preserve class ratios. | `0.5` |  |

---

### `bert_bilstm/embeddings.py`
**Description:** Build vocabulary & fastText embeddings (for the hybrid branch).

| Argument | Help | Default | Action |
|---|---|---|---|
| `--train_csv` | CSV used to extract vocabulary. | `"datasets/split-sentiment/train.csv"` |  |
| `--fasttext_path` | Pretrained fastText binary file. | `"embedding/cc.fa.300.bin"` |  |
| `--out_dir` | Output dir for vocab & embeddings. | `"models/bert-bilstm"` |  |
| `--min_freq` | Min token frequency to keep. | `2` |  |
| `--seed` | Random seed. | `42` |  |

---

### `bert_bilstm/data_module.py`
**Description:** DataLoaders with fixed padding + safe caching (Hybrid: BERT + fastText).

| Argument | Help | Default | Action |
|---|---|---|---|
| `--train_csv` | Training CSV. | `"datasets/bert_bilstm/train.csv"` |  |
| `--val_csv` | Validation CSV. | `"datasets/bert_bilstm/val.csv"` |  |
| `--test_csv` | Test CSV. | `"datasets/bert_bilstm/test.csv"` |  |
| `--vocab_json` | Vocabulary file. | `"models/bert_bilstm/vocab.json"` |  |
| `--batch_size` | Batch size. | `128` |  |
| `--num_workers` | DataLoader workers. | `0` |  |
| `--sampler_mode` | Sampling strategy. | `"none"` |  |
| `--max_len` | Max sequence length. | `150` |  |
| `--cache_dir` | Cache directory for preprocessed samples. | `"artifacts/cache"` |  |
| `--no_cache` | Disable caching. |  | `store_true` |
| `--rebuild_cache` | Ignore existing cache and rebuild. |  | `store_true` |

---

### `bert_bilstm/model.py`
**Description:** Defines the BERT‑augmented BiLSTM sentiment model.

| Argument | Help | Default | Action |
|---|---|---|---|
| `--embeddings_path` | Precomputed fastText embeddings. | `"models/bert_bilstm/embeddings.pt"` |  |
| `--train_csv` | Training CSV. | `"datasets/bert_bilstm/train.csv"` |  |
| `--val_csv` | Validation CSV. | `"datasets/bert_bilstm/val.csv"` |  |
| `--test_csv` | Test CSV. | `"datasets/bert_bilstm/test.csv"` |  |
| `--vocab_json` | Vocabulary file. | `"models/bert_bilstm/vocab.json"` |  |
| `--batch_size` | DataLoader batch size. | `128` |  |
| `--num_workers` | DataLoader workers. | `0` |  |
| `--sampler_mode` | Sampling strategy (e.g., `none`, class‑balanced). | `"none"` |  |
| `--max_len` | Max sequence length. | `150` |  |
| `--hidden_size` | LSTM hidden size. | `256` |  |
| `--num_layers` | Number of stacked LSTM layers. | `1` |  |
| `--dropout_embed` | Dropout on embedding layer. | `0.1` |  |
| `--dropout_out` | Dropout after LSTM / on classifier. | `0.3` |  |
| `--pooling` | Feature pooling (`attn`/`max`/`avg`). | `"attn"` |  |
| `--freeze_embeddings` | Freeze fastText embeddings. |  | `store_true` |
| `--freeze_bert` | Freeze BERT parameters. |  | `store_true` |

---

### `bert_bilstm/train.py`
**Description:** Train **BiLSTM + BERT + fastText** (hybrid sentiment).

| Argument | Help | Default | Action |
|---|---|---|---|
| `--train_csv` | Training CSV. | `"datasets/bert_bilstm/train.csv"` |  |
| `--val_csv` | Validation CSV. | `"datasets/bert_bilstm/val.csv"` |  |
| `--test_csv` | Test CSV. | `"datasets/bert_bilstm/test.csv"` |  |
| `--vocab_json` | Vocabulary file. | `"models/bert_bilstm/vocab.json"` |  |
| `--embeddings_path` | fastText embeddings. | `"models/bert_bilstm/embeddings.pt"` |  |
| `--epochs` | Training epochs. | `8` |  |
| `--batch_size` | Train/eval batch size. | `128` |  |
| `--lr` | Learning rate. | `2e-3` |  |
| `--weight_decay` | L2 regularization. | `0.01` |  |
| `--grad_accum_steps` | Gradient accumulation steps. | `1` |  |
| `--max_len` | Max sequence length. | `150` |  |
| `--sampler_mode` | Data sampling strategy. | `"none"` |  |
| `--hidden_size` | LSTM hidden size. | `256` |  |
| `--num_layers` | LSTM layers. | `1` |  |
| `--dropout_embed` | Embedding dropout. | `0.1` |  |
| `--dropout_out` | Post‑LSTM / classifier dropout. | `0.3` |  |
| `--pooling` | Feature pooling. | `"attn"` |  |
| `--freeze_embeddings` | Freeze fastText embeddings. |  | `store_true` |
| `--freeze_bert` | Freeze BERT encoder. |  | `store_true` |
| `--loss` | Loss function (`ce` or `focal`). | `"ce"` |  |
| `--gamma` | Gamma for Focal Loss. | `2.0` |  |
| `--scheduler` | LR scheduler (`onecycle`, …). | `"onecycle"` |  |
| `--max_grad_norm` | Gradient clipping. | `1.0` |  |
| `--amp` | Automatic Mixed Precision. |  | `store_true` |
| `--log_grad` | Log branch gradients. |  | `store_true` |
| `--log_grad_every` | Gradient logging frequency (in steps). | `1` |  |
| `--out_dir` | Model/checkpoint output dir. | `"models/bert-bilstm"` |  |
| `--chart_dir` | Training charts output dir. | `"chart/bert-ilstm"` |  |
| `--logs_dir` | Logs directory. | `"logs/bert-ilstm"` |  |
| `--seed` | Random seed. | `42` |  |

---

### `bert_bilstm/infer.py`
**Description:** Inference for the **Hybrid** (BERT + fastText + BiLSTM).

| Argument | Help | Default | Action |
|---|---|---|---|
| `--checkpoint` | Trained hybrid checkpoint. | `"models/bert_bilstm/bert_bilstm_best.pt"` |  |
| `--vocab_json` | Vocab compatible with the checkpoint. | `"models/bert_bilstm/vocab.json"` |  |
| `--embeddings_path` | fastText embeddings for the BiLSTM branch. | `"models/bert_bilstm/embeddings.pt"` |  |
| `--bert_model_name` | BERT backbone used in training. | `"bert-base-multilingual-cased"` |  |
| `--max_len` | Max input length (falls back to training value if `None`). | `None` |  |
| `--batch_size` | Inference batch size. | `128` |  |
| `--device` | Compute device. | `"cuda" if torch.cuda.is_available() else "cpu"` |  |
| `--text` | Single or multiple inputs (custom separator supported). | `None` |  |

---

### `bilstm/split_dataset.py`
**Description:** Stratified 80/10/10 split from `sentiment.csv` (stand‑alone).

| Argument | Help | Default | Action |
|---|---|---|---|
| `--input_csv` | Full input CSV. | `"datasets/sentiment.csv"` |  |
| `--output_dir` | Output folder with splits. | `"datasets/bilstm"` |  |
| `--random_state` | Seed. | `42` |  |
| `--tolerance_pct` | Allowed class‑ratio deviation. | `0.5` |  |

---

### `bilstm/embeddings.py`
**Description:** Build vocabulary & fastText embeddings (for stand‑alone BiLSTM).

| Argument | Help | Default | Action |
|---|---|---|---|
| `--train_csv` | Training CSV to build vocab. | `"datasets/bilstm/train.csv"` |  |
| `--fasttext_path` | Pretrained fastText model path. | `"embedding/cc.fa.300.bin"` |  |
| `--out_dir` | Output dir for vocab & embeddings. | `"models/bilstm"` |  |
| `--min_freq` | Min token frequency. | `2` |  |
| `--seed` | Random seed. | `42` |  |

---

### `bilstm/data_module.py`
**Description:** DataLoaders with pad/trim, padding masks, and configurable sampling.

| Argument | Help | Default | Action |
|---|---|---|---|
| `--train_csv` | Training CSV. | `"datasets/bilstm/train.csv"` |  |
| `--val_csv` | Validation CSV. | `"datasets/bilstm/val.csv"` |  |
| `--test_csv` | Test CSV. | `"datasets/bilstm/test.csv"` |  |
| `--vocab_json` | Vocabulary file. | `"models/bilstm/vocab.json"` |  |
| `--batch_size` | Batch size. | `128` |  |
| `--num_workers` | DataLoader workers. | `2` |  |
| `--sampler_mode` | Sampling strategy. | `"none"` |  |
| `--max_len` | Max sequence length. | `150` |  |

---

### `bilstm/model.py`
**Description:** BiLSTM classifier definition.

| Argument | Help | Default | Action |
|---|---|---|---|
| `--embeddings_path` | fastText embeddings path. | `"models/bilstm/embeddings.pt"` |  |
| `--train_csv` | Training CSV. | `"datasets/bilstm/train.csv"` |  |
| `--val_csv` | Validation CSV. | `"datasets/bilstm/val.csv"` |  |
| `--test_csv` | Test CSV. | `"datasets/bilstm/test.csv"` |  |
| `--vocab_json` | Vocabulary path. | `"models/bilstm/vocab.json"` |  |
| `--batch_size` | Batch size. | `128` |  |
| `--num_workers` | DataLoader workers. | `2` |  |
| `--sampler_mode` | Sampling strategy. | `"none"` |  |
| `--max_len` | Max sequence length. | `150` |  |
| `--hidden_size` | LSTM hidden size. | `256` |  |
| `--num_layers` | LSTM layers. | `1` |  |
| `--dropout_embed` | Embedding dropout. | `0.1` |  |
| `--dropout_out` | Post‑LSTM / classifier dropout. | `0.3` |  |
| `--pooling` | Pooling type (`attn`/`max`/`avg`). | `"attn"` |  |
| `--freeze_embeddings` | Freeze fastText embeddings. |  | `store_true` |

---

### `bilstm/train.py`
**Description:** Train **BiLSTM** (stand‑alone).

| Argument | Help | Default | Action |
|---|---|---|---|
| `--train_csv` | Training CSV. | `"datasets/bilstm/train.csv"` |  |
| `--val_csv` | Validation CSV. | `"datasets/bilstm/val.csv"` |  |
| `--test_csv` | Test CSV. | `"datasets/bilstm/test.csv"` |  |
| `--vocab_json` | Vocabulary. | `"models/bilstm/vocab.json"` |  |
| `--embeddings_path` | fastText embeddings. | `"models/bilstm/embeddings.pt"` |  |
| `--epochs` | Training epochs. | `8` |  |
| `--batch_size` | Batch size. | `128` |  |
| `--lr` | Learning rate. | `2e-3` |  |
| `--weight_decay` | L2 regularization. | `0.01` |  |
| `--grad_accum_steps` | Gradient accumulation steps. | `1` |  |
| `--max_len` | Max sequence length. | `150` |  |
| `--sampler_mode` | Sampling strategy. | `"none"` |  |
| `--hidden_size` | LSTM hidden size. | `256` |  |
| `--num_layers` | LSTM layers. | `1` |  |
| `--dropout_embed` | Embedding dropout. | `0.1` |  |
| `--dropout_out` | Post‑LSTM / classifier dropout. | `0.3` |  |
| `--pooling` | Feature pooling (`attn`/`max`/`avg`). | `"attn"` |  |
| `--freeze_embeddings` | Freeze fastText embeddings. |  | `store_true` |
| `--loss` | Loss function (`ce` or `focal`). | `"ce"` |  |
| `--gamma` | Gamma for Focal Loss. | `2.0` |  |
| `--scheduler` | LR scheduler. | `"onecycle"` |  |
| `--max_grad_norm` | Gradient clipping. | `1.0` |  |
| `--amp` | Automatic Mixed Precision. |  | `store_true` |
| `--out_dir` | Output dir for models. | `"models/bilstm"` |  |
| `--chart_dir` | Training charts dir. | `"chart/bilstm"` |  |
| `--logs_dir` | Logs dir. | `"logs/bilstm"` |  |
| `--seed` | Random seed. | `42` |  |

---

### `bilstm/infer.py`
**Description:** Inference for **BiLSTM** (stand‑alone sentiment).

| Argument | Help | Default | Action |
|---|---|---|---|
| `--model_dir` | BiLSTM checkpoint. | `"models/bilstm/bilstm_best.pt"` |  |
| `--vocab_json` | BiLSTM vocabulary. | `"models/bilstm/vocab.json"` |  |
| `--embeddings_path` | fastText embeddings. | `"models/bilstm/embeddings.pt"` |  |
| `--max_len` | Max length (falls back to training value if `None`). | `None` |  |
| `--batch_size` | Inference batch size. | `256` |  |
| `--device` | Compute device. | `"cuda" if torch.cuda.is_available() else "cpu"` |  |
| `--text` | Input text(s). | `None` |  |

---

### `mt5/tokenize_model.py`
**Description:** Prepare tokenized dataset for mT5 training.

| Argument | Help | Default | Action |
|---|---|---|---|
| `--min_input_tokens` | Minimum input tokens to keep a sample. | `50` |  |
| `--max_input_tokens` | Maximum input tokens; longer inputs are truncated. | `512` |  |
| `--min_target_tokens` | Minimum target summary length. | `20` |  |
| `--max_target_tokens` | Maximum target summary length. | `150` |  |
| `--model_name` | Base mT5 model. | `"google/mt5-small"` |  |
| `--seed` | Random seed. | `42` |  |
| `--input_csv` | CSV with `article` and `summary` columns. | `"datasets/summarize.csv"` |  |
| `--output_dir` | Output dir for tokenized dataset. | `"datasets/mt5"` |  |
| `--text_col` | Input text column name. | `"article"` |  |
| `--summary_col` | Summary column name. | `"summary"` |  |

---

### `mt5/train.py`
**Description:** mT5 training (tokenize & fine‑tune setup).

| Argument | Help | Default | Action |
|---|---|---|---|
| `--model_name` | Base mT5 model. | `"google/mt5-small"` |  |
| `--seed` | Random seed. | `42` |  |
| `--tokenized_dir` | Output dir for tokenized dataset. | `"datasets/mt5"` |  |
| `--output_dir` | Output dir for model weights. | `"models/mt5"` |  |
| `--log_dir` | Logs dir. | `"logs/mt5"` |  |
| `--num_workers` | DataLoader workers. | `0` |  |
| `--epochs` | Training epochs. | `8` |  |

---

### `mt5/infer.py`
**Description:** Inference for **mT5 summarization**.

| Argument | Help | Default | Action |
|---|---|---|---|
| `--device` | Compute device. | `"cuda" if torch.cuda.is_available() else "cpu"` |  |
| `--model_dir` | Trained/merged mT5 weights. | `"models/mt5/lora-merged"` |  |
| `--text` | Input text to summarize. | `None` |  |
| `--max_new_tokens` | Max generated tokens. | `150` |  |
| `--num_beams` | Beam size. | `4` |  |

---

## Reproducibility

- All training/evaluation scripts expose `--seed` and deterministic length limits.
- Default **max length** is **150 tokens** for classification (and for the summarize‑then‑classify path).

## License

This repository is released under the MIT license. See `LICENSE` for details.
