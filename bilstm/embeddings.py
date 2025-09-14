# 2
# python embeddings.py --train_csv datasets/train.csv --fasttext_path embedding/cc.fa.300.bin --out_dir artifacts --min_freq 2 --seed 42
# -*- coding: utf-8 -*-
import os
import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
import torch
from gensim.models.fasttext import load_facebook_vectors
from gensim.models import KeyedVectors


SPECIAL_TOKENS = ["<pad>", "<unk>"]  


def simple_tokenize(s: str):
    return str(s).strip().split()


def build_vocab(train_csv: str, min_freq: int = 2):
    df = pd.read_csv(train_csv, encoding="utf-8-sig")
    if not {"text", "score"}.issubset(df.columns):
        raise ValueError("Required columns (text, score) not found in train.csv.")

    counter = Counter()
    for text in df["text"]:
        counter.update(simple_tokenize(text))

    itos = list(SPECIAL_TOKENS)
    for tok, freq in counter.most_common():
        if freq >= min_freq:
            itos.append(tok)

    stoi = {tok: idx for idx, tok in enumerate(itos)}
    vocab_stats = {
        "total_unique_tokens_in_train": len(counter),
        "kept_tokens_with_min_freq": len(itos) - len(SPECIAL_TOKENS),
        "min_freq": min_freq,
    }
    return stoi, itos, counter, vocab_stats


def load_fasttext_vectors(ft_path: str):

    if not os.path.exists(ft_path):
        raise FileNotFoundError(f"fastText file not found: {ft_path}")

    if ft_path.endswith(".bin"):
        kv = load_facebook_vectors(ft_path)
        has_subword = True
    else:
        kv = KeyedVectors.load_word2vec_format(ft_path, binary=False)
        has_subword = False

    emb_dim = kv.vector_size
    if emb_dim != 300:
        print(f"[WARN] fastText model dim = {emb_dim}, not 300. Continuing with this value.")
    return kv, has_subword, emb_dim


def build_embedding_matrix(stoi, kv, has_subword: bool, emb_dim: int = 300, seed: int = 42):
    rng = np.random.default_rng(seed)
    vocab_size = len(stoi)
    matrix = np.zeros((vocab_size, emb_dim), dtype=np.float32)
    try:
        mean_vec = kv.get_mean_vector(kv.index_to_key)
        if mean_vec.shape[0] != emb_dim:
            mean_vec = None
    except Exception:
        mean_vec = None

    used_types = 0
    oov_types = 0
    oov_list = []

    for token, idx in stoi.items():
        if token == "<pad>":
            matrix[idx] = np.zeros(emb_dim, dtype=np.float32)
            continue
        if token == "<unk>":
            if mean_vec is not None:
                matrix[idx] = mean_vec.astype(np.float32)
            else:
                matrix[idx] = rng.normal(0, 0.5, size=(emb_dim,)).astype(np.float32)
            continue

        vec = None
        if has_subword:
            try:
                vec = kv.get_vector(token)
            except KeyError:
                vec = None
        else:
            if token in kv.key_to_index:
                vec = kv.get_vector(token)

        if vec is not None:
            matrix[idx] = vec.astype(np.float32)
            used_types += 1
        else:
            matrix[idx] = rng.normal(0, 0.5, size=(emb_dim,)).astype(np.float32)
            oov_types += 1
            if len(oov_list) < 30:
                oov_list.append(token)

    stats = {
        "vocab_size_including_specials": vocab_size,
        "types_with_pretrained_vectors": used_types,
        "types_random_init": oov_types,
        "oov_sample": oov_list,
    }
    return matrix, stats


def compute_token_coverage(counter: Counter, kv, has_subword: bool):
    total_tokens = sum(counter.values())
    covered_tokens = 0
    if has_subword:
        covered_tokens = total_tokens
    else:
        for tok, freq in counter.items():
            if tok in kv.key_to_index:
                covered_tokens += freq
    coverage = covered_tokens / total_tokens if total_tokens > 0 else 0.0
    return coverage


def save_artifacts(out_dir: str, stoi, itos, embedding_matrix: np.ndarray, config: dict):
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(stoi, f, ensure_ascii=False)

    with open(os.path.join(out_dir, "itos.txt"), "w", encoding="utf-8") as f:
        for tok in itos:
            f.write(tok + "\n")

    torch.save(torch.from_numpy(embedding_matrix), os.path.join(out_dir, "embeddings.pt"))

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Step 2: Build vocab & fastText embeddings")
    parser.add_argument("--train_csv", type=str, default="datasets/bilstm/train.csv")
    parser.add_argument("--fasttext_path", type=str, default="embedding/cc.fa.300.bin")
    parser.add_argument("--out_dir", type=str, default="models/bilstm")
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("==> Building vocabulary from TRAIN ...")
    stoi, itos, counter, vocab_stats = build_vocab(args.train_csv, min_freq=args.min_freq)
    print(f"[OK] vocab size (with specials): {len(stoi):,} | kept≥{args.min_freq}: {vocab_stats['kept_tokens_with_min_freq']:,} / unique: {vocab_stats['total_unique_tokens_in_train']:,}")

    print("\n==> Loading fastText vectors ...")
    kv, has_subword, emb_dim = load_fasttext_vectors(args.fasttext_path)
    print(f"[OK] fastText loaded | dim={emb_dim} | subword_support={has_subword}")

    print("\n==> Building embedding matrix ...")
    embedding_matrix, emb_stats = build_embedding_matrix(stoi, kv, has_subword, emb_dim=emb_dim, seed=args.seed)
    type_cov = emb_stats["types_with_pretrained_vectors"] / max(1, (len(stoi) - len(SPECIAL_TOKENS)))
    token_cov = compute_token_coverage(counter, kv, has_subword)

    print(f"[STATS] vocab_size={emb_stats['vocab_size_including_specials']:,}")
    print(f"[STATS] type_coverage (pretrained) = {type_cov*100:.2f}%")
    print(f"[STATS] token_coverage (corpus-weighted) = {token_cov*100:.2f}%")
    if emb_stats["oov_sample"]:
        print(f"[STATS] OOV sample (random-initialized): {emb_stats['oov_sample']}")

    print("\n==> Saving artifacts ...")
    config = {
        "min_freq": args.min_freq,
        "embedding_dim": emb_dim,
        "vocab_size": len(stoi),
        "special_tokens": SPECIAL_TOKENS,
        "fasttext_path": os.path.abspath(args.fasttext_path),
        "train_csv": os.path.abspath(args.train_csv),
        "has_subword": has_subword,
        "seed": args.seed,
    }
    save_artifacts(args.out_dir, stoi, itos, embedding_matrix, config)
    print(f"[OK] Saved to: {os.path.abspath(args.out_dir)}")
    print("Files: vocab.json, itos.txt, embeddings.pt, config.json")

if __name__ == "__main__":
    main()
