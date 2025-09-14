# 3
# python data_module.py --sampler_mode none
# python data_module.py --sampler_mode weighted
# python data_module.py --sampler_mode balanced
# data_module.py
# -*- coding: utf-8 -*-
import os
import json
import math
import random
import argparse
from typing import List, Tuple, Dict, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler

PAD_ID = 0
UNK_ID = 1

# ---------------- Utils ----------------
def simple_tokenize(s: str) -> List[str]:
    return str(s).strip().split()

def load_vocab(vocab_path: str) -> Dict[str, int]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        stoi = json.load(f)
    assert stoi.get("<pad>", None) == PAD_ID and stoi.get("<unk>", None) == UNK_ID, \
        "vocab.json must have <pad>=0 and <unk>=1."
    return stoi

def encode_text(text: str, stoi: Dict[str, int]) -> List[int]:
    toks = simple_tokenize(text)
    return [stoi.get(t, UNK_ID) for t in toks]

class SentimentDataset(Dataset):
    def __init__(self, csv_path: str, stoi: Dict[str, int], max_len: int = 150):
        self.df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if not {"text", "score"}.issubset(self.df.columns):
            raise ValueError(f"ستون‌های لازم (text, score) در {csv_path} پیدا نشد.")
        self.stoi = stoi
        self.max_len = max_len

        self.seqs: List[List[int]] = []
        self.labels: List[int] = []

        for text, y in zip(self.df["text"].tolist(), self.df["score"].tolist()):
            ids = encode_text(text, stoi)
            if len(ids) == 0:
                ids = [UNK_ID]
            ids = ids[: self.max_len]
            self.seqs.append(ids)
            self.labels.append(int(y))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        return self.seqs[idx], self.labels[idx]

def collate_fn(batch: List[Tuple[List[int], int]], pad_id: int = PAD_ID, max_len: Optional[int] = None):
    seqs, labels = zip(*batch)
    max_in_batch = max(len(s) for s in seqs)
    tgt_len = min(max_in_batch, max_len) if max_len is not None else max_in_batch

    B = len(seqs)
    X = torch.full((B, tgt_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((B, tgt_len), dtype=torch.bool)
    for i, s in enumerate(seqs):
        L = min(len(s), tgt_len)
        if L > 0:
            X[i, :L] = torch.tensor(s[:L], dtype=torch.long)
            attn_mask[i, :L] = True

    y = torch.tensor(labels, dtype=torch.long)
    return {
        "input_ids": X,
        "attention_mask": attn_mask,
        "labels": y,
        "lengths": attn_mask.sum(dim=1)
    }

class Collator:
    def __init__(self, pad_id: int, max_len: int):
        self.pad_id = pad_id
        self.max_len = max_len
    def __call__(self, batch):
        return collate_fn(batch, pad_id=self.pad_id, max_len=self.max_len)

# ---------------- Class stats & weights ----------------
def compute_class_counts(labels: List[int], num_classes: int = 3) -> Dict[int, int]:
    counts = {c: 0 for c in range(num_classes)}
    for y in labels:
        counts[int(y)] += 1
    return counts

def balanced_class_weights_from_counts(counts: Dict[int, int]) -> Dict[int, float]:
    N = sum(counts.values())
    K = len(counts)
    weights = {}
    for c in counts:
        if counts[c] > 0:
            weights[c] = N / (K * counts[c])
        else:
            weights[c] = 0.0
    return weights

def build_weighted_sampler(labels: List[int], class_weights: Dict[int, float]) -> WeightedRandomSampler:
    sample_weights = [class_weights[int(y)] for y in labels]
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights=sample_weights_tensor, num_samples=len(sample_weights), replacement=True)

class BalancedBatchSampler(Sampler[List[int]]):

    def __init__(self, labels: List[int], batch_size: int, drop_last: bool = False,
                 seed: int = 42, shuffle: bool = True):
        super().__init__(None)
        self.labels = list(map(int, labels))
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.rng = random.Random(seed)

        self.idx_by_class: Dict[int, List[int]] = {}
        for i, y in enumerate(self.labels):
            self.idx_by_class.setdefault(y, []).append(i)
        self.classes = sorted(self.idx_by_class.keys())
        self.k = len(self.classes)

        base = batch_size // self.k
        rem = batch_size - base * self.k
        self.per_class = {c: base + (1 if j < rem else 0) for j, c in enumerate(self.classes)}
        self._len = len(self.labels) // self.batch_size if self.drop_last else math.ceil(len(self.labels) / self.batch_size)

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        pools: Dict[int, List[int]] = {}
        ptr: Dict[int, int] = {}
        for c in self.classes:
            arr = self.idx_by_class[c][:]
            if self.shuffle:
                self.rng.shuffle(arr)
            pools[c] = arr
            ptr[c] = 0

        produced_batches = 0
        while produced_batches < self._len:
            batch: List[int] = []
            for c in self.classes:
                take = self.per_class[c]
                for _ in range(take):
                    if ptr[c] >= len(pools[c]):
                        if self.shuffle:
                            self.rng.shuffle(pools[c])
                        ptr[c] = 0
                    batch.append(pools[c][ptr[c]])
                    ptr[c] += 1
            yield batch
            produced_batches += 1

def get_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    vocab_json: str,
    batch_size: int = 128,
    num_workers: int = 2,
    sampler_mode: str = "none",   # "none" | "weighted" | "balanced"
    max_len: int = 150,
):
    stoi = load_vocab(vocab_json)

    train_ds = SentimentDataset(train_csv, stoi, max_len=max_len)
    val_ds   = SentimentDataset(val_csv,   stoi, max_len=max_len)
    test_ds  = SentimentDataset(test_csv,  stoi, max_len=max_len)

    train_counts = compute_class_counts(train_ds.labels)
    val_counts   = compute_class_counts(val_ds.labels)
    test_counts  = compute_class_counts(test_ds.labels)
    class_weights = balanced_class_weights_from_counts(train_counts) 

    pin_mem = torch.cuda.is_available()
    collate = Collator(PAD_ID, max_len)

    if sampler_mode == "weighted":
        train_sampler = build_weighted_sampler(train_ds.labels, class_weights)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler, shuffle=False,
            num_workers=num_workers, pin_memory=pin_mem, collate_fn=collate, drop_last=False
        )
    elif sampler_mode == "balanced":
        batch_sampler = BalancedBatchSampler(train_ds.labels, batch_size=batch_size, drop_last=False)
        train_loader = DataLoader(
            train_ds, batch_sampler=batch_sampler,
            num_workers=num_workers, pin_memory=pin_mem, collate_fn=collate
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_mem, collate_fn=collate, drop_last=False
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem, collate_fn=collate, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem, collate_fn=collate, drop_last=False
    )

    meta = {
        "train_counts": train_counts,
        "val_counts": val_counts,
        "test_counts": test_counts,
        "class_weights": class_weights,   
        "class_weights_source": "computed from TRAIN (balanced N/(K*n_c))",
        "vocab_size": len(stoi),
        "pad_id": PAD_ID,
        "unk_id": UNK_ID,
        "max_len": max_len,
        "sampler_mode": sampler_mode,
    }
    return train_loader, val_loader, test_loader, meta

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Step 3: DataLoaders with fixed padding")
    parser.add_argument("--train_csv", type=str, default="datasets/bilstm/train.csv")
    parser.add_argument("--val_csv",   type=str, default="datasets/bilstm/val.csv")
    parser.add_argument("--test_csv",  type=str, default="datasets/bilstm/test.csv")
    parser.add_argument("--vocab_json", type=str, default="models/bilstm/vocab.json")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--sampler_mode", type=str, default="none", choices=["none", "weighted", "balanced"])
    parser.add_argument("--max_len", type=int, default=150)
    args = parser.parse_args()

    train_loader, val_loader, test_loader, meta = get_dataloaders(
        args.train_csv, args.val_csv, args.test_csv, args.vocab_json,
        batch_size=args.batch_size, num_workers=args.num_workers,
        sampler_mode=args.sampler_mode, max_len=args.max_len
    )

    print("==> Dataloaders ready.")
    print("Counts:", meta["train_counts"], meta["val_counts"], meta["test_counts"])
    print(f"Class weights ({meta['class_weights_source']}): {meta['class_weights']}")
    print(f"sampler_mode={meta['sampler_mode']}, vocab_size={meta['vocab_size']}, max_len={meta['max_len']}, cuda={torch.cuda.is_available()}")

    # یک batch تست
    batch = next(iter(train_loader))
    shapes = {k: tuple(v.shape) for k, v in batch.items()}
    print("Batch shapes:", shapes)
    print("Sample first row:", batch["input_ids"][0, :10].tolist(),
          "| mask:", batch["attention_mask"][0, :10].int().tolist(),
          "| label:", int(batch["labels"][0]))

    out_dir = os.path.dirname(os.path.abspath(args.vocab_json)) or "."
    os.makedirs(out_dir, exist_ok=True)
    class_weights_path = os.path.join(out_dir, "class_weights.json")
    with open(class_weights_path, "w", encoding="utf-8") as f:
        json.dump(meta["class_weights"], f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved class weights to: {class_weights_path}")

if __name__ == "__main__":
    # برای Windows از guard استفاده می‌کنیم تا مولتی‌پروسس ایمن باشد.
    main()
