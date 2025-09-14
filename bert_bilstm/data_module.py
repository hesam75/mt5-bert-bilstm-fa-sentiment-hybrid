# 3
# python data_module.py --sampler_mode none
# python data_module.py --sampler_mode weighted
# python data_module.py --sampler_mode balanced
# data_module.py
# -*- coding: utf-8 -*-
import os
import json
import math
import time
import random
import argparse
import hashlib
from typing import List, Tuple, Dict, Optional, Any

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler
from transformers import BertTokenizer

PAD_ID = 0
UNK_ID = 1
CACHE_SCHEMA_VERSION = 2  

def simple_tokenize(s: str) -> List[str]:
    return str(s).strip().split()

def load_vocab(vocab_path: str) -> Dict[str, int]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        stoi = json.load(f)
    assert stoi.get("<pad>", None) == PAD_ID and stoi.get("<unk>", None) == UNK_ID, \
        "vocab.json must have <pad>=0 and <unk>=1."
    return stoi

class PreTokenizedTensorDataset(Dataset):
    def __init__(self, tensors: Dict[str, torch.Tensor]):
        req = ["input_ids","attention_mask","lengths","labels","text_input"]
        for k in req:
            if k not in tensors:
                raise ValueError(f"Missing tensor in cache: {k}")
        n = tensors["labels"].shape[0]
        for k,v in tensors.items():
            if v.shape[0] != n:
                raise ValueError(f"Tensor {k} first dim {v.shape[0]} != {n}")
        self.tensors = tensors
        self.n = n

    def __len__(self): return self.n

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.tensors["input_ids"][idx],
            "attention_mask": self.tensors["attention_mask"][idx],
            "lengths": self.tensors["lengths"][idx],
            "labels": self.tensors["labels"][idx],
            "text_input": self.tensors["text_input"][idx],
        }

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
        weights[c] = (N / (K * counts[c])) if counts[c] > 0 else 0.0
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

        produced = 0
        while produced < self._len:
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
            produced += 1

def _file_fingerprint(p: str) -> Dict[str, Any]:
    p = os.path.abspath(p)
    return {"path": p, "size": os.path.getsize(p), "mtime": os.path.getmtime(p)}

def _make_cache_key(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]

def _cache_paths(cache_dir: str, split: str, key: str) -> Tuple[str, str]:
    os.makedirs(cache_dir, exist_ok=True)
    data_path = os.path.join(cache_dir, f"{split}_{key}.pt")
    meta_path = os.path.join(cache_dir, f"{split}_{key}.json")
    return data_path, meta_path

def _atomic_save(path: str, obj: Any):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path) 

def _save_cache(cache_dir: str, split: str, key: str,
                tensors: Dict[str, torch.Tensor], meta: Dict[str, Any]):
    data_path, meta_path = _cache_paths(cache_dir, split, key)
    pack = {
        "input_ids": tensors["input_ids"].to(torch.int32),
        "attention_mask": tensors["attention_mask"].to(torch.uint8),
        "lengths": tensors["lengths"].to(torch.int32),
        "labels": tensors["labels"].to(torch.int64),
        "text_input": tensors["text_input"].to(torch.int32),
    }
    _atomic_save(data_path, pack)
    meta = dict(meta)
    meta["cache_schema"] = CACHE_SCHEMA_VERSION
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def _validate_and_fix_loaded(tensors: Dict[str, torch.Tensor], max_len: int) -> Dict[str, torch.Tensor]:
    tensors["input_ids"] = tensors["input_ids"].to(torch.long)
    tensors["attention_mask"] = tensors["attention_mask"].to(torch.bool)
    tensors["lengths"] = tensors["lengths"].to(torch.long)
    tensors["labels"] = tensors["labels"].to(torch.long)
    tensors["text_input"] = tensors["text_input"].to(torch.long)

    N = tensors["labels"].shape[0]
    for k in ["input_ids","attention_mask","text_input"]:
        if tensors[k].ndim != 2 or tensors[k].shape[0] != N:
            raise ValueError(f"Tensor {k} has wrong shape {tuple(tensors[k].shape)}")
        if tensors[k].shape[1] != max_len:
            raise ValueError(f"Tensor {k} second dim {tensors[k].shape[1]} != max_len={max_len}")
    if tensors["lengths"].shape != (N,):
        raise ValueError("lengths must be shape (N,)")

    tensors["lengths"] = tensors["attention_mask"].sum(dim=1).to(torch.long)
    return tensors

def _try_load_cache(cache_dir: str, split: str, key: str, max_len: int):
    data_path, meta_path = _cache_paths(cache_dir, split, key)
    if not (os.path.exists(data_path) and os.path.exists(meta_path)):
        return None, None
    try:
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        if meta.get("cache_schema") != CACHE_SCHEMA_VERSION:
            return None, None
        tensors = torch.load(data_path, map_location="cpu")
        tensors = _validate_and_fix_loaded(tensors, max_len)
        return tensors, meta
    except Exception:
        return None, None

def _build_and_cache_split(csv_path: str,
                           tokenizer: "BertTokenizer",
                           stoi: Dict[str,int],
                           max_len: int,
                           cache_dir: str,
                           split: str,
                           key: str) -> Dict[str, torch.Tensor]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if not {"text","score"}.issubset(df.columns):
        raise ValueError(f"Columns (text, score) not found in {csv_path}")
    texts = df["text"].astype(str).tolist()
    labels = torch.tensor(df["score"].astype(int).values, dtype=torch.long)

    enc = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_attention_mask=True,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"]               
    attention_mask = enc["attention_mask"].bool()
    lengths = attention_mask.sum(dim=1).to(torch.long)  

    def _ft_encode_line(s: str) -> List[int]:
        toks = simple_tokenize(s)
        ids = [stoi.get(t, UNK_ID) for t in toks][:max_len]
        if not ids: ids = [UNK_ID]
        if len(ids) < max_len:
            ids = ids + [PAD_ID]*(max_len-len(ids))
        return ids
    ft_ids = torch.tensor([_ft_encode_line(s) for s in texts], dtype=torch.long)

    tensors = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "labels": labels,
        "text_input": ft_ids,
    }
    meta = {"rows": len(df), "max_len": max_len, "split": split}
    _save_cache(cache_dir, split, key, tensors, meta)
    return tensors

def get_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    vocab_json: str,
    batch_size: int = 128,
    num_workers: int = 2,
    sampler_mode: str = "none",   # "none" | "weighted" | "balanced"
    max_len: int = 150,
    cache_dir: str = "models/bert_bilstm/cache",
    use_cache: bool = True,
    rebuild_cache: bool = False,
):
    stoi = load_vocab(vocab_json)
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    fp = {
        "schema": CACHE_SCHEMA_VERSION,
        "train": _file_fingerprint(train_csv),
        "val":   _file_fingerprint(val_csv),
        "test":  _file_fingerprint(test_csv),
        "vocab": _file_fingerprint(vocab_json),
        "tokenizer": getattr(tokenizer, "name_or_path", "bert-base-multilingual-cased"),
        "max_len": max_len,
    }
    key = _make_cache_key(fp)

    def load_or_build(split, csv_path):
        if use_cache and not rebuild_cache:
            tensors, meta = _try_load_cache(cache_dir, split, key, max_len)
            if tensors is not None:
                return tensors
        return _build_and_cache_split(csv_path, tokenizer, stoi, max_len, cache_dir, split, key)

    train_t = load_or_build("train", train_csv)
    val_t   = load_or_build("val",   val_csv)
    test_t  = load_or_build("test",  test_csv)

    train_labels = train_t["labels"].tolist()
    val_labels   = val_t["labels"].tolist()
    test_labels  = test_t["labels"].tolist()
    train_counts = compute_class_counts(train_labels)
    val_counts   = compute_class_counts(val_labels)
    test_counts  = compute_class_counts(test_labels)
    class_weights = balanced_class_weights_from_counts(train_counts)

    pin_mem = torch.cuda.is_available()
    pw = (num_workers > 0)

    train_ds = PreTokenizedTensorDataset(train_t)
    val_ds   = PreTokenizedTensorDataset(val_t)
    test_ds  = PreTokenizedTensorDataset(test_t)

    if sampler_mode == "weighted":
        sampler = build_weighted_sampler(train_labels, class_weights)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, shuffle=False,
            num_workers=num_workers, pin_memory=pin_mem, drop_last=False, persistent_workers=pw
        )
    elif sampler_mode == "balanced":
        batch_sampler = BalancedBatchSampler(train_labels, batch_size=batch_size, drop_last=False)
        train_loader = DataLoader(
            train_ds, batch_sampler=batch_sampler,
            num_workers=num_workers, pin_memory=pin_mem, persistent_workers=pw
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_mem, drop_last=False, persistent_workers=pw
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem, drop_last=False, persistent_workers=pw
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem, drop_last=False, persistent_workers=pw
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
        "cache_key": key,
        "cache_dir": os.path.abspath(cache_dir),
    }
    return train_loader, val_loader, test_loader, meta

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Step 3: DataLoaders with fixed padding + secure cache (Hybrid BERT + fastText)")
    parser.add_argument("--train_csv", type=str, default="datasets/bert_bilstm/train.csv")
    parser.add_argument("--val_csv",   type=str, default="datasets/bert_bilstm/val.csv")
    parser.add_argument("--test_csv",  type=str, default="datasets/bert_bilstm/test.csv")
    parser.add_argument("--vocab_json", type=str, default="models/bert_bilstm/vocab.json")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--sampler_mode", type=str, default="none", choices=["none", "weighted", "balanced"])
    parser.add_argument("--max_len", type=int, default=150)
    parser.add_argument("--cache_dir", type=str, default="artifacts/cache")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--rebuild_cache", action="store_true")
    args = parser.parse_args()
    args.sampler_mode = "balanced"
    print("start dataloaders")
    start = time.time()
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        args.train_csv, args.val_csv, args.test_csv, args.vocab_json,
        batch_size=args.batch_size, num_workers=args.num_workers,
        sampler_mode=args.sampler_mode, max_len=args.max_len,
        cache_dir=args.cache_dir, use_cache=not args.no_cache, rebuild_cache=args.rebuild_cache
    )
    end = time.time()
    print("dataloaders time:", f"{end - start:.2f} Sec")
    print("==> Dataloaders ready.")
    print("Counts:", meta["train_counts"], meta["val_counts"], meta["test_counts"])
    print(f"Class weights ({meta['class_weights_source']}): {meta['class_weights']}")
    print(f"sampler_mode={meta['sampler_mode']}, vocab_size={meta['vocab_size']}, max_len={meta['max_len']}, cuda={torch.cuda.is_available()}")
    print(f"cache_dir={meta['cache_dir']} | cache_key={meta['cache_key']}")

    batch = next(iter(train_loader))
    shapes = {k: tuple(v.shape) for k, v in batch.items()}
    print("Batch shapes:", shapes)
    print("Sample first row:",
          "bert_ids:", batch["input_ids"][0, :10].tolist(),
          "| bert_mask:", batch["attention_mask"][0, :10].int().tolist(),
          "| fasttext_ids:", batch["text_input"][0, :10].tolist(),
          "| label:", int(batch["labels"][0]))

    out_dir = os.path.dirname(os.path.abspath(args.vocab_json)) or "."
    os.makedirs(out_dir, exist_ok=True)
        
    class_weights_path = os.path.join(out_dir, "class_weights.json")
    with open(class_weights_path, "w", encoding="utf-8") as f:
        json.dump(meta["class_weights"], f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved class weights to: {class_weights_path}")

    class_counts_path = os.path.join(out_dir, "class_counts.json")
    with open(class_counts_path, "w", encoding="utf-8") as f:
        json.dump(meta["train_counts"], f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved class counts to: {class_counts_path}")
    
if __name__ == "__main__":
    main()
