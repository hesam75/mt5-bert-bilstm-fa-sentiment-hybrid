# 6
# infer.py
# -*- coding: utf-8 -*-
import os
import json
import argparse
from typing import List, Tuple, Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from transformers import BertTokenizer

from model import HybridBERTFastTextBiLSTM, load_embeddings, count_params
from data_module import PAD_ID 

LABEL_MAP = {0: "neutral", 1: "positive", 2: "negative"}

def load_vocab(vocab_path: str) -> Dict[str, int]:
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_special_ids(vocab: Dict[str, int]) -> Tuple[int, int]:
    pad_id = vocab.get("<pad>", 0)
    unk_id = vocab.get("<unk>", 1)
    return pad_id, unk_id

def simple_tokenize(text: str) -> List[str]:
    return str(text).strip().split()

def ft_encode_texts(
    texts: List[str], vocab: Dict[str, int], max_len: int, pad_id: int, unk_id: int
) -> torch.Tensor:
    out = []
    for t in texts:
        toks = simple_tokenize(t)
        ids = [vocab.get(tok, unk_id) for tok in toks][:max_len]
        if not ids:
            ids = [unk_id]
        if len(ids) < max_len:
            ids = ids + [pad_id] * (max_len - len(ids))
        out.append(ids)
    return torch.tensor(out, dtype=torch.long)

def bert_encode_texts(
    texts: List[str], tokenizer: BertTokenizer, max_len: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Output: input_ids (B,T), attention_mask (B,T) bool, lengths (B,)
    """
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
    attention_mask = enc["attention_mask"].to(torch.bool)
    lengths = attention_mask.sum(dim=1).to(torch.long)  
    return input_ids, attention_mask, lengths

def load_model(
    checkpoint_path: str,
    embeddings_path: str,
    bert_model_name: str,
    device: torch.device,
    pooling_override: str = None
) -> Tuple[HybridBERTFastTextBiLSTM, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})

    E = load_embeddings(embeddings_path) 

    # Hypers
    hidden_size   = int(cfg.get("hidden_size", 256))
    num_layers    = int(cfg.get("num_layers", 1))
    dropout_embed = float(cfg.get("dropout_embed", 0.1))
    dropout_out   = float(cfg.get("dropout_out", 0.3))
    pooling       = pooling_override or cfg.get("pooling", "attn")
    max_len       = int(cfg.get("max_len", 150))

    model = HybridBERTFastTextBiLSTM(
        bert_model_name=bert_model_name or cfg.get("bert_model_name", "bert-base-multilingual-cased"),
        embedding_weight=E,
        num_classes=3,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_embed=dropout_embed,
        dropout_out=dropout_out,
        pooling=pooling,
        freeze_embeddings=True,  
        pad_idx=PAD_ID,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, {"max_len": max_len, "cfg": cfg}


@torch.no_grad()
def predict_texts(
    model: HybridBERTFastTextBiLSTM,
    texts: List[str],
    tokenizer: BertTokenizer,
    vocab: Dict[str, int],
    max_len: int,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    pad_id, unk_id = get_special_ids(vocab)

    bert_ids, bert_mask, lengths = bert_encode_texts(texts, tokenizer, max_len) 
    ft_ids = ft_encode_texts(texts, vocab, max_len, pad_id, unk_id)      

    bert_ids = bert_ids.to(device)
    bert_mask = bert_mask.to(device)
    ft_ids = ft_ids.to(device)
    lengths = lengths.to(device)

    L = int(lengths.max().item())
    if L <= 0: L = 1
    bert_ids = bert_ids[:, :L]
    bert_mask = bert_mask[:, :L]
    ft_ids = ft_ids[:, :L] 
    logits = model(bert_ids, lengths, bert_mask, ft_ids)
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    preds = probs.argmax(axis=1)
    return preds, probs
def sentiment(text,model_name,model_dir,device_name,vocab_json,embeddings_path,max_len):
    device = torch.device(device_name)
    vocab = load_vocab(vocab_json)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model, meta = load_model(model_dir, embeddings_path, model_name, device)
    max_len = max_len or int(meta["max_len"])

    preds, probs = predict_texts(model, [text], tokenizer, vocab, max_len, device)
    pred_id = int(preds[0])
    return pred_id

def main():
    parser = argparse.ArgumentParser("Step 6: Inference BERT-BiLSTM")
    parser.add_argument("--checkpoint", type=str, default="models/bert_bilstm/bert_bilstm_best.pt")
    parser.add_argument("--vocab_json", type=str, default="models/bert_bilstm/vocab.json")
    parser.add_argument("--embeddings_path", type=str, default="models/bert_bilstm/embeddings.pt")
    parser.add_argument("--bert_model_name", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text", type=str, default=None)

    args = parser.parse_args()

    device = torch.device(args.device)
    vocab = load_vocab(args.vocab_json)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)

    model, meta = load_model(args.checkpoint, args.embeddings_path, args.bert_model_name, device)
    max_len = args.max_len or int(meta["max_len"])

    preds, probs = predict_texts(model, [args.text], tokenizer, vocab, max_len, device)
    pred_id = int(preds[0])
    print(f"[TEXT] Pred: {LABEL_MAP.get(pred_id, str(pred_id))} ({pred_id}) | "
            f"probs = {probs[0].round(3).tolist()}")



if __name__ == "__main__":
    main()
