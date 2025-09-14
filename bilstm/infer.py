# 6
# infer_bilstm.py
# -*- coding: utf-8 -*-

import json
import argparse
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F

from model import BiLSTMClassifier, load_embeddings
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
    return text.strip().split()

def encode_texts(texts: List[str], vocab: Dict[str, int], max_len: int, pad_id: int, unk_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    B = len(texts)
    ids = np.full((B, max_len), fill_value=pad_id, dtype=np.int64)
    lens = np.zeros(B, dtype=np.int64)

    for i, t in enumerate(texts):
        toks = simple_tokenize(t)
        seq = [vocab.get(tok, unk_id) for tok in toks][:max_len]
        if len(seq) == 0:
            seq = [unk_id]
        L = min(len(seq), max_len)
        ids[i, :L] = np.array(seq[:L], dtype=np.int64)
        lens[i] = L

    input_ids = torch.from_numpy(ids)
    lengths = torch.from_numpy(lens)
    attn = (input_ids != pad_id)
    return input_ids, lengths, attn



def load_model(checkpoint_path: str, embeddings_path: str, device: torch.device, pooling_override: str = None) -> Tuple[BiLSTMClassifier, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    emb_path = embeddings_path

    E = load_embeddings(emb_path)  # (V, 300)
    hidden_size = int(cfg.get("hidden_size", 256))
    num_layers = int(cfg.get("num_layers", 1))
    dropout_embed = float(cfg.get("dropout_embed", 0.1))
    dropout_out = float(cfg.get("dropout_out", 0.3))
    pooling = pooling_override or cfg.get("pooling", "attn")
    max_len = int(cfg.get("max_len", 150))

    model = BiLSTMClassifier(
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
    model: BiLSTMClassifier,
    texts: List[str],
    vocab: Dict[str, int],
    max_len: int,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    pad_id, unk_id = get_special_ids(vocab)
    input_ids, lengths, attn = encode_texts(texts, vocab, max_len, pad_id, unk_id)
    input_ids = input_ids.to(device)
    lengths   = lengths.to(device)
    attn      = attn.to(device)

    L = int(lengths.max().item())
    if L <= 0:
        L = 1
    input_ids = input_ids[:, :L]   
    attn      = attn[:, :L]        
    logits = model(input_ids, lengths, attn)      
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()
    preds = probs.argmax(axis=1)
    return preds, probs

def sentiment(text,model_dir,device_name,vocab_json,embeddings_path,max_len):

    device = torch.device(device_name)
    vocab = load_vocab(vocab_json)

    model, meta = load_model(model_dir, embeddings_path, device)
    max_len = max_len or int(meta["max_len"])

    preds, probs = predict_texts(model, [text], vocab, max_len, device)
    pred_id = int(preds[0])
    return pred_id

# -------------------- CLI --------------------
def main():
    parser = argparse.ArgumentParser("Step 6: BiLSTM Inference")
    parser.add_argument("--model_dir", type=str, default="models/bilstm/bilstm_best.pt")
    parser.add_argument("--vocab_json", type=str, default="models/bilstm/vocab.json")
    parser.add_argument("--embeddings_path", type=str, default="models/bilstm/embeddings.pt")
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--text", type=str, default=None)

    args = parser.parse_args()

    device = torch.device(args.device)
    vocab = load_vocab(args.vocab_json)

    model, meta = load_model(args.model_dir, args.embeddings_path, device)
    max_len = args.max_len or int(meta["max_len"])

    preds, probs = predict_texts(model, [args.text], vocab, max_len, device)
    pred_id = int(preds[0])
    print(f"[TEXT] Pred: {LABEL_MAP.get(pred_id, str(pred_id))} ({pred_id}) | " f"probs = {probs[0].round(3).tolist()}")




if __name__ == "__main__":
    main()
