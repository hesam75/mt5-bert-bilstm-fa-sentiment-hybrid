# 4
# model.py
import os
import json
import argparse
from typing import Literal, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertTokenizer

from data_module import get_dataloaders, PAD_ID


class HybridBERTFastTextBiLSTM(nn.Module):
    def __init__(
        self,
        bert_model_name="bert-base-multilingual-cased",  
        embedding_weight: torch.Tensor = None, 
        num_classes: int = 3,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout_embed: float = 0.1,
        dropout_out: float = 0.3,
        pooling: Literal["attn", "last", "mean", "max"] = "attn",
        freeze_embeddings: bool = True,
        freeze_bert:bool = True,
        pad_idx: int = PAD_ID,
    ):
        super(HybridBERTFastTextBiLSTM, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.freeze_bert = freeze_bert
        if self.freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
            self.bert.eval() 

        # fastText Embedding
        self.embedding = nn.Embedding.from_pretrained(embedding_weight,freeze=freeze_embeddings, padding_idx=pad_idx,)
        self.dropout_embed = nn.Dropout(dropout_embed)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size + embedding_weight.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if num_layers == 1 else 0.2,
        )

        self.pooling_mode = pooling
        if pooling == "attn":
            self.pool = AttentionPooling(hidden_dim=hidden_size * 2)
        else:
            self.pool = None

        self.dropout_out = nn.Dropout(dropout_out)

        self.classifier = nn.Linear(hidden_size * 2, num_classes)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor, attention_mask: torch.Tensor, text_input: torch.Tensor):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_hidden_states = bert_outputs.last_hidden_state 

        fasttext_embeds = self.embedding(text_input)  
        combined_features = torch.cat((bert_hidden_states, fasttext_embeds), dim=-1) 
        packed = pack_padded_sequence(combined_features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        T = input_ids.size(1)
        h, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        if attention_mask.size(1) != h.size(1):
          attention_mask = attention_mask[:, :h.size(1)]

        # Pooling
        if self.pooling_mode == "attn":
            pooled = self.pool(h, attention_mask)
        elif self.pooling_mode == "last":
            pooled = self._select_last_valid(h, lengths)
        elif self.pooling_mode == "mean":
            pooled = masked_mean(h, attention_mask)
        elif self.pooling_mode == "max":
            pooled = masked_max(h, attention_mask)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling_mode}")
        out = self.dropout_out(pooled)
        logits = self.classifier(out) 
        return logits

    def _select_last_valid(self, h: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, H2 = h.size()
        idx = (lengths - 1).clamp(min=0).view(B, 1, 1).expand(B, 1, H2) 
        gathered = h.gather(dim=1, index=idx).squeeze(1) 
        return gathered


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        s = torch.tanh(self.proj(h))    
        scores = self.v(s)                
        alpha = masked_softmax(scores, mask, dim=1)
        alpha = alpha.unsqueeze(-1)         
        ctx = (alpha * h).sum(dim=1)      
        return ctx


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float().unsqueeze(-1)  
    s = (x * mask_f).sum(dim=1)        
    denom = mask_f.sum(dim=1).clamp(min=1e-6) 
    return s / denom


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    very_neg = torch.finfo(x.dtype).min
    mask_f = mask.unsqueeze(-1) 
    x_masked = x.masked_fill(~mask_f, very_neg)
    return x_masked.max(dim=1).values


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if logits.dim() == 3:
        logits = logits.squeeze(-1)
    logits = logits.masked_fill(~mask, float('-inf'))
    probs = torch.softmax(logits, dim=dim)
    probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
    return probs

def load_embeddings(emb_path: str) -> torch.Tensor:
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embedding file not found: {emb_path}")
    E = torch.load(emb_path, map_location="cpu")
    if not isinstance(E, torch.Tensor):
        E = torch.as_tensor(E)
    if E.dtype != torch.float32:
        E = E.float()
    return E
def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
# ---------------------- Main ----------------------
def main():
    parser = argparse.ArgumentParser(description="Step 4: BiLSTM + BERT Model")
    parser.add_argument("--embeddings_path", type=str, default="models/bert_bilstm/embeddings.pt")
    parser.add_argument("--train_csv", type=str, default="datasets/bert_bilstm/train.csv")
    parser.add_argument("--val_csv", type=str, default="datasets/bert_bilstm/val.csv")
    parser.add_argument("--test_csv", type=str, default="datasets/bert_bilstm/test.csv")
    parser.add_argument("--vocab_json", type=str, default="models/bert_bilstm/vocab.json")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--sampler_mode", type=str, default="none", choices=["none", "weighted", "balanced"])
    parser.add_argument("--max_len", type=int, default=150)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout_embed", type=float, default=0.1)
    parser.add_argument("--dropout_out", type=float, default=0.3)
    parser.add_argument("--pooling", type=str, default="attn", choices=["attn", "last", "mean", "max"])
    parser.add_argument("--freeze_embeddings", action="store_true")
    parser.add_argument("--freeze_bert", action="store_true") 
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.sampler_mode="balanced"
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        args.train_csv, args.val_csv, args.test_csv, args.vocab_json,
        batch_size=args.batch_size, num_workers=args.num_workers,
        sampler_mode=args.sampler_mode, max_len=args.max_len
    )

    E = load_embeddings(args.embeddings_path)

    model = HybridBERTFastTextBiLSTM(
        bert_model_name="bert-base-multilingual-cased",
        num_classes=3,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_embed=args.dropout_embed,
        dropout_out=args.dropout_out,
        pooling=args.pooling,
        freeze_embeddings=args.freeze_embeddings,
        freeze_bert=args.freeze_bert,
        pad_idx=PAD_ID,
    ).to(device)

    total, trainable = count_params(model)
    print(f"Model ready on {device}. params: total={total:,}, trainable={trainable:,}")
    print(f"Embedding frozen: {args.freeze_embeddings} | Pooling: {args.pooling} | hidden={args.hidden_size} | layers={args.num_layers}")

    batch = next(iter(train_loader))
    X = batch["input_ids"].to(device)
    M = batch["attention_mask"].to(device)
    L = batch["lengths"].to(device)
    with torch.no_grad():
        logits = model(X, L, M)
    print("Logits shape:", tuple(logits.shape))

    os.makedirs("models/bert_bilstm", exist_ok=True)
    config_path = os.path.join("models/bert_bilstm", "model_config.json")
    cfg = {
        "embeddings_path": os.path.abspath(args.embeddings_path),
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout_embed": args.dropout_embed,
        "dropout_out": args.dropout_out,
        "pooling": args.pooling,
        "freeze_embeddings": args.freeze_embeddings,
        "sampler_mode": args.sampler_mode,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved model config to: {config_path}")


if __name__ == "__main__":
    main()
