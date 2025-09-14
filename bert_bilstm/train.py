# 5
# train.py
# -*- coding: utf-8 -*-
import os
import json
import math
import argparse
from typing import Dict, Tuple, List
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import pandas as pd
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_module import get_dataloaders, PAD_ID
from model import HybridBERTFastTextBiLSTM, load_embeddings, count_params


# ------------------------- Utils -------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True 

def to_device(batch: dict, device: torch.device) -> dict:
    return {
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device, non_blocking=True),
        "lengths": batch["lengths"].to(device, non_blocking=True),
        "text_input": batch["text_input"].to(device, non_blocking=True),
    }

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# --------- Gradient logging helpers ----------
def grad_l2_of_module(module: nn.Module) -> float:
    if module is None:
        return 0.0
    sq = 0.0
    for p in module.parameters(recurse=True):
        if p.grad is not None:
            g = p.grad.detach()
            val = torch.norm(g, p=2).item()
            sq += val * val
    return math.sqrt(sq) if sq > 0 else 0.0

def maybe_make_dir(p: str):
    os.makedirs(p, exist_ok=True)

# ------------------------- Losses -------------------------
class FocalLoss(nn.Module):
    def __init__(self, num_classes: int, gamma: float = 2.0, alpha: Dict[int, float] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is None:
            self.alpha = torch.ones(num_classes, dtype=torch.float32)
        else:
            self.alpha = torch.tensor([alpha.get(c, 1.0) for c in range(num_classes)], dtype=torch.float32)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        targets = targets.view(-1, 1)
        pt = probs.gather(1, targets).squeeze(1)
        log_pt = log_probs.gather(1, targets).squeeze(1)
        alpha_t = self.alpha.to(logits.device)[targets.squeeze(1)] 
        loss = -alpha_t * (1 - pt).pow(self.gamma) * log_pt
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

# ------------------------- Metrics & Plots -------------------------
def evaluate_model(model: nn.Module, loader, device: torch.device) -> Tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    ce = nn.CrossEntropyLoss(reduction="mean").to(device)

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            logits = model(
                batch["input_ids"],
                batch["lengths"],
                batch["attention_mask"],
                batch["text_input"]
            )
            loss = ce(logits, batch["labels"])
            total_loss += loss.item() * batch["labels"].size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    # per-class
    p_c, r_c, f1_c, supp_c = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1,2], zero_division=0)

    metrics = {
        "loss": total_loss / len(y_true),
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_micro": p_micro,
        "recall_micro": r_micro,
        "f1_micro": f1_micro,
        "precision_weighted": p_weighted,
        "recall_weighted": r_weighted,
        "f1_weighted": f1_weighted,
        "per_class_precision": p_c.tolist(),
        "per_class_recall": r_c.tolist(),
        "per_class_f1": f1_c.tolist(),
        "per_class_support": supp_c.tolist(),
        "classification_report": classification_report(y_true, y_pred, digits=4, zero_division=0)
    }
    return metrics, y_true, y_pred

def save_confusion_csv(cm: np.ndarray, out_path: str, labels=("0","1","2")):
    df = pd.DataFrame(cm, index=list(labels), columns=list(labels))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, encoding="utf-8-sig", index_label="True \\ Pred")

def save_full_metric_csv(history: dict, val_best: dict, test_metrics: dict, cm: np.ndarray, out_path: str):
    rows = []
    for i in range(len(history.get("train_loss", []))):
        ep = i + 1
        rows.append({"split":"train","epoch":ep,"metric":"loss","value":history["train_loss"][i]})
        rows.append({"split":"train","epoch":ep,"metric":"acc","value":history["train_acc"][i]})
        rows.append({"split":"val","epoch":ep,"metric":"loss","value":history["val_loss"][i]})
        rows.append({"split":"val","epoch":ep,"metric":"acc","value":history["val_acc"][i]})
        if "val_f1_macro" in history and len(history["val_f1_macro"])>=ep:
            rows.append({"split":"val","epoch":ep,"metric":"f1_macro","value":history["val_f1_macro"][i]})

    if val_best:
        rows.append({"split":"val_best","epoch":val_best.get("epoch",-1),"metric":"f1_macro","value":val_best.get("f1_macro",np.nan)})

    keep = ["accuracy","precision_macro","recall_macro","f1_macro",
            "precision_micro","recall_micro","f1_micro",
            "precision_weighted","recall_weighted","f1_weighted"]
    for k in keep:
        rows.append({"split":"test","epoch":"","metric":k,"value":test_metrics[k]})

    for cls in [0,1,2]:
        rows.append({"split":"test","epoch":"","metric":f"precision_class_{cls}","value":test_metrics["per_class_precision"][cls]})
        rows.append({"split":"test","epoch":"","metric":f"recall_class_{cls}","value":test_metrics["per_class_recall"][cls]})
        rows.append({"split":"test","epoch":"","metric":f"f1_class_{cls}","value":test_metrics["per_class_f1"][cls]})
        rows.append({"split":"test","epoch":"","metric":f"support_class_{cls}","value":test_metrics["per_class_support"][cls]})

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            rows.append({"split":"test","epoch":"","metric":f"confusion[{i},{j}]","value":int(cm[i,j])})

    df = pd.DataFrame(rows, columns=["split","epoch","metric","value"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, encoding="utf-8-sig", index=False)


def plot_curves(history: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss (train/val)"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "loss_curve.png")); plt.close()

    plt.figure()
    plt.plot(history["train_acc"], label="train")
    plt.plot(history["val_acc"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy (train/val)"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "acc_curve.png")); plt.close()

    if "val_f1_macro" in history and len(history["val_f1_macro"]) > 0:
        plt.figure()
        plt.plot(history["val_f1_macro"], label="val F1-macro")
        plt.xlabel("Epoch"); plt.ylabel("F1-macro"); plt.title("F1-macro (validation)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "f1_curve.png")); plt.close()


def plot_lr(lr_history: List[float], out_dir: str):
    if not lr_history:
        return
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(lr_history)
    plt.xlabel("Step"); plt.ylabel("Learning Rate"); plt.title("LR Schedule")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "lr_curve.png")); plt.close()


def plot_confusion(cm: np.ndarray, out_path: str, class_names=("0","1","2")):
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path); plt.close()


def plot_confusion_normalized(cm: np.ndarray, out_path: str, class_names=("0","1","2")):
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, where=row_sums!=0)
    plt.figure()
    plt.imshow(cm_norm, interpolation="nearest")
    plt.title("Confusion Matrix (Normalized)")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]*100:.1f}%", ha="center", va="center")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path); plt.close()


def plot_per_class_bars(prec, rec, f1, out_path: str, class_names=("0","1","2")):
    idx = np.arange(len(class_names))
    width = 0.25
    plt.figure()
    plt.bar(idx - width, prec, width, label="Precision")
    plt.bar(idx,         rec,  width, label="Recall")
    plt.bar(idx + width, f1,   width, label="F1")
    plt.xticks(idx, class_names)
    plt.ylabel("Score")
    plt.ylim(0, 1.0)
    plt.title("Per-class Metrics (test)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path); plt.close()

# ------------------------- Train loop -------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None,
                    grad_accum_steps=1, max_norm=None, scheduler=None, step_scheduler_per_batch=False,
                    epoch:int=None, epochs:int=None, show_progress:bool=True,
                    log_grad: bool=False, log_grad_every: int=10, grad_csv_dir: str=None):
    model.train()
    running_loss = 0.0
    running_correct = 0
    n_samples = 0
    lr_track: List[float] = []
    grad_logs: List[dict] = []

    last_grads = {"gBERT": None, "gFT": None, "gLSTM": None, "gCLS": None}
    last_grad_step = -1

    optimizer.zero_grad(set_to_none=True)
    desc = f"Epoch {epoch}/{epochs}" if (epoch is not None and epochs is not None) else "Training"
    iterator = tqdm(loader, total=len(loader), desc=desc, ncols=110, leave=False)

    for step, batch in enumerate(iterator, start=1):
        batch = to_device(batch, device)
        with autocast(enabled=(scaler is not None)):
            logits = model(
                batch["input_ids"],
                batch["lengths"],
                batch["attention_mask"],
                batch["text_input"]
            )
            loss = criterion(logits, batch["labels"]) / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        did_step = False
        if step % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)

            global_step = step // grad_accum_steps
            if log_grad and (global_step % max(1, log_grad_every) == 0):
                g_bert = grad_l2_of_module(getattr(model, "bert", None))
                g_fast = grad_l2_of_module(getattr(model, "embedding", None))
                g_lstm = grad_l2_of_module(getattr(model, "lstm", None))
                g_cls  = grad_l2_of_module(getattr(model, "classifier", None))
                last_grads.update({"gBERT": g_bert, "gFT": g_fast, "gLSTM": g_lstm, "gCLS": g_cls})
                last_grad_step = step
                lr_now_dbg = optimizer.param_groups[0]["lr"]
                avg_loss_dbg = (running_loss + loss.item() * batch["labels"].size(0) * grad_accum_steps) / max(1, n_samples + batch["labels"].size(0))
                avg_acc_dbg  = (running_correct / max(1, n_samples)) if n_samples > 0 else 0.0
                grad_logs.append({
                    "epoch": epoch, "step": step, "lr": lr_now_dbg,
                    "running_loss": avg_loss_dbg, "running_acc": avg_acc_dbg,
                    "g_bert": g_bert, "g_fasttext": g_fast, "g_lstm": g_lstm, "g_classifier": g_cls
                })

            if max_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            did_step = True

        if step_scheduler_per_batch and scheduler is not None and did_step:
            scheduler.step()

        running_loss += loss.item() * batch["labels"].size(0) * grad_accum_steps
        preds = torch.argmax(logits, dim=1)
        running_correct += (preds == batch["labels"]).sum().item()
        n_samples += batch["labels"].size(0)

        lr_now = optimizer.param_groups[0]["lr"]
        lr_track.append(lr_now)

        if show_progress:
            avg_loss = running_loss / max(1, n_samples)
            avg_acc  = running_correct / max(1, n_samples)
            postfix = {"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.3f}", "lr": f"{lr_now:.2e}"}
            if log_grad and last_grads["gBERT"] is not None:
                postfix.update({
                    "gBERT": f"{last_grads['gBERT']:.2f}",
                    "gFT":   f"{last_grads['gFT']:.2f}",
                    "gLSTM": f"{last_grads['gLSTM']:.2f}",
                    "gCLS":  f"{last_grads['gCLS']:.2f}",
                })
            iterator.set_postfix(postfix)

    if show_progress and hasattr(iterator, "close"):
        iterator.close()
    epoch_loss = running_loss / max(1, n_samples)
    epoch_acc = running_correct / max(1, n_samples)

    if log_grad and grad_logs and grad_csv_dir is not None:
        maybe_make_dir(grad_csv_dir)
        df = pd.DataFrame(grad_logs)
        out_csv = os.path.join(grad_csv_dir, f"epoch_{epoch:02d}.csv")
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return epoch_loss, epoch_acc, lr_track

# ------------------------- Main -------------------------
def main():
    parser = argparse.ArgumentParser(description="Traing BiLSTM + BERT")
    parser.add_argument("--train_csv", type=str, default="datasets/bert_bilstm/train.csv")
    parser.add_argument("--val_csv", type=str, default="datasets/bert_bilstm/val.csv")
    parser.add_argument("--test_csv", type=str, default="datasets/bert_bilstm/test.csv")
    parser.add_argument("--vocab_json", type=str, default="models/bert_bilstm/vocab.json")
    parser.add_argument("--embeddings_path", type=str, default="models/bert_bilstm/embeddings.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=150)
    parser.add_argument("--sampler_mode", type=str, default="none", choices=["none", "weighted", "balanced"])
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout_embed", type=float, default=0.1)
    parser.add_argument("--dropout_out", type=float, default=0.3)
    parser.add_argument("--pooling", type=str, default="attn", choices=["attn", "last", "mean", "max"])
    parser.add_argument("--freeze_embeddings", action="store_true", help="Freeze embeddings")
    parser.add_argument("--freeze_bert", action="store_true", help="Freeze BERT")
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"])
    parser.add_argument("--gamma", type=float, default=2.0, help="Only for focal")
    parser.add_argument("--scheduler", type=str, default="onecycle", choices=["none", "onecycle", "plateau"])
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Mixed precision training")
    parser.add_argument("--log_grad", action="store_true", help="Enable gradient logging for branches")
    parser.add_argument("--log_grad_every", type=int, default=1, help="Log gradient every N steps (at the moment of step)")
    parser.add_argument("--out_dir", type=str, default="models/bert-bilstm")
    parser.add_argument("--chart_dir", type=str, default="chart/bert-ilstm")
    parser.add_argument("--logs_dir", type=str, default="logs/bert-ilstm")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.chart_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    train_loader, val_loader, test_loader, meta = get_dataloaders(
        args.train_csv, args.val_csv, args.test_csv, args.vocab_json,
        batch_size=args.batch_size, num_workers=2, sampler_mode=args.sampler_mode, max_len=args.max_len
    )
    class_weights = meta["class_weights"]
    num_classes = 3

    E = load_embeddings(args.embeddings_path)
    model = HybridBERTFastTextBiLSTM(
        bert_model_name="bert-base-multilingual-cased",  
        embedding_weight=E,
        num_classes=num_classes,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout_embed=args.dropout_embed,
        dropout_out=args.dropout_out,
        pooling=args.pooling,
        freeze_embeddings=args.freeze_embeddings,
        freeze_bert=args.freeze_bert,
        pad_idx=PAD_ID,
    ).to(device)
    total_p, trainable_p = count_params(model)
    print(f"Model params: total={total_p:,}, trainable={trainable_p:,} | device={device}")

    use_plain_ce = (args.sampler_mode in {"balanced", "weighted"})  

    if args.loss == "ce":
        if use_plain_ce:
            criterion = nn.CrossEntropyLoss(reduction="mean")
        else:
            weight_vec = torch.tensor([class_weights[i] for i in range(num_classes)],
                                      dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=weight_vec, reduction="mean")
    else:
        alpha = None if use_plain_ce else class_weights
        criterion = FocalLoss(num_classes=num_classes, gamma=args.gamma, alpha=alpha)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "onecycle":
        steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum_steps))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch
        )
        step_scheduler_per_batch = True
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1, verbose=True)
        step_scheduler_per_batch = False
    else:
        scheduler = None
        step_scheduler_per_batch = False

    scaler = GradScaler(enabled=args.amp)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1_macro": [], "lr": []}

    best = {"f1_macro": -1.0, "epoch": -1}
    best_path = os.path.join(args.out_dir, "bert_bilstm_best.pt")
    grad_csv_dir = os.path.join(args.out_dir, "grad_logs")

    # ----------------- Training Loop -----------------
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, lr_track = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler=scaler if args.amp else None,
            grad_accum_steps=args.grad_accum_steps,
            max_norm=args.max_grad_norm,
            scheduler=scheduler,
            step_scheduler_per_batch=step_scheduler_per_batch,
            epoch=epoch, epochs=args.epochs, show_progress=True,
            log_grad=args.log_grad, log_grad_every=args.log_grad_every, grad_csv_dir=grad_csv_dir
        )
        history["lr"].extend(lr_track)

        val_metrics, _, _ = evaluate_model(model, val_loader, device)

        if scheduler is not None and args.scheduler == "plateau":
            scheduler.step(val_metrics["f1_macro"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])

        print(f"[Epoch {epoch:02d}] "
              f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} || "
              f"val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f} | val_f1_macro={val_metrics['f1_macro']:.4f}")

        if val_metrics["f1_macro"] > best["f1_macro"]:
            best.update({"f1_macro": val_metrics["f1_macro"], "epoch": epoch})
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": vars(args),
            }, best_path)
            print(f"[BEST] Saved new best model to {best_path}")

    save_json(history, os.path.join(args.logs_dir, "train_history.json"))
    plot_curves(history, args.chart_dir)
    plot_lr(history.get("lr", []), args.chart_dir)

    # ----------------- Test Evaluation -----------------
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[LOAD] Best model from epoch {ckpt['epoch']} (val_f1_macro={ckpt['val_metrics']['f1_macro']:.4f}) loaded.")

    test_metrics, y_true, y_pred = evaluate_model(model, test_loader, device)
    print("\n[TEST] Metrics:")
    for k in ["accuracy","precision_macro","recall_macro","f1_macro",
              "precision_micro","recall_micro","f1_micro",
              "precision_weighted","recall_weighted","f1_weighted"]:
        print(f"  {k}: {test_metrics[k]:.4f}")
    print("\n[TEST] Classification report:\n", test_metrics["classification_report"])

    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    cm_path = os.path.join(args.chart_dir, "confusion_matrix.png")
    plot_confusion(cm, cm_path, class_names=("0","1","2"))
    cmn_path = os.path.join(args.chart_dir, "confusion_matrix_normalized.png")
    plot_confusion_normalized(cm, cmn_path, class_names=("0","1","2"))

    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, where=row_sums!=0)
    cm_csv_path = os.path.join(args.logs_dir, "confusion_matrix.csv")
    save_confusion_csv(cm, cm_csv_path, labels=("0","1","2"))
    cmn_csv_path = os.path.join(args.logs_dir, "confusion_matrix_normalized.csv")
    save_confusion_csv(cm_norm, cmn_csv_path, labels=("0","1","2"))

    best_info = {"epoch": best["epoch"], "f1_macro": best["f1_macro"]}
    full_csv_path = os.path.join(args.logs_dir, "full_metric.csv")
    save_full_metric_csv(history, best_info, test_metrics, cm, full_csv_path)

    per_class_png = os.path.join(args.chart_dir, "per_class_metrics.png")
    prec_c = test_metrics["per_class_precision"]
    rec_c  = test_metrics["per_class_recall"]
    f1_c   = test_metrics["per_class_f1"]
    plot_per_class_bars(prec_c, rec_c, f1_c, per_class_png, class_names=("0","1","2"))
    print(f"[OK] Saved per-class metrics to: {per_class_png}")

    save_json(test_metrics, os.path.join(args.logs_dir, "test_metrics.json"))
    print(f"[OK] Curves saved to: {args.chart_dir}/loss_curve.png, acc_curve.png, f1_curve.png, lr_curve.png")
    print(f"[OK] Confusion matrices saved to: {cm_path}, {cmn_path}")
    print(f"[OK] Confusion matrices CSV saved to: {cm_csv_path}, {cmn_csv_path}")
    print(f"[OK] Full metrics saved to: {full_csv_path}")

    print("[DONE]")


if __name__ == "__main__":
    main()

