import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup


class TextClassificationDataset(Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor], labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _autocast_ctx(device: str, enabled: bool):
    if enabled and device == "cuda" and torch.cuda.is_available():
        # Prefer bf16 if supported, otherwise fallback to fp16
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        amp_autocast = getattr(torch.amp, "autocast", None)
        if amp_autocast is not None:
            return amp_autocast(device_type="cuda", dtype=dtype)
        # Fallback for older torch versions (may emit deprecation warning)
        return torch.cuda.amp.autocast(dtype=dtype)
    return nullcontext()


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    fp16: bool,
    num_labels: int,
):
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with _autocast_ctx(device, fp16):
                outputs = model(**batch)
                logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    acc = accuracy_score(all_labels, all_preds)
    if num_labels == 2:
        p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    else:
        p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def build_label_map(series: pd.Series) -> dict[str, int]:
    # 保持出现顺序，兼容中文标注（如“是/否”）
    unique = list(pd.unique(series.astype(str)))
    return {lab: i for i, lab in enumerate(unique)}


def compute_metrics_builder(num_labels: int):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        if num_labels == 2:
            p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        else:
            p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a HF sequence classification model on a CSV file.")
    parser.add_argument("--base-model", type=str, default="richinfoai/ritrieve_zh_v1", help="HF model name or path")
    parser.add_argument("--csv", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--output", type=str, required=True, help="Path to save fine-tuned model")
    parser.add_argument("--text-col", type=str, default="text", help="Text column name")
    parser.add_argument("--label-col", type=str, default="label", help="Label column name")
    parser.add_argument("--test-size", type=float, default=0.1, help="Eval split size")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 if supported")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) 读取数据
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=[args.text_col, args.label_col])
    df[args.text_col] = df[args.text_col].astype(str)
    df[args.label_col] = df[args.label_col].astype(str)

    # 2) 标签映射（全量构建，后续划分不会丢标签）
    label2id = build_label_map(df[args.label_col])
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)
    df["label_id"] = df[args.label_col].map(label2id).astype(int)
    print(f"Labels: {label2id}")

    # 3) 划分数据
    train_df, eval_df = train_test_split(df, test_size=args.test_size, random_state=args.seed, stratify=df["label_id"])

    # 4) 分词器与编码
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    def tokenize_texts(texts: list[str]):
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )

    train_enc = tokenize_texts(train_df[args.text_col].tolist())
    eval_enc = tokenize_texts(eval_df[args.text_col].tolist())
    train_ds = TextClassificationDataset(train_enc, train_df["label_id"].tolist())
    eval_ds = TextClassificationDataset(eval_enc, eval_df["label_id"].tolist())

    # 5) 模型
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    # 6) 类别权重（缓解不平衡）
    counts = train_df["label_id"].value_counts().sort_index()
    class_weights = torch.tensor(
        [len(train_df) / max(1, int(counts.get(i, 0))) for i in range(num_labels)], dtype=torch.float
    )
    print(f"Class weights: {class_weights.tolist()}")

    # 7) DataLoader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size)

    # 8) 优化器与调度器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    total_steps = max(1, (len(train_loader) // max(1, args.gradient_accumulation_steps)) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # 9) 训练循环
    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and torch.cuda.is_available()))
    ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_metric = -1.0
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with _autocast_ctx(device, args.fp16):
                outputs = model(**batch)
                logits = outputs.logits
                loss = ce_loss(logits, labels)
                loss = loss / max(1, args.gradient_accumulation_steps)
            scaler.scale(loss).backward()

            if step % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            running_loss += loss.item()
            if global_step % args.logging_steps == 0:
                avg_loss = running_loss / max(1, args.logging_steps)
                print(f"epoch {epoch + 1} step {global_step}: loss={avg_loss:.4f}")
                running_loss = 0.0

        # 每个 epoch 结束做一次评估
        metrics = evaluate(model, eval_loader, device, args.fp16, num_labels)
        print(f"Epoch {epoch + 1} eval: {metrics}")
        main_metric = metrics["f1"] if num_labels == 2 else metrics["accuracy"]
        if main_metric > best_metric:
            best_metric = main_metric
            model.save_pretrained(str(out_dir))
            tokenizer.save_pretrained(str(out_dir))
            (out_dir / "label_map.json").write_text(pd.Series(label2id).to_json(force_ascii=False, indent=2))
            print(f"✅ Saved best model to {out_dir} (metric={best_metric:.4f})")

    print("Training finished.")


if __name__ == "__main__":
    main()
