# KoELECTRA 파인튜닝 (NER / 의도분류 공용) — 로컬 CPU 스모크 테스트, 실학습은 Colab GPU
# NER:    python scripts/train_koelectra.py --task ner --out models/ner-drug
# 분류:   python scripts/train_koelectra.py --task intent --out models/intent-drug
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

ROOT = Path(__file__).resolve().parent.parent
NER_LABELS = ["O", "B-DRUG", "I-DRUG"]
INTENT_LABELS = ["기타", "병용", "복용법", "부작용", "성분"]
MAX_LEN = 64


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class NerDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.enc = tokenizer([r["text"] for r in rows], truncation=True, max_length=MAX_LEN,
                             padding="max_length", return_offsets_mapping=True)
        self.labels = []
        for i, r in enumerate(rows):
            ents = [(s, e) for s, e, _ in r["entities"]]
            labs = []
            for (ts, te) in self.enc["offset_mapping"][i]:
                if ts == te:
                    labs.append(-100)
                    continue
                lab = 0
                for (es, ee) in ents:
                    if ts >= es and te <= ee:
                        lab = 1 if ts == es else 2
                        break
                labs.append(lab)
            self.labels.append(labs)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": torch.tensor(self.enc["input_ids"][i]),
                "attention_mask": torch.tensor(self.enc["attention_mask"][i]),
                "labels": torch.tensor(self.labels[i])}


class IntentDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.enc = tokenizer([r["text"] for r in rows], truncation=True, max_length=MAX_LEN,
                             padding="max_length")
        self.labels = [INTENT_LABELS.index(r["intent"]) for r in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": torch.tensor(self.enc["input_ids"][i]),
                "attention_mask": torch.tensor(self.enc["attention_mask"][i]),
                "labels": torch.tensor(self.labels[i])}


def ner_spans(label_ids):
    """라벨 시퀀스 → (start_tok, end_tok) 스팬 집합."""
    spans, start = set(), None
    for i, lab in enumerate(list(label_ids) + [0]):
        if lab == 1:
            if start is not None:
                spans.add((start, i))
            start = i
        elif lab != 2 and start is not None:
            spans.add((start, i))
            start = None
    return spans


@torch.no_grad()
def evaluate(model, loader, task, device):
    model.eval()
    if task == "ner":
        tp = fp = fn = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(input_ids=batch["input_ids"],
                          attention_mask=batch["attention_mask"]).logits.argmax(-1)
            for p, g in zip(preds.tolist(), batch["labels"].tolist()):
                valid = [i for i, x in enumerate(g) if x != -100]
                ps = ner_spans([p[i] for i in valid])
                gs = ner_spans([g[i] for i in valid])
                tp += len(ps & gs)
                fp += len(ps - gs)
                fn += len(gs - ps)
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        return {"precision": prec, "recall": rec, "f1": f1}
    correct = total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds = model(input_ids=batch["input_ids"],
                      attention_mask=batch["attention_mask"]).logits.argmax(-1)
        correct += (preds == batch["labels"]).sum().item()
        total += len(preds)
    return {"accuracy": correct / total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["ner", "intent"], required=True)
    ap.add_argument("--base", default=str(ROOT / "models" / "koelectra-small-v3-discriminator"))
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "synth"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-samples", type=int, default=0, help="스모크 테스트용 샘플 제한")
    args = ap.parse_args()

    # 로컬 경로가 없으면(예: Colab) HF 저장소 ID로 폴백
    if not Path(args.base).exists():
        candidates = [Path("models/koelectra-small-v3-discriminator")]
        found = next((c for c in candidates if c.exists()), None)
        args.base = str(found) if found else "monologg/koelectra-small-v3-discriminator"
        print(f"베이스 모델 경로 폴백: {args.base}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, task={args.task}")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    prefix = "ner" if args.task == "ner" else "intent"
    data = {s: read_jsonl(Path(args.data_dir) / f"{prefix}_{s}.jsonl") for s in ("train", "val", "test")}
    if args.max_samples:
        data = {s: rows[: args.max_samples] for s, rows in data.items()}

    DS = NerDataset if args.task == "ner" else IntentDataset
    loaders = {s: DataLoader(DS(rows, tokenizer), batch_size=args.batch, shuffle=(s == "train"))
               for s, rows in data.items()}

    if args.task == "ner":
        model = AutoModelForTokenClassification.from_pretrained(
            args.base, num_labels=len(NER_LABELS),
            id2label=dict(enumerate(NER_LABELS)),
            label2id={l: i for i, l in enumerate(NER_LABELS)})
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base, num_labels=len(INTENT_LABELS),
            id2label=dict(enumerate(INTENT_LABELS)),
            label2id={l: i for i, l in enumerate(INTENT_LABELS)})
    model.to(device)

    steps = len(loaders["train"]) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, int(steps * 0.1), steps)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(loaders["train"], 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            running += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(loaders['train'])} loss={running/step:.4f}", flush=True)
        print(f"epoch {epoch} | val: {evaluate(model, loaders['val'], args.task, device)}", flush=True)

    print(f"test: {evaluate(model, loaders['test'], args.task, device)}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"저장: {out}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
