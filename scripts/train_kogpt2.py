# KoGPT2 파인튜닝 (검사결과 → 설명 문장) — 실학습은 Colab GPU (예상 15~20분)
# 실행: python scripts/train_kogpt2.py --out models/kogpt2-drug
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForCausalLM, PreTrainedTokenizerFast,
                          get_linear_schedule_with_warmup)

ROOT = Path(__file__).resolve().parent.parent
MAX_LEN = 256


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class PairDataset(Dataset):
    """프롬프트(입력) 부분은 loss에서 제외(-100)하고 출력 문장만 학습한다."""

    def __init__(self, rows, tok):
        self.items = []
        eos = tok.eos_token
        for r in rows:
            prompt_ids = tok(r["input"], add_special_tokens=False)["input_ids"]
            target_ids = tok(" " + r["output"] + eos, add_special_tokens=False)["input_ids"]
            ids = (prompt_ids + target_ids)[:MAX_LEN]
            labels = ([-100] * len(prompt_ids) + target_ids)[:MAX_LEN]
            pad = MAX_LEN - len(ids)
            attn = [1] * len(ids) + [0] * pad
            ids += [tok.pad_token_id] * pad
            labels += [-100] * pad
            self.items.append((ids, attn, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, attn, labels = self.items[i]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(attn),
                "labels": torch.tensor(labels)}


@torch.no_grad()
def val_loss(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        total += model(**batch).loss.item()
        n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(ROOT / "models" / "kogpt2-base-v2"))
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "synth"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    if not Path(args.base).exists():
        fallback = Path("models/kogpt2-base-v2")
        args.base = str(fallback) if fallback.exists() else "skt/kogpt2-base-v2"
        print(f"베이스 모델 경로 폴백: {args.base}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    # SKT KoGPT2 공식 로딩 방식 — 특수 토큰을 명시하지 않으면 GPT2 기본 토큰이
    # 어휘 범위(51200) 밖에 추가돼 임베딩 오류가 난다
    tok = PreTrainedTokenizerFast.from_pretrained(
        args.base, bos_token="</s>", eos_token="</s>", unk_token="<unk>",
        pad_token="<pad>", mask_token="<mask>")
    data = {s: read_jsonl(Path(args.data_dir) / f"gpt_{s}.jsonl") for s in ("train", "val", "test")}
    if args.max_samples:
        data = {s: rows[: args.max_samples] for s, rows in data.items()}

    loaders = {s: DataLoader(PairDataset(rows, tok), batch_size=args.batch, shuffle=(s == "train"))
               for s, rows in data.items()}

    model = AutoModelForCausalLM.from_pretrained(args.base).to(device)
    steps = len(loaders["train"]) * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = get_linear_schedule_with_warmup(opt, int(steps * 0.05), steps)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(loaders["train"], 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            running += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(loaders['train'])} loss={running/step:.4f}", flush=True)
        print(f"epoch {epoch} | val_loss={val_loss(model, loaders['val'], device):.4f}", flush=True)

    print(f"test_loss={val_loss(model, loaders['test'], device):.4f}")

    # 생성 샘플 확인 (보고서 캡처용)
    model.eval()
    for r in data["test"][:3]:
        enc = tok(r["input"], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=120, do_sample=False,
                                 no_repeat_ngram_size=3, pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"\nIN : {r['input'][:110]}\nGEN: {gen[:160]}\nREF: {r['output'][:160]}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"\n저장: {out_dir}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
