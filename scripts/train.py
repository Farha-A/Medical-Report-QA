"""LoRA fine-tune MedGemma on MIMIC-CXR image/report pairs."""
from __future__ import annotations

import logging
import warnings

warnings.filterwarnings("ignore", message=r".*Accessing `__path__` from.*")


class _DropPathDeprecation(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Accessing `__path__` from" not in record.getMessage()


logging.getLogger("transformers").addFilter(_DropPathDeprecation())

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

PROMPT = (
    "You are a radiologist. Examine the chest X-ray and write a concise "
    "report covering FINDINGS and IMPRESSION."
)


class CXRDataset(Dataset):
    def __init__(self, manifest_path: str, processor, max_length: int = 512):
        self.records = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image = Image.open(rec["image"]).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": rec["report"]}],
            },
        ]
        batch = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        item = {k: v.squeeze(0) for k, v in batch.items()}
        labels = item["input_ids"].clone()
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        item["labels"] = labels
        return item


def collate(batch, pad_id: int):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        tensors = [b[k] for b in batch]
        if k in {"input_ids", "labels", "attention_mask"}:
            max_len = max(t.shape[-1] for t in tensors)
            padded = []
            for t in tensors:
                pad_amt = max_len - t.shape[-1]
                if pad_amt > 0:
                    fill = -100 if k == "labels" else (0 if k == "attention_mask" else pad_id)
                    t = torch.cat([t, torch.full((pad_amt,), fill, dtype=t.dtype)])
                padded.append(t)
            out[k] = torch.stack(padded)
        else:
            out[k] = torch.stack(tensors)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/manifest.json")
    p.add_argument("--model", default="google/medgemma-4b-it")
    p.add_argument("--output_dir", default="checkpoints/medgemma-cxr-lora")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_samples", type=int, default=0, help="0 = use all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true", help="force CPU even when CUDA is available")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    processor = AutoProcessor.from_pretrained(args.model)
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            quantization_config=bnb,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        print("[train] No CUDA — running on CPU in float32. Smoke-test only; use --max_samples 4-16.")
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        model.to("cpu")

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    dataset = CXRDataset(args.manifest, processor)
    if args.max_samples > 0:
        idxs = list(range(len(dataset)))
        random.shuffle(idxs)
        dataset.records = [dataset.records[i] for i in idxs[: args.max_samples]]

    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=use_cuda,
        fp16=False,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        gradient_checkpointing=use_cuda,
        remove_unused_columns=False,
        report_to="none",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda b: collate(b, pad_id),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
