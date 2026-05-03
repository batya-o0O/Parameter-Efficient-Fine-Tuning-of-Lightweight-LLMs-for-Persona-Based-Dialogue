#!/usr/bin/env python3
import os
import math
import argparse
import importlib.util
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    p.add_argument("--dataset_repo", type=str, default="ZenMoore/RoleBench")

    # Full dataset split (you can change to other RoleBench subsets later)
    p.add_argument("--train_file", type=str, default="rolebench-eng/role-generalization/general/train.jsonl")
    p.add_argument("--eval_file", type=str, default="rolebench-eng/role-generalization/general/test.jsonl")

    p.add_argument("--output_dir", type=str, default="./outputs/phi3_rolebench_phase1_full")
    p.add_argument("--cache_dir", type=str, default=os.path.expanduser("~/hf_cache"))
    p.add_argument("--datasets_cache", type=str, default=os.path.expanduser("~/hf_datasets"))

    # Training
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--max_length", type=int, default=1024)

    # Checkpointing / logging
    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--dataloader_workers", type=int, default=4)
    p.add_argument("--packing", action="store_true", help="Enable sequence packing for higher throughput")
    p.add_argument("--group_by_length", action="store_true", help="Group samples by length to reduce padding")

    # Precision
    p.add_argument("--bf16", action="store_true", help="Use bf16 mixed precision (recommended on H100)")
    p.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision")

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Optional: tensorboard
    p.add_argument("--report_to", type=str, default="none", choices=["none", "tensorboard"])

    return p.parse_args()


def pick_answer(ex):
    gen = ex.get("generated", [])
    if isinstance(gen, list) and len(gen) > 0:
        return gen[0]
    return ""


def to_text(ex):
    # RoleBench fields: role, question, generated(list) are typical
    role = ex.get("role", "UnknownRole")
    q = ex.get("question", "")
    a = pick_answer(ex)

    text = (
        f"<|system|>\nYou are role-playing as {role}. Stay in character.\n<|end|>\n"
        f"<|user|>\n{q}\n<|end|>\n"
        f"<|assistant|>\n{a}\n<|end|>\n"
    )
    return {"text": text}


def main():
    args = parse_args()

    # ---- Cache dirs ----
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.datasets_cache, exist_ok=True)
    os.environ["HF_HOME"] = args.cache_dir
    os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
    os.environ["HF_DATASETS_CACHE"] = args.datasets_cache

    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # ---- Load dataset (full split) ----
    raw = load_dataset(
        args.dataset_repo,
        data_files={"train": args.train_file, "test": args.eval_file},
        cache_dir=args.datasets_cache,
    )

    # Map to plain-text SFT format (robust with TRL 0.27)
    ds = raw.map(to_text, remove_columns=raw["train"].column_names)

    print(ds)
    print("Sample text:\n", ds["train"][0]["text"][:400])

    # ---- Quantization config (QLoRA) ----
    # Use bf16 compute on H100 if you pass --bf16, otherwise fp16
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    # For H100: bf16 is ideal. For fp16: also ok.
    model_dtype = torch.bfloat16 if args.bf16 else torch.float16
    has_flash_attn = importlib.util.find_spec("flash_attn") is not None
    attn_impl = "flash_attention_2" if has_flash_attn else "eager"

    # Packing is only safe with flash attention variants that support flattened sequences.
    if args.packing and not has_flash_attn:
        print("flash-attn is not installed; disabling --packing for safety.")
        args.packing = False

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_cfg,
        dtype=model_dtype,
        attn_implementation=attn_impl,
    )

    model = prepare_model_for_kbit_training(model)

    # ---- LoRA config (explicit target modules) ----
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # ---- Training config ----
    os.makedirs(args.output_dir, exist_ok=True)

    # Mixed precision flags:
    # On H100: prefer bf16. Set either bf16 OR fp16, not both.
    use_bf16 = bool(args.bf16)
    use_fp16 = bool(args.fp16) and not use_bf16
    report_to = args.report_to

    # Gracefully disable tensorboard logging if the package is unavailable.
    if report_to == "tensorboard":
        has_tb = (importlib.util.find_spec("tensorboard") is not None) or (
            importlib.util.find_spec("tensorboardX") is not None
        )
        if not has_tb:
            print("TensorBoard not installed; falling back to --report_to none")
            report_to = "none"

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_length=args.max_length,

        logging_steps=args.logging_steps,

        # Checkpoints + eval
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,

        # Robust optimizer for QLoRA
        optim="paged_adamw_8bit",

        # precision
        bf16=use_bf16,
        fp16=use_fp16,

        # dataset field name
        dataset_text_field="text",
        dataloader_num_workers=args.dataloader_workers,
        packing=args.packing,
        group_by_length=args.group_by_length,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Logging destinations
        report_to=report_to,

        # Safety
        max_grad_norm=1.0,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        peft_config=lora_cfg,
        processing_class=tokenizer,
    )

    # ---- Estimate steps (nice to print) ----
    train_len = len(ds["train"])
    eff_batch = args.batch_size * args.grad_accum
    steps_per_epoch = math.ceil(train_len / eff_batch)
    total_steps = int(steps_per_epoch * args.epochs)
    print(f"Train examples: {train_len}")
    print(f"Effective batch size: {eff_batch}")
    print(f"Steps/epoch: {steps_per_epoch}, total steps ~ {total_steps}")

    # ---- Train ----
    if args.resume_from_checkpoint:
        print("Resuming from checkpoint:", args.resume_from_checkpoint)
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    trainer.save_model(args.output_dir)
    print("Final saved to:", args.output_dir)


if __name__ == "__main__":
    main()
