import argparse
import gc
import importlib.util
import os
import torch
from datasets import load_dataset
from src.utils.io import load_json, ensure_dir, save_json
from src.utils.logging import setup_logger

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def _cluster_sort_key(v):
    return int(v) if str(v).isdigit() else str(v)


def parse_cluster_ids(value):
    if not value:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def pick_answer(ex):
    gen = ex.get("generated", [])
    if isinstance(gen, list) and len(gen) > 0:
        return gen[0]
    return ""

def to_text(ex):
    role = ex.get("role", "UnknownRole")
    q = ex.get("question", "")
    a = pick_answer(ex)
    text = (
        f"<|system|>\nYou are role-playing as {role}. Stay in character.\n<|end|>\n"
        f"<|user|>\n{q}\n<|end|>\n"
        f"<|assistant|>\n{a}\n<|end|>\n"
    )
    return {"text": text}

def train_one(cluster_id, roles, cfg, logger, raw):
    roles_set = set(roles)
    roles_list = sorted(roles_set)
    out_dir = os.path.join(cfg["output_root"], f"cluster_{cluster_id}")
    ensure_dir(out_dir)

    ds = raw.filter(lambda x: x["role"] in roles_set)
    ds = ds.map(to_text, remove_columns=raw["train"].column_names)

    train_count = len(ds["train"])
    eval_count = len(ds["test"])
    if train_count == 0:
        logger.warning(f"Cluster {cluster_id}: no train rows after filtering; skipping.")
        return

    logger.info(
        f"Cluster {cluster_id}: roles={len(roles_list)} train={train_count} eval={eval_count}"
    )
    if eval_count == 0:
        logger.warning(
            f"Cluster {cluster_id}: no eval rows after filtering; training will continue without evaluation."
        )

    use_bf16 = bool(cfg.get("use_bf16", True))
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16
    report_to = cfg.get("report_to", "none")
    has_tb = (importlib.util.find_spec("tensorboard") is not None) or (
        importlib.util.find_spec("tensorboardX") is not None
    )
    if report_to == "tensorboard" and not has_tb:
        logger.info("TensorBoard not installed; falling back to report_to=none")
        report_to = "none"

    has_flash_attn = importlib.util.find_spec("flash_attn") is not None
    attn_impl = "flash_attention_2" if has_flash_attn else "eager"
    use_packing = bool(cfg.get("packing", False))
    if use_packing and not has_flash_attn:
        logger.info("flash-attn missing; disabling packing for safety")
        use_packing = False

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_cfg,
        dtype=model_dtype,
        attn_implementation=attn_impl,
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    sft_kwargs = dict(
        output_dir=out_dir,
        per_device_train_batch_size=int(cfg["batch_size"]),
        gradient_accumulation_steps=int(cfg["grad_accum"]),
        learning_rate=float(cfg["lr"]),
        num_train_epochs=float(cfg["epochs"]),
        max_length=int(cfg["max_length"]),
        logging_steps=int(cfg["logging_steps"]),
        save_steps=int(cfg["save_steps"]),
        save_strategy="steps",
        save_total_limit=int(cfg["save_total_limit"]),
        dataset_text_field="text",
        dataloader_num_workers=int(cfg["dataloader_workers"]),
        packing=use_packing,
        group_by_length=bool(cfg["group_by_length"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        bf16=use_bf16,
        fp16=(not use_bf16),
        report_to=report_to,
        max_grad_norm=1.0,
    )
    if eval_count > 0:
        sft_kwargs["eval_steps"] = int(cfg["eval_steps"])
        sft_kwargs["eval_strategy"] = "steps"
        eval_dataset = ds["test"]
    else:
        sft_kwargs["eval_strategy"] = "no"
        eval_dataset = None

    sft_args = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds["train"],
        eval_dataset=eval_dataset,
        peft_config=lora_cfg,
        processing_class=tok,
    )
    save_json(
        os.path.join(out_dir, "cluster_meta.json"),
        {"cluster_id": cluster_id, "roles": roles_list, "train_rows": train_count, "eval_rows": eval_count, "cfg": cfg},
    )
    trainer.train()
    trainer.save_model(out_dir)
    del trainer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", required=True)  # data/clusters_k10.json
    ap.add_argument("--output_root", default="outputs/phase2_clusters_k10")
    ap.add_argument("--model_id", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--dataset_repo", default="ZenMoore/RoleBench")
    ap.add_argument("--train_file", default="rolebench-eng/role-generalization/general/train.jsonl")
    ap.add_argument("--eval_file", default="rolebench-eng/role-generalization/general/test.jsonl")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--logging_steps", type=int, default=25)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--eval_steps", type=int, default=500)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--dataloader_workers", type=int, default=4)
    ap.add_argument("--packing", action="store_true")
    ap.add_argument("--group_by_length", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--use_bf16", action="store_true")
    ap.add_argument("--report_to", default="none", choices=["none","tensorboard"])
    ap.add_argument(
        "--cluster_ids",
        type=str,
        default="",
        help="Comma-separated cluster ids to train (default: all clusters). Example: 0,3,9",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_root)
    logger = setup_logger(os.path.join(args.output_root, "train_clusters.log"), name="cluster-train")

    data = load_json(args.clusters)
    clusters = data["clusters"]
    selected_cluster_ids = parse_cluster_ids(args.cluster_ids)
    if selected_cluster_ids:
        missing = [cid for cid in selected_cluster_ids if cid not in clusters]
        if missing:
            raise ValueError(f"Requested cluster_ids not found: {missing}")
        cluster_ids = selected_cluster_ids
    else:
        cluster_ids = sorted(clusters.keys(), key=_cluster_sort_key)

    logger.info(f"Training clusters: {cluster_ids}")
    raw = load_dataset(
        args.dataset_repo,
        data_files={"train": args.train_file, "test": args.eval_file},
        cache_dir=os.environ.get("HF_DATASETS_CACHE", None),
    )

    cfg = vars(args)
    cfg["output_root"] = args.output_root

    for cluster_id in cluster_ids:
        members = clusters[cluster_id]
        roles = [m["persona_id"] for m in members]  # role names
        train_one(cluster_id, roles, cfg, logger, raw)

if __name__ == "__main__":
    main()
